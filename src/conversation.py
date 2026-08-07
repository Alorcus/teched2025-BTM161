import json
import logging
import re
import uuid
from pathlib import Path
from typing import Callable

import mlflow

from src.agents import reset_inventory
from src.agents.context_isolation import ContextOverflowError
from src.agents.customer_agent import CustomerAgent
from src.agents.tray import get_tray, clear_tray
from src.agents.order_store import load_recent_order, set_order_status
from src.agents.shared_components import OrderStatus
from src.stream import extract_messages

logger = logging.getLogger("coffee_shop.conversation")

FEEDBACK_STORE_PATH = Path("./feedback_store.json")


class ConversationEngine:
    """Headless conversation runner for the coffee shop multi-agent system."""

    def __init__(
        self,
        app,
        mlflow_enabled=True,
        setup_name: str | None = None,
    ):
        self.app = app
        self.mlflow_enabled = mlflow_enabled
        self.setup_name = setup_name
        self.traces_of_latest_conversations: list[str] = []
        self.feedback_log: dict[str, dict] = {}

    def _get_config(self, thread_id):
        return {"configurable": {"thread_id": thread_id}}

    def send_message(
        self, thread_id: str, message: str, scenario_index: int | None = None
    ) -> str | None:
        """Send a message through the swarm and return the last customer-facing agent response.

        When `mlflow_enabled` and `setup_name` are set, tags the produced MLflow
        trace with `setup` and `scenario_index` so downstream analytics can filter
        by them.
        """
        config = self._get_config(thread_id)
        stream_input = {
            "messages": [{"role": "user", "content": message}],
            "handoff_context": None,
        }
        last_agent_message = None
        try:
            for sm in extract_messages(
                self.app.stream(stream_input, config, subgraphs=True)
            ):
                if sm.is_agent_reply:
                    last_agent_message = sm.content
        finally:
            # Tag even when the turn blew up: an abandoned conversation still
            # belongs to its setup/scenario in the event log.
            if self.mlflow_enabled:
                trace_id = mlflow.get_last_active_trace_id()
                self.traces_of_latest_conversations.append(trace_id)
                if trace_id is not None and self.setup_name is not None:
                    _tag_trace(trace_id, self.setup_name, scenario_index)

        return last_agent_message

    def run_automated(
        self,
        customer_agent: CustomerAgent,
        scenario_index=None,
        on_message: Callable[[str, str], None] | None = None,
    ) -> list[str]:
        """Run a full automated conversation using the CustomerAgent.

        Returns the list of trace IDs collected during this conversation.
        """
        reset_inventory()
        customer_agent.reset(scenario_index)
        thread_id = str(uuid.uuid4())
        trace_start = len(self.traces_of_latest_conversations)

        message = customer_agent.get_initial_message()
        if on_message:
            on_message("customer", message)

        abort_reason = None
        while message:
            try:
                agent_reply = self.send_message(
                    thread_id, message, scenario_index=customer_agent.scenario_index
                )
            except ContextOverflowError as exc:
                abort_reason = str(exc)
                logger.error(f"Conversation {thread_id} abandoned: {exc}")
                break

            if on_message and agent_reply:
                on_message("agent", agent_reply)

            if not agent_reply:
                break

            message = customer_agent.respond_to(agent_reply)
            if on_message and message:
                on_message("customer", message)
            elif on_message and customer_agent.last_terminating_message:
                on_message("customer", customer_agent.last_terminating_message)

        # An abandoned conversation never reaches the counter, so the tray is
        # left untouched and its order keeps whatever status it had — a dangling
        # order is the truthful trace of a swarm that never finished.
        if abort_reason is None:
            self._consume_tray(customer_agent)

        order_id = _extract_order_id_from_history(customer_agent.history)
        feedback = (
            _abort_feedback(abort_reason)
            if abort_reason
            else customer_agent.get_feedback()
        )
        self.feedback_log[thread_id] = {
            "thread_id": thread_id,
            "order_id": order_id,
            "scenario_index": getattr(customer_agent, "scenario_index", None),
            **feedback,
        }
        self._save_feedback_store()
        score = feedback["feedback_score"]
        score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
        logger.info(f"Customer feedback [{score_text}]: {feedback['feedback_reason']}")

        return self.traces_of_latest_conversations[trace_start:]

    def _save_feedback_store(self):
        existing = {}
        if FEEDBACK_STORE_PATH.exists():
            with open(FEEDBACK_STORE_PATH) as f:
                existing = json.load(f)
        existing.update(self.feedback_log)
        with open(FEEDBACK_STORE_PATH, "w") as f:
            json.dump(existing, f, indent=2)

    def _consume_tray(self, customer_agent: CustomerAgent):
        """Customer takes the tray — apply effects and mark order complete."""
        order = load_recent_order()
        if not order:
            return
        order_id = order.order_id_str

        tray_items = get_tray(order_id)
        if not tray_items:
            return

        has_contaminated = any(entry.contaminated for entry in tray_items)
        if has_contaminated:
            customer_agent.inject_experience(
                "You received your coffee but it tastes slightly off — almost metallic. Something isn't right."
            )

        if order.status == OrderStatus.IN_PREPARATION:
            set_order_status(
                order, OrderStatus.COMPLETED, context="tray pickup by customer"
            )
        elif order.status != OrderStatus.COMPLETED:
            logger.warning(
                "Tray pickup skipped completion for %s: status=%s not IN_PREPARATION",
                order_id,
                order.status.value,
            )

        clear_tray(order_id)


def _abort_feedback(reason: str) -> dict:
    """Feedback for a conversation abandoned mid-flight.

    A hard 0.0 rather than an LLM judgement: there is no completed interaction
    to judge, and the customer demonstrably never got what they asked for.
    `aborted` lets the simulator count these separately from crashes.
    """
    return {
        "feedback_score": 0.0,
        "feedback_reason": f"Conversation aborted — {reason}",
        "raw_feedback_response": "",
        "valid": True,
        "aborted": True,
    }


def _extract_order_id_from_history(history: list) -> str | None:
    """Return the last order ID (e.g. ORD0001) mentioned in agent messages."""
    order_id = None
    for role, content in history:
        if role == "agent":
            match = re.search(r"ORD\d{4}", content)
            if match:
                order_id = match.group()
    return order_id


def _tag_trace(trace_id: str, setup_name: str, scenario_index: int | None) -> None:
    """Attach `setup` and `scenario_index` tags to an MLflow trace by id.

    Values are cast to strings — MLflow stringifies tag values server-side but
    logs a warning for non-string inputs. `scenario_index=None` is stored as
    "-1" (the "custom / unspecified" sentinel used across the pipeline).
    """
    resolved = -1 if scenario_index is None else int(scenario_index)
    mlflow.set_trace_tag(trace_id, "setup", setup_name)
    mlflow.set_trace_tag(trace_id, "scenario_index", str(resolved))
