import json
import logging
import re
import uuid
from pathlib import Path
from typing import Callable

import mlflow

from src.agents import reset_inventory
from src.agents.customer_agent import CustomerAgent
from src.agents.tray import get_tray, clear_tray
from src.agents.order_store import load_recent_order
from src.agents.shared_components import OrderStatus
from src.agents.order_state_machine import state_machine, InvalidTransitionError
from src.stream import extract_messages

logger = logging.getLogger("coffee_shop.conversation")

FEEDBACK_STORE_PATH = Path("./feedback_store.json")


class ConversationEngine:
    """Headless conversation runner for the coffee shop multi-agent system."""

    def __init__(self, app, mlflow_enabled=True, retrospective=None,
                 supervisor_log_path: str | None = None):
        self.app = app
        self.mlflow_enabled = mlflow_enabled
        self.traces_of_latest_conversations: list[str] = []
        self.feedback_log: dict[str, dict] = {}
        self.retrospective = retrospective
        self.supervisor_log_path = supervisor_log_path
        # Per-conversation transcript: list of (agent_name, content) tuples in
        # turn order. Used to build per-agent views for the retrospective.
        self._transcript: list[tuple[str, str]] = []

    def _get_config(self, thread_id):
        return {"configurable": {"thread_id": thread_id}}

    def send_message(self, thread_id: str, message: str) -> str | None:
        """Send a message through the swarm and return the last customer-facing agent response."""
        config = self._get_config(thread_id)
        last_agent_message = None

        stream = self.app.stream(
            {"messages": [{"role": "user", "content": message}], "handoff_context": None},
            config,
            subgraphs=True,
        )
        for sm in extract_messages(stream):
            if sm.content:
                self._transcript.append((sm.agent_name or "unknown", sm.content))
            if sm.is_agent_reply:
                last_agent_message = sm.content

        if self.mlflow_enabled:
            trace_id = mlflow.get_last_active_trace_id()
            self.traces_of_latest_conversations.append(trace_id)

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
        self._transcript = []

        message = customer_agent.get_initial_message()
        if on_message:
            on_message("customer", message)
        self._transcript.append(("customer", message))

        while message:
            agent_reply = self.send_message(thread_id, message)
            if on_message and agent_reply:
                on_message("agent", agent_reply)

            if not agent_reply:
                break

            message = customer_agent.respond_to(agent_reply)
            if on_message and message:
                on_message("customer", message)
            if message:
                self._transcript.append(("customer", message))

        self._consume_tray(customer_agent)

        order_id = _extract_order_id_from_history(customer_agent.history)
        feedback = customer_agent.get_feedback()
        self.feedback_log[thread_id] = {"thread_id": thread_id, "order_id": order_id, **feedback}
        self._save_feedback_store()
        logger.info(
            "Customer feedback [%.2f]: %s", feedback["feedback_score"], feedback["feedback_reason"]
        )

        if self.retrospective is not None:
            try:
                agents, transcripts = _build_retrospective_views(
                    self._transcript, self.supervisor_log_path
                )
                self.retrospective.run(thread_id, agents, transcripts)
            except Exception:
                logger.exception("retrospective failed; continuing")

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

        if order.status != OrderStatus.COMPLETED:
            try:
                state_machine.transition(order, OrderStatus.COMPLETED, context="tray pickup by customer")
            except InvalidTransitionError:
                pass

        clear_tray(order_id)


def _extract_order_id_from_history(history: list) -> str | None:
    """Return the last order ID (e.g. ORD0001) mentioned in agent messages."""
    order_id = None
    for role, content in history:
        if role == "agent":
            match = re.search(r"ORD\d{4}", content)
            if match:
                order_id = match.group()
    return order_id


# Operator agents that can receive a retrospective. Customer is excluded — its
# feedback is captured separately via CustomerAgent.get_feedback(). The process
# supervisor receives a different transcript view (its own critique log tail),
# handled in _build_retrospective_views.
_OPERATOR_AGENTS = ("order_agent", "inventory_agent", "barista_agent", "customer_service_agent")


def _build_retrospective_views(
    transcript: list[tuple[str, str]],
    supervisor_log_path: str | None,
) -> tuple[list[str], dict[str, str]]:
    """Build per-agent transcript views for the retrospective.

    Each operator agent sees only the messages it produced and the customer
    messages that immediately preceded its turns, so it has enough context to
    explain its own actions without seeing other agents' reasoning.

    The process supervisor sees the tail of its own critique log
    (process_meta.log) — its decisions about activities and violations — not
    the conversation transcript, since its job is to judge the process flow.
    """
    agents: list[str] = []
    views: dict[str, str] = {}

    seen_agents = {a for a, _ in transcript if a in _OPERATOR_AGENTS}
    for agent in _OPERATOR_AGENTS:
        if agent not in seen_agents:
            continue
        lines: list[str] = []
        last_customer: str | None = None
        for who, content in transcript:
            if who == "customer":
                last_customer = content
                continue
            if who == agent:
                if last_customer is not None:
                    lines.append(f"Customer: {last_customer}")
                    last_customer = None
                lines.append(f"You ({agent}): {content}")
        if lines:
            agents.append(agent)
            views[agent] = "\n".join(lines)

    if supervisor_log_path:
        tail = _read_log_tail(Path(supervisor_log_path), max_lines=80)
        if tail:
            agents.append("process_supervisor")
            views["process_supervisor"] = tail

    return agents, views


def _read_log_tail(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return ""
    return "".join(lines[-max_lines:]).strip()
