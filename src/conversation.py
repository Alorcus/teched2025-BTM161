import logging
import uuid
from typing import Callable

import mlflow

from src.agents import reset_inventory
from src.agents.customer_agent import CustomerAgent
from src.agents.tray import get_tray, clear_tray
from src.agents.order_store import load_recent_order, save_order
from src.agents.shared_components import OrderStatus
from src.stream import extract_messages

logger = logging.getLogger("coffee_shop.conversation")


class ConversationEngine:
    """Headless conversation runner for the coffee shop multi-agent system."""

    def __init__(self, app, mlflow_enabled=True):
        self.app = app
        self.mlflow_enabled = mlflow_enabled
        self.traces_of_latest_conversations: list[str] = []

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

        message = customer_agent.get_initial_message()
        if on_message:
            on_message("customer", message)

        while message:
            agent_reply = self.send_message(thread_id, message)
            if on_message and agent_reply:
                on_message("agent", agent_reply)

            if not agent_reply:
                break

            message = customer_agent.respond_to(agent_reply)
            if on_message and message:
                on_message("customer", message)

        self._consume_tray(customer_agent)

        return self.traces_of_latest_conversations[trace_start:]

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
            order.status = OrderStatus.COMPLETED
            save_order(order)

        clear_tray(order_id)
