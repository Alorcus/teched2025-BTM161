import logging

from langchain_core.messages import HumanMessage, ToolMessage

logger = logging.getLogger("coffee_shop.context_isolation")

AGENT_TO_HANDOFF_TOOL = {
    "order_agent": "transfer_to_order_agent",
    "inventory_agent": "transfer_to_inventory",
    "barista_agent": "transfer_to_barista",
    "customer_service_agent": "transfer_to_customer_service",
}


def _extract_current_turn_messages(messages: list, agent_name: str) -> list:
    """Extract messages belonging to this agent's current turn.

    Scans backward from the end to find the last handoff boundary (a ToolMessage
    from the transfer tool that routes to this agent). Returns all messages after
    that boundary. If no boundary is found, returns all messages (entry agent case).
    """
    handoff_tool_name = AGENT_TO_HANDOFF_TOOL.get(agent_name, f"transfer_to_{agent_name}")
    boundary_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "") or ""
            if name == handoff_tool_name:
                boundary_idx = i
                break

    if boundary_idx >= 0:
        return list(messages[boundary_idx + 1:])
    return list(messages)


def create_context_isolation_hook(agent_name: str):
    """Create a pre_model_hook that gives each agent only its relevant context.

    The agent's LLM receives:
    - A synthetic briefing message (from the handoff context), if this agent was
      entered via a handoff
    - All messages from this agent's current turn (after the handoff boundary)

    For the entry agent (no handoff), all messages are passed through directly.
    """
    def hook(state):
        messages = state.get("messages", [])
        handoff_context = state.get("handoff_context", None)

        logger.debug("%s: %d own messages, handoff_context=%s",
                     agent_name, len(messages),
                     handoff_context.get("from_agent") if isinstance(handoff_context, dict) else None)

        own_messages = _extract_current_turn_messages(messages, agent_name)

        if isinstance(handoff_context, dict) and handoff_context.get("from_agent"):
            briefing = HumanMessage(content=(
                f"[Handoff from {handoff_context['from_agent']}]\n"
                f"Context: {handoff_context['context_summary']}\n"
                f"Your task: {handoff_context['expectation']}"
            ))
            return {"llm_input_messages": [briefing] + own_messages}

        return {"llm_input_messages": own_messages}

    return hook
