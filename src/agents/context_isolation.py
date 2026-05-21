import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

logger = logging.getLogger("coffee_shop.context_isolation")

AGENT_TO_HANDOFF_TOOL = {
    "order_agent": "transfer_to_order_agent",
    "inventory_agent": "transfer_to_inventory",
    "barista_agent": "transfer_to_barista",
    "customer_service_agent": "transfer_to_customer_service",
}


def _find_boundary(messages: list, agent_name: str) -> int:
    """Find the index of the last handoff ToolMessage that routes to this agent.

    Returns -1 if no boundary is found (entry agent case).
    """
    handoff_tool_name = AGENT_TO_HANDOFF_TOOL.get(agent_name, f"transfer_to_{agent_name}")
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "") or ""
            if name == handoff_tool_name:
                return i
    return -1


def _extract_current_turn_messages(messages: list, agent_name: str) -> list:
    """Extract messages belonging to this agent's current turn.

    Scans backward from the end to find the last handoff boundary (a ToolMessage
    from the transfer tool that routes to this agent). Returns all messages after
    that boundary. If no boundary is found, returns all messages (entry agent case).
    """
    boundary_idx = _find_boundary(messages, agent_name)
    if boundary_idx >= 0:
        return list(messages[boundary_idx + 1:])
    return list(messages)


def _extract_handoff_context_from_boundary(messages: list, agent_name: str) -> dict | None:
    """Extract handoff context from the boundary ToolMessage content.

    The transfer tools write: "Successfully transferred to <agent>. Context: <summary>"
    This is the only reliable source when handoff_context isn't in the subgraph state.
    """
    boundary_idx = _find_boundary(messages, agent_name)
    if boundary_idx < 0:
        return None

    boundary_msg = messages[boundary_idx]
    content = getattr(boundary_msg, "content", "") or ""

    # Extract the context from the tool message
    match = re.search(r"Context:\s*(.+)", content)
    if match:
        context_summary = match.group(1).strip()
        # Determine source agent from content
        from_match = re.search(r"transferred to (\w+)", content)
        from_agent = "previous_agent"
        if from_match:
            # The content says "transferred to X" where X is THIS agent,
            # so we need to look at what came before
            pass

        # Look backward to find the AI message that initiated the transfer
        for i in range(boundary_idx - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage):
                from_agent = getattr(msg, "name", "") or "previous_agent"
                break

        return {
            "from_agent": from_agent,
            "context_summary": context_summary,
        }
    return None


def _strip_orphaned_tool_messages(messages: list) -> list:
    """Remove ToolMessages whose tool_call_id has no matching tool_use in a preceding AIMessage.

    The Anthropic API requires every tool_result to reference a tool_use_id from
    the immediately preceding assistant message. When context isolation slices the
    message history, ToolMessages can become orphaned. This function removes them.
    """
    valid_tool_call_ids = set()
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", []):
                if tc.get("id"):
                    valid_tool_call_ids.add(tc["id"])

    result = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tcid = getattr(msg, "tool_call_id", None)
            if tcid and tcid not in valid_tool_call_ids:
                continue
        result.append(msg)
    return result


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

        own_messages = _extract_current_turn_messages(messages, agent_name)
        own_messages = _strip_orphaned_tool_messages(own_messages)

        # Try state-level handoff_context first (works in tests),
        # fall back to extracting from boundary ToolMessage (works in runtime).
        handoff_context = state.get("handoff_context", None)
        if not isinstance(handoff_context, dict) or not handoff_context.get("from_agent"):
            handoff_context = _extract_handoff_context_from_boundary(messages, agent_name)

        logger.debug("%s: %d own messages, handoff_context=%s",
                     agent_name, len(own_messages),
                     handoff_context.get("from_agent") if isinstance(handoff_context, dict) else None)

        if isinstance(handoff_context, dict) and handoff_context.get("from_agent"):
            briefing_parts = [
                f"[Handoff from {handoff_context['from_agent']}]",
                f"Context: {handoff_context['context_summary']}",
            ]
            if handoff_context.get("expectation"):
                briefing_parts.append(f"Your task: {handoff_context['expectation']}")
            briefing = HumanMessage(content="\n".join(briefing_parts))
            return {"llm_input_messages": [briefing] + own_messages}

        # Ensure non-empty: LangGraph falls back to raw state messages when
        # llm_input_messages is empty (falsy). Provide a minimal prompt.
        if not own_messages:
            own_messages = [HumanMessage(content="You have been activated. Proceed with your task.")]

        return {"llm_input_messages": own_messages}

    return hook
