"""Per-agent guarded subgraph that replaces `create_react_agent`.

Topology:
    START → llm → response_gateway → (cond)
                                     ├─ correction issued → llm (loop with corrective HumanMessage)
                                     └─ passed          → (cond)
                                                          ├─ no tool_calls   → END
                                                          └─ has tool_calls  → gateway → (cond)
                                                                                         ├─ batch-denied → llm (loop, synthetic ToolMessages for every tool_call_id)
                                                                                         └─ batch-allowed → tools → llm (loop)

The response_gateway inspects the LLM's assistant message text (before any
tool_call routing) against response-scoped guardrails — the ones that declare
themselves for the synthetic `assistant_message` "tool". Detected off-menu
recommendations are blocked at the graph layer: the offending AIMessage is
removed from state, a corrective HumanMessage is appended, and the LLM
re-runs. A per-turn retry cap prevents infinite loops.

Batch-verdict policy (all-or-nothing per LLM turn): per-call verdicts are still
evaluated and logged individually, but if ANY call in the batch is denied the
entire batch is rejected — synthetic ToolMessages are emitted for every
tool_call_id in the AIMessage so the Anthropic tool_use↔tool_result invariant
holds, and control returns to the LLM with denial reasons. Only when every
proposed call is allowed (or flagged) does the batch reach `tools`.
"""
import logging
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.context_isolation import create_context_isolation_hook
from src.agents.shared_components import CoffeeShopState
from src.llm import bind_tools_sequential

from .gateway import Gateway
from .types import Effect

logger = logging.getLogger("coffee_shop.control_plane.subgraph")

_RESPONSE_GUARDRAIL_TOOL_NAME = "assistant_message"
_CORRECTION_KWARG = "response_guardrail_correction"
_MAX_RESPONSE_GUARDRAIL_RETRIES = 3


def _last_ai_with_tool_calls(messages) -> AIMessage | None:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            return m
    return None


def _thread_id_of(config: RunnableConfig | None) -> str | None:
    if not config:
        return None
    return (config.get("configurable") or {}).get("thread_id")


def _is_correction_message(msg) -> bool:
    return (
        isinstance(msg, HumanMessage)
        and (msg.additional_kwargs or {}).get(_CORRECTION_KWARG) is True
    )


def _corrections_since_last_user_turn(messages) -> int:
    """Count corrective HumanMessages appended after the most recent genuine user
    HumanMessage. This is the retry-budget counter — a new user turn implicitly
    resets it to zero because we only walk the tail up to the last non-correction
    HumanMessage.
    """
    count = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and not _is_correction_message(msg):
            break
        if _is_correction_message(msg):
            count += 1
    return count


def create_agent_subgraph(
    agent_id: str,
    llm,
    tools,
    prompt: str,
    gateway: Gateway,
):
    """Build a compiled per-agent subgraph wired with the given Gateway."""
    llm_with_tools = bind_tools_sequential(llm, tools)
    context_hook = create_context_isolation_hook(agent_id)
    tool_node = ToolNode(tools)

    def llm_node(state: CoffeeShopState, config: RunnableConfig):
        hook_update = context_hook(state)
        llm_input = hook_update.get("llm_input_messages") or state.get("messages", [])
        sys_msgs = [SystemMessage(content=prompt)] if prompt else []
        ai = llm_with_tools.invoke(sys_msgs + list(llm_input), config=config)
        # create_react_agent set this for us; we need to set it explicitly so
        # _resolve_from_agent and context-isolation boundary detection keep working.
        ai.name = agent_id
        return {"messages": [ai]}

    def route_after_llm(state: CoffeeShopState) -> str:
        last = state["messages"][-1] if state.get("messages") else None
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "gateway"
        return END

    def response_gateway_node(state: CoffeeShopState, config: RunnableConfig):
        """Evaluate the outgoing AIMessage against response-scoped guardrails.

        Synthesizes a pseudo tool call `assistant_message` whose args carry the
        message content. Reuses `Gateway.evaluate_call` so the verdict, JSONL log
        record, and OCEL projection are identical to any other guardrail decision.
        On DENY (under the retry cap): removes the offending AIMessage from state
        and appends a corrective HumanMessage carrying the guardrail's
        `reason_for_llm`. On ALLOW/FLAG or cap-exhausted: leaves state untouched.
        """
        messages = state.get("messages", [])
        if not messages:
            return {}
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return {}

        applicable = [
            gr for gr in gateway.guardrails
            if gr.applies_to(_RESPONSE_GUARDRAIL_TOOL_NAME)
        ]
        if not applicable:
            return {}

        content = last.content if isinstance(last.content, str) else ""
        synthetic_call = {
            "name": _RESPONSE_GUARDRAIL_TOOL_NAME,
            "args": {"content": content},
            "id": f"resp-{getattr(last, 'id', 'unknown')}",
        }
        decision = gateway.evaluate_call(
            synthetic_call, dict(state), thread_id=_thread_id_of(config),
        )

        if decision.final_decision != Effect.DENY:
            return {}

        prior_corrections = _corrections_since_last_user_turn(messages)
        if prior_corrections >= _MAX_RESPONSE_GUARDRAIL_RETRIES:
            logger.warning(
                "response guardrail retry cap (%d) reached for %s; publishing last attempt",
                _MAX_RESPONSE_GUARDRAIL_RETRIES, agent_id,
            )
            return {}

        correction_text = (
            decision.deny_reason_for_llm
            or "Your last message violated a response guardrail. Try again."
        )
        correction = HumanMessage(
            content=correction_text,
            additional_kwargs={_CORRECTION_KWARG: True},
        )
        target_id = getattr(last, "id", None)
        patch: list = [correction]
        if target_id is not None:
            patch.insert(0, RemoveMessage(id=target_id))
        return {"messages": patch}

    def route_after_response_gateway(state: CoffeeShopState) -> str:
        last = state["messages"][-1] if state.get("messages") else None
        if _is_correction_message(last):
            return "llm"
        return route_after_llm(state)

    def gateway_node(state: CoffeeShopState, config: RunnableConfig):
        ai = _last_ai_with_tool_calls(state.get("messages", []))
        if ai is None:
            return {}
        thread_id = _thread_id_of(config)

        decisions = [
            gateway.evaluate_call(tc, dict(state), thread_id=thread_id)
            for tc in ai.tool_calls
        ]
        any_denied = any(d.final_decision == Effect.DENY for d in decisions)
        if not any_denied:
            # All allowed — no state update; tools_node reads state.messages directly.
            return {}

        # Batch denied: one synthetic ToolMessage per tool_call_id.
        sibling_reasons = [
            d.deny_reason_for_llm for d in decisions if d.final_decision == Effect.DENY
        ]
        sibling_blurb = "; ".join(sibling_reasons)
        synth: list[ToolMessage] = []
        for d in decisions:
            if d.final_decision == Effect.DENY:
                content = d.deny_reason_for_llm or "Tool call denied by guardrail."
            else:
                content = (
                    f"Tool call {d.tool_name!r} was not executed because a sibling tool "
                    f"call in the same batch was denied by a guardrail: {sibling_blurb}"
                )
            synth.append(ToolMessage(
                content=content,
                name=d.tool_name,
                tool_call_id=d.tool_call_id,
                status="error",
            ))
        return {"messages": synth}

    def route_after_gateway(state: CoffeeShopState) -> str:
        # If the last message is an AIMessage with un-answered tool_calls, route to tools;
        # otherwise (gateway emitted synthetic ToolMessages) route back to llm.
        msgs = state.get("messages", [])
        if not msgs:
            return END
        last = msgs[-1]
        if isinstance(last, ToolMessage):
            return "llm"
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    g = StateGraph(CoffeeShopState)
    g.add_node("llm", llm_node)
    g.add_node("response_gateway", response_gateway_node)
    g.add_node("gateway", gateway_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "llm")
    g.add_edge("llm", "response_gateway")
    g.add_conditional_edges(
        "response_gateway",
        route_after_response_gateway,
        {"llm": "llm", "gateway": "gateway", END: END},
    )
    g.add_conditional_edges("gateway", route_after_gateway, {"tools": "tools", "llm": "llm", END: END})
    g.add_edge("tools", "llm")
    return g.compile()
