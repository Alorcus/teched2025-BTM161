"""Per-agent guarded subgraph that replaces `create_react_agent`.

Topology:
    START → llm → (cond)
                  ├─ no tool_calls   → END
                  └─ has tool_calls  → gateway → (cond)
                                                 ├─ batch-denied → llm (loop, synthetic ToolMessages for every tool_call_id)
                                                 └─ batch-allowed → tools → llm (loop)

Batch-verdict policy (all-or-nothing per LLM turn): per-call verdicts are still
evaluated and logged individually, but if ANY call in the batch is denied the
entire batch is rejected — synthetic ToolMessages are emitted for every
tool_call_id in the AIMessage so the Anthropic tool_use↔tool_result invariant
holds, and control returns to the LLM with denial reasons. Only when every
proposed call is allowed (or flagged) does the batch reach `tools`.
"""

import logging
from typing import Callable

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.context_isolation import create_context_isolation_hook
from src.agents.shared_components import CoffeeShopState
from src.llm import bind_tools_sequential

from .gateway import Gateway
from .types import Effect

logger = logging.getLogger("coffee_shop.control_plane.subgraph")


def _last_ai_with_tool_calls(messages) -> AIMessage | None:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            return m
    return None


def _thread_id_of(config: RunnableConfig | None) -> str | None:
    if not config:
        return None
    return (config.get("configurable") or {}).get("thread_id")


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
            synth.append(
                ToolMessage(
                    content=content,
                    name=d.tool_name,
                    tool_call_id=d.tool_call_id,
                    status="error",
                )
            )
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

    def tools_node_wrapped(state: CoffeeShopState, config: RunnableConfig):
        # Log tool_execution for each call before running. Result preview comes
        # post-execution; for MVP we log the args here and rely on MLflow span
        # mirroring later for the actual outputs.
        ai = _last_ai_with_tool_calls(state.get("messages", []))
        thread_id = _thread_id_of(config)
        if ai is not None:
            for tc in ai.tool_calls:
                gateway.log_tool_execution(
                    tool_call_id=tc.get("id", ""),
                    tool_name=tc.get("name", ""),
                    tool_args=dict(tc.get("args", {})),
                    result_preview="(see MLflow trace for tool output)",
                    thread_id=thread_id,
                )
        return tool_node.invoke(state, config=config)

    g = StateGraph(CoffeeShopState)
    g.add_node("llm", llm_node)
    g.add_node("gateway", gateway_node)
    g.add_node("tools", tools_node_wrapped)
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", route_after_llm, {"gateway": "gateway", END: END})
    g.add_conditional_edges(
        "gateway", route_after_gateway, {"tools": "tools", "llm": "llm", END: END}
    )
    g.add_edge("tools", "llm")
    return g.compile()
