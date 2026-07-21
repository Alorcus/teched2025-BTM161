"""Tests 1-8: Agent handoff mechanics.

Validates handoff tools produce correct Command updates, resolve from_agent,
clear handoff_context between turns, and don't duplicate messages.
"""
import unittest
from unittest.mock import patch, MagicMock

from langgraph.prebuilt.tool_node import ToolNode
from langgraph.runtime import CONF, CONFIG_KEY_RUNTIME, DEFAULT_RUNTIME
from langgraph.types import Command
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from src.agents.shared_components import (
    transfer_to_agent,
    _resolve_from_agent,
)


TOOLS_AND_TARGETS = [
    (transfer_to_agent, "inventory_agent"),
    (transfer_to_agent, "barista_agent"),
    (transfer_to_agent, "customer_service_agent"),
    (transfer_to_agent, "order_agent"),
]


_TOOL_NODE_CONFIG = {CONF: {CONFIG_KEY_RUNTIME: DEFAULT_RUNTIME}}


def _invoke_handoff(tool, target="inventory_agent", from_agent="order_agent"):
    """Helper: invoke a handoff tool via ToolNode and return the Command."""
    tn = ToolNode([tool])
    tool_call = {
        "name": tool.name,
        "args": {
            "target_agent": target,
            "context_summary": "Test context summary",
            "expectation": "Test expectation",
        },
        "id": f"call_{tool.name}_test",
        "type": "tool_call",
    }
    state = {
        "messages": [
            HumanMessage(content="hello"),
            AIMessage(content="", tool_calls=[tool_call]),
        ],
        "active_agent": from_agent,
    }
    result = tn.invoke(state, config=_TOOL_NODE_CONFIG)
    return result[0]


class TestHandoffToolReturnsOnlyNewMessage(unittest.TestCase):
    """Test 1: Handoff tools pass [tool_message] not [*state['messages'], tool_message]."""

    def test_messages_update_contains_only_tool_message(self):
        for tool, target in TOOLS_AND_TARGETS:
            with self.subTest(tool=tool.name):
                cmd = _invoke_handoff(tool, target=target)
                msgs = cmd.update["messages"]
                self.assertEqual(len(msgs), 1,
                                 f"{tool.name} should emit exactly 1 message, got {len(msgs)}")
                self.assertIsInstance(msgs[0], ToolMessage)


class TestHandoffSetsActiveAgent(unittest.TestCase):
    """Test 2: Each transfer tool sets active_agent to correct target."""

    def test_active_agent_set_correctly(self):
        for tool, target in TOOLS_AND_TARGETS:
            with self.subTest(tool=tool.name):
                cmd = _invoke_handoff(tool, target=target)
                self.assertEqual(cmd.update["active_agent"], target)


class TestHandoffContextPopulated(unittest.TestCase):
    """Test 3: Handoff context carries from_agent, context_summary, expectation."""

    def test_handoff_context_has_all_keys(self):
        for tool, target in TOOLS_AND_TARGETS:
            with self.subTest(tool=tool.name):
                cmd = _invoke_handoff(tool, target=target, from_agent="source_agent")
                hc = cmd.update["handoff_context"]
                self.assertEqual(hc["from_agent"], "source_agent")
                self.assertEqual(hc["context_summary"], "Test context summary")
                self.assertEqual(hc["expectation"], "Test expectation")


class TestResolveFromAgent(unittest.TestCase):
    """Tests 4-6: _resolve_from_agent logic."""

    def test_uses_active_agent_when_present(self):
        state = {
            "active_agent": "inventory_agent",
            "messages": [AIMessage(content="hi", name="order_agent")],
        }
        self.assertEqual(_resolve_from_agent(state), "inventory_agent")

    def test_fallback_to_message_name(self):
        state = {
            "active_agent": "unknown",
            "messages": [
                AIMessage(content="a", name="order_agent"),
                AIMessage(content="b", name="barista_agent"),
            ],
        }
        self.assertEqual(_resolve_from_agent(state), "barista_agent")

    def test_returns_unknown_when_no_info(self):
        state = {"messages": [HumanMessage(content="hi")]}
        self.assertEqual(_resolve_from_agent(state), "unknown")


class TestHandoffContextClearedOnNewTurn(unittest.TestCase):
    """Test 7: handoff_context: None in stream input clears stale context."""

    def test_context_cleared_between_turns(self):
        from langgraph.graph import StateGraph
        from langgraph_swarm import add_active_agent_router
        from langgraph.checkpoint.memory import InMemorySaver
        from src.agents.shared_components import CoffeeShopState

        checkpointer = InMemorySaver()
        builder = StateGraph(CoffeeShopState)

        def dummy_node(state):
            return {"messages": [AIMessage(content="hello", name="order_agent")]}

        builder.add_node("order_agent", dummy_node)
        builder.set_entry_point("order_agent")
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "test-clear-ctx"}}

        graph.invoke(
            {
                "messages": [HumanMessage(content="turn 1")],
                "handoff_context": {
                    "from_agent": "barista_agent",
                    "context_summary": "stale ctx",
                    "expectation": "stale exp",
                },
            },
            config,
        )

        graph.invoke(
            {
                "messages": [HumanMessage(content="turn 2")],
                "handoff_context": None,
            },
            config,
        )

        snapshot = graph.get_state(config)
        self.assertIsNone(snapshot.values.get("handoff_context"))


class TestNoMessageDuplicationAcrossHandoffs(unittest.TestCase):
    """Test 8: Multiple handoffs don't cause message explosion."""

    def test_message_count_grows_linearly(self):
        initial_messages = [HumanMessage(content="order please")]

        total_messages = list(initial_messages)
        for tool, target in TOOLS_AND_TARGETS[:3]:
            cmd = _invoke_handoff(tool, target=target)
            total_messages.extend(cmd.update["messages"])

        self.assertEqual(len(total_messages), 4)
        for msg in total_messages[1:]:
            self.assertIsInstance(msg, ToolMessage)


if __name__ == "__main__":
    unittest.main()
