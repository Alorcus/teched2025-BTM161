"""Tests for the multi-agent handoff system.

Verifies that handoff tools execute correctly through LangGraph's ToolNode,
that state injection works, and that the full graph compiles and routes properly.
"""

import unittest

from langchain_core.messages import AIMessage
from langgraph.prebuilt.tool_node import ToolNode, _get_state_args
from langgraph.types import Command

from src.agents.shared_components import (
    transfer_to_agent,
)


class TestHandoffToolInjection(unittest.TestCase):
    """Verify InjectedState and InjectedToolCallId are properly injected at runtime."""

    ALL_TOOLS = [
        transfer_to_agent,
    ]

    def test_state_args_detected(self):
        """ToolNode must detect InjectedState on all handoff tools."""
        for tool in self.ALL_TOOLS:
            state_args = _get_state_args(tool)
            self.assertIn(
                "state",
                state_args,
                f"{tool.name} missing 'state' in state_args — "
                f"InjectedState not detected (got {state_args})",
            )

    def test_tool_call_schema_excludes_injected_params(self):
        """LLM-facing schema must expose target_agent, context_summary, and expectation only."""
        for tool in self.ALL_TOOLS:
            schema = tool.tool_call_schema.model_json_schema()
            props = set(schema["properties"].keys())
            self.assertEqual(
                props,
                {"target_agent", "context_summary", "expectation"},
                f"{tool.name} schema exposes wrong fields: {props}",
            )

    def test_handoff_executes_through_tool_node(self):
        """Handoff tools must execute without TypeError when called via ToolNode."""
        tn = ToolNode([transfer_to_agent])
        tool_call = {
            "name": "transfer_to_agent",
            "args": {
                "target_agent": "inventory_agent",
                "context_summary": "Customer ordered 1 espresso, ORD0001 created.",
                "expectation": "Check espresso stock availability.",
            },
            "id": "call_test_001",
            "type": "tool_call",
        }
        state = {
            "messages": [AIMessage(content="", tool_calls=[tool_call])],
            "active_agent": "order_agent",
        }

        result = tn.invoke(state)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        cmd = result[0]
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.goto, "inventory_agent")
        self.assertEqual(cmd.update["active_agent"], "inventory_agent")
        self.assertEqual(cmd.update["handoff_context"]["from_agent"], "order_agent")
        self.assertEqual(
            cmd.update["handoff_context"]["context_summary"],
            "Customer ordered 1 espresso, ORD0001 created.",
        )
        # Messages must contain only the new ToolMessage (not the full state copy)
        self.assertIn("messages", cmd.update)
        forwarded_msgs = cmd.update["messages"]
        self.assertEqual(len(forwarded_msgs), 1)
        tool_msg = forwarded_msgs[0]
        self.assertEqual(tool_msg.name, "transfer_to_agent")
        self.assertIn("Successfully transferred", tool_msg.content)

    def test_all_handoff_tools_execute(self):
        """All four handoff tools must execute without error."""
        tools_and_targets = [
            (transfer_to_agent, "inventory_agent"),
            (transfer_to_agent, "barista_agent"),
            (transfer_to_agent, "customer_service_agent"),
            (transfer_to_agent, "order_agent"),
        ]
        for tool, expected_target in tools_and_targets:
            with self.subTest(tool=tool.name):
                tn = ToolNode([tool])
                tool_call = {
                    "name": tool.name,
                    "args": {
                        "target_agent": expected_target,
                        "context_summary": "Test context",
                        "expectation": "Test expectation",
                    },
                    "id": f"call_{tool.name}",
                    "type": "tool_call",
                }
                state = {
                    "messages": [AIMessage(content="", tool_calls=[tool_call])],
                    "active_agent": "source_agent",
                }
                result = tn.invoke(state)
                self.assertIsInstance(result, list)
                cmd = result[0]
                self.assertEqual(cmd.goto, expected_target)
                self.assertEqual(
                    cmd.update["handoff_context"]["from_agent"], "source_agent"
                )


class TestGraphCompilation(unittest.TestCase):
    """Verify the full CoffeeShop graph compiles and has correct structure."""

    def test_coffee_shop_compiles(self):
        """CoffeeShop.open_shop() must compile without error."""
        from src.coffee_shop import CoffeeShop
        from src.config import CoffeeShopConfig

        shop = CoffeeShop(CoffeeShopConfig(setup_name="baseline"))
        shop.open_shop()
        self.assertIsNotNone(shop.app)

    def test_graph_nodes(self):
        """Graph must contain all expected agent nodes."""
        from src.coffee_shop import CoffeeShop
        from src.config import CoffeeShopConfig

        shop = CoffeeShop(CoffeeShopConfig(setup_name="baseline"))
        shop.open_shop()
        nodes = set(shop.app.get_graph().nodes.keys())
        expected = {
            "__start__",
            "order_agent",
            "inventory_agent",
            "barista_agent",
            "customer_service_agent",
        }
        self.assertEqual(nodes, expected)

    def test_graph_routing_edges(self):
        """Each agent must have correct outgoing edges (destinations)."""
        from src.coffee_shop import CoffeeShop
        from src.config import CoffeeShopConfig

        shop = CoffeeShop(CoffeeShopConfig(setup_name="baseline"))
        shop.open_shop()
        edges = shop.app.get_graph().edges
        edge_map = {}
        for edge in edges:
            edge_map.setdefault(edge.source, set()).add(edge.target)

        self.assertIn("inventory_agent", edge_map.get("order_agent", set()))
        self.assertIn("customer_service_agent", edge_map.get("order_agent", set()))
        self.assertIn("barista_agent", edge_map.get("inventory_agent", set()))
        self.assertIn("customer_service_agent", edge_map.get("barista_agent", set()))
        self.assertIn("order_agent", edge_map.get("customer_service_agent", set()))


class TestContextIsolationHook(unittest.TestCase):
    """Verify context isolation hook filters messages correctly."""

    def test_entry_agent_sees_all_messages(self):
        """Order agent (entry) should see all messages when no handoff context."""
        from langchain_core.messages import HumanMessage

        from src.agents.context_isolation import create_context_isolation_hook

        hook = create_context_isolation_hook("order_agent")
        state = {
            "messages": [HumanMessage(content="I want a latte")],
            "handoff_context": None,
        }
        result = hook(state)
        self.assertEqual(len(result["llm_input_messages"]), 1)
        self.assertEqual(result["llm_input_messages"][0].content, "I want a latte")

    def test_receiving_agent_sees_only_briefing_and_own_messages(self):
        """Inventory agent should see handoff briefing + own-turn messages only."""
        from langchain_core.messages import HumanMessage, ToolMessage

        from src.agents.context_isolation import create_context_isolation_hook

        hook = create_context_isolation_hook("inventory_agent")
        state = {
            "messages": [
                HumanMessage(content="I want a latte"),
                AIMessage(content="Processing your order..."),
                ToolMessage(
                    content="Successfully transferred to inventory_agent. Context: Order ORD0001 for 1 latte",
                    name="transfer_to_agent",
                    tool_call_id="tc1",
                ),
                AIMessage(content="Checking stock..."),
            ],
            "handoff_context": {
                "from_agent": "order_agent",
                "context_summary": "Order ORD0001 for 1 latte",
                "expectation": "Check latte availability",
            },
        }
        result = hook(state)
        msgs = result["llm_input_messages"]
        # Should be: briefing + 1 own message (AIMessage after transfer)
        self.assertEqual(len(msgs), 2)
        self.assertIn("[Handoff from order_agent]", msgs[0].content)
        self.assertEqual(msgs[1].content, "Checking stock...")


if __name__ == "__main__":
    unittest.main()
