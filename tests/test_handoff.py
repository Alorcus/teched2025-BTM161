"""Tests for the multi-agent handoff system.

Verifies that handoff tools execute correctly through LangGraph's ToolNode,
that state injection works, and that the full graph compiles and routes properly.
"""
import unittest
from langgraph.prebuilt.tool_node import ToolNode, _get_state_args
from langgraph.types import Command
from langchain_core.messages import AIMessage

from src.agents.shared_components import (
    transfer_to_inventory, transfer_to_barista,
    transfer_to_customer_service, transfer_to_order_agent,
)


class TestHandoffToolInjection(unittest.TestCase):
    """Verify InjectedState and InjectedToolCallId are properly injected at runtime."""

    ALL_TOOLS = [
        transfer_to_inventory,
        transfer_to_barista,
        transfer_to_customer_service,
        transfer_to_order_agent,
    ]

    def test_state_args_detected(self):
        """ToolNode must detect InjectedState on all handoff tools."""
        for tool in self.ALL_TOOLS:
            state_args = _get_state_args(tool)
            self.assertIn("state", state_args,
                          f"{tool.name} missing 'state' in state_args — "
                          f"InjectedState not detected (got {state_args})")

    def test_tool_call_schema_excludes_injected_params(self):
        """LLM-facing schema must only expose context_summary and expectation."""
        for tool in self.ALL_TOOLS:
            schema = tool.tool_call_schema.model_json_schema()
            props = set(schema["properties"].keys())
            self.assertEqual(props, {"context_summary", "expectation"},
                             f"{tool.name} schema exposes wrong fields: {props}")

    def test_handoff_executes_through_tool_node(self):
        """Handoff tools must execute without TypeError when called via ToolNode."""
        tn = ToolNode([transfer_to_inventory])
        tool_call = {
            "name": "transfer_to_inventory",
            "args": {
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
        self.assertEqual(cmd.update["handoff_context"]["context_summary"],
                         "Customer ordered 1 espresso, ORD0001 created.")
        # Messages must contain only the new ToolMessage (not the full state copy)
        self.assertIn("messages", cmd.update)
        forwarded_msgs = cmd.update["messages"]
        self.assertEqual(len(forwarded_msgs), 1)
        tool_msg = forwarded_msgs[0]
        self.assertEqual(tool_msg.name, "transfer_to_inventory")
        self.assertIn("Successfully transferred", tool_msg.content)

    def test_all_handoff_tools_execute(self):
        """All four handoff tools must execute without error."""
        tools_and_targets = [
            (transfer_to_inventory, "inventory_agent"),
            (transfer_to_barista, "barista_agent"),
            (transfer_to_customer_service, "customer_service_agent"),
            (transfer_to_order_agent, "order_agent"),
        ]
        for tool, expected_target in tools_and_targets:
            with self.subTest(tool=tool.name):
                tn = ToolNode([tool])
                tool_call = {
                    "name": tool.name,
                    "args": {
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
                self.assertEqual(cmd.update["handoff_context"]["from_agent"], "source_agent")


class TestGraphCompilation(unittest.TestCase):
    """Verify the full CoffeeShop graph compiles and has correct structure."""

    def test_coffee_shop_compiles(self):
        """CoffeeShop.open_shop() must compile without error."""
        from src.coffee_shop import CoffeeShop
        shop = CoffeeShop()
        shop.open_shop()
        self.assertIsNotNone(shop.app)

    def test_graph_nodes(self):
        """Graph must contain all expected agent nodes."""
        from src.coffee_shop import CoffeeShop
        shop = CoffeeShop()
        shop.open_shop()
        nodes = set(shop.app.get_graph().nodes.keys())
        expected = {"__start__", "order_agent", "inventory_agent", "barista_agent", "customer_service_agent"}
        self.assertEqual(nodes, expected)

    def test_graph_routing_edges(self):
        """Each agent must have correct outgoing edges (destinations)."""
        from src.coffee_shop import CoffeeShop
        shop = CoffeeShop()
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
        from src.agents.context_isolation import create_context_isolation_hook
        from langchain_core.messages import HumanMessage

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
        from src.agents.context_isolation import create_context_isolation_hook
        from langchain_core.messages import HumanMessage, ToolMessage

        hook = create_context_isolation_hook("inventory_agent")
        state = {
            "messages": [
                HumanMessage(content="I want a latte"),
                AIMessage(content="Processing your order..."),
                ToolMessage(content="Transferred", name="transfer_to_inventory", tool_call_id="tc1"),
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


class TestSimulationEndToEnd(unittest.TestCase):
    """Integration test: run a full simulation and verify order completes."""

    def test_scenario_0_order_reaches_terminal_status(self):
        """Scenario 0 must produce an order that reaches completed or preparation_error."""
        from src.coffee_shop import CoffeeShop
        from src.agents.order_store import engine
        from src.agents.shared_components import Order, OrderStatus
        from sqlmodel import Session, select

        shop = CoffeeShop()
        shop.open_shop()

        trace_ids = shop.run_conversation(scenario_index=0)

        self.assertTrue(len(trace_ids) > 0, "No traces generated")

        with Session(engine) as session:
            order = session.exec(select(Order).order_by(Order.id.desc())).first()
            self.assertIsNotNone(order, "No order found in database")
            terminal_statuses = {
                OrderStatus.COMPLETED,
                OrderStatus.PREPARATION_ERROR,
                OrderStatus.REFUNDED,
            }
            self.assertIn(
                order.status, terminal_statuses,
                f"Order {order.order_id_str} stuck at '{order.status.value}' — "
                f"expected one of {[s.value for s in terminal_statuses]}",
            )

    def test_scenario_0_inventory_updated(self):
        """Scenario 0 must trigger at least one inventory stock update."""
        from src.coffee_shop import CoffeeShop
        from src.agents.order_store import engine
        from src.agents.shared_components import Order, OrderStatus
        from sqlmodel import Session, select

        shop = CoffeeShop()
        shop.open_shop()

        shop.run_conversation(scenario_index=0)

        with Session(engine) as session:
            order = session.exec(select(Order).order_by(Order.id.desc())).first()
            self.assertIsNotNone(order)
            # If order reached barista or beyond, inventory was confirmed
            past_inventory = {
                OrderStatus.INVENTORY_CONFIRMED,
                OrderStatus.IN_PREPARATION,
                OrderStatus.COMPLETED,
                OrderStatus.PREPARATION_ERROR,
            }
            self.assertIn(
                order.status, past_inventory,
                f"Order never reached inventory confirmation — stuck at '{order.status.value}'",
            )


if __name__ == "__main__":
    unittest.main()
