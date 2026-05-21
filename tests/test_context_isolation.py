"""Tests 39-44: Context isolation hook.

Validates entry agent sees all messages, handoff boundary slicing,
briefing prepend/absence, and defensive guards for non-dict handoff_context.
"""
import unittest

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.agents.context_isolation import create_context_isolation_hook


class TestEntryAgentGetsAllMessages(unittest.TestCase):
    """Test 39: Order agent (no handoff boundary) receives full history."""

    def test_all_messages_passed(self):
        hook = create_context_isolation_hook("order_agent")
        messages = [
            HumanMessage(content="I want a latte"),
            AIMessage(content="Sure! What size?", name="order_agent"),
            HumanMessage(content="Large please"),
        ]
        state = {"messages": messages, "handoff_context": None}
        result = hook(state)
        self.assertEqual(len(result["llm_input_messages"]), 3)
        self.assertEqual(result["llm_input_messages"][0].content, "I want a latte")
        self.assertEqual(result["llm_input_messages"][2].content, "Large please")


class TestHandoffAgentGetsOnlyPostBoundaryMessages(unittest.TestCase):
    """Test 40: Agent entered via handoff sees only messages after its transfer tool message."""

    def test_post_boundary_only(self):
        hook = create_context_isolation_hook("inventory_agent")
        messages = [
            HumanMessage(content="I want 2 espressos"),
            AIMessage(content="Processing...", name="order_agent"),
            ToolMessage(content="Transferred", name="transfer_to_inventory", tool_call_id="tc1"),
            AIMessage(content="Checking stock for espresso", name="inventory_agent"),
            HumanMessage(content="extra message"),
        ]
        state = {
            "messages": messages,
            "handoff_context": {
                "from_agent": "order_agent",
                "context_summary": "Order ORD0001",
                "expectation": "Check espresso availability",
            },
        }
        result = hook(state)
        # Should be: briefing + 2 messages after boundary
        own_msgs = result["llm_input_messages"]
        # First is briefing, then the 2 post-boundary messages
        self.assertEqual(len(own_msgs), 3)
        self.assertIn("[Handoff from order_agent]", own_msgs[0].content)
        self.assertEqual(own_msgs[1].content, "Checking stock for espresso")
        self.assertEqual(own_msgs[2].content, "extra message")


class TestHandoffAgentGetsBriefingPrepended(unittest.TestCase):
    """Test 41: When handoff_context is set, a synthetic HumanMessage is prepended."""

    def test_briefing_structure(self):
        hook = create_context_isolation_hook("barista_agent")
        messages = [
            ToolMessage(content="Transferred", name="transfer_to_barista", tool_call_id="tc2"),
            AIMessage(content="Preparing order", name="barista_agent"),
        ]
        state = {
            "messages": messages,
            "handoff_context": {
                "from_agent": "inventory_agent",
                "context_summary": "All items confirmed, stock deducted",
                "expectation": "Prepare order ORD0001",
            },
        }
        result = hook(state)
        briefing = result["llm_input_messages"][0]
        self.assertIsInstance(briefing, HumanMessage)
        self.assertIn("[Handoff from inventory_agent]", briefing.content)
        self.assertIn("All items confirmed, stock deducted", briefing.content)
        self.assertIn("Prepare order ORD0001", briefing.content)


class TestNoBriefingWhenHandoffContextIsNone(unittest.TestCase):
    """Test 42: Cleared handoff_context produces no briefing."""

    def test_no_briefing(self):
        hook = create_context_isolation_hook("order_agent")
        messages = [HumanMessage(content="Hi there")]
        state = {"messages": messages, "handoff_context": None}
        result = hook(state)
        # No briefing — just the original message
        self.assertEqual(len(result["llm_input_messages"]), 1)
        self.assertEqual(result["llm_input_messages"][0].content, "Hi there")


class TestNoBriefingWhenHandoffContextIsEmptyDict(unittest.TestCase):
    """Test 43: Empty dict {} treated as no handoff (no briefing)."""

    def test_empty_dict_no_briefing(self):
        hook = create_context_isolation_hook("order_agent")
        messages = [HumanMessage(content="Hello")]
        state = {"messages": messages, "handoff_context": {}}
        result = hook(state)
        self.assertEqual(len(result["llm_input_messages"]), 1)
        self.assertEqual(result["llm_input_messages"][0].content, "Hello")


class TestNonDictHandoffContextHandledSafely(unittest.TestCase):
    """Test 44: If handoff_context is a non-dict truthy value, no AttributeError."""

    def test_string_handoff_context(self):
        hook = create_context_isolation_hook("order_agent")
        messages = [HumanMessage(content="Hey")]
        state = {"messages": messages, "handoff_context": "some_stale_value"}
        # Should not raise AttributeError
        result = hook(state)
        self.assertEqual(len(result["llm_input_messages"]), 1)

    def test_list_handoff_context(self):
        hook = create_context_isolation_hook("order_agent")
        messages = [HumanMessage(content="Hey")]
        state = {"messages": messages, "handoff_context": ["stale"]}
        result = hook(state)
        self.assertEqual(len(result["llm_input_messages"]), 1)

    def test_int_handoff_context(self):
        hook = create_context_isolation_hook("order_agent")
        messages = [HumanMessage(content="Hey")]
        state = {"messages": messages, "handoff_context": 42}
        result = hook(state)
        self.assertEqual(len(result["llm_input_messages"]), 1)


if __name__ == "__main__":
    unittest.main()
