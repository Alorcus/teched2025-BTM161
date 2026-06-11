"""Tests 39-44: Context isolation hook.

Validates entry agent sees all messages, handoff boundary slicing,
briefing prepend/absence, and defensive guards for non-dict handoff_context.
"""

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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
            ToolMessage(
                content="Successfully transferred to inventory_agent. Context: Order ORD0001",
                name="transfer_to_agent",
                tool_call_id="tc1",
            ),
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
            ToolMessage(
                content="Successfully transferred to barista_agent. Context: All items confirmed",
                name="transfer_to_agent",
                tool_call_id="tc2",
            ),
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


class TestOrphanedToolMessagesAreStripped(unittest.TestCase):
    """Test 45: ToolMessages without matching tool_use in AIMessage are removed.

    This reproduces the real failure: the parent graph state contains a
    ToolMessage(name='transfer_to_agent') but the AIMessage with the
    corresponding tool_use was not propagated from the subgraph. The hook
    must strip orphaned ToolMessages to avoid Anthropic API errors.
    """

    def test_orphaned_tool_message_stripped_from_entry_agent(self):
        hook = create_context_isolation_hook("order_agent")
        messages = [
            HumanMessage(content="Ring it up"),
            # This ToolMessage has no preceding AIMessage with matching tool_use
            ToolMessage(
                content="Successfully transferred to inventory_agent. Context: test",
                name="transfer_to_agent",
                tool_call_id="tc-orphan",
            ),
        ]
        state = {"messages": messages, "handoff_context": None}
        result = hook(state)
        # The orphaned ToolMessage should be removed
        self.assertEqual(len(result["llm_input_messages"]), 1)
        self.assertEqual(result["llm_input_messages"][0].content, "Ring it up")

    def test_valid_tool_message_kept(self):
        hook = create_context_isolation_hook("inventory_agent")
        messages = [
            ToolMessage(
                content="Successfully transferred to inventory_agent. Context: test",
                name="transfer_to_agent",
                tool_call_id="tc-boundary",
            ),
            AIMessage(
                content="",
                name="inventory_agent",
                tool_calls=[{"id": "tc-check", "name": "check_inventory", "args": {}}],
            ),
            ToolMessage(content="All available", tool_call_id="tc-check"),
        ]
        state = {"messages": messages, "handoff_context": None}
        result = hook(state)
        # After boundary: briefing (from boundary content fallback) + AIMessage + ToolMessage
        own = result["llm_input_messages"]
        self.assertEqual(len(own), 3)
        self.assertIsInstance(own[0], HumanMessage)  # briefing from boundary extraction
        self.assertIsInstance(own[1], AIMessage)
        self.assertIsInstance(own[2], ToolMessage)

    def test_mixed_orphaned_and_valid(self):
        hook = create_context_isolation_hook("order_agent")
        messages = [
            HumanMessage(content="Go ahead"),
            AIMessage(
                content="",
                name="order_agent",
                tool_calls=[{"id": "tc-proc", "name": "process_order", "args": {}}],
            ),
            ToolMessage(content="Order created", tool_call_id="tc-proc"),
            # Orphaned: no AIMessage has tool_use with id="tc-ghost"
            ToolMessage(content="Ghost result", tool_call_id="tc-ghost"),
        ]
        state = {"messages": messages, "handoff_context": None}
        result = hook(state)
        own = result["llm_input_messages"]
        # Should keep Human, AI, valid Tool; strip orphaned Tool
        self.assertEqual(len(own), 3)
        self.assertEqual(own[2].content, "Order created")


class TestEmptyMessagesAfterBoundaryNeverFalsy(unittest.TestCase):
    """Test 46: When no messages exist after boundary, hook returns a non-empty
    llm_input_messages to prevent LangGraph from falling back to raw state."""

    def test_no_messages_after_boundary_returns_synthetic(self):
        hook = create_context_isolation_hook("inventory_agent")
        messages = [
            HumanMessage(content="Order something"),
            ToolMessage(
                content="Successfully transferred to inventory_agent. Context: test",
                name="transfer_to_agent",
                tool_call_id="tc1",
            ),
        ]
        state = {"messages": messages, "handoff_context": None}
        result = hook(state)
        # Must be non-empty (truthy) to avoid LangGraph fallback
        self.assertTrue(len(result["llm_input_messages"]) > 0)
        # Should be a HumanMessage with activation prompt
        self.assertIsInstance(result["llm_input_messages"][0], HumanMessage)

    def test_with_handoff_context_prepends_briefing(self):
        hook = create_context_isolation_hook("inventory_agent")
        messages = [
            HumanMessage(content="Order something"),
            ToolMessage(
                content="Successfully transferred to inventory_agent. Context: test",
                name="transfer_to_agent",
                tool_call_id="tc1",
            ),
        ]
        state = {
            "messages": messages,
            "handoff_context": {
                "from_agent": "order_agent",
                "context_summary": "Order ORD0001 placed",
                "expectation": "Check inventory",
            },
        }
        result = hook(state)
        # Should have briefing even with 0 own messages
        self.assertTrue(len(result["llm_input_messages"]) >= 1)
        self.assertIn(
            "[Handoff from order_agent]", result["llm_input_messages"][0].content
        )


if __name__ == "__main__":
    unittest.main()
