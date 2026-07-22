"""Tests for `stream.extract_messages` including response-guardrail rejection.

Verifies that when the response_gateway emits a corrective HumanMessage
identifying a rejected AIMessage, `extract_messages` downgrades the rejected
message's `is_agent_reply` flag so the customer never receives the rejected
text through `send_message`.
"""
import unittest

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from src.control_plane.subgraph import (
    CORRECTION_KWARG,
    REJECTED_CONTENT_KWARG,
)
from src.stream import extract_messages


def _fake_stream(steps: list[tuple[str, dict]]):
    """Convert a list of (namespace_key, node_update) into the shape LangGraph
    stream yields: `[(ns_tuple, {node_name: {"messages": [...]}}), ...]`.

    Each step's `node_update` is the raw dict a node returned from its callback.
    """
    for ns_key, node_map in steps:
        yield (ns_key,), node_map


class TestExtractMessagesBasic(unittest.TestCase):
    def test_clean_reply_yielded_with_is_agent_reply_true(self):
        ai = AIMessage(content="Welcome", name="order_agent", id="ai-1")
        stream = _fake_stream([
            ("order_agent", {"llm": {"messages": [ai]}}),
        ])
        results = list(extract_messages(stream))
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].is_agent_reply)
        self.assertEqual(results[0].content, "Welcome")

    def test_tool_call_message_not_agent_reply(self):
        ai = AIMessage(
            content="",
            name="order_agent",
            id="ai-1",
            tool_calls=[{"name": "process_order", "args": {}, "id": "tc-1"}],
        )
        stream = _fake_stream([
            ("order_agent", {"llm": {"messages": [ai]}}),
        ])
        results = list(extract_messages(stream))
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].is_agent_reply)


class TestExtractMessagesRejection(unittest.TestCase):
    def test_rejected_message_downgraded(self):
        bad = AIMessage(content="Try our hazelnut latte!", name="order_agent", id="ai-bad")
        correction = HumanMessage(
            content="not on menu",
            additional_kwargs={
                CORRECTION_KWARG: True,
                REJECTED_CONTENT_KWARG: "Try our hazelnut latte!",
            },
        )
        good = AIMessage(content="Would you like a latte?", name="order_agent", id="ai-good")

        stream = _fake_stream([
            ("order_agent", {"llm": {"messages": [bad]}}),
            ("order_agent", {"response_gateway": {"messages": [
                RemoveMessage(id="ai-bad"), correction,
            ]}}),
            ("order_agent", {"llm": {"messages": [good]}}),
        ])
        results = list(extract_messages(stream))

        by_content = {r.content: r for r in results}
        self.assertIn("Try our hazelnut latte!", by_content)
        self.assertIn("Would you like a latte?", by_content)
        self.assertFalse(by_content["Try our hazelnut latte!"].is_agent_reply)
        self.assertTrue(by_content["Would you like a latte?"].is_agent_reply)

    def test_id_based_pairing_survives_duplicate_content(self):
        """When two agents produce identical short text ('Sure!') and only one
        is rejected, id-based pairing must downgrade the correct message.
        The correction stamps REJECTED_MESSAGE_ID_KWARG so the extractor can
        match on id rather than falling back to content."""
        from src.control_plane.subgraph import REJECTED_MESSAGE_ID_KWARG

        first_sure = AIMessage(content="Sure!", name="order_agent", id="ai-first")
        second_sure = AIMessage(content="Sure!", name="order_agent", id="ai-second")
        # A correction that identifies second_sure by id.
        correction = HumanMessage(
            content="policy violation",
            additional_kwargs={
                CORRECTION_KWARG: True,
                REJECTED_MESSAGE_ID_KWARG: "ai-second",
                REJECTED_CONTENT_KWARG: "Sure!",
            },
        )
        # Note: the second AIMessage will not add a new StreamMessage because
        # extract_messages dedupes by (content, name). So the pending buffer
        # remains the first "Sure!" — id-match against "ai-second" must NOT
        # downgrade it.
        stream = _fake_stream([
            ("order_agent", {"llm": {"messages": [first_sure]}}),
            ("order_agent", {"llm": {"messages": [second_sure]}}),
            ("order_agent", {"response_gateway": {"messages": [correction]}}),
        ])
        results = list(extract_messages(stream))
        # The first "Sure!" must remain a valid agent reply — id mismatch
        # prevents the correction from downgrading a different message.
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].is_agent_reply)

    def test_last_agent_reply_semantics_customer_never_sees_rejected(self):
        """Simulates the ConversationEngine.send_message behavior: keep the
        last StreamMessage where is_agent_reply is True.
        """
        bad = AIMessage(content="Have a caramel macchiato!", name="order_agent", id="ai-bad")
        correction = HumanMessage(
            content="not on menu",
            additional_kwargs={
                CORRECTION_KWARG: True,
                REJECTED_CONTENT_KWARG: "Have a caramel macchiato!",
            },
        )
        good = AIMessage(content="How about a cappuccino?", name="order_agent", id="ai-good")

        stream = _fake_stream([
            ("order_agent", {"llm": {"messages": [bad]}}),
            ("order_agent", {"response_gateway": {"messages": [correction]}}),
            ("order_agent", {"llm": {"messages": [good]}}),
        ])

        last_reply = None
        for sm in extract_messages(stream):
            if sm.is_agent_reply:
                last_reply = sm.content

        self.assertEqual(last_reply, "How about a cappuccino?")

    def test_retry_cap_replaces_offending_with_fallback(self):
        """When the retry cap is hit, the response_gateway must emit a
        canned-fallback AIMessage stamped as a correction. The stream extractor
        pairs the fallback's rejected-message-id with the offending AIMessage
        and downgrades it; the customer then sees only the fallback, never the
        rejected content.
        """
        from src.control_plane.subgraph import (
            CAP_EXHAUSTED_FALLBACK,
            REJECTED_MESSAGE_ID_KWARG,
        )

        bad1 = AIMessage(content="Hazelnut latte?", name="order_agent", id="ai-1")
        correction1 = HumanMessage(
            content="fix",
            additional_kwargs={
                CORRECTION_KWARG: True,
                REJECTED_MESSAGE_ID_KWARG: "ai-1",
                REJECTED_CONTENT_KWARG: "Hazelnut latte?",
            },
        )
        bad2 = AIMessage(content="Caramel macchiato?", name="order_agent", id="ai-2")
        correction2 = HumanMessage(
            content="fix",
            additional_kwargs={
                CORRECTION_KWARG: True,
                REJECTED_MESSAGE_ID_KWARG: "ai-2",
                REJECTED_CONTENT_KWARG: "Caramel macchiato?",
            },
        )
        bad3 = AIMessage(content="Vanilla frappe?", name="order_agent", id="ai-3")
        fallback = AIMessage(
            content=CAP_EXHAUSTED_FALLBACK,
            name="order_agent",
            id="ai-fallback",
            additional_kwargs={
                CORRECTION_KWARG: True,
                REJECTED_MESSAGE_ID_KWARG: "ai-3",
                REJECTED_CONTENT_KWARG: "Vanilla frappe?",
            },
        )

        stream = _fake_stream([
            ("order_agent", {"llm": {"messages": [bad1]}}),
            ("order_agent", {"response_gateway": {"messages": [correction1]}}),
            ("order_agent", {"llm": {"messages": [bad2]}}),
            ("order_agent", {"response_gateway": {"messages": [correction2]}}),
            ("order_agent", {"llm": {"messages": [bad3]}}),
            ("order_agent", {"response_gateway": {"messages": [fallback]}}),
        ])

        last_reply = None
        for sm in extract_messages(stream):
            if sm.is_agent_reply:
                last_reply = sm.content

        self.assertEqual(last_reply, CAP_EXHAUSTED_FALLBACK)


if __name__ == "__main__":
    unittest.main()
