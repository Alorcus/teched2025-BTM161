"""Tests 31-38: ConversationRunner.

Validates concurrency (lock, double-start, flag reset), error handling,
deduplication, max turns, and messages key matching.
"""
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from langchain_core.messages import AIMessage, ToolMessage

from src.dashboard.event_bus import EventBus, EventType
from src.dashboard.conversation_runner import ConversationRunner, MAX_CONVERSATION_TURNS


def _make_mock_shop():
    """Create a mock CoffeeShop with controllable stream."""
    shop = MagicMock()
    shop._get_config.return_value = {"configurable": {"thread_id": "test"}}
    shop.customer_agent = MagicMock()
    return shop


class TestRunnerStartSetsIsRunning(unittest.TestCase):
    """Test 31: Start sets flag atomically before thread runs."""

    def test_flag_set_immediately(self):
        shop = _make_mock_shop()
        # Make stream block until we release it
        block = threading.Event()
        shop.app.stream.side_effect = lambda *a, **kw: iter([]) if block.wait(0.5) else iter([])
        shop.customer_agent.get_initial_message.return_value = "hello"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)

        self.assertTrue(runner.is_running)
        block.set()
        runner._thread.join(timeout=2)


class TestRunnerDoubleStartPrevented(unittest.TestCase):
    """Test 32: Second call to start() is a no-op."""

    def test_no_second_thread(self):
        shop = _make_mock_shop()
        block = threading.Event()

        def slow_stream(*a, **kw):
            block.wait(1)
            return iter([])

        shop.app.stream.side_effect = slow_stream
        shop.customer_agent.get_initial_message.return_value = "hi"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        first_thread = runner._thread

        runner.start(scenario_index=1)
        self.assertIs(runner._thread, first_thread)

        block.set()
        first_thread.join(timeout=2)


class TestRunnerIsRunningClearedOnCompletion(unittest.TestCase):
    """Test 33: Flag resets after conversation finishes."""

    def test_flag_cleared(self):
        shop = _make_mock_shop()
        shop.app.stream.return_value = iter([])
        shop.customer_agent.get_initial_message.return_value = "hi"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        self.assertFalse(runner.is_running)


class TestRunnerIsRunningClearedOnError(unittest.TestCase):
    """Test 34: Flag resets even when stream raises."""

    def test_flag_cleared_on_error(self):
        shop = _make_mock_shop()
        shop.app.stream.side_effect = RuntimeError("LLM timeout")
        shop.customer_agent.get_initial_message.return_value = "hi"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        self.assertFalse(runner.is_running)
        # Should have published an error event
        events = bus.drain()
        error_events = [e for e in events if "error" in (e.content or "").lower()]
        self.assertTrue(len(error_events) > 0)


class TestStreamErrorPublishesEventAndReturnsNone(unittest.TestCase):
    """Test 35: LLM errors during streaming are caught gracefully."""

    def test_mid_stream_error(self):
        shop = _make_mock_shop()

        # Stream yields one item then raises
        def failing_stream(*a, **kw):
            yield (("order_agent:abc",), {"agent": {"messages": [AIMessage(content="hi", name="order_agent")]}})
            raise RuntimeError("API rate limit")

        shop.app.stream.side_effect = failing_stream
        shop.customer_agent.get_initial_message.return_value = "hello"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        error_events = [e for e in events if "stream error" in (e.content or "").lower()]
        self.assertTrue(len(error_events) > 0)


class TestStreamDeduplicatesMessages(unittest.TestCase):
    """Test 36: Same message emitted twice by stream is only dispatched once."""

    def test_dedup(self):
        shop = _make_mock_shop()

        msg = AIMessage(content="Order received!", name="order_agent", id="msg-001")

        def dup_stream(*a, **kw):
            # Same message appears twice
            yield (("order_agent:abc",), {"agent": {"messages": [msg]}})
            yield (("order_agent:abc",), {"agent": {"messages": [msg]}})

        shop.app.stream.side_effect = dup_stream
        shop.customer_agent.get_initial_message.return_value = "hello"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE
                      and e.content == "Order received!"]
        self.assertEqual(len(agent_msgs), 1)


class TestSameContentDifferentIdNotDeduplicated(unittest.TestCase):
    """Test 36b: Two messages with same content but different IDs are both dispatched."""

    def test_same_content_different_id(self):
        shop = _make_mock_shop()

        msg1 = AIMessage(content="OK", name="order_agent", id="msg-aaa")
        msg2 = AIMessage(content="OK", name="order_agent", id="msg-bbb")

        def stream_with_same_content(*a, **kw):
            yield (("order_agent:abc",), {"agent": {"messages": [msg1]}})
            yield (("order_agent:abc",), {"agent": {"messages": [msg2]}})

        shop.app.stream.side_effect = stream_with_same_content
        shop.customer_agent.get_initial_message.return_value = "hello"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE
                      and e.content == "OK"]
        self.assertEqual(len(agent_msgs), 2)


class TestMaxTurnsLimit(unittest.TestCase):
    """Test 37: Conversation stops at MAX_CONVERSATION_TURNS."""

    def test_stops_at_max(self):
        shop = _make_mock_shop()

        call_count = [0]

        def stream_reply(*a, **kw):
            call_count[0] += 1
            msg = AIMessage(content=f"reply {call_count[0]}", name="order_agent",
                            id=f"msg-{call_count[0]}")
            yield (("order_agent:abc",), {"agent": {"messages": [msg]}})

        shop.app.stream.side_effect = stream_reply
        shop.customer_agent.get_initial_message.return_value = "hi"
        # Customer always responds (would loop forever without limit)
        shop.customer_agent.respond_to.return_value = "more please"

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=10)

        self.assertEqual(call_count[0], MAX_CONVERSATION_TURNS)


class TestMessagesKeyExactMatch(unittest.TestCase):
    """Test 38: Only k == 'messages' is matched, not substrings."""

    def test_error_messages_key_ignored(self):
        shop = _make_mock_shop()

        real_msg = AIMessage(content="real", name="order_agent", id="msg-real")
        fake_msg = AIMessage(content="should be ignored", name="order_agent", id="msg-fake")

        def stream_with_bad_key(*a, **kw):
            yield (("order_agent:abc",), {"agent": {
                "messages": [real_msg],
                "error_messages": [fake_msg],
            }})

        shop.app.stream.side_effect = stream_with_bad_key
        shop.customer_agent.get_initial_message.return_value = "hi"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE]
        contents = [e.content for e in agent_msgs]
        self.assertIn("real", contents)
        self.assertNotIn("should be ignored", contents)


if __name__ == "__main__":
    unittest.main()
