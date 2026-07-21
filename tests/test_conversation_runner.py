"""Tests for ConversationRunner.

Validates concurrency (lock, double-start, flag reset), error handling,
deduplication, max turns, messages key matching, and handover pause seam.
"""
import threading
import time
import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from src.config import CoffeeShopConfig
from src.dashboard.interaction.event_bus import EventBus, EventType
from src.dashboard.interaction.conversation_runner import (
    ConversationRunner,
    MAX_CONVERSATION_TURNS,
    _extract_text,
)


def _make_mock_shop():
    """Create a mock CoffeeShop with controllable stream."""
    shop = MagicMock()
    shop._get_config.return_value = {"configurable": {"thread_id": "test"}}
    shop.customer_agent = MagicMock()
    return shop


class TestRunnerStartSetsIsRunning(unittest.TestCase):
    """Start sets flag atomically before thread runs."""

    def test_flag_set_immediately(self):
        shop = _make_mock_shop()
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
    """Second call to start() is a no-op."""

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
    """Flag resets after conversation finishes."""

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
    """Flag resets even when stream raises."""

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
        events = bus.drain()
        error_events = [e for e in events if "error" in (e.content or "").lower()]
        self.assertTrue(len(error_events) > 0)


class TestStreamErrorPublishesEventAndReturnsNone(unittest.TestCase):
    """LLM errors during streaming are caught gracefully."""

    def test_mid_stream_error(self):
        shop = _make_mock_shop()

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
    """Same message emitted twice by stream is only dispatched once."""

    def test_dedup(self):
        shop = _make_mock_shop()

        msg = AIMessage(content="Order received!", name="order_agent", id="msg-001")

        def dup_stream(*a, **kw):
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
    """Two messages with same content but different IDs are both dispatched."""

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
    """Conversation stops at MAX_CONVERSATION_TURNS."""

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
        shop.customer_agent.respond_to.return_value = "more please"
        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=10)

        self.assertEqual(call_count[0], MAX_CONVERSATION_TURNS)


class TestMessagesKeyExactMatch(unittest.TestCase):
    """Only k == 'messages' is matched, not substrings."""

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


class TestUserVisibleEmittedToOrderAgent(unittest.TestCase):
    """USER_VISIBLE is emitted targeting order_agent for initial customer message."""

    def test_initial_message_targets_order_agent(self):
        shop = _make_mock_shop()

        msg = AIMessage(content="Welcome!", name="order_agent", id="msg-w")

        def stream_reply(*a, **kw):
            yield (("order_agent:abc",), {"agent": {"messages": [msg]}})

        shop.app.stream.side_effect = stream_reply
        shop.customer_agent.get_initial_message.return_value = "I want a latte"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        user_visible = [e for e in events if e.event_type == EventType.USER_VISIBLE]
        self.assertEqual(len(user_visible), 1)
        self.assertEqual(user_visible[0].agent_name, "order_agent")
        self.assertEqual(user_visible[0].content, "I want a latte")


class TestUserVisibleFollowsHandoff(unittest.TestCase):
    """After a handoff, USER_VISIBLE targets the new agent."""

    def test_targets_new_agent_after_handoff(self):
        shop = _make_mock_shop()

        call_count = [0]

        def stream_with_handoff(*a, **kw):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                msg = AIMessage(content="Let me brew that", name="order_agent", id="msg-1")
                yield (("order_agent:abc",), {"agent": {
                    "messages": [msg],
                    "active_agent": "barista_agent",
                    "handoff_context": {
                        "from_agent": "order_agent",
                        "context_summary": "Customer wants a latte",
                        "expectation": "Brew the latte",
                    },
                }})
            else:
                msg = AIMessage(content="Coffee is ready!", name="barista_agent", id="msg-2")
                yield (("barista_agent:def",), {"agent": {"messages": [msg]}})

        shop.app.stream.side_effect = stream_with_handoff
        shop.customer_agent.get_initial_message.return_value = "I want a latte"

        respond_calls = [0]

        def mock_respond(reply):
            respond_calls[0] += 1
            if respond_calls[0] == 1:
                return "Thanks, sounds good"
            return None

        shop.customer_agent.respond_to.side_effect = mock_respond

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        user_visible = [e for e in events if e.event_type == EventType.USER_VISIBLE]
        self.assertEqual(len(user_visible), 2)
        self.assertEqual(user_visible[0].agent_name, "order_agent")
        self.assertEqual(user_visible[0].content, "I want a latte")
        self.assertEqual(user_visible[1].agent_name, "barista_agent")
        self.assertEqual(user_visible[1].content, "Thanks, sounds good")


class TestHandoffNotDuplicatedOnEcho(unittest.TestCase):
    """A single transfer must produce one HANDOFF event even when the parent
    graph re-surfaces the same handoff_context in a later state snapshot."""

    def test_repeated_handoff_context_is_deduped(self):
        shop = _make_mock_shop()

        hc = {
            "from_agent": "inventory_agent",
            "context_summary": "Order ORD0048 verified",
            "expectation": "Brew the latte",
        }

        call_count = [0]

        def stream_with_echo(*a, **kw):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] == 1:
                handoff_msg = AIMessage(
                    content="Handing off",
                    name="inventory_agent",
                    id="msg-handoff",
                )
                yield (("inventory_agent:abc",), {"agent": {
                    "messages": [handoff_msg],
                    "active_agent": "barista_agent",
                    "handoff_context": hc,
                }})
                final_msg = AIMessage(
                    content="Order complete!",
                    name="barista_agent",
                    id="msg-final",
                )
                yield (("barista_agent:def",), {"agent": {"messages": [final_msg]}})
                yield ((), {"router": {
                    "active_agent": "barista_agent",
                    "handoff_context": hc,
                }})
            else:
                yield (("barista_agent:def",), {"agent": {
                    "messages": [AIMessage(content="bye", name="barista_agent", id="msg-bye")],
                }})

        shop.app.stream.side_effect = stream_with_echo
        shop.customer_agent.get_initial_message.return_value = "I want a latte"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        handoffs = [e for e in events if e.event_type == EventType.HANDOFF]
        self.assertEqual(len(handoffs), 1)
        self.assertEqual(handoffs[0].agent_name, "inventory_agent")
        self.assertEqual(handoffs[0].target_agent, "barista_agent")

    def test_distinct_handoffs_are_not_deduped(self):
        """Two genuinely different transfers in the same turn must both fire."""
        shop = _make_mock_shop()

        def stream_two_handoffs(*a, **kw):
            yield (("order_agent:abc",), {"agent": {
                "messages": [AIMessage(content="to inv", name="order_agent", id="m1")],
                "active_agent": "inventory_agent",
                "handoff_context": {
                    "from_agent": "order_agent",
                    "context_summary": "check stock",
                    "expectation": "verify",
                },
            }})
            yield (("inventory_agent:def",), {"agent": {
                "messages": [AIMessage(content="to barista", name="inventory_agent", id="m2")],
                "active_agent": "barista_agent",
                "handoff_context": {
                    "from_agent": "inventory_agent",
                    "context_summary": "stock ok",
                    "expectation": "brew",
                },
            }})

        shop.app.stream.side_effect = stream_two_handoffs
        shop.customer_agent.get_initial_message.return_value = "hi"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        handoffs = [e for e in events if e.event_type == EventType.HANDOFF]
        self.assertEqual(len(handoffs), 2)
        self.assertEqual(handoffs[0].target_agent, "inventory_agent")
        self.assertEqual(handoffs[1].target_agent, "barista_agent")


class TestActiveAgentResetsOnNewConversation(unittest.TestCase):
    """_active_agent resets to order_agent at the start of each conversation."""

    def test_resets_on_new_run(self):
        shop = _make_mock_shop()

        call_count = [0]

        def stream_handoff(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                msg = AIMessage(content="Handing off", name="order_agent", id=f"msg-h{call_count[0]}")
                yield (("order_agent:abc",), {"agent": {
                    "messages": [msg],
                    "active_agent": "barista_agent",
                    "handoff_context": {
                        "from_agent": "order_agent",
                        "context_summary": "ctx",
                        "expectation": "brew",
                    },
                }})
            else:
                msg = AIMessage(content="Hello!", name="order_agent", id=f"msg-s{call_count[0]}")
                yield (("order_agent:abc",), {"agent": {"messages": [msg]}})

        shop.app.stream.side_effect = stream_handoff
        shop.customer_agent.get_initial_message.return_value = "hi"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)

        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)
        bus.drain()

        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        user_visible = [e for e in events if e.event_type == EventType.USER_VISIBLE]
        self.assertTrue(len(user_visible) >= 1)
        self.assertEqual(user_visible[0].agent_name, "order_agent")


class TestHandoverPauseAndResume(unittest.TestCase):
    """End-to-end test for the dashboard's pause/go toggle."""

    def test_pause_then_resume_in_single_flow(self):
        shop = _make_mock_shop()
        receiver_gate = threading.Event()

        sender_msg = AIMessage(
            content="Handing off to barista",
            name="order_agent",
            id="sender-msg-1",
        )
        receiver_msg = AIMessage(
            content="Brewing the latte!",
            name="barista_agent",
            id="receiver-msg-1",
        )

        def streaming_handover(*_a, **_kw):
            yield (("order_agent:abc",), {"agent": {
                "messages": [sender_msg],
                "active_agent": "barista_agent",
                "handoff_context": {
                    "from_agent": "order_agent",
                    "context_summary": "one latte for the customer",
                    "expectation": "brew it",
                },
            }})
            assert receiver_gate.wait(timeout=10), (
                "Test bug: receiver_gate was never released."
            )
            yield (("barista_agent:def",), {"agent": {
                "messages": [receiver_msg],
            }})

        shop.app.stream.side_effect = streaming_handover
        shop.customer_agent.get_initial_message.return_value = "I want a latte"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.pause_on_next_handover = True
        runner.start(scenario_index=0)

        deadline = time.time() + 5.0
        while time.time() < deadline and not runner.is_paused:
            time.sleep(0.01)
        self.assertTrue(
            runner.is_paused,
            "Runner did not pause at the handover seam; pause toggle not honored.",
        )

        mid_events = bus.drain()
        handoffs = [e for e in mid_events if e.event_type == EventType.HANDOFF]
        receiver_msgs = [
            e for e in mid_events
            if e.event_type == EventType.AGENT_MESSAGE
            and e.agent_name == "barista_agent"
        ]
        self.assertEqual(len(handoffs), 1)
        self.assertEqual(handoffs[0].target_agent, "barista_agent")
        self.assertEqual(len(receiver_msgs), 0)

        self.assertEqual(shop.app.stream.call_count, 1)

        receiver_gate.set()
        runner.resume()

        runner._thread.join(timeout=5)
        self.assertFalse(runner.is_running)

        post_events = bus.drain()
        receiver_msgs_post = [
            e for e in post_events
            if e.event_type == EventType.AGENT_MESSAGE
            and e.agent_name == "barista_agent"
        ]
        self.assertEqual(len(receiver_msgs_post), 1)
        self.assertEqual(receiver_msgs_post[0].content, "Brewing the latte!")
        self.assertEqual(runner._active_agent, "barista_agent")

    def test_pause_off_proceeds_without_blocking(self):
        shop = _make_mock_shop()

        sender_msg = AIMessage(content="ok", name="order_agent", id="s1")
        receiver_msg = AIMessage(content="ok", name="barista_agent", id="r1")

        def stream(*_a, **_kw):
            yield (("order_agent:abc",), {"agent": {
                "messages": [sender_msg],
                "active_agent": "barista_agent",
                "handoff_context": {
                    "from_agent": "order_agent",
                    "context_summary": "go",
                    "expectation": "brew",
                },
            }})
            yield (("barista_agent:def",), {"agent": {"messages": [receiver_msg]}})

        shop.app.stream.side_effect = stream
        shop.customer_agent.get_initial_message.return_value = "hi"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.pause_on_next_handover = False
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        self.assertFalse(runner.is_running)
        self.assertFalse(runner.is_paused)
        events = bus.drain()
        receiver_msgs = [
            e for e in events
            if e.event_type == EventType.AGENT_MESSAGE
            and e.agent_name == "barista_agent"
        ]
        self.assertEqual(len(receiver_msgs), 1)

    def test_pause_default_seeded_from_config(self):
        """CoffeeShopConfig.handover_pause_default seeds the runner flag."""
        shop = _make_mock_shop()
        shop.config = CoffeeShopConfig(handover_pause_default=True)
        runner = ConversationRunner(shop, EventBus())
        self.assertTrue(runner.pause_on_next_handover)

        shop2 = _make_mock_shop()
        shop2.config = CoffeeShopConfig(handover_pause_default=False)
        runner2 = ConversationRunner(shop2, EventBus())
        self.assertFalse(runner2.pause_on_next_handover)

    def test_pause_skipped_on_dedup_echo(self):
        """A router echo of the same handoff_context must NOT trigger a
        second pause — pause sits inside the dedup branch."""
        shop = _make_mock_shop()
        sender_msg = AIMessage(content="hand off", name="order_agent", id="s1")
        receiver_msg = AIMessage(content="brewing", name="barista_agent", id="r1")
        hc = {
            "from_agent": "order_agent",
            "context_summary": "one latte",
            "expectation": "brew",
        }

        def stream(*_a, **_kw):
            yield (("order_agent:abc",), {"agent": {
                "messages": [sender_msg],
                "active_agent": "barista_agent",
                "handoff_context": hc,
            }})
            yield ((), {"router": {
                "active_agent": "barista_agent",
                "handoff_context": hc,
            }})
            yield (("barista_agent:def",), {"agent": {"messages": [receiver_msg]}})

        shop.app.stream.side_effect = stream
        shop.customer_agent.get_initial_message.return_value = "hi"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.pause_on_next_handover = True
        runner.start(scenario_index=0)

        deadline = time.time() + 5.0
        while time.time() < deadline and not runner.is_paused:
            time.sleep(0.01)
        self.assertTrue(runner.is_paused)

        runner.resume()
        runner._thread.join(timeout=5)
        self.assertFalse(runner.is_running)

        events = bus.drain()
        handoffs = [e for e in events if e.event_type == EventType.HANDOFF]
        self.assertEqual(len(handoffs), 1)


class TestExtractTextHelper(unittest.TestCase):
    """_extract_text flattens both str and list-of-blocks content."""

    def test_str_content_passthrough(self):
        self.assertEqual(_extract_text("hello"), "hello")

    def test_empty_str(self):
        self.assertEqual(_extract_text(""), "")

    def test_list_of_blocks_extracts_text(self):
        content = [
            {"type": "text", "text": "Let me check inventory first"},
            {"type": "tool_use", "name": "check_inventory", "input": {}},
        ]
        self.assertEqual(_extract_text(content), "Let me check inventory first")

    def test_list_with_only_tool_use(self):
        content = [{"type": "tool_use", "name": "check_inventory", "input": {}}]
        self.assertEqual(_extract_text(content), "")

    def test_list_multiple_text_blocks_joined(self):
        content = [
            {"type": "text", "text": "First thought."},
            {"type": "text", "text": "Second thought."},
        ]
        self.assertEqual(_extract_text(content), "First thought.\nSecond thought.")

    def test_unknown_shape_returns_empty(self):
        self.assertEqual(_extract_text(None), "")
        self.assertEqual(_extract_text(42), "")


class TestPublishMessageNormallyThoughtSalvage(unittest.TestCase):
    """AGENT_THOUGHT is emitted before TOOL_CALL when an AIMessage carries
    both prose and tool_calls; without prose, only TOOL_CALL fires."""

    def _make_runner(self):
        shop = _make_mock_shop()
        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        return runner, bus

    def test_thought_emitted_before_tool_call_str_content(self):
        runner, bus = self._make_runner()
        msg = AIMessage(
            content="Let me check inventory first",
            tool_calls=[{"name": "check_inventory", "args": {}, "id": "tc1"}],
        )
        runner._publish_message(msg, "barista")
        events = bus.drain()
        self.assertEqual(
            [e.event_type for e in events],
            [EventType.AGENT_THOUGHT, EventType.TOOL_CALL],
        )
        self.assertEqual(events[0].content, "Let me check inventory first")
        self.assertEqual(events[0].tool_name, "check_inventory")
        self.assertEqual(events[0].agent_name, "barista")
        self.assertEqual(events[1].tool_name, "check_inventory")

    def test_thought_emitted_for_list_of_blocks_content(self):
        runner, bus = self._make_runner()
        msg = AIMessage(
            content=[
                {"type": "text", "text": "Espresso needs a fresh shot."},
                {"type": "tool_use", "name": "start_preparation",
                 "input": {"drink": "espresso"}, "id": "tc1"},
            ],
            tool_calls=[{"name": "start_preparation",
                         "args": {"drink": "espresso"}, "id": "tc1"}],
        )
        runner._publish_message(msg, "barista")
        events = bus.drain()
        self.assertEqual(
            [e.event_type for e in events],
            [EventType.AGENT_THOUGHT, EventType.TOOL_CALL],
        )
        self.assertEqual(events[0].content, "Espresso needs a fresh shot.")
        self.assertEqual(events[0].tool_name, "start_preparation")

    def test_no_thought_when_content_empty_str(self):
        runner, bus = self._make_runner()
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "check_inventory", "args": {}, "id": "tc1"}],
        )
        runner._publish_message(msg, "barista")
        events = bus.drain()
        self.assertEqual([e.event_type for e in events], [EventType.TOOL_CALL])

    def test_no_thought_when_content_only_tool_use_block(self):
        runner, bus = self._make_runner()
        msg = AIMessage(
            content=[
                {"type": "tool_use", "name": "check_inventory",
                 "input": {}, "id": "tc1"},
            ],
            tool_calls=[{"name": "check_inventory", "args": {}, "id": "tc1"}],
        )
        runner._publish_message(msg, "barista")
        events = bus.drain()
        self.assertEqual([e.event_type for e in events], [EventType.TOOL_CALL])

    def test_no_thought_when_content_only_whitespace(self):
        runner, bus = self._make_runner()
        msg = AIMessage(
            content="   \n  \t ",
            tool_calls=[{"name": "check_inventory", "args": {}, "id": "tc1"}],
        )
        runner._publish_message(msg, "barista")
        events = bus.drain()
        self.assertEqual([e.event_type for e in events], [EventType.TOOL_CALL])

    def test_text_only_message_still_emits_agent_message(self):
        """The elif msg.content branch is untouched — text-only turns still
        emit AGENT_MESSAGE, not AGENT_THOUGHT."""
        runner, bus = self._make_runner()
        msg = AIMessage(content="Your latte is ready.", tool_calls=[])
        runner._publish_message(msg, "barista")
        events = bus.drain()
        self.assertEqual([e.event_type for e in events], [EventType.AGENT_MESSAGE])
        self.assertEqual(events[0].content, "Your latte is ready.")


if __name__ == "__main__":
    unittest.main()
