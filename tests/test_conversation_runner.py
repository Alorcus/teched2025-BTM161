"""Tests 31-38: ConversationRunner.

Validates concurrency (lock, double-start, flag reset), error handling,
deduplication, max turns, and messages key matching.
"""
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from src.config import CoffeeShopConfig
from src.dashboard.interaction.event_bus import EventBus, EventType
from src.dashboard.interaction.conversation_runner import (
    ConversationRunner,
    MAX_CONVERSATION_TURNS,
    _summarize_tool_calls,
    _rejected_content,
    _extract_text,
)


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
        events = bus.drain()
        error_events = [e for e in events if "error" in (e.content or "").lower()]
        self.assertTrue(len(error_events) > 0)


class TestStreamErrorPublishesEventAndReturnsNone(unittest.TestCase):
    """Test 35: LLM errors during streaming are caught gracefully."""

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
    """Test 36: Same message emitted twice by stream is only dispatched once."""

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
                # Real handoff from inventory to barista.
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
                # Barista produces its final reply.
                final_msg = AIMessage(
                    content="Order complete!",
                    name="barista_agent",
                    id="msg-final",
                )
                yield (("barista_agent:def",), {"agent": {"messages": [final_msg]}})
                # Parent-level echo: the same handoff_context is still in state
                # and gets re-emitted by a terminal/router update. This is the
                # scenario that produced the duplicate HANDOFF in the global log.
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


def _active_shop():
    """Mock shop with an active CoffeeShopConfig and a configurable supervisor."""
    shop = _make_mock_shop()
    shop.config = CoffeeShopConfig(
        process_supervisor_active=True,
        process_supervisor_max_retries=3,
    )
    # supervisor is a real attribute, not a MagicMock auto-attr, so the runner
    # exercises the active path.
    shop.process_supervisor = MagicMock()
    shop.process_supervisor.observe.return_value = None
    shop.process_supervisor.critique.return_value = "supervisor-critique-text"
    shop.process_supervisor.append_violation.return_value = (
        "Violation:supervisor_retry_exhausted"
    )
    # By default get_state returns no messages — the runner falls back to
    # injecting the critique without a RemoveMessage. Tests that want to
    # exercise the RemoveMessage path override get_state explicitly.
    state_snapshot = MagicMock()
    state_snapshot.values = {"messages": []}
    shop.app.get_state.return_value = state_snapshot
    return shop


class TestActiveSupervisorSuppressesViolation(unittest.TestCase):
    """A Violation on an AIMessage suppresses the normal publish, emits
    AGENT_MESSAGE_REJECTED, patches state, and re-streams."""

    def test_suppress_publish_and_resume(self):
        shop = _active_shop()
        bad_msg = AIMessage(content="off-topic chitchat", name="order_agent", id="bad-1")
        good_msg = AIMessage(content="Got it: one latte!", name="order_agent", id="good-1")
        # Simulate the parent-graph checkpoint storing the offending AIMessage
        # under a different id (typical when streaming through subgraphs).
        bad_in_state = AIMessage(content="off-topic chitchat", name="order_agent", id="state-bad-1")
        snap = MagicMock()
        snap.values = {"messages": [bad_in_state]}
        shop.app.get_state.return_value = snap

        verdicts = iter([
            "Violation:llm_unknown_activity_A99 | AIMessage[order_agent] text=off-topic",
            "Execution:A01:identify_customer_request | AIMessage[order_agent] text=Got it",
        ])
        shop.process_supervisor.observe.side_effect = lambda *_a, **_k: next(verdicts)

        call_count = [0]

        def stream(*_a, **_kw):
            call_count[0] += 1
            if call_count[0] == 1:
                yield (("order_agent:abc",), {"agent": {"messages": [bad_msg]}})
            else:
                yield (("order_agent:abc",), {"agent": {"messages": [good_msg]}})

        shop.app.stream.side_effect = stream
        shop.customer_agent.get_initial_message.return_value = "I want a latte"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE
                      and e.agent_name == "order_agent"]
        rejected = [e for e in events if e.event_type == EventType.AGENT_MESSAGE_REJECTED]

        self.assertEqual(len(rejected), 1, f"want 1 rejection, got {len(rejected)}")
        self.assertEqual(rejected[0].content, "off-topic chitchat")
        self.assertIn("Violation:", rejected[0].supervisor_line or "")
        self.assertEqual(rejected[0].rejection_reason, "supervisor-critique-text")

        # Exactly one normal AGENT_MESSAGE — the corrected attempt.
        self.assertEqual(len(agent_msgs), 1)
        self.assertEqual(agent_msgs[0].content, "Got it: one latte!")

        # update_state was called once with a RemoveMessage targeting the
        # PARENT-graph state id (not the subgraph stream id). The critique
        # itself rides into the next stream() call as input, NOT inside the
        # state patch — verified separately on shop.app.stream's call_args.
        self.assertEqual(shop.app.update_state.call_count, 1)
        patch_dict = shop.app.update_state.call_args[0][1]
        patch_msgs = patch_dict["messages"]
        self.assertTrue(any(isinstance(m, RemoveMessage) and m.id == "state-bad-1" for m in patch_msgs))
        # The second stream() call was invoked with the critique HumanMessage
        # as fresh input.
        self.assertEqual(shop.app.stream.call_count, 2)
        second_input = shop.app.stream.call_args_list[1][0][0]
        self.assertIsNotNone(second_input)
        self.assertIn("messages", second_input)
        crit = next(m for m in second_input["messages"] if isinstance(m, HumanMessage))
        self.assertIn("off-topic chitchat", crit.content)
        self.assertIn("supervisor-critique-text", crit.content)
        self.assertIn("SYSTEM CONTROL", crit.content)


class TestCritiqueAccumulatesInSingleHumanMessage(unittest.TestCase):
    """Two consecutive violations from the same agent in the same turn must
    accumulate into a SINGLE HumanMessage (same id, growing content)."""

    def test_accumulate(self):
        shop = _active_shop()
        attempts = [
            AIMessage(content="weather report", name="order_agent", id="a-1"),
            AIMessage(content="haiku about leaves", name="order_agent", id="a-2"),
            AIMessage(content="One latte coming up!", name="order_agent", id="a-3"),
        ]
        verdicts = iter([
            "Violation:r1 | x",
            "Violation:r2 | y",
            "Execution:A01:identify_customer_request | z",
        ])
        critiques = iter(["crit-1", "crit-2"])
        shop.process_supervisor.observe.side_effect = lambda *_a, **_k: next(verdicts)
        shop.process_supervisor.critique.side_effect = lambda *_a, **_k: next(critiques)

        idx = [0]

        def stream(*_a, **_kw):
            i = idx[0]
            idx[0] += 1
            yield (("order_agent:abc",), {"agent": {"messages": [attempts[i]]}})

        shop.app.stream.side_effect = stream
        shop.customer_agent.get_initial_message.return_value = "hi"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        # Two rejections were published.
        events = bus.drain()
        rejected = [e for e in events if e.event_type == EventType.AGENT_MESSAGE_REJECTED]
        self.assertEqual(len(rejected), 2)

        # Each rejection should have produced a stream() invocation whose
        # input contains the critique HumanMessage. The SECOND invocation's
        # critique must reflect BOTH attempts (accumulated).
        self.assertEqual(shop.app.stream.call_count, 3)  # initial + 2 retries
        second_input = shop.app.stream.call_args_list[1][0][0]
        third_input = shop.app.stream.call_args_list[2][0][0]
        first_crit = next(m for m in second_input["messages"] if isinstance(m, HumanMessage))
        second_crit = next(m for m in third_input["messages"] if isinstance(m, HumanMessage))

        self.assertEqual(first_crit.id, second_crit.id)  # same HumanMessage
        self.assertIn("weather report", second_crit.content)
        self.assertIn("haiku about leaves", second_crit.content)
        self.assertIn("crit-1", second_crit.content)
        self.assertIn("crit-2", second_crit.content)
        # update_state was called for each rejection (with RemoveMessage when
        # the parent state contains a matching AIMessage; the default mock
        # snapshot has none so we expect at most call_count==2 update_states
        # carrying ToolMessage stubs or RemoveMessage. Verify it's at most 2.
        self.assertLessEqual(shop.app.update_state.call_count, 2)


class TestRetryCapDeadlocksInsteadOfBypass(unittest.TestCase):
    """Cap at max_retries: after N rejections the (N+1)th violation also
    publishes as REJECTED (suppressed, not normal), supervisor.append_violation
    is called with supervisor_retry_exhausted, and the conversation halts for
    that agent. Letting the let-through publish normally would let a
    non-compliant worker bypass the supervisor entirely, which is exactly the
    bug we're closing."""

    def test_cap(self):
        shop = _active_shop()
        # max_retries = 3 → expect 4 rejections (the 4th is the cap-hit).
        attempts = [
            AIMessage(content=f"bad-{i}", name="order_agent", id=f"id-{i}")
            for i in range(4)
        ]
        verdicts = iter([f"Violation:r{i} | x" for i in range(4)])
        shop.process_supervisor.observe.side_effect = lambda *_a, **_k: next(verdicts)

        idx = [0]

        def stream(*_a, **_kw):
            i = idx[0]
            idx[0] += 1
            if i >= len(attempts):
                return
            yield (("order_agent:abc",), {"agent": {"messages": [attempts[i]]}})

        shop.app.stream.side_effect = stream
        shop.customer_agent.get_initial_message.return_value = "hi"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        rejected = [e for e in events if e.event_type == EventType.AGENT_MESSAGE_REJECTED]
        agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE
                      and e.agent_name == "order_agent"]

        self.assertEqual(len(rejected), 4, f"want 4 rejections (3 retries + 1 cap-hit), got {len(rejected)}")
        self.assertEqual(len(agent_msgs), 0, "no normal AGENT_MESSAGE after cap")
        # supervisor_retry_exhausted was logged.
        shop.process_supervisor.append_violation.assert_called_once_with(
            "supervisor_retry_exhausted"
        )


class TestInactiveFlagPreservesPassiveBehavior(unittest.TestCase):
    """When process_supervisor_active=False, a Violation:* on an AIMessage
    publishes a normal AGENT_MESSAGE with supervisor_line stamped — current
    behaviour. No AGENT_MESSAGE_REJECTED, no update_state."""

    def test_passive_behavior(self):
        shop = _make_mock_shop()
        shop.config = CoffeeShopConfig(process_supervisor_active=False)
        shop.process_supervisor = MagicMock()
        shop.process_supervisor.observe.return_value = "Violation:llm_unknown_activity_A99 | x"

        msg = AIMessage(content="off-topic", name="order_agent", id="x")

        def stream(*_a, **_kw):
            yield (("order_agent:abc",), {"agent": {"messages": [msg]}})

        shop.app.stream.side_effect = stream
        shop.customer_agent.get_initial_message.return_value = "hi"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        rejected = [e for e in events if e.event_type == EventType.AGENT_MESSAGE_REJECTED]
        agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE
                      and e.agent_name == "order_agent"]

        self.assertEqual(len(rejected), 0)
        self.assertEqual(len(agent_msgs), 1)
        self.assertEqual(agent_msgs[0].content, "off-topic")
        self.assertTrue(
            (agent_msgs[0].supervisor_line or "").startswith("Violation:")
        )
        shop.app.update_state.assert_not_called()
        shop.process_supervisor.critique.assert_not_called()


class TestViolationOnUserOrToolResultIgnored(unittest.TestCase):
    """Even with active=True, a Violation:* supervisor verdict on a non-AI
    message (or a non-swarm-agent author) does NOT trigger suppression."""

    def test_tool_message_passes_through(self):
        shop = _active_shop()
        # Supervisor (per regex) returns Violation only for the AI message we
        # don't actually generate; we just verify a ToolMessage that arrives
        # with a Violation line still publishes normally.
        shop.process_supervisor.observe.return_value = "Violation:weird | x"

        tool_msg = ToolMessage(
            content='{"ok": true}', name="check_inventory",
            tool_call_id="tc-1", id="tm-1",
        )

        def stream(*_a, **_kw):
            yield (("order_agent:abc",), {"agent": {"messages": [tool_msg]}})

        shop.app.stream.side_effect = stream
        shop.customer_agent.get_initial_message.return_value = "hi"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        rejected = [e for e in events if e.event_type == EventType.AGENT_MESSAGE_REJECTED]
        tool_results = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        self.assertEqual(len(rejected), 0)
        self.assertEqual(len(tool_results), 1)
        shop.app.update_state.assert_not_called()


class TestToolCallViolationSummarizedToProse(unittest.TestCase):
    """An AIMessage with tool_calls (and no content) that violates the model
    must surface a PROSE summary as the rejected event content (not raw JSON)
    and that prose must appear inside the quoted-critique HumanMessage."""

    def test_tool_call_summary(self):
        shop = _active_shop()
        bad = AIMessage(
            content="",
            name="order_agent",
            id="bad-tc",
            tool_calls=[{
                "name": "transfer_to_barista",
                "args": {
                    "target_agent": "barista_agent",
                    "context_summary": "premature handoff",
                    "expectation": "brew",
                },
                "id": "tc-1",
            }],
        )
        good = AIMessage(content="Sure thing!", name="order_agent", id="good-tc")
        verdicts = iter([
            "Violation:premature_handoff | AIMessage[order_agent] tool_calls=[...]",
            "Execution:A01:identify_customer_request | x",
        ])
        shop.process_supervisor.observe.side_effect = lambda *_a, **_k: next(verdicts)

        idx = [0]
        msgs = [bad, good]

        def stream(*_a, **_kw):
            i = idx[0]; idx[0] += 1
            yield (("order_agent:abc",), {"agent": {"messages": [msgs[i]]}})

        shop.app.stream.side_effect = stream
        shop.customer_agent.get_initial_message.return_value = "hi"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        rejected = [e for e in events if e.event_type == EventType.AGENT_MESSAGE_REJECTED]
        tool_calls = [e for e in events if e.event_type == EventType.TOOL_CALL
                      and e.agent_name == "order_agent"]
        executed_tool_calls = [
            e for e in tool_calls
            if not (e.supervisor_line or "").startswith("REJECTED")
        ]
        rejected_tool_calls = [
            e for e in tool_calls
            if (e.supervisor_line or "").startswith("REJECTED")
        ]

        self.assertEqual(len(rejected), 1)
        self.assertIn("hand off to barista_agent", rejected[0].content)
        self.assertNotIn("{", rejected[0].content[:5])  # not raw JSON
        self.assertEqual(len(executed_tool_calls), 0,
                         "no executed tool_call for rejected attempt")
        self.assertEqual(len(rejected_tool_calls), 1,
                         "exactly one render-only TOOL_CALL row for the rejected tool_call")
        self.assertEqual(rejected_tool_calls[0].tool_name, "transfer_to_barista")

        # Critique HumanMessage rides in the resume input, not the state patch.
        second_input = shop.app.stream.call_args_list[1][0][0]
        crit = next(m for m in second_input["messages"] if isinstance(m, HumanMessage))
        self.assertIn("hand off to barista_agent", crit.content)


class TestSummarizeToolCalls(unittest.TestCase):
    """Pure function: tool_calls → prose summary."""

    def test_handoff(self):
        s = _summarize_tool_calls([{
            "name": "transfer_to_barista",
            "args": {"target_agent": "barista_agent", "context_summary": "one latte"},
        }])
        self.assertIn("hand off to barista_agent", s)
        self.assertIn("one latte", s)

    def test_regular_tool(self):
        s = _summarize_tool_calls([{
            "name": "process_order",
            "args": {"item": "latte"},
        }])
        self.assertIn("process_order", s)
        self.assertIn('"item"', s)

    def test_multi(self):
        s = _summarize_tool_calls([
            {"name": "a", "args": {}},
            {"name": "b", "args": {}},
        ])
        self.assertIn(";", s)
        self.assertIn("call a", s)
        self.assertIn("call b", s)

    def test_long_args_truncated(self):
        s = _summarize_tool_calls([{"name": "x", "args": {"k": "v" * 500}}])
        # 120 char + ellipsis cap
        self.assertLess(len(s), 220)


class TestRejectedContentFallsBackToToolCalls(unittest.TestCase):
    """_rejected_content uses text content if present, otherwise summarizes
    tool_calls. Required so AGENT_MESSAGE_REJECTED.content is never empty."""

    def test_text_preferred(self):
        msg = AIMessage(content="hello", tool_calls=[], id="m1")
        self.assertEqual(_rejected_content(msg), "hello")

    def test_tool_calls_fallback(self):
        msg = AIMessage(
            content="",
            id="m2",
            tool_calls=[{"name": "f", "args": {"a": 1}, "id": "tc1"}],
        )
        self.assertIn("call f", _rejected_content(msg))


class TestSupervisorDisabledNoOp(unittest.TestCase):
    """When process_supervisor_enabled=False, CoffeeShop.process_supervisor is
    None. The runner must publish messages normally with supervisor_line=None
    and never invoke supervisor methods (because there are none)."""

    def test_runner_publishes_without_supervisor(self):
        shop = _make_mock_shop()
        shop.config = CoffeeShopConfig(process_supervisor_enabled=False)
        shop.process_supervisor = None

        msg = AIMessage(content="welcome!", name="order_agent", id="m1")

        def stream(*_a, **_kw):
            yield (("order_agent:abc",), {"agent": {"messages": [msg]}})

        shop.app.stream.side_effect = stream
        shop.customer_agent.get_initial_message.return_value = "hi"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        rejected = [e for e in events if e.event_type == EventType.AGENT_MESSAGE_REJECTED]
        agent_msgs = [e for e in events if e.event_type == EventType.AGENT_MESSAGE
                      and e.agent_name == "order_agent"]

        self.assertEqual(len(rejected), 0)
        self.assertEqual(len(agent_msgs), 1)
        self.assertEqual(agent_msgs[0].content, "welcome!")
        self.assertIsNone(agent_msgs[0].supervisor_line)
        shop.app.update_state.assert_not_called()


class TestHandoverPauseAndResume(unittest.TestCase):
    """End-to-end test for the dashboard's pause/go toggle.

    Drives the ConversationRunner with a stream that emits one handover, then
    asserts (1) with pause_on_next_handover=True the runner halts AFTER the
    sender's handoff is published and BEFORE the receiver's first message is
    published; (2) calling runner.resume() releases the wait, the receiver's
    message is then published, and the conversation completes normally.
    Both halves run in a single end-to-end flow so a regression in either
    direction fails the test.
    """

    def test_pause_then_resume_in_single_flow(self):
        shop = _make_mock_shop()

        # Block the stream generator between yielding the handoff update and
        # yielding the receiver's first message. This proves the runner has
        # actually stopped *consuming* the stream, not just that we didn't
        # call its callbacks — i.e. real backpressure on the receiver.
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
            # 1. Sender's outgoing message + handoff_context — runner publishes
            #    HANDOFF here and (if toggle on) enters _wait_for_resume.
            yield (("order_agent:abc",), {"agent": {
                "messages": [sender_msg],
                "active_agent": "barista_agent",
                "handoff_context": {
                    "from_agent": "order_agent",
                    "context_summary": "one latte for the customer",
                    "expectation": "brew it",
                },
            }})
            # 2. Block before the receiver speaks. Without a pause seam in
            #    the runner, this would still gate the receiver — but the
            #    HANDOFF event would have published, the wait would not, and
            #    the assertion below (that no AGENT_MESSAGE from barista has
            #    been published yet but the runner thread is parked) would
            #    still hold for the WRONG reason (parked on receiver_gate
            #    instead of on _resume_event). To distinguish, we wait only
            #    a short time on receiver_gate, then assert the runner is
            #    paused via runner.is_paused — that property is True iff the
            #    runner is parked at _resume_event specifically.
            assert receiver_gate.wait(timeout=10), (
                "Test bug: receiver_gate was never released; the runner "
                "either never reached the receiver-message yield or the "
                "stream generator was abandoned."
            )
            # 3. Receiver's first message.
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

        # --- Half 1: assert pause is honored ---
        # Wait for the runner to reach the pause seam. The HANDOFF event is
        # published immediately before _wait_for_resume() clears the resume
        # Event, so we poll for is_paused becoming True with a tight bound.
        # Failing here means the toggle was ignored (test fails if pause is
        # not honored, per the spec).
        deadline = time.time() + 5.0
        while time.time() < deadline and not runner.is_paused:
            time.sleep(0.01)
        self.assertTrue(
            runner.is_paused,
            "Runner did not pause at the handover seam; pause toggle not honored.",
        )

        # The HANDOFF event must be on the bus (sender emitted), but the
        # receiver's AGENT_MESSAGE must NOT be there yet (receiver has not
        # accepted). We drain partially via a peek that preserves bus order.
        mid_events = bus.drain()
        handoffs = [e for e in mid_events if e.event_type == EventType.HANDOFF]
        receiver_msgs = [
            e for e in mid_events
            if e.event_type == EventType.AGENT_MESSAGE
            and e.agent_name == "barista_agent"
        ]
        self.assertEqual(
            len(handoffs), 1,
            f"Expected 1 HANDOFF event after pause, got {len(handoffs)}",
        )
        self.assertEqual(handoffs[0].target_agent, "barista_agent")
        self.assertEqual(
            len(receiver_msgs), 0,
            "Receiver's AGENT_MESSAGE was published before resume — "
            "pause did not halt at the correct seam.",
        )

        # Sanity: stream() must not have been called a second time yet
        # (the generator is still parked on receiver_gate.wait inside the
        # ongoing call).
        self.assertEqual(shop.app.stream.call_count, 1)

        # --- Half 2: toggle to Go and assert resume completes the handover ---
        # First release the upstream gate so the receiver_msg yield can fire
        # once the runner consumes it; then resume the runner. If resume()
        # does nothing, the runner stays parked at _resume_event.wait and
        # the join below times out — which fails this half of the test.
        receiver_gate.set()
        runner.resume()

        runner._thread.join(timeout=5)
        self.assertFalse(
            runner.is_running,
            "Runner did not finish after resume — pending handoff did not complete.",
        )

        # After resume, the receiver's AGENT_MESSAGE must have been published.
        post_events = bus.drain()
        receiver_msgs_post = [
            e for e in post_events
            if e.event_type == EventType.AGENT_MESSAGE
            and e.agent_name == "barista_agent"
        ]
        self.assertEqual(
            len(receiver_msgs_post), 1,
            "Receiver's AGENT_MESSAGE was not published after resume — "
            "conversation did not progress past the paused handover.",
        )
        self.assertEqual(receiver_msgs_post[0].content, "Brewing the latte!")

        # The runner reflects the post-handoff active agent.
        self.assertEqual(runner._active_agent, "barista_agent")

    def test_pause_off_proceeds_without_blocking(self):
        """Sanity guard: with the toggle off (default), the runner must NOT
        pause — a regression that always pauses would deadlock real use."""
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

        # Both updates carry the SAME handoff_context — the second is a
        # router echo and must be deduped.
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

        # Wait for the (single) pause to occur.
        deadline = time.time() + 5.0
        while time.time() < deadline and not runner.is_paused:
            time.sleep(0.01)
        self.assertTrue(runner.is_paused)

        # Resume; runner must finish without parking a second time.
        runner.resume()
        runner._thread.join(timeout=5)
        self.assertFalse(runner.is_running)

        events = bus.drain()
        handoffs = [e for e in events if e.event_type == EventType.HANDOFF]
        # Dedup means exactly one HANDOFF was published even though the
        # stream emitted handoff_context twice.
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
        runner._publish_message_normally(msg, "barista", None)
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
        runner._publish_message_normally(msg, "barista", None)
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
        runner._publish_message_normally(msg, "barista", None)
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
        runner._publish_message_normally(msg, "barista", None)
        events = bus.drain()
        self.assertEqual([e.event_type for e in events], [EventType.TOOL_CALL])

    def test_no_thought_when_content_only_whitespace(self):
        runner, bus = self._make_runner()
        msg = AIMessage(
            content="   \n  \t ",
            tool_calls=[{"name": "check_inventory", "args": {}, "id": "tc1"}],
        )
        runner._publish_message_normally(msg, "barista", None)
        events = bus.drain()
        self.assertEqual([e.event_type for e in events], [EventType.TOOL_CALL])

    def test_text_only_message_still_emits_agent_message(self):
        """The elif msg.content branch is untouched — text-only turns still
        emit AGENT_MESSAGE, not AGENT_THOUGHT."""
        runner, bus = self._make_runner()
        msg = AIMessage(content="Your latte is ready.", tool_calls=[])
        runner._publish_message_normally(msg, "barista", None)
        events = bus.drain()
        self.assertEqual([e.event_type for e in events], [EventType.AGENT_MESSAGE])
        self.assertEqual(events[0].content, "Your latte is ready.")


if __name__ == "__main__":
    unittest.main()