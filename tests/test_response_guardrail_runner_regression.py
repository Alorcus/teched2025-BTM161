"""Regression: `_apply_state_patch_for_rejection` must not abort the pushback
loop when `update_state` raises.

The real failure surfaced as:
    ValueError: Attempting to delete a message with an ID that doesn't exist ('lc_run--...')

which came from the langgraph `add_messages` reducer during a `RemoveMessage`
update — the id we resolved from `get_state` doesn't match the id inside
whatever channel a routing branch reads from. Before this fix the exception
propagated up to `_stream_with_events`, was logged as "Failed to apply
supervisor state patch; aborting active-mode loop", and the retry loop broke
so the user saw a REJECTED message and no pushback.

After the fix the exception is swallowed at the `update_state` call site: the
rejected message stays in state (best-effort cleanup) but the critique still
drives re-invocation, so the loop recovers.
"""
import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.config import CoffeeShopConfig
from src.control_plane.gateway import Gateway
from src.control_plane.guardrails import HardGuardrail
from src.control_plane.log_sink import NullLogSink
from src.control_plane.predicates import off_menu_recommendation_predicate
from src.control_plane.types import Effect
from src.dashboard.interaction.conversation_runner import ConversationRunner
from src.dashboard.interaction.event_bus import EventBus, EventType


def _gateway() -> Gateway:
    gr = HardGuardrail(
        name="off_menu_recommendation",
        version="v1",
        tools=[Gateway.RESPONSE_TOOL_NAME],
        effect=Effect.DENY,
        predicate=off_menu_recommendation_predicate("deny"),
        predicate_args={},
    )
    return Gateway(
        agent_id="order_agent",
        guardrails=[gr],
        allowed_handovers=[],
        snapshot_id="snap-test",
        log_sink=NullLogSink(),
    )


def _make_shop_with_failing_update_state():
    """Mock shop whose `update_state` raises the exact ValueError langgraph
    surfaces when a `RemoveMessage` targets an id absent from a routing-branch
    channel."""
    shop = MagicMock()
    shop._get_config.return_value = {"configurable": {"thread_id": "test"}}
    shop.customer_agent = MagicMock()
    shop.config = CoffeeShopConfig(
        process_supervisor_active=True,
        process_supervisor_max_retries=3,
    )
    supervisor = MagicMock()
    supervisor.observe.return_value = None
    supervisor.critique.return_value = "supervisor-critique"
    shop.process_supervisor = supervisor
    shop.gateways = {"order_agent": _gateway()}
    # get_state returns a state with an AIMessage whose content matches the
    # streamed offender — so `_find_state_message_id` resolves an id and the
    # runner will attempt a RemoveMessage. That's what makes update_state
    # actually get called (and fail) in this scenario.
    state_snap = MagicMock()
    state_snap.values = {
        "messages": [
            AIMessage(
                content="Try our hazelnut latte!",
                name="order_agent",
                id="state-bad-1",
            ),
        ],
    }
    shop.app.get_state.return_value = state_snap
    shop.app.update_state.side_effect = ValueError(
        "Attempting to delete a message with an ID that doesn't exist ('state-bad-1')"
    )
    return shop


class TestResponseGuardrailAbortRegression(unittest.TestCase):
    def test_update_state_failure_does_not_abort_pushback(self):
        shop = _make_shop_with_failing_update_state()

        bad = AIMessage(
            content="Try our hazelnut latte!",
            name="order_agent",
            id="bad-1",
        )
        good = AIMessage(
            content="One large latte coming up.",
            name="order_agent",
            id="good-1",
        )

        call_count = [0]

        def stream(*_a, **_kw):
            call_count[0] += 1
            if call_count[0] == 1:
                yield (("order_agent:abc",), {"agent": {"messages": [bad]}})
            else:
                yield (("order_agent:abc",), {"agent": {"messages": [good]}})

        shop.app.stream.side_effect = stream
        shop.customer_agent.get_initial_message.return_value = "recommend something"
        shop.customer_agent.respond_to.return_value = None

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        runner.start(scenario_index=0)
        runner._thread.join(timeout=5)

        events = bus.drain()
        rejected = [e for e in events if e.event_type == EventType.AGENT_MESSAGE_REJECTED]
        agent_msgs = [
            e for e in events
            if e.event_type == EventType.AGENT_MESSAGE and e.agent_name == "order_agent"
        ]
        log_msgs = [e for e in events if e.event_type == EventType.LOG_MESSAGE]

        self.assertFalse(
            any("aborting active-mode loop" in (e.content or "") for e in log_msgs),
            "runner aborted the active-mode loop instead of continuing pushback",
        )
        self.assertGreaterEqual(
            len(rejected), 1,
            "expected the off-menu response to publish an AGENT_MESSAGE_REJECTED",
        )
        self.assertGreaterEqual(
            shop.app.stream.call_count, 2,
            "expected the runner to re-invoke stream() after rejection despite "
            "update_state failing",
        )
        self.assertTrue(
            any("large latte" in (m.content or "") for m in agent_msgs),
            f"expected the corrected on-menu reply to publish; got "
            f"{[m.content for m in agent_msgs]}",
        )
        second_input = shop.app.stream.call_args_list[1][0][0]
        self.assertIsNotNone(second_input)
        self.assertIn("messages", second_input)
        crit = next(
            (m for m in second_input["messages"] if isinstance(m, HumanMessage)),
            None,
        )
        self.assertIsNotNone(crit, "expected a critique HumanMessage on the retry")
        self.assertIn("menu", crit.content.lower())


if __name__ == "__main__":
    unittest.main()
