"""Tests for the response-guardrail block-and-pushback loop.

The enforcement points changed: response guardrails are now evaluated by
`Gateway.evaluate_assistant_message` and applied by the runners consuming
`app.stream(...)`, not by an in-subgraph node. This suite covers:

  * `Gateway.evaluate_assistant_message` returns the right decision shape.
  * `ConversationEngine` (headless / simulate) suppresses the offending
    AIMessage from `last_agent_message` and retries with a critique.
  * `ConversationRunner._evaluate_response_guardrails` returns the deny
    reason so `_handle_response_guardrail_violation` can suppress + critique.
"""
import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.conversation import ConversationEngine
from src.control_plane.gateway import Gateway
from src.control_plane.guardrails import HardGuardrail
from src.control_plane.log_sink import NullLogSink
from src.control_plane.predicates import off_menu_recommendation_predicate
from src.control_plane.types import Effect


def _build_gateway(agent_id: str = "order_agent") -> Gateway:
    gr = HardGuardrail(
        name="off_menu_recommendation",
        version="v1",
        tools=[Gateway.RESPONSE_TOOL_NAME],
        effect=Effect.DENY,
        predicate=off_menu_recommendation_predicate("deny"),
        predicate_args={},
    )
    return Gateway(
        agent_id=agent_id,
        guardrails=[gr],
        allowed_handovers=[],
        snapshot_id="snap-test",
        log_sink=NullLogSink(),
    )


class TestGatewayEvaluateAssistantMessage(unittest.TestCase):
    def test_clean_message_allowed(self):
        gateway = _build_gateway()
        decision = gateway.evaluate_assistant_message(
            content="One large latte coming up.",
            message_id="ai-1",
            state={},
        )
        self.assertEqual(decision.final_decision, Effect.ALLOW)
        self.assertEqual(decision.tool_name, Gateway.RESPONSE_TOOL_NAME)
        self.assertEqual(decision.deny_reason_for_llm, "")

    def test_off_menu_message_denied_with_reason(self):
        gateway = _build_gateway()
        decision = gateway.evaluate_assistant_message(
            content="How about a hazelnut latte?",
            message_id="ai-1",
            state={},
        )
        self.assertEqual(decision.final_decision, Effect.DENY)
        self.assertIn("menu", decision.deny_reason_for_llm.lower())

    def test_empty_content_allowed(self):
        gateway = _build_gateway()
        decision = gateway.evaluate_assistant_message(
            content="",
            message_id="ai-1",
            state={},
        )
        self.assertEqual(decision.final_decision, Effect.ALLOW)


class TestConversationEngineResponseGuardrail(unittest.TestCase):
    """Drive `ConversationEngine.send_message` with a scripted app stream to
    verify that a denied assistant message is suppressed from
    `last_agent_message` and triggers a re-invoke with a critique."""

    def _make_stream_output(self, contents: list[str]):
        """One stream() invocation yields all its (ns, update) pairs and stops.
        Each string in `contents` becomes one AIMessage from `order_agent`."""
        updates = []
        for i, text in enumerate(contents):
            ai = AIMessage(content=text, name="order_agent", id=f"ai-{i}")
            updates.append((("order_agent",), {"llm": {"messages": [ai]}}))
        return iter(updates)

    def _engine(self, script: list[list[str]]):
        """`script` is a list of stream outputs — one per expected app.stream()
        call. We pop from the head on each call."""
        remaining = list(script)

        app = MagicMock()
        app.stream.side_effect = lambda *a, **kw: self._make_stream_output(remaining.pop(0))
        # get_state/update_state are used to remove the denied message from
        # checkpointed state. Return an empty snapshot so the lookup is a no-op.
        snapshot = MagicMock()
        snapshot.values = {"messages": []}
        app.get_state.return_value = snapshot

        engine = ConversationEngine(
            app,
            mlflow_enabled=False,
            setup_name="baseline",
            gateways={"order_agent": _build_gateway()},
        )
        return engine, app

    def test_clean_message_passes_through(self):
        engine, app = self._engine([["One large latte coming up."]])
        reply = engine.send_message("t1", "I want a latte")
        self.assertEqual(reply, "One large latte coming up.")
        self.assertEqual(app.stream.call_count, 1)

    def test_denied_message_triggers_retry_and_returns_clean_second_reply(self):
        engine, app = self._engine([
            ["How about a hazelnut latte?"],
            ["One large latte coming up."],
        ])
        reply = engine.send_message("t2", "Recommend something")
        self.assertEqual(reply, "One large latte coming up.")
        self.assertEqual(app.stream.call_count, 2)
        # Second call must have received a critique HumanMessage as its input
        # (as a fresh turn), not None.
        second_input = app.stream.call_args_list[1][0][0]
        self.assertIsNotNone(second_input)
        msgs = second_input["messages"]
        self.assertEqual(len(msgs), 1)
        self.assertIsInstance(msgs[0], HumanMessage)
        self.assertIn("menu", msgs[0].content.lower())

    def test_retry_cap_stops_the_loop(self):
        # Off-menu every time: engine must give up after the retry cap and
        # return None (last successful reply is None).
        from src.conversation import MAX_RESPONSE_GUARDRAIL_RETRIES

        script = [["Try our mocha."]] * (MAX_RESPONSE_GUARDRAIL_RETRIES + 2)
        engine, app = self._engine(script)
        reply = engine.send_message("t3", "Recommend something")
        self.assertIsNone(reply)
        # Initial call + MAX_RESPONSE_GUARDRAIL_RETRIES retries.
        self.assertEqual(app.stream.call_count, MAX_RESPONSE_GUARDRAIL_RETRIES + 1)


class TestConversationRunnerResponseGuardrail(unittest.TestCase):
    """Unit-level check that the runner's response-guardrail hook returns the
    deny reason for offending assistant messages and None otherwise. Full
    stream-loop wiring is covered by test_conversation_runner.py."""

    def _make_runner(self, gateway: Gateway | None):
        # Import inside the test to avoid import-time coupling to Panel/pn.
        from src.dashboard.interaction.conversation_runner import ConversationRunner

        shop = MagicMock()
        shop.config = None  # forces defaults in ConversationRunner.__init__
        shop.process_supervisor = None
        shop.gateways = {"order_agent": gateway} if gateway else {}
        event_bus = MagicMock()
        return ConversationRunner(shop, event_bus)

    def test_returns_none_for_clean_message(self):
        runner = self._make_runner(_build_gateway())
        msg = AIMessage(content="One large latte coming up.", name="order_agent")
        self.assertIsNone(runner._evaluate_response_guardrails(msg, "order_agent"))

    def test_returns_deny_reason_for_off_menu(self):
        runner = self._make_runner(_build_gateway())
        msg = AIMessage(content="How about a hazelnut latte?", name="order_agent")
        reason = runner._evaluate_response_guardrails(msg, "order_agent")
        self.assertIsNotNone(reason)
        self.assertIn("menu", reason.lower())

    def test_returns_none_when_no_gateway_registered(self):
        runner = self._make_runner(None)
        msg = AIMessage(content="How about a hazelnut latte?", name="order_agent")
        self.assertIsNone(runner._evaluate_response_guardrails(msg, "order_agent"))

    def test_returns_none_for_non_swarm_agent(self):
        runner = self._make_runner(_build_gateway())
        msg = AIMessage(content="How about a hazelnut latte?", name="customer")
        self.assertIsNone(runner._evaluate_response_guardrails(msg, "customer"))


if __name__ == "__main__":
    unittest.main()
