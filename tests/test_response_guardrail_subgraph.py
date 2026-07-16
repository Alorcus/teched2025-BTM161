"""Graph-level tests for the response guardrail block-and-pushback loop.

Exercises `response_gateway_node` (built inside `create_agent_subgraph`) by
driving state through it with a stub LLM. Verifies:
  * A clean AIMessage passes through unchanged.
  * An off-menu AIMessage is removed from state, a corrective HumanMessage
    is appended with the `response_guardrail_correction` marker, and the
    graph re-invokes the LLM.
  * After the configured retry cap, the last (still-offending) AIMessage
    is emitted instead of retrying forever.
"""
import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from src.control_plane.gateway import Gateway
from src.control_plane.guardrails import HardGuardrail
from src.control_plane.log_sink import NullLogSink
from src.control_plane.predicates import off_menu_recommendation_predicate
from src.control_plane.subgraph import (
    _CORRECTION_KWARG,
    _MAX_RESPONSE_GUARDRAIL_RETRIES,
    _corrections_since_last_user_turn,
    _is_correction_message,
    create_agent_subgraph,
)
from src.control_plane.types import Effect


def _build_gateway() -> Gateway:
    gr = HardGuardrail(
        name="off_menu_recommendation",
        version="v1",
        tools=["assistant_message"],
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


def _stub_llm(responses):
    """Return a fake chat model that yields `responses[i]` on the i-th invoke."""
    state = {"count": 0}

    class _Stub:
        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, messages, config=None):
            i = state["count"]
            state["count"] = i + 1
            return responses[min(i, len(responses) - 1)]

    return _Stub(), state


class TestCorrectionCounter(unittest.TestCase):
    def test_no_corrections_returns_zero(self):
        msgs = [HumanMessage(content="I want a latte")]
        self.assertEqual(_corrections_since_last_user_turn(msgs), 0)

    def test_correction_marker_recognised(self):
        correction = HumanMessage(
            content="You recommended off-menu items.",
            additional_kwargs={_CORRECTION_KWARG: True},
        )
        self.assertTrue(_is_correction_message(correction))
        self.assertFalse(_is_correction_message(HumanMessage(content="hi")))

    def test_counts_corrections_since_last_user_turn(self):
        correction = lambda: HumanMessage(
            content="fix it", additional_kwargs={_CORRECTION_KWARG: True},
        )
        msgs = [
            HumanMessage(content="first user turn"),
            AIMessage(content="Try our mocha"),
            correction(),
            AIMessage(content="Try our chai"),
            correction(),
        ]
        self.assertEqual(_corrections_since_last_user_turn(msgs), 2)

    def test_new_user_turn_resets_counter(self):
        correction = HumanMessage(
            content="fix it", additional_kwargs={_CORRECTION_KWARG: True},
        )
        msgs = [
            HumanMessage(content="first user turn"),
            AIMessage(content="Try our mocha"),
            correction,
            HumanMessage(content="second user turn"),
        ]
        self.assertEqual(_corrections_since_last_user_turn(msgs), 0)


class TestResponseGuardrailInSubgraph(unittest.TestCase):
    """Compile the real subgraph with a stub LLM and drive it end-to-end."""

    def _compile(self, ai_responses):
        llm, calls = _stub_llm(ai_responses)
        gateway = _build_gateway()
        graph = create_agent_subgraph(
            agent_id="order_agent",
            llm=llm,
            tools=[],
            prompt="You are the order agent.",
            gateway=gateway,
        )
        return graph, calls

    def test_clean_message_passes_through(self):
        graph, calls = self._compile([
            AIMessage(content="One large latte coming up."),
        ])
        result = graph.invoke(
            {"messages": [HumanMessage(content="I want a latte")], "handoff_context": None},
            config={"configurable": {"thread_id": "t1"}},
        )
        last = result["messages"][-1]
        self.assertIsInstance(last, AIMessage)
        self.assertEqual(last.content, "One large latte coming up.")
        self.assertEqual(calls["count"], 1)

    def test_off_menu_message_triggers_retry(self):
        graph, calls = self._compile([
            AIMessage(content="How about a hazelnut latte?"),
            AIMessage(content="How about a latte?"),
        ])
        result = graph.invoke(
            {"messages": [HumanMessage(content="Recommend something")], "handoff_context": None},
            config={"configurable": {"thread_id": "t2"}},
        )
        self.assertEqual(calls["count"], 2)
        last = result["messages"][-1]
        self.assertIsInstance(last, AIMessage)
        self.assertEqual(last.content, "How about a latte?")
        # A correction HumanMessage should have been injected between the two AI turns.
        corrections = [m for m in result["messages"] if _is_correction_message(m)]
        self.assertEqual(len(corrections), 1)
        self.assertIn("menu", corrections[0].content.lower())
        # The first (offending) AIMessage should not remain in the emitted history.
        ai_contents = [m.content for m in result["messages"] if isinstance(m, AIMessage)]
        self.assertNotIn("How about a hazelnut latte?", ai_contents)

    def test_retry_cap_publishes_last_attempt(self):
        # LLM stubbornly recommends off-menu on every retry; cap = 3.
        # We supply 5 attempts (initial + 4 corrections) — after 3 corrections
        # the guardrail must give up and let the AIMessage through.
        graph, calls = self._compile([
            AIMessage(content="Try our mocha."),
            AIMessage(content="Try our mocha."),
            AIMessage(content="Try our mocha."),
            AIMessage(content="Try our mocha."),
            AIMessage(content="Try our mocha."),
        ])
        result = graph.invoke(
            {"messages": [HumanMessage(content="Recommend something")], "handoff_context": None},
            config={"configurable": {"thread_id": "t3"}},
        )
        # After the cap, an offending AIMessage is emitted; no further retries.
        last = result["messages"][-1]
        self.assertIsInstance(last, AIMessage)
        self.assertEqual(last.content, "Try our mocha.")
        # We should see the retry-cap worth of corrections and no more.
        corrections = [m for m in result["messages"] if _is_correction_message(m)]
        self.assertEqual(len(corrections), _MAX_RESPONSE_GUARDRAIL_RETRIES)


if __name__ == "__main__":
    unittest.main()
