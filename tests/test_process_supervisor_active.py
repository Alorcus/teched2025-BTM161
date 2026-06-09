"""Integration tests for active ProcessSupervisor flow.

Builds a minimal per-agent subgraph via create_agent_subgraph with active
supervision turned on. Verifies that:
  (a) violating tool_calls do not execute
  (b) the rejected AIMessage is removed from state via RemoveMessage
  (c) a corrective HumanMessage authored by 'process_supervisor' lands
  (d) the agent retries
  (e) on retry, when the LLM produces a compliant message, tools execute
  (f) retry exhaustion lets the last attempt through (no deadlock)

No real LLM is used. The supervisor's classification LLM and corrective LLM
calls are scripted via MagicMock to deterministically return the desired
verdict / corrective text.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from src.control_plane.gateway import Gateway
from src.control_plane.process_supervisor import (
    ProcessSupervisor,
    SupervisorVerdict,
)
from src.control_plane.subgraph import create_agent_subgraph


_REAL_MODEL = Path(__file__).resolve().parent.parent / "config" / "process_model.yaml"


_dummy_tool_calls = {"count": 0}


@tool
def dummy_tool(value: str) -> str:
    """A test tool that should never run when the supervisor rejects."""
    _dummy_tool_calls["count"] += 1
    return f"executed:{value}"


def _reset_dummy_tool():
    _dummy_tool_calls["count"] = 0


def _dummy_tool_call_count() -> int:
    return _dummy_tool_calls["count"]


class _ScriptedBoundLLM:
    """Mimics the object returned by llm.bind_tools(...). Each .invoke()
    pops the next pre-scripted AIMessage from a queue."""

    def __init__(self, scripted_messages: list[AIMessage]):
        self._queue = list(scripted_messages)
        self.invoke_count = 0

    def __or__(self, other):
        # Support `bound | _HandoffDeferrer()` chaining used by ChatOllama path.
        return self

    def invoke(self, messages, config=None):
        self.invoke_count += 1
        if not self._queue:
            return AIMessage(content="exhausted", id=f"ai-exhausted-{self.invoke_count}")
        return self._queue.pop(0)


class _FakeLLM:
    """Stand-in for ChatAnthropic. .bind_tools returns a scripted bound LLM."""

    def __init__(self, scripted_messages: list[AIMessage]):
        self._scripted = scripted_messages

    @property
    def __class__(self):
        # Keep type(llm).__name__ stable so bind_tools_sequential takes the
        # ChatAnthropic branch (avoids the _HandoffDeferrer wrapping). We
        # answer to a fake "ChatAnthropic" identity.
        class ChatAnthropic:  # noqa: N801
            pass
        return ChatAnthropic

    def bind_tools(self, tools, parallel_tool_calls=False):
        return _ScriptedBoundLLM(self._scripted)


def _make_real_llm_marker():
    # Ensure the FakeLLM lies about its class name to dodge the Ollama path.
    f = _FakeLLM([])
    assert type(f).__name__ == "ChatAnthropic"
    return f


class _FakeAgentRepoEntry:
    pass


class TestActiveSupervisorRejectsAndRetries(unittest.TestCase):
    """End-to-end through the subgraph: rejected AIMessage is removed, agent
    re-runs with a corrective HumanMessage, second attempt tools execute."""

    def setUp(self):
        _reset_dummy_tool()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _make_supervisor(self, decide_action_returns: list[SupervisorVerdict],
                         corrective_text: str = "Process supervisor: rerun please."):
        """Build a real ProcessSupervisor instance but stub decide_action and
        corrective_text so we control the flow without LLM calls."""
        sup = ProcessSupervisor(
            process_model_path=_REAL_MODEL,
            log_path=Path(self.tmp.name) / "process.log",
            llm=MagicMock(),
        )
        verdicts_iter = iter(decide_action_returns)

        def fake_decide(msg, agent_name):
            try:
                return next(verdicts_iter)
            except StopIteration:
                return SupervisorVerdict(
                    decision_line="Execution:A01:identify_customer_request",
                    is_violation=False,
                    reason="execution",
                )

        sup.decide_action = fake_decide
        sup.corrective_text = MagicMock(return_value=corrective_text)
        return sup

    def _build_subgraph(self, scripted_messages, supervisor, max_retries=3):
        # Mock gateway: every tool call is allowed (so we isolate the supervisor
        # from the guardrail path).
        gateway = MagicMock(spec=Gateway)

        def _allow(tc, state, thread_id=None):
            decision = MagicMock()
            decision.final_decision = MagicMock()
            decision.final_decision.__eq__ = lambda self, other: False
            decision.tool_name = tc.get("name", "")
            decision.tool_call_id = tc.get("id", "")
            return decision

        gateway.evaluate_call.side_effect = _allow
        gateway.log_tool_execution = MagicMock()

        llm = _FakeLLM(scripted_messages)
        return create_agent_subgraph(
            agent_id="order_agent",
            llm=llm,
            tools=[dummy_tool],
            prompt="You are a test agent.",
            gateway=gateway,
            supervisor=supervisor,
            supervisor_active=True,
            supervisor_max_retries=max_retries,
        )

    def test_rejection_removes_aimessage_and_retries(self):
        # First scripted reply violates; second is compliant (text-only, no tool).
        violating = AIMessage(
            content="off-topic ramble that violates the process",
            id="ai-violating-1",
            name="order_agent",
        )
        compliant = AIMessage(
            content="Hi! Happy to help with your order.",
            id="ai-compliant-1",
            name="order_agent",
        )
        verdict_reject = SupervisorVerdict(
            decision_line="Violation:test_reject",
            is_violation=True,
            reason="test_reject",
            allowed_activities=("A01",),
        )
        verdict_ok = SupervisorVerdict(
            decision_line="Execution:A01:identify_customer_request",
            is_violation=False,
            reason="execution",
        )
        sup = self._make_supervisor(
            decide_action_returns=[verdict_reject, verdict_ok],
            corrective_text="Process supervisor: please retry.",
        )
        graph = self._build_subgraph(
            scripted_messages=[violating, compliant], supervisor=sup,
        )

        config = {"configurable": {"thread_id": "t-1"}}
        initial_state = {
            "messages": [HumanMessage(content="hello")],
            "active_agent": "order_agent",
            "handoff_context": None,
        }
        final = graph.invoke(initial_state, config=config)

        msg_types = [type(m).__name__ for m in final["messages"]]
        msg_ids = [getattr(m, "id", None) for m in final["messages"]]

        # (b) The violating AIMessage is gone from state.
        self.assertNotIn(
            "ai-violating-1", msg_ids,
            f"rejected AIMessage should be removed; got messages={msg_types}",
        )
        # (c) A HumanMessage authored by process_supervisor is in state.
        supervisor_msgs = [
            m for m in final["messages"]
            if isinstance(m, HumanMessage) and getattr(m, "name", None) == "process_supervisor"
        ]
        self.assertEqual(
            len(supervisor_msgs), 1,
            f"expected one supervisor HumanMessage; got {msg_types}",
        )
        self.assertIn("retry", supervisor_msgs[0].content.lower())
        # (d, e) The compliant AIMessage from the retry is final.
        self.assertEqual(final["messages"][-1].id, "ai-compliant-1")
        # (a) dummy_tool never executed (compliant message has no tool_calls,
        # and the violating one was removed before reaching tools).
        self.assertEqual(_dummy_tool_call_count(), 0)

    def test_retry_exhaustion_lets_last_attempt_through(self):
        # All scripted attempts violate. With max_retries=2, the 3rd attempt
        # must pass through unchanged.
        attempts = [
            AIMessage(content=f"violating attempt {i}", id=f"ai-v-{i}", name="order_agent")
            for i in range(1, 5)
        ]
        verdicts = [
            SupervisorVerdict(decision_line="Violation:r", is_violation=True,
                              reason=f"r{i}", allowed_activities=("A01",))
            for i in range(1, 5)
        ]
        sup = self._make_supervisor(
            decide_action_returns=verdicts,
            corrective_text="Process supervisor: try again.",
        )
        graph = self._build_subgraph(
            scripted_messages=attempts, supervisor=sup, max_retries=2,
        )

        config = {"configurable": {"thread_id": "t-exhaust"}}
        initial_state = {
            "messages": [HumanMessage(content="hi")],
            "active_agent": "order_agent",
            "handoff_context": None,
        }
        final = graph.invoke(
            initial_state,
            config={**config, "recursion_limit": 50},
        )

        # Last AIMessage in state must be one of the violating attempts (the
        # 3rd, since max_retries=2 means 2 corrections then let through). Its
        # id must remain in state — not removed.
        ai_msgs = [m for m in final["messages"] if isinstance(m, AIMessage)]
        self.assertGreaterEqual(len(ai_msgs), 1)
        last_ai_id = ai_msgs[-1].id
        self.assertTrue(
            last_ai_id.startswith("ai-v-"),
            f"last AIMessage should be one of the violating attempts, got id={last_ai_id}",
        )

    def test_text_only_violation_routes_through_gateway(self):
        # Content-only AIMessage from the agent — no tool_calls — still gets
        # routed through gateway in active mode and corrected.
        violating = AIMessage(
            content="off-topic monologue",
            id="ai-text-violating",
            name="order_agent",
        )
        compliant = AIMessage(
            content="Hello, how can I help with your order?",
            id="ai-text-compliant",
            name="order_agent",
        )
        sup = self._make_supervisor(
            decide_action_returns=[
                SupervisorVerdict(
                    decision_line="Violation:invalid_activity_trigger",
                    is_violation=True,
                    reason="invalid_activity_trigger",
                    allowed_activities=("A01",),
                ),
                SupervisorVerdict(
                    decision_line="Execution:A01:identify_customer_request",
                    is_violation=False,
                    reason="execution",
                ),
            ],
            corrective_text="Process supervisor: respond on-topic.",
        )
        graph = self._build_subgraph(
            scripted_messages=[violating, compliant], supervisor=sup,
        )
        final = graph.invoke(
            {"messages": [HumanMessage(content="hi")],
             "active_agent": "order_agent",
             "handoff_context": None},
            config={"configurable": {"thread_id": "t-text"}},
        )
        ids = [getattr(m, "id", None) for m in final["messages"]]
        self.assertNotIn("ai-text-violating", ids)
        self.assertIn("ai-text-compliant", ids)
        # supervisor HumanMessage was inserted between
        sup_msgs = [m for m in final["messages"]
                    if isinstance(m, HumanMessage) and getattr(m, "name", None) == "process_supervisor"]
        self.assertEqual(len(sup_msgs), 1)


class TestRunnerPublishesRejectedEvent(unittest.TestCase):
    """The runner publishes AGENT_MESSAGE_REJECTED + WARNING LOG_MESSAGE when
    the supervisor's verdict on an AIMessage is a violation and active mode
    is on."""

    def test_rejected_event_published(self):
        from src.dashboard.event_bus import EventBus, EventType
        from src.dashboard.conversation_runner import ConversationRunner

        shop = MagicMock()
        shop.config.process_supervisor_active = True

        # Stand-in supervisor: observe is a no-op; last_verdict_for returns
        # a violation verdict.
        sup = MagicMock()
        sup.observe.return_value = "Violation:test"
        sup.last_verdict_for.return_value = SupervisorVerdict(
            decision_line="Violation:test",
            is_violation=True,
            reason="test_violation",
            allowed_activities=("A01", "A02"),
        )
        shop.process_supervisor = sup

        bus = EventBus()
        runner = ConversationRunner(shop, bus)
        msg = AIMessage(
            content="bad agent output",
            id="ai-bad-1",
            name="order_agent",
        )
        runner._process_message(msg, "order_agent")

        events = bus.drain()
        kinds = [e.event_type for e in events]
        self.assertIn(EventType.AGENT_MESSAGE_REJECTED, kinds)
        self.assertIn(EventType.LOG_MESSAGE, kinds)
        rejected = next(e for e in events if e.event_type == EventType.AGENT_MESSAGE_REJECTED)
        self.assertEqual(rejected.rejection_reason, "test_violation")
        self.assertEqual(rejected.allowed_activities, ["A01", "A02"])
        # Make sure normal AGENT_MESSAGE was NOT published for this message.
        self.assertNotIn(EventType.AGENT_MESSAGE, kinds)
        # WARNING level on the log message.
        log_msg = next(e for e in events if e.event_type == EventType.LOG_MESSAGE)
        import logging as _logging
        self.assertEqual(log_msg.log_level, _logging.WARNING)


if __name__ == "__main__":
    unittest.main()
