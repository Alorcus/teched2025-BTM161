"""Unit tests for ProcessSupervisor._allowed_next_activities and the static
successor table. These run without an LLM — the helper is purely log-driven."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.control_plane.process_supervisor import ProcessSupervisor


# Reuse the project's full process_model.yaml so the test exercises the same
# activity catalogue the production code does.
_REAL_MODEL = Path(__file__).resolve().parent.parent / "config" / "process_model.yaml"
_DUMMY_PROMPT = "test prompt {activity_catalog} {prior_log_tail} {message_brief}"


class TestAllowedNextActivities(unittest.TestCase):

    def _make_supervisor(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        log_path = Path(td.name) / "process.log"
        # llm is required by ProcessSupervisor but unused by
        # _allowed_next_activities — a MagicMock satisfies the constructor.
        return ProcessSupervisor(
            process_model_path=_REAL_MODEL,
            log_path=log_path,
            llm=MagicMock(),
            prompt_template=_DUMMY_PROMPT,
        )

    def test_initial_state_allows_a01(self):
        sup = self._make_supervisor()
        self.assertEqual(sup._allowed_next_activities("order_agent"), ["A01"])

    def test_after_a01_allows_a02_or_a08(self):
        sup = self._make_supervisor()
        sup._lines.append("Execution:A01:identify_customer_request | AIMessage[order_agent] text=hi")
        self.assertEqual(
            sorted(sup._allowed_next_activities("order_agent")),
            sorted(["A02", "A08"]),
        )

    def test_after_a02_allows_a03(self):
        sup = self._make_supervisor()
        sup._lines.extend([
            "Execution:A01:identify_customer_request | AIMessage[order_agent] text=hi",
            "Execution:A02:create_order | AIMessage[order_agent] tool_calls=[process_order(...)]",
        ])
        self.assertEqual(sup._allowed_next_activities("order_agent"), ["A03"])

    def test_after_a03_allows_parallel_split(self):
        sup = self._make_supervisor()
        sup._lines.extend([
            "Execution:A01:identify_customer_request | AIMessage[order_agent] text=hi",
            "Execution:A02:create_order | AIMessage[order_agent] tool_calls=[process_order(...)]",
            "Execution:A03:check_stock | AIMessage[inventory_agent] tool_calls=[check_inventory(...)]",
        ])
        self.assertEqual(
            sorted(sup._allowed_next_activities("inventory_agent")),
            sorted(["A04", "A05", "A06"]),
        )

    def test_terminated_branch_filtered_out(self):
        sup = self._make_supervisor()
        sup._lines.extend([
            "Execution:A03:check_stock | AIMessage[inventory_agent] tool_calls=[check_inventory(...)]",
            "Execution:A04:place_food_on_tray | AIMessage[inventory_agent] tool_calls=[place_on_tray(...)]",
            "Termination:A04:place_food_on_tray:via_handoff_to_barista_agent | AIMessage[inventory_agent] tool_calls=[transfer_to_agent(...)]",
        ])
        # After A04 terminates, the most recent event is the termination → its
        # successor set is consulted; A04 is filtered out as already terminated.
        allowed = sup._allowed_next_activities("inventory_agent")
        self.assertNotIn("A04", allowed)

    def test_complaint_branch_after_a01(self):
        sup = self._make_supervisor()
        sup._lines.append(
            "Execution:A01:identify_customer_request | AIMessage[order_agent] text=hi"
        )
        self.assertIn("A08", sup._allowed_next_activities("customer_service_agent"))


class TestVerdictCacheRoundTrip(unittest.TestCase):
    """Verdict cache reuses observe() output so decide_action does not double-classify."""

    def test_cache_hit_after_observe(self):
        from langchain_core.messages import AIMessage

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        log_path = Path(td.name) / "process.log"
        # Force the LLM to return a Violation for any classification so we can
        # detect duplicate calls.
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="Violation:test_only")
        sup = ProcessSupervisor(
            process_model_path=_REAL_MODEL, log_path=log_path, llm=llm,
            prompt_template=_DUMMY_PROMPT,
        )
        msg = AIMessage(content="some text", id="ai-test-1", name="order_agent")
        sup.observe(msg, agent_name="order_agent")
        first_call_count = llm.invoke.call_count
        verdict = sup.decide_action(msg, agent_name="order_agent")
        self.assertTrue(verdict.is_violation)
        self.assertEqual(llm.invoke.call_count, first_call_count,
                         "decide_action should reuse cached verdict, not re-classify")

    def test_last_verdict_for_returns_cached(self):
        from langchain_core.messages import AIMessage

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        log_path = Path(td.name) / "process.log"
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="Violation:foo")
        sup = ProcessSupervisor(
            process_model_path=_REAL_MODEL, log_path=log_path, llm=llm,
            prompt_template=_DUMMY_PROMPT,
        )
        msg = AIMessage(content="x", id="ai-test-2", name="order_agent")
        self.assertIsNone(sup.last_verdict_for(msg))
        sup.observe(msg, agent_name="order_agent")
        v = sup.last_verdict_for(msg)
        self.assertIsNotNone(v)
        self.assertTrue(v.is_violation)
        self.assertEqual(v.reason, "foo")


if __name__ == "__main__":
    unittest.main()
