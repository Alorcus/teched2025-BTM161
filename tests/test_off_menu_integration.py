"""Integration test: soft `assistant_message:on_menu_only` guardrail wired end-to-end.

Exercises the full baseline setup (catalog → gateway → subgraph → stream) with
a stub LLM plus a stub judge, verifying that:

  1. When the order_agent proposes an off-menu drink, the response guardrail
     denies it and the customer never receives the rejected text.
  2. The gateway_decision log records the DENY verdict for
     `assistant_message:on_menu_only` on `assistant_message`.

This test is intentionally offline: the real LLM is replaced with a stubbed
sequence and the soft guardrail's judge_invoker is overridden via monkeypatch
so the deny/allow is deterministic. A separate subprocess test in
`test_off_menu_scenario_e2e.py` (skipped when the LLM proxy is unreachable)
runs scenario 4 for real.
"""
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from src.agents import init_db, reset_inventory
from src.control_plane import AgentRepo, Catalog, JsonlLogSink, build
from src.control_plane.subgraph import CORRECTION_KWARG, REJECTED_CONTENT_KWARG
from src.stream import extract_messages


class _SequenceLLM:
    """Replays a fixed sequence of AIMessages for the order_agent."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def bind_tools(self, tools, **_):
        return self

    def invoke(self, messages, config=None):
        i = min(self.call_count, len(self._responses) - 1)
        self.call_count += 1
        return self._responses[i]


def _stub_judge(_system: str, user: str) -> str:
    """Deny anything mentioning 'hazelnut', 'macchiato', 'frappe', or 'matcha'."""
    lowered = user.lower()
    forbidden = ("hazelnut", "macchiato", "frappe", "frappé", "matcha", "mocha")
    if any(word in lowered for word in forbidden):
        return json.dumps({
            "decision": "deny",
            "reason": "That item is not on our menu.",
        })
    return json.dumps({"decision": "allow", "reason": ""})


class TestOffMenuGuardrailIntegration(unittest.TestCase):
    """Baseline order_agent subgraph wired with the real Catalog config +
    a stub judge for the soft guardrail. Confirms the block-and-pushback
    reaches the customer-facing reply correctly."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = Path(self._tmpdir) / "coffee.db"
        self._log_path = Path(self._tmpdir) / "gate.jsonl"

        import os
        os.environ["COFFEE_SHOP_DB"] = str(self._db_path)
        init_db()
        reset_inventory()

        self._repo = AgentRepo(Path("config/setups/baseline"))
        self._catalog = Catalog(Path("config/setups/baseline"))
        self._log_sink = JsonlLogSink(self._log_path, setup_name="baseline")

        for guardrail in self._catalog.guardrails(["assistant_message:on_menu_only"]):
            guardrail.judge_invoker = _stub_judge

    def tearDown(self):
        import os
        os.environ.pop("COFFEE_SHOP_DB", None)

    def _build_order_agent(self, llm):
        subgraph, _definition, _snapshot, _gateway = build(
            agent_id="order_agent",
            llm=llm,
            repo=self._repo,
            catalog=self._catalog,
            log_sink=self._log_sink,
        )
        return subgraph

    def test_assistant_message_on_menu_only_denied_and_retried(self):
        bad = AIMessage(
            content="I recommend our house special: a hazelnut latte!",
            name="order_agent", id="ai-bad",
        )
        good = AIMessage(
            content="Would you like a latte? It is one of our most popular drinks.",
            name="order_agent", id="ai-good",
        )
        llm = _SequenceLLM([bad, good])
        graph = self._build_order_agent(llm)

        thread_id = str(uuid.uuid4())
        input_state = {
            "messages": [HumanMessage(content="Can you recommend a drink?")],
            "handoff_context": None,
        }
        config = {"configurable": {"thread_id": thread_id}}

        stream = graph.stream(input_state, config, subgraphs=True)

        last_reply = None
        for stream_msg in extract_messages(stream):
            if stream_msg.is_agent_reply:
                last_reply = stream_msg.content

        self.assertEqual(llm.call_count, 2, "LLM should have been retried once")
        self.assertEqual(
            last_reply,
            "Would you like a latte? It is one of our most popular drinks.",
        )
        self.assertNotIn("hazelnut latte", (last_reply or "").lower())

    def test_gateway_decision_logged_with_deny_verdict(self):
        bad = AIMessage(
            content="How about a caramel macchiato?", name="order_agent", id="ai-bad",
        )
        good = AIMessage(
            content="Would you like an americano?", name="order_agent", id="ai-good",
        )
        llm = _SequenceLLM([bad, good])
        graph = self._build_order_agent(llm)

        thread_id = str(uuid.uuid4())
        stream = graph.stream(
            {"messages": [HumanMessage(content="Recommend me something")], "handoff_context": None},
            {"configurable": {"thread_id": thread_id}},
            subgraphs=True,
        )
        for _ in extract_messages(stream):
            pass

        entries = [
            json.loads(line)
            for line in self._log_path.read_text().splitlines()
            if line.strip()
        ]
        deny_records = [
            e for e in entries
            if e.get("event_type") == "gateway_decision"
            and e.get("final_decision") == "deny"
            and any(
                v.get("guardrail_name") == "assistant_message:on_menu_only"
                and v.get("effect") == "deny"
                for v in e.get("verdicts", [])
            )
        ]
        self.assertEqual(len(deny_records), 1, f"expected 1 deny record, got {len(deny_records)}: {entries}")
        record = deny_records[0]
        self.assertEqual(record["tool_name"], "assistant_message")
        self.assertEqual(record["thread_id"], thread_id)


if __name__ == "__main__":
    unittest.main()
