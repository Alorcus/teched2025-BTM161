"""LogGenerator regression tests.

Fixtures are real MLflow traces captured from the current LangChain autolog
shape (LangGraph outer → agent → inner LangGraph → llm → ChatAnthropic).
Two fixtures cover the shapes we care about:

- `modern_langgraph_two_turn.json`: single agent, two llm turns, no tools.
- `modern_langgraph_with_transfer.json`: multi-agent with a
  `transfer_to_agent` tool span (Part 1 regression + Part 2 integration).

Both were captured on branch rg/extend-metrics-for-evaluation; do not
regenerate blindly — they encode the extraction contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from src.trace_processing.log_generator import LogGenerator

FIXTURES = Path(__file__).parent / "fixtures" / "traces"


def _load(name: str) -> dict:
    with (FIXTURES / name).open() as f:
        return json.load(f)


@pytest.fixture
def two_turn_trace() -> dict:
    return _load("modern_langgraph_two_turn.json")


@pytest.fixture
def transfer_trace() -> dict:
    return _load("modern_langgraph_with_transfer.json")


class TestModernLLMExtraction:
    """Part 2: LogGenerator must emit call_llm rows from `llm` spans."""

    def test_two_turn_trace_emits_call_llm_rows(self, two_turn_trace):
        df = LogGenerator().generate_event_log_df(two_turn_trace)

        call_llm = df.filter(pl.col("concept:name") == "call_llm")
        assert call_llm.height == 2, f"expected 2 call_llm rows, got {call_llm.height}"

    def test_call_llm_carries_agent_response_message(self, two_turn_trace):
        df = LogGenerator().generate_event_log_df(two_turn_trace)

        call_llm = df.filter(pl.col("concept:name") == "call_llm")
        for msg in call_llm["message"].to_list():
            assert msg is not None and msg != "", "call_llm message must be populated"

    def test_call_llm_carries_model_and_token_counts(self, two_turn_trace):
        df = LogGenerator().generate_event_log_df(two_turn_trace)

        call_llm = df.filter(pl.col("concept:name") == "call_llm")
        for row in call_llm.iter_rows(named=True):
            assert row["model"], f"missing model on call_llm row: {row}"
            assert row["input_tokens"] > 0, "input_tokens must be > 0"
            assert row["response_tokens"] > 0, "response_tokens must be > 0"

    def test_call_llm_agent_matches_langgraph_node(self, two_turn_trace):
        df = LogGenerator().generate_event_log_df(two_turn_trace)

        call_llm = df.filter(pl.col("concept:name") == "call_llm")
        # This fixture is a customer-service-only conversation — every llm turn
        # belongs to customer_service_agent.
        agents = set(call_llm["org:resource"].unique().to_list())
        assert agents == {"customer_service_agent"}, f"unexpected agents: {agents}"


class TestUserPromptExtraction:
    """User prompt always emits from the outer LangGraph root, regardless of shape."""

    def test_user_prompt_present(self, two_turn_trace):
        df = LogGenerator().generate_event_log_df(two_turn_trace)

        prompts = df.filter(pl.col("concept:name") == "user_prompt")
        assert prompts.height == 1
        first = prompts.row(0, named=True)
        assert first["org:resource"] == "user"
        assert first["message"], "user prompt message must be populated"


class TestPart1TransferHandover:
    """Part 1 regression: transfer_to_* tool calls emit as execute_tool rows."""

    def test_transfer_trace_emits_execute_tool_for_transfer(self, transfer_trace):
        df = LogGenerator().generate_event_log_df(transfer_trace)

        transfers = df.filter(
            (pl.col("concept:name") == "execute_tool")
            & (pl.col("tool") == "transfer_to_agent")
        )
        if transfers.height < 1:
            all_execute = df.filter(pl.col("concept:name") == "execute_tool").select(
                "org:resource", "tool"
            )
            raise AssertionError(
                "expected at least one execute_tool row for transfer_to_agent; "
                f"got {transfers.height}. All execute_tool rows: {all_execute.to_dicts()}"
            )

    def test_transfer_row_org_resource_is_source_agent(self, transfer_trace):
        df = LogGenerator().generate_event_log_df(transfer_trace)

        transfers = df.filter(
            (pl.col("concept:name") == "execute_tool")
            & (pl.col("tool") == "transfer_to_agent")
        )
        # Every transfer's org:resource should be a *_agent name — the source
        # agent making the handover.
        for row in transfers.iter_rows(named=True):
            assert row["org:resource"].endswith("_agent"), (
                f"transfer source not an agent: {row['org:resource']}"
            )


class TestModernShapeIntegration:
    """Combined Part 1 + Part 2 on a real multi-agent handover trace."""

    def test_transfer_trace_has_all_row_types(self, transfer_trace):
        df = LogGenerator().generate_event_log_df(transfer_trace)

        names = set(df["concept:name"].unique().to_list())
        assert "user_prompt" in names
        assert "call_llm" in names
        assert "execute_tool" in names

    def test_call_llm_rows_precede_and_follow_transfer(self, transfer_trace):
        df = LogGenerator().generate_event_log_df(transfer_trace).sort("time:timestamp")

        # A transfer_to_agent handover must have at least one call_llm from
        # a different agent afterwards — the receiving agent's first turn.
        transfers = df.filter(
            (pl.col("concept:name") == "execute_tool")
            & (pl.col("tool") == "transfer_to_agent")
        )
        assert transfers.height > 0, "no transfer rows present"

        first_transfer = transfers.row(0, named=True)
        source_agent = first_transfer["org:resource"]
        transfer_ts = first_transfer["time:timestamp"]

        later_call_llm = df.filter(
            (pl.col("time:timestamp") > transfer_ts)
            & (pl.col("concept:name") == "call_llm")
        )
        assert later_call_llm.height > 0, (
            "no call_llm after transfer — handover extraction broken"
        )
        assert (later_call_llm["org:resource"] != source_agent).any(), (
            "expected a call_llm from a different agent after the transfer"
        )

    def test_no_row_missing_case_id(self, transfer_trace):
        df = LogGenerator().generate_event_log_df(transfer_trace)
        assert df["case_id"].is_not_null().all(), "every row must carry the thread_id"
        assert df["case_id"].n_unique() == 1, "single trace must map to a single case_id"


class TestEmptyTraceFallback:
    """Traces without a LangGraph root return an empty frame, not a crash."""

    def test_empty_spans_returns_empty_frame(self):
        df = LogGenerator().generate_event_log_df({"spans": []})
        assert df.is_empty()

    def test_missing_langgraph_root_returns_empty_frame(self):
        # A standalone ChatAnthropic trace (e.g. the feedback path) has no
        # LangGraph root and must not raise.
        df = LogGenerator().generate_event_log_df({
            "spans": [
                {
                    "name": "ChatAnthropic",
                    "span_id": "abc",
                    "parent_span_id": None,
                    "start_time_unix_nano": 0,
                    "end_time_unix_nano": 1,
                    "attributes": {},
                }
            ]
        })
        assert df.is_empty()
