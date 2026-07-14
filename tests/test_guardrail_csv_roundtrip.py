"""Round-trip tests for guardrail_log <-> _all_traces.csv.

`_all_traces.csv` is shared with users who don't have
`guardrail_log/events.jsonl` on disk; the OCEL extension built from
CSV-embedded rows must be identical to the one built from the JSONL.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from src.trace_processing.eventlog_conversion import _resolve_guardrail_extension
from src.trace_processing.guardrail_log_loader import (
    GuardrailOcelExtension,
    load_guardrail_events,
    load_guardrail_events_from_eventlog,
)
from src.trace_processing.trace_processor import _load_gateway_rows


def _decision(
    *,
    ts: float,
    thread_id: str,
    agent_id: str = "order_agent",
    setup_name: str = "baseline",
    snapshot_id: str = "order_agent@v1+abc123",
    tool_name: str = "process_order",
    tool_call_id: str | None = None,
    final_decision: str = "allow",
    verdicts: list | None = None,
    tool_args: dict | None = None,
) -> dict:
    return {
        "ts": ts,
        "setup_name": setup_name,
        "event_type": "gateway_decision",
        "snapshot_id": snapshot_id,
        "agent_id": agent_id,
        "thread_id": thread_id,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id or f"toolu_{ts}",
        "tool_args": tool_args if tool_args is not None else {"customer": "Alice"},
        "final_decision": final_decision,
        "verdicts": verdicts or [],
    }


def _ext_signature(ext: GuardrailOcelExtension) -> dict:
    """Reduce a GuardrailOcelExtension to a comparable structure.

    Compares on the sorted set of relationships / attributes rather than
    on ocel_ids that carry a random uuid — event_id is generated fresh at
    projection time, so the two projections never share those.
    """
    def _events_by_type(evt_type: str) -> list[dict]:
        df = ext.event_tables.get(evt_type)
        if df is None or df.is_empty():
            return []
        cols = [c for c in df.columns if c != "ocel_id"]
        return sorted(
            df.select(cols).to_dicts(), key=lambda r: (r.get("tool_call_id"), r.get("agent_id"))
        )

    def _objects_by_type(obj_type: str) -> list[dict]:
        df = ext.object_tables.get(obj_type)
        if df is None or df.is_empty():
            return []
        return sorted(df.to_dicts(), key=lambda r: r["ocel_id"])

    def _e2o_signature() -> list[tuple]:
        if ext.event_object_rows.is_empty():
            return []
        tool_by_event: dict[str, str] = {}
        for evt_type in ("gateway_flag", "gateway_deny"):
            df = ext.event_tables.get(evt_type)
            if df is None or df.is_empty():
                continue
            for row in df.to_dicts():
                tool_by_event[row["ocel_id"]] = row.get("tool_call_id", "")
        return sorted(
            (
                tool_by_event.get(r["ocel_event_id"], ""),
                r["ocel_object_id"],
                r["ocel_qualifier"],
            )
            for r in ext.event_object_rows.to_dicts()
        )

    def _o2o_signature() -> list[tuple]:
        if ext.object_object_rows.is_empty():
            return []
        return sorted(
            (r["ocel_source_id"], r["ocel_target_id"], r["ocel_qualifier"])
            for r in ext.object_object_rows.to_dicts()
        )

    return {
        "gateway_flag": _events_by_type("gateway_flag"),
        "gateway_deny": _events_by_type("gateway_deny"),
        "guardrail": _objects_by_type("guardrail"),
        "setup": _objects_by_type("setup"),
        "snapshot": _objects_by_type("snapshot"),
        "tool_call": _objects_by_type("tool_call"),
        "case_setup_map": dict(sorted(ext.case_setup_map.items())),
        "case_agent_snapshot_map": {
            f"{k[0]}|{k[1]}": v for k, v in sorted(ext.case_agent_snapshot_map.items())
        },
        "tool_call_ids": sorted(ext.tool_call_ids),
        "event_object_rows": _e2o_signature(),
        "object_object_rows": _o2o_signature(),
    }


class TestGuardrailCsvRoundTrip(unittest.TestCase):

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def _project_via_csv(self, records: list[dict]) -> GuardrailOcelExtension:
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "events.jsonl"
            self._write_jsonl(jsonl_path, records)
            rows = _load_gateway_rows(jsonl_path)
            csv_path = Path(tmp) / "eventlog.csv"
            rows.write_csv(str(csv_path))
            el = pl.read_csv(str(csv_path), infer_schema_length=10_000)
        return load_guardrail_events_from_eventlog(el)

    def test_allow_only_produces_empty_events_but_populates_maps(self):
        records = [
            _decision(
                ts=1783868902.16553,
                thread_id="thread-1",
                final_decision="allow",
                verdicts=[],
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "events.jsonl"
            self._write_jsonl(jsonl_path, records)
            jsonl_ext = load_guardrail_events(jsonl_path)
        csv_ext = self._project_via_csv(records)
        self.assertEqual(_ext_signature(jsonl_ext), _ext_signature(csv_ext))

    def test_deny_decision_round_trip(self):
        records = [
            _decision(
                ts=1783868902.16553,
                thread_id="thread-1",
                final_decision="deny",
                verdicts=[
                    {
                        "guardrail_name": "no_freebies",
                        "guardrail_version": "v1",
                        "guardrail_type": "policy",
                        "effect": "deny",
                        "reason_internal": "cost > 0",
                        "reason_for_llm": "Free items disabled.",
                    }
                ],
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "events.jsonl"
            self._write_jsonl(jsonl_path, records)
            jsonl_ext = load_guardrail_events(jsonl_path)
        csv_ext = self._project_via_csv(records)
        self.assertEqual(_ext_signature(jsonl_ext), _ext_signature(csv_ext))

    def test_mixed_flag_and_deny_across_threads(self):
        records = [
            _decision(
                ts=1783868902.1,
                thread_id="thread-a",
                agent_id="order_agent",
                tool_call_id="tc-1",
                final_decision="allow",
                verdicts=[
                    {
                        "guardrail_name": "menu_check",
                        "guardrail_version": "v2",
                        "guardrail_type": "content",
                        "effect": "flag",
                        "reason_internal": "unusual item",
                        "reason_for_llm": "Item is uncommon.",
                    }
                ],
            ),
            _decision(
                ts=1783868903.4,
                thread_id="thread-a",
                agent_id="barista_agent",
                snapshot_id="barista_agent@v1+xyz",
                tool_call_id="tc-2",
                final_decision="deny",
                verdicts=[
                    {
                        "guardrail_name": "safety",
                        "guardrail_version": "v3",
                        "guardrail_type": "policy",
                        "effect": "deny",
                        "reason_internal": "unsafe temp",
                        "reason_for_llm": "Too hot.",
                    }
                ],
            ),
            _decision(
                ts=1783868904.9,
                thread_id="thread-b",
                agent_id="order_agent",
                tool_call_id="tc-3",
                final_decision="allow",
                verdicts=[],
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "events.jsonl"
            self._write_jsonl(jsonl_path, records)
            jsonl_ext = load_guardrail_events(jsonl_path)
        csv_ext = self._project_via_csv(records)
        self.assertEqual(_ext_signature(jsonl_ext), _ext_signature(csv_ext))


class TestGatewayAppendScoping(unittest.TestCase):

    def test_load_gateway_rows_returns_case_id_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "events.jsonl"
            jsonl.write_text(
                json.dumps(_decision(ts=1783868902.0, thread_id="t-1")) + "\n"
                + json.dumps(_decision(ts=1783868903.0, thread_id="t-2")) + "\n"
            )
            rows = _load_gateway_rows(jsonl)
        self.assertEqual(set(rows["case_id"].to_list()), {"t-1", "t-2"})
        filtered = rows.filter(pl.col("case_id").is_in(["t-1"]))
        self.assertEqual(filtered.height, 1)

    def test_load_gateway_rows_writes_naive_utc_timestamps(self):
        """Pins UTC convention on gateway CSV rows.

        The Metrics Dashboard treats every CSV `time:timestamp` as
        naive-UTC. If gateway rows carried naive-LOCAL strings instead,
        `_load_combined_eventlog`'s UTC→local conversion would double-shift
        them by the local offset — pushing gateway events hours into the
        future and knocking them out of "Last 10 min" filters.

        `ts = 1783868902.0` is 2026-07-12T15:08:22 UTC. In naive-LOCAL for
        a CET/CEST host the string would come out as 17:08:22 (+02:00).
        """
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "events.jsonl"
            jsonl.write_text(
                json.dumps(_decision(ts=1783868902.0, thread_id="t-1")) + "\n"
            )
            rows = _load_gateway_rows(jsonl)
        ts_str = rows["time:timestamp"][0]
        # Anchoring on a UTC-rendered string (not a local datetime
        # comparison) makes this fail identically on WSL/CET, macOS/PST,
        # and UTC CI runners.
        self.assertTrue(
            ts_str.startswith("2026-07-12T15:08:22"),
            f"Expected naive-UTC prefix '2026-07-12T15:08:22', got {ts_str!r}. "
            f"If this fails with a +Nh shift, _load_gateway_rows regressed "
            f"back to `datetime.fromtimestamp(...)` without `tz=timezone.utc`.",
        )


class TestNaiveUtcEquivalence(unittest.TestCase):
    """`load_guardrail_events_from_eventlog` must produce identical epoch
    values whether `time:timestamp` is a naive-UTC string or a naive-UTC
    Datetime — but ONLY when the Datetime path has NOT been through
    timezone conversion. Downstream code handing the loader a naive-LOCAL
    Datetime (as the dashboard's `_load_combined_eventlog` produces) MUST
    route through `_resolve_guardrail_extension`, which preserves the
    naive-UTC sibling column.
    """

    def _make_row(self, ts_iso: str) -> dict:
        return {
            "case_id": "thread-1",
            "identity:id": "id-1",
            "time:timestamp": ts_iso,
            "time_finished": ts_iso,
            "concept:name": "gateway_decision",
            "concept:instance": "gateway deny: process_order",
            "org:resource": "order_agent",
            "gateway_setup_name": "baseline",
            "gateway_snapshot_id": "order_agent@v1+abc123",
            "gateway_tool_name": "process_order",
            "gateway_tool_call_id": "toolu_1",
            "gateway_final_decision": "deny",
            "gateway_tool_args_json": json.dumps({"customer": "Alice"}, sort_keys=True),
            "gateway_verdicts_json": json.dumps([
                {
                    "guardrail_name": "no_freebies",
                    "guardrail_version": "v1",
                    "guardrail_type": "policy",
                    "effect": "deny",
                    "reason_internal": "cost > 0",
                    "reason_for_llm": "Free items disabled.",
                }
            ]),
        }

    def test_string_and_naive_utc_datetime_paths_produce_same_epoch(self):
        """Path A: `time:timestamp` as a naive-UTC ISO string.
        Path B: same column parsed via `str.to_datetime()` — no timezone
        conversion, so the Datetime stays naive-UTC.

        Both paths must resolve to the same `ocel_time`.
        """
        ts_iso = "2026-07-12T15:08:22.000000"
        row = self._make_row(ts_iso)

        el_string = pl.DataFrame([row])
        ext_string = load_guardrail_events_from_eventlog(el_string)

        el_dt = el_string.with_columns(
            pl.col("time:timestamp").str.to_datetime(strict=False)
        )
        self.assertIn(el_dt.schema["time:timestamp"], (pl.Datetime, pl.Datetime("us"), pl.Datetime("ms")))
        ext_dt = load_guardrail_events_from_eventlog(el_dt)

        deny_str = ext_string.event_tables["gateway_deny"]
        deny_dt = ext_dt.event_tables["gateway_deny"]
        self.assertEqual(deny_str.height, 1)
        self.assertEqual(deny_dt.height, 1)
        self.assertEqual(
            deny_str["ocel_time"].to_list(),
            deny_dt["ocel_time"].to_list(),
            "String and naive-UTC Datetime input paths must resolve to the "
            "same ocel_time — the loader tags naive datetimes as UTC.",
        )

        # If the loader silently interpreted the naive datetime as LOCAL,
        # a CET/CEST host would emit 13:08:22 or 14:08:22 instead of 15:08:22.
        expected = datetime(2026, 7, 12, 15, 8, 22)
        self.assertEqual(deny_str["ocel_time"][0], expected)


class TestDashboardEndToEndPath(unittest.TestCase):
    """When the dashboard's `_load_combined_eventlog` has already converted
    `time:timestamp` to naive-LOCAL, `_resolve_guardrail_extension` must
    still yield an unshifted gateway `ocel_time` by consulting the
    preserved `time:timestamp_utc_naive` sibling column.
    """

    def _make_row(self, ts_iso: str) -> dict:
        return {
            "case_id": "thread-1",
            "identity:id": "id-1",
            "time:timestamp": ts_iso,
            "time_finished": ts_iso,
            "concept:name": "gateway_decision",
            "concept:instance": "gateway deny: process_order",
            "org:resource": "order_agent",
            "gateway_setup_name": "baseline",
            "gateway_snapshot_id": "order_agent@v1+abc123",
            "gateway_tool_name": "process_order",
            "gateway_tool_call_id": "toolu_1",
            "gateway_final_decision": "deny",
            "gateway_tool_args_json": json.dumps({"customer": "Alice"}, sort_keys=True),
            "gateway_verdicts_json": json.dumps([
                {
                    "guardrail_name": "no_freebies",
                    "guardrail_version": "v1",
                    "guardrail_type": "policy",
                    "effect": "deny",
                    "reason_internal": "cost > 0",
                    "reason_for_llm": "Free items disabled.",
                }
            ]),
        }

    def test_utc_naive_sibling_column_prevents_double_shift(self):
        """Simulate a post-`_load_combined_eventlog` frame where
        `time:timestamp` is Datetime shifted to naive-LOCAL and
        `time:timestamp_utc_naive` holds the same instant as naive-UTC.
        The resolver must consult the sibling column.
        """
        ts_iso = "2026-07-12T15:08:22.000000"
        utc_naive = datetime(2026, 7, 12, 15, 8, 22)
        # Fixed +2h offset (not the host tz) so the shifted vs. unshifted
        # comparison works regardless of where the test runs.
        local_offset = timedelta(hours=2)
        shifted_local = utc_naive + local_offset

        row = self._make_row(ts_iso)
        el = pl.DataFrame([row])
        el_dashboard = el.with_columns(
            pl.col("time:timestamp")
              .str.to_datetime(strict=False)
              .alias("time:timestamp_utc_naive"),
        ).with_columns(
            pl.Series("time:timestamp", [shifted_local], dtype=pl.Datetime("us")),
        )

        ext = _resolve_guardrail_extension(el_dashboard, guardrail_log_path=None)
        deny = ext.event_tables["gateway_deny"]
        self.assertEqual(deny.height, 1)
        self.assertEqual(
            deny["ocel_time"][0],
            utc_naive,
            "Resolver must consult time:timestamp_utc_naive so the gateway "
            "event's ocel_time is naive-UTC, not the shifted local time.",
        )

    def test_no_sibling_column_falls_back_to_jsonl(self):
        """Datetime `time:timestamp` + no sibling column + JSONL on disk →
        resolver must prefer the JSONL."""
        ts_iso = "2026-07-12T15:08:22.000000"
        utc_naive = datetime(2026, 7, 12, 15, 8, 22)
        shifted_local = utc_naive + timedelta(hours=2)

        row = self._make_row(ts_iso)
        el = pl.DataFrame([row]).with_columns(
            pl.Series("time:timestamp", [shifted_local], dtype=pl.Datetime("us")),
        )

        # 2026-07-12T15:08:22 UTC == epoch 1783868902.0
        record = {
            "ts": 1783868902.0,
            "setup_name": "baseline",
            "event_type": "gateway_decision",
            "snapshot_id": "order_agent@v1+abc123",
            "agent_id": "order_agent",
            "thread_id": "thread-1",
            "tool_name": "process_order",
            "tool_call_id": "toolu_1",
            "tool_args": {"customer": "Alice"},
            "final_decision": "deny",
            "verdicts": [
                {
                    "guardrail_name": "no_freebies",
                    "guardrail_version": "v1",
                    "guardrail_type": "policy",
                    "effect": "deny",
                    "reason_internal": "cost > 0",
                    "reason_for_llm": "Free items disabled.",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "events.jsonl"
            jsonl_path.write_text(json.dumps(record) + "\n")
            ext = _resolve_guardrail_extension(el, guardrail_log_path=jsonl_path)

        deny = ext.event_tables["gateway_deny"]
        self.assertEqual(deny.height, 1)
        self.assertEqual(deny["ocel_time"][0], utc_naive)


if __name__ == "__main__":
    unittest.main()
