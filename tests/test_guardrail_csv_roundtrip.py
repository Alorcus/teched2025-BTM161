"""Round-trip tests for guardrail_log <-> _all_traces.csv.

Motivation: `_all_traces.csv` is designed to be shared with users who don't
have `guardrail_log/events.jsonl` on disk. That contract only holds if the
guardrail extension the OCEL converter builds from CSV-embedded rows is
identical to the one it would build from the original JSONL. These tests
pin that equivalence.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
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
    """One realistic gateway_decision record. Mirrors what
    control_plane.log_sink writes."""
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

    We compare on the sorted set of relationships / attributes rather than
    on ocel_ids that carry a random uuid (event_id is generated fresh at
    projection time — the two projections will never share those). The
    signature captures what an OCEL consumer actually observes: event
    types, object types, and E2O / O2O edges keyed by stable identifiers.
    """
    def _events_by_type(evt_type: str) -> list[dict]:
        df = ext.event_tables.get(evt_type)
        if df is None or df.is_empty():
            return []
        # Drop the random ocel_id; keep everything else. Sort by tool_call_id
        # so ordering isn't a false diff.
        cols = [c for c in df.columns if c != "ocel_id"]
        return sorted(
            df.select(cols).to_dicts(), key=lambda r: (r.get("tool_call_id"), r.get("agent_id"))
        )

    def _objects_by_type(obj_type: str) -> list[dict]:
        df = ext.object_tables.get(obj_type)
        if df is None or df.is_empty():
            return []
        return sorted(df.to_dicts(), key=lambda r: r["ocel_id"])

    # E2O edges: strip event_id (uuid) but keep object_id + qualifier, and
    # rekey each edge by the (tool_call_id, object_id, qualifier) triple so
    # events line up across the two extensions.
    def _e2o_signature() -> list[tuple]:
        if ext.event_object_rows.is_empty():
            return []
        # Look up each event's tool_call_id via the event tables.
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
    """CSV-embedded gateway rows must decode to the same OCEL extension as
    the JSONL they came from."""

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def _project_via_csv(self, records: list[dict]) -> GuardrailOcelExtension:
        """JSONL → _load_gateway_rows → in-memory CSV round-trip →
        load_guardrail_events_from_eventlog. Round-tripping through a CSV
        write/read is what the shared-file recipient actually experiences."""
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "events.jsonl"
            self._write_jsonl(jsonl_path, records)
            rows = _load_gateway_rows(jsonl_path)
            csv_path = Path(tmp) / "eventlog.csv"
            rows.to_csv(csv_path, index=False)
            el = pl.read_csv(str(csv_path), infer_schema_length=10_000)
        return load_guardrail_events_from_eventlog(el)

    def test_allow_only_produces_empty_events_but_populates_maps(self):
        """Allowed decisions emit no gateway_flag/deny events but still
        register tool_call objects and populate the backfill maps."""
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
        """A deny decision must reproduce the same event/object/edge set."""
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
        """Multiple decisions across threads, agents, and effects — the most
        realistic shape. Verifies that consulting/flagging/denying verdicts
        all round-trip and that per-thread setup+snapshot maps stay
        consistent."""
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
    """`_load_gateway_rows` reads the entire JSONL; scoping to new case_ids
    happens in the merge step in extract_new_traces. This is a small
    behavioural check on that step (via the standalone helper) so the row
    contract downstream can rely on a case_id column being present."""

    def test_load_gateway_rows_returns_case_id_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "events.jsonl"
            jsonl.write_text(
                json.dumps(_decision(ts=1783868902.0, thread_id="t-1")) + "\n"
                + json.dumps(_decision(ts=1783868903.0, thread_id="t-2")) + "\n"
            )
            rows = _load_gateway_rows(jsonl)
        self.assertEqual(set(rows["case_id"]), {"t-1", "t-2"})
        # Downstream (extract_new_traces) filters by case_id — column must
        # exist and be indexable with `.isin()`.
        filtered = rows[rows["case_id"].isin({"t-1"})]
        self.assertEqual(len(filtered), 1)

    def test_load_gateway_rows_writes_naive_utc_timestamps(self):
        """The Metrics Dashboard treats every CSV `time:timestamp` as
        naive-UTC (LogGenerator writes them that way, from OpenTelemetry
        `start_time_unix_nano`). If gateway rows carried naive-LOCAL strings
        instead, `_load_combined_eventlog`'s UTC→local conversion would
        double-shift them by the local offset — pushing current-run
        gateway events hours into the future and knocking them out of
        "Last 10 min" filters. Pin the UTC convention with a fixed epoch
        that renders differently in every non-UTC zone.

        `ts = 1783868902.0` is 2026-07-12T15:08:22 UTC (deterministic).
        In naive-LOCAL for a CET/CEST host the string would come out as
        17:08:22 (+02:00); in UTC it must be 15:08:22.
        """
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "events.jsonl"
            jsonl.write_text(
                json.dumps(_decision(ts=1783868902.0, thread_id="t-1")) + "\n"
            )
            rows = _load_gateway_rows(jsonl)
        ts_str = rows["time:timestamp"].iloc[0]
        # The exact UTC rendering for this epoch. Anchoring on a string
        # (not a local datetime comparison) means this test fails
        # identically on WSL/CET, macOS/PST, and UTC CI runners.
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
    timezone conversion. This is the contract expansion for todo 017:
    downstream code that hands the loader a naive-LOCAL Datetime (as the
    dashboard's `_load_combined_eventlog` produces) is REQUIRED to route
    through `_resolve_guardrail_extension`, which preserves the naive-UTC
    sibling column.
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
        """Path A: `time:timestamp` as a naive-UTC ISO string (what
        _load_gateway_rows writes).
        Path B: same column parsed with `pl.col.str.to_datetime()` — no
        `dt.replace_time_zone("UTC").dt.convert_time_zone(...)` chain, so
        the Datetime is still naive-UTC.

        Both paths must resolve to the same `ocel_time` on the resulting
        gateway_deny event.
        """
        ts_iso = "2026-07-12T15:08:22.000000"
        row = self._make_row(ts_iso)

        # Path A: string.
        el_string = pl.DataFrame([row])
        ext_string = load_guardrail_events_from_eventlog(el_string)

        # Path B: Datetime, but naive-UTC (no timezone conversion applied).
        el_dt = el_string.with_columns(
            pl.col("time:timestamp").str.to_datetime(strict=False)
        )
        # Sanity: the column really is a naive Datetime.
        self.assertIn(el_dt.schema["time:timestamp"], (pl.Datetime, pl.Datetime("us"), pl.Datetime("ms")))
        ext_dt = load_guardrail_events_from_eventlog(el_dt)

        # The event tables should carry the same ocel_time in both paths.
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

        # And that ocel_time must equal the intended UTC wall clock: 15:08:22
        # UTC on 2026-07-12. If the loader silently interpreted the naive
        # datetime as LOCAL, a CET/CEST host would emit 13:08:22 or 14:08:22
        # instead — this pins the invariant.
        expected = datetime(2026, 7, 12, 15, 8, 22)
        self.assertEqual(deny_str["ocel_time"][0], expected)


class TestDashboardEndToEndPath(unittest.TestCase):
    """End-to-end acceptance: when the dashboard's `_load_combined_eventlog`
    has already converted `time:timestamp` to naive-LOCAL, the resolver in
    `eventlog_conversion._resolve_guardrail_extension` must still yield an
    unshifted gateway `ocel_time` by consulting the preserved
    `time:timestamp_utc_naive` sibling column.
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
        """Simulate a post-`_load_combined_eventlog` frame:
        - `time:timestamp` is a Datetime that has been shifted to naive-LOCAL.
        - `time:timestamp_utc_naive` is the same instant kept as naive-UTC.

        `_resolve_guardrail_extension` MUST use the sibling column, so the
        resulting gateway_deny event's `ocel_time` matches the original
        naive-UTC value — not the shifted local one.
        """
        ts_iso = "2026-07-12T15:08:22.000000"
        utc_naive = datetime(2026, 7, 12, 15, 8, 22)
        # Fake a non-trivial local offset regardless of host tz so the test
        # actually distinguishes shifted from unshifted. +2h ≈ Europe/Berlin
        # summer; we just need any non-zero delta.
        local_offset = timedelta(hours=2)
        shifted_local = utc_naive + local_offset

        row = self._make_row(ts_iso)
        # Base frame (naive-UTC string column, as the CSV holds).
        el = pl.DataFrame([row])
        # Now simulate what `_load_combined_eventlog` produces: parse
        # `time:timestamp` and shift it, AND write a sibling naive-UTC
        # Datetime column.
        el_dashboard = el.with_columns(
            pl.col("time:timestamp")
              .str.to_datetime(strict=False)
              .alias("time:timestamp_utc_naive"),
        ).with_columns(
            # Overwrite time:timestamp with a shifted Datetime. We build the
            # shifted column from a Python literal so the test doesn't depend
            # on the host's actual timezone.
            pl.Series("time:timestamp", [shifted_local], dtype=pl.Datetime("us")),
        )

        ext = _resolve_guardrail_extension(el_dashboard, guardrail_log_path=None)
        deny = ext.event_tables["gateway_deny"]
        self.assertEqual(deny.height, 1)
        # Must reflect the untouched UTC wall clock — 15:08:22, not 17:08:22.
        self.assertEqual(
            deny["ocel_time"][0],
            utc_naive,
            "Resolver must consult time:timestamp_utc_naive so the gateway "
            "event's ocel_time is naive-UTC, not the shifted local time.",
        )

    def test_no_sibling_column_falls_back_to_jsonl(self):
        """When `time:timestamp` is a Datetime AND no sibling column exists
        AND a JSONL is on disk, the resolver must prefer the JSONL — that
        was the original guard in `_resolve_guardrail_extension` and it still
        holds as the fallback path."""
        ts_iso = "2026-07-12T15:08:22.000000"
        utc_naive = datetime(2026, 7, 12, 15, 8, 22)
        shifted_local = utc_naive + timedelta(hours=2)

        row = self._make_row(ts_iso)
        el = pl.DataFrame([row]).with_columns(
            pl.Series("time:timestamp", [shifted_local], dtype=pl.Datetime("us")),
        )

        # Corresponding JSONL record with the same UTC instant as ts epoch.
        # 2026-07-12T15:08:22 UTC == 1783868902.0
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
        # JSONL loader converts epoch to naive-UTC — same 15:08:22 target.
        self.assertEqual(deny["ocel_time"][0], utc_naive)


if __name__ == "__main__":
    unittest.main()
