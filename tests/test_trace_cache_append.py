"""Unit tests for trace_cache append semantics.

`_all_traces.csv` is a shareable artefact: it must grow in append mode
and never rewrite rows that are already present.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import polars as pl

from src.dashboard.metrics import trace_cache


def _make_row(case_id: str, name: str = "call_llm") -> dict:
    return {
        "case_id": case_id,
        "identity:id": f"id-{case_id}-{name}",
        "time:timestamp": "2026-07-12T10:00:00.000",
        "time_finished": "2026-07-12T10:00:00.100",
        "concept:name": name,
        "concept:instance": name,
        "org:resource": "test",
        "duration": 100000000,
        "case_setup": "baseline",
        "case_scenario_index": 0,
    }


class _FakeTrace:
    """Stand-in for mlflow.entities.Trace.to_dict() output."""

    def __init__(self, case_id: str, trace_id: str):
        self._case_id = case_id
        self._trace_id = trace_id

    def to_dict(self) -> dict:
        return {
            "info": {"trace_id": self._trace_id, "tags": {"setup": "baseline", "scenario_index": "0"}},
            "spans": [
                {
                    "name": "LangGraph",
                    "span_id": "root",
                    "parent_span_id": None,
                    "attributes": {"metadata": f'{{"thread_id": "{self._case_id}"}}'},
                },
            ],
        }


class TestTraceCacheAppend(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _seed_cache(self, case_ids: list[str]) -> None:
        rows = [_make_row(cid) for cid in case_ids]
        pd.DataFrame(rows).to_csv(self.log_dir / trace_cache.CACHE_FILENAME, index=False)
        (self.log_dir / trace_cache.META_FILENAME).write_text(
            f"{trace_cache._SCHEMA_VERSION}\n"
        )

    def test_appends_new_case_and_skips_covered(self):
        self._seed_cache(["case-old-1", "case-old-2"])

        traces = [
            _FakeTrace("case-old-1", "trace-a"),
            _FakeTrace("case-new-1", "trace-b"),
        ]

        generate_calls: list[str] = []

        def fake_generate(trace_dict):
            cid = json.loads(trace_dict["spans"][0]["attributes"]["metadata"])["thread_id"]
            generate_calls.append(cid)
            return pd.DataFrame([_make_row(cid, name="new_event")])

        with mock.patch.object(
            trace_cache.TraceProcessor, "_get_all_traces", return_value=traces
        ), mock.patch(
            "src.trace_processing.trace_processor.LogGenerator"
        ) as MockLogGen, mock.patch(
            "src.trace_processing.trace_processor._mlflow_trace_count", create=True, return_value=2
        ), mock.patch(
            "src.dashboard.metrics.trace_cache._mlflow_trace_count", return_value=2
        ):
            MockLogGen.return_value.generate_event_log_df.side_effect = fake_generate
            trace_cache.ensure_trace_cache(self.log_dir)

        self.assertEqual(generate_calls, ["case-new-1"])

        result = pl.read_csv(str(self.log_dir / trace_cache.CACHE_FILENAME))
        case_ids = set(result["case_id"].to_list())
        self.assertEqual(case_ids, {"case-old-1", "case-old-2", "case-new-1"})

    def test_preserves_imported_rows_when_mlflow_empty(self):
        self._seed_cache(["case-imported-1", "case-imported-2"])

        with mock.patch(
            "src.dashboard.metrics.trace_cache._mlflow_trace_count", return_value=0
        ):
            result = trace_cache.ensure_trace_cache(self.log_dir)

        self.assertIsNotNone(result)
        df = pl.read_csv(str(self.log_dir / trace_cache.CACHE_FILENAME))
        self.assertEqual(
            set(df["case_id"].to_list()), {"case-imported-1", "case-imported-2"}
        )

    def test_preserves_imported_rows_when_new_traces_appear(self):
        self._seed_cache(["case-imported-1"])

        traces = [_FakeTrace("case-fresh-1", "trace-fresh")]

        def fake_generate(trace_dict):
            cid = json.loads(trace_dict["spans"][0]["attributes"]["metadata"])["thread_id"]
            return pd.DataFrame([_make_row(cid)])

        with mock.patch.object(
            trace_cache.TraceProcessor, "_get_all_traces", return_value=traces
        ), mock.patch(
            "src.trace_processing.trace_processor.LogGenerator"
        ) as MockLogGen, mock.patch(
            "src.dashboard.metrics.trace_cache._mlflow_trace_count", return_value=1
        ):
            MockLogGen.return_value.generate_event_log_df.side_effect = fake_generate
            trace_cache.ensure_trace_cache(self.log_dir)

        df = pl.read_csv(str(self.log_dir / trace_cache.CACHE_FILENAME))
        self.assertEqual(
            set(df["case_id"].to_list()), {"case-imported-1", "case-fresh-1"}
        )

    def test_schema_bump_preserves_stale_rows_via_quarantine(self):
        """On schema-version mismatch, old rows are moved to a versioned
        backup filename rather than deleted, then the canonical cache is
        rebuilt from MLflow."""
        old_schema = trace_cache._SCHEMA_VERSION - 1
        rows = [_make_row("case-stale-1"), _make_row("case-stale-2")]
        pd.DataFrame(rows).to_csv(self.log_dir / trace_cache.CACHE_FILENAME, index=False)
        # Legacy two-line sidecar format — also covers the backward-compat
        # read path.
        (self.log_dir / trace_cache.META_FILENAME).write_text(
            f"2\n{old_schema}\n"
        )

        traces = [_FakeTrace("case-new-after-bump", "trace-x")]

        def fake_generate(trace_dict):
            cid = json.loads(trace_dict["spans"][0]["attributes"]["metadata"])["thread_id"]
            return pd.DataFrame([_make_row(cid)])

        with mock.patch.object(
            trace_cache.TraceProcessor, "_get_all_traces", return_value=traces
        ), mock.patch(
            "src.trace_processing.trace_processor.LogGenerator"
        ) as MockLogGen, mock.patch(
            "src.dashboard.metrics.trace_cache._mlflow_trace_count", return_value=1
        ):
            MockLogGen.return_value.generate_event_log_df.side_effect = fake_generate
            trace_cache.ensure_trace_cache(self.log_dir)

        df = pl.read_csv(str(self.log_dir / trace_cache.CACHE_FILENAME))
        self.assertEqual(set(df["case_id"].to_list()), {"case-new-after-bump"})

        backup = self.log_dir / f"_all_traces.v{old_schema}.csv"
        self.assertTrue(backup.exists(), f"expected quarantine backup at {backup}")
        stale = pl.read_csv(str(backup))
        self.assertEqual(
            set(stale["case_id"].to_list()), {"case-stale-1", "case-stale-2"}
        )

        self.assertEqual(
            trace_cache._cached_schema_version(
                self.log_dir / trace_cache.META_FILENAME
            ),
            trace_cache._SCHEMA_VERSION,
        )

    def test_missing_sidecar_quarantines_existing_cache(self):
        """A cache file with no sidecar has an unknown schema version — its
        shape may not match the current writer, so quarantine it rather
        than risk mixing shapes."""
        rows = [_make_row("case-orphan-1"), _make_row("case-orphan-2")]
        pd.DataFrame(rows).to_csv(
            self.log_dir / trace_cache.CACHE_FILENAME, index=False
        )
        self.assertFalse((self.log_dir / trace_cache.META_FILENAME).exists())

        traces = [_FakeTrace("case-fresh", "trace-fresh")]

        def fake_generate(trace_dict):
            cid = json.loads(trace_dict["spans"][0]["attributes"]["metadata"])["thread_id"]
            return pd.DataFrame([_make_row(cid)])

        with mock.patch.object(
            trace_cache.TraceProcessor, "_get_all_traces", return_value=traces
        ), mock.patch(
            "src.trace_processing.trace_processor.LogGenerator"
        ) as MockLogGen, mock.patch(
            "src.dashboard.metrics.trace_cache._mlflow_trace_count", return_value=1
        ):
            MockLogGen.return_value.generate_event_log_df.side_effect = fake_generate
            trace_cache.ensure_trace_cache(self.log_dir)

        quarantine = self.log_dir / "_all_traces.unknown-schema.csv"
        self.assertTrue(
            quarantine.exists(),
            f"expected unknown-schema quarantine at {quarantine}",
        )
        stale = pl.read_csv(str(quarantine))
        self.assertEqual(
            set(stale["case_id"].to_list()), {"case-orphan-1", "case-orphan-2"}
        )

        canonical = self.log_dir / trace_cache.CACHE_FILENAME
        if canonical.exists():
            fresh = pl.read_csv(str(canonical))
            self.assertEqual(
                set(fresh["case_id"].to_list()),
                {"case-fresh"},
                "canonical cache must not contain pre-existing orphan rows",
            )


if __name__ == "__main__":
    unittest.main()
