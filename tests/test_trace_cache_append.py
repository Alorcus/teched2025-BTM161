"""Unit tests for trace_cache append semantics.

The Metrics Dashboard's `_all_traces.csv` is a **shareable** artefact: a
user hands the file to a colleague who then explores those traces without
needing the original MLflow store. That contract requires the cache to
grow in append mode — never rewriting rows that are already present. These
tests exercise that behavior end-to-end at the module boundary (stubbing
MLflow and LogGenerator), so a regression that reintroduces the old
full-rebuild would fail here.
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
    """One canonical event-log row. The exact column set doesn't matter for
    these tests — we only care that case_id round-trips."""
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
    """Minimal stand-in for mlflow.entities.Trace.to_dict() output."""

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
    """The cache MUST NOT rewrite existing rows on refresh.

    Why: `_all_traces.csv` is intended for cross-user sharing; a rebuild-on-
    load loop erases curation and drops case_ids that the recipient's MLflow
    doesn't know about. Confirming append semantics here keeps that contract
    intact.
    """

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
        """New case_ids are added; already-covered ones are NOT re-extracted."""
        self._seed_cache(["case-old-1", "case-old-2"])

        traces = [
            _FakeTrace("case-old-1", "trace-a"),
            _FakeTrace("case-new-1", "trace-b"),
        ]

        # LogGenerator returns per-trace event frames. We spy on it to make
        # sure it is NEVER invoked for the already-covered case_id.
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

        # LogGenerator was called only for the NEW case.
        self.assertEqual(generate_calls, ["case-new-1"])

        # The resulting CSV covers both old cases AND the new one.
        result = pl.read_csv(str(self.log_dir / trace_cache.CACHE_FILENAME))
        case_ids = set(result["case_id"].to_list())
        self.assertEqual(case_ids, {"case-old-1", "case-old-2", "case-new-1"})

    def test_preserves_imported_rows_when_mlflow_empty(self):
        """A CSV whose case_ids are not in the local MLflow store must survive
        a dashboard refresh — that is the whole point of the shareable file."""
        self._seed_cache(["case-imported-1", "case-imported-2"])

        with mock.patch(
            "src.dashboard.metrics.trace_cache._mlflow_trace_count", return_value=0
        ):
            result = trace_cache.ensure_trace_cache(self.log_dir)

        # Non-None (file still exists) and untouched.
        self.assertIsNotNone(result)
        df = pl.read_csv(str(self.log_dir / trace_cache.CACHE_FILENAME))
        self.assertEqual(
            set(df["case_id"].to_list()), {"case-imported-1", "case-imported-2"}
        )

    def test_preserves_imported_rows_when_new_traces_appear(self):
        """MLflow has traces the imported CSV doesn't know about → those are
        appended; imported rows are still present unchanged."""
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
        """When the sidecar's schema version differs from the current one the
        row shape has changed, so we cannot append safely. The old rows are
        preserved under a versioned backup filename; the canonical cache is
        rebuilt from MLflow. The shareable-CSV contract says imported rows
        must survive — quarantine, don't delete."""
        # Seed with rows tagged at a stale schema version.
        old_schema = trace_cache._SCHEMA_VERSION - 1
        rows = [_make_row("case-stale-1"), _make_row("case-stale-2")]
        pd.DataFrame(rows).to_csv(self.log_dir / trace_cache.CACHE_FILENAME, index=False)
        # Legacy two-line sidecar is still accepted; use it here to also cover
        # the backward-compat read path.
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

        # Canonical cache: ONLY the fresh row from MLflow.
        df = pl.read_csv(str(self.log_dir / trace_cache.CACHE_FILENAME))
        self.assertEqual(set(df["case_id"].to_list()), {"case-new-after-bump"})

        # Quarantine backup exists at the old-schema name and holds the
        # stale rows verbatim.
        backup = self.log_dir / f"_all_traces.v{old_schema}.csv"
        self.assertTrue(backup.exists(), f"expected quarantine backup at {backup}")
        stale = pl.read_csv(str(backup))
        self.assertEqual(
            set(stale["case_id"].to_list()), {"case-stale-1", "case-stale-2"}
        )

        # Sidecar now reflects the current schema version.
        self.assertEqual(
            trace_cache._cached_schema_version(
                self.log_dir / trace_cache.META_FILENAME
            ),
            trace_cache._SCHEMA_VERSION,
        )

    def test_missing_sidecar_quarantines_existing_cache(self):
        """A cache with rows but no sidecar has an unknown schema version.
        Silently appending v6 rows to a possibly-v5 file would mix shapes,
        so we quarantine the existing file and rebuild fresh. The user's
        escape hatch: rename the quarantined file back and write a matching
        sidecar to opt into reuse."""
        # Seed rows WITHOUT the sidecar.
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

        # Quarantine file holds the pre-existing rows unchanged.
        quarantine = self.log_dir / "_all_traces.unknown-schema.csv"
        self.assertTrue(
            quarantine.exists(),
            f"expected unknown-schema quarantine at {quarantine}",
        )
        stale = pl.read_csv(str(quarantine))
        self.assertEqual(
            set(stale["case_id"].to_list()), {"case-orphan-1", "case-orphan-2"}
        )

        # Canonical cache is either fresh-from-MLflow or absent. No silent
        # mixing of the pre-existing rows with new rows.
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
