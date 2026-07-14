"""Tests for the SQLite fast path in `trace_cache.ensure_trace_cache`.

Two optimisations combined:

  A. Freshness sentinel — fingerprint the MLflow sqlite (mtime, size,
     row count, schema version). If it matches the previous sync's
     fingerprint AND the CSV exists, return immediately without any
     MLflow work.

  B. SQLite bypass — when a sync IS needed, run a single SQL query
     against `trace_info` + `trace_request_metadata` to find the
     `request_id`s whose thread_id isn't already covered. Only those
     are fetched via `client.get_trace()`.

These tests build a real (tiny) mlflow sqlite fixture on disk so we can
exercise the actual query paths, then wrap the trace-fetching in a mock
so we don't need a real MLflow server.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import polars as pl

from src.dashboard.metrics import trace_cache


def _seed_mlflow_sqlite(db_path: Path, traces: list[dict]) -> None:
    """Create the minimal MLflow sqlite schema the SQLite bypass touches.

    Each trace is a dict with:
      request_id: str
      thread_id: str | None   (None → no `mlflow.trace.session` row)
    """
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE trace_info (
                request_id TEXT PRIMARY KEY,
                timestamp_ms INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE trace_request_metadata (
                request_id TEXT,
                key TEXT,
                value TEXT
            )
            """
        )
        for i, t in enumerate(traces):
            conn.execute(
                "INSERT INTO trace_info(request_id, timestamp_ms) VALUES (?, ?)",
                (t["request_id"], 1000 + i),
            )
            if t.get("thread_id") is not None:
                conn.execute(
                    "INSERT INTO trace_request_metadata VALUES (?, 'mlflow.trace.session', ?)",
                    (t["request_id"], t["thread_id"]),
                )
        conn.commit()


def _fake_trace(request_id: str, thread_id: str):
    """Stand-in for mlflow.entities.Trace that carries just enough shape
    for the extraction to reach LogGenerator."""
    class _T:
        def __init__(self):
            self.info = mock.MagicMock()
            self.info.trace_metadata = (
                {"mlflow.trace.session": thread_id} if thread_id else {}
            )

        def to_dict(self):
            return {
                "info": {
                    "trace_id": request_id,
                    "request_id": request_id,
                    "tags": {"setup": "baseline", "scenario_index": "0"},
                },
                "spans": [
                    {
                        "name": "LangGraph",
                        "span_id": "root",
                        "parent_span_id": None,
                        "attributes": {"metadata": f'{{"thread_id": "{thread_id}"}}'} if thread_id else {},
                    }
                ],
            }
    return _T()


def _fake_trace_no_session(request_id: str):
    """A trace with no `mlflow.trace.session` metadata — LogGenerator
    returns empty (mirrors real feedback / standalone ChatAnthropic
    traces)."""
    class _T:
        def __init__(self):
            self.info = mock.MagicMock()
            self.info.trace_metadata = {}

        def to_dict(self):
            return {
                "info": {"trace_id": request_id, "request_id": request_id, "tags": {}},
                "spans": [],
            }
    return _T()


class TestFreshnessSentinel(unittest.TestCase):
    """Sentinel A: fingerprint matches → skip all MLflow work."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self._tmp.name)
        self.db_path = self.log_dir / "mlflow.db"
        self.tracking_uri = f"sqlite:///{self.db_path}"
        _seed_mlflow_sqlite(
            self.db_path,
            [{"request_id": "req-1", "thread_id": "case-1"}],
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_second_call_skips_mlflow_when_fingerprint_unchanged(self):
        """First call runs full sync; second call short-circuits — the
        `_get_traces_by_request_ids` mock must be invoked exactly once."""
        fetches = []

        def fake_fetch(self, ids):
            fetches.append(list(ids))
            return [_fake_trace(rid, "case-1") for rid in ids]

        def fake_generate(trace_dict):
            cid = trace_dict.get("info", {}).get("trace_id", "case-?")
            # Return a trivially-valid event log for case-1.
            return pl.DataFrame([
                {
                    "case_id": "case-1",
                    "identity:id": "id-1",
                    "time:timestamp": "2026-07-12T10:00:00.000",
                    "time_finished": "2026-07-12T10:00:00.100",
                    "concept:name": "call_llm",
                    "concept:instance": "call_llm",
                    "org:resource": "test",
                    "duration": 100000000,
                }
            ])

        with mock.patch(
            "src.trace_processing.trace_processor.TraceProcessor._get_traces_by_request_ids",
            fake_fetch,
        ), mock.patch(
            "src.trace_processing.trace_processor.LogGenerator"
        ) as MockLogGen:
            MockLogGen.return_value.generate_event_log_df.side_effect = fake_generate

            trace_cache.ensure_trace_cache(self.log_dir, tracking_uri=self.tracking_uri)
            self.assertEqual(len(fetches), 1, "first call should fetch")

            trace_cache.ensure_trace_cache(self.log_dir, tracking_uri=self.tracking_uri)
            self.assertEqual(
                len(fetches), 1,
                "second call must skip fetch when fingerprint unchanged",
            )

    def test_sync_reruns_when_db_mtime_changes(self):
        """Bumping the DB's mtime invalidates the fingerprint."""
        fetches = []

        def fake_fetch(self, ids):
            fetches.append(list(ids))
            return [_fake_trace(rid, "case-1" if rid == "req-1" else "case-2") for rid in ids]

        def fake_generate(trace_dict):
            cid = trace_dict.get("info", {}).get("trace_id", "case-?")
            # Map trace_id → distinct case_id so each new trace produces
            # a new row and isn't filtered out as already-covered.
            case_map = {"req-1": "case-1", "req-2": "case-2"}
            case_id = case_map.get(cid, cid)
            return pl.DataFrame([
                {
                    "case_id": case_id,
                    "identity:id": f"id-{case_id}",
                    "time:timestamp": "2026-07-12T10:00:00.000",
                    "time_finished": "2026-07-12T10:00:00.100",
                    "concept:name": "call_llm",
                    "concept:instance": "call_llm",
                    "org:resource": "test",
                    "duration": 100000000,
                }
            ])

        with mock.patch(
            "src.trace_processing.trace_processor.TraceProcessor._get_traces_by_request_ids",
            fake_fetch,
        ), mock.patch(
            "src.trace_processing.trace_processor.LogGenerator"
        ) as MockLogGen:
            MockLogGen.return_value.generate_event_log_df.side_effect = fake_generate

            trace_cache.ensure_trace_cache(self.log_dir, tracking_uri=self.tracking_uri)
            self.assertEqual(len(fetches), 1)

            # Insert a genuinely-new trace, then bump mtime so the
            # fingerprint (mtime + row count) definitely changes.
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    "INSERT INTO trace_info(request_id, timestamp_ms) VALUES ('req-2', 2000)"
                )
                conn.execute(
                    "INSERT INTO trace_request_metadata VALUES ('req-2', 'mlflow.trace.session', 'case-2')"
                )
                conn.commit()
            future = time.time() + 10
            os.utime(self.db_path, (future, future))
            trace_cache.ensure_trace_cache(self.log_dir, tracking_uri=self.tracking_uri)
            self.assertGreater(
                len(fetches), 1,
                "new-trace + mtime change must invalidate the fingerprint",
            )
            # The second fetch should target only the new request_id, not
            # everything — that's the point of the SQLite bypass.
            self.assertEqual(fetches[1], ["req-2"])


class TestNoSessionLedger(unittest.TestCase):
    """Bypass B: traces without `mlflow.trace.session` should only be
    fetched once — a ledger records their request_ids so subsequent
    syncs skip them."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self._tmp.name)
        self.db_path = self.log_dir / "mlflow.db"
        self.tracking_uri = f"sqlite:///{self.db_path}"

    def tearDown(self):
        self._tmp.cleanup()

    def test_no_session_traces_not_refetched_after_first_sync(self):
        _seed_mlflow_sqlite(
            self.db_path,
            [
                {"request_id": "req-feedback-1", "thread_id": None},
                {"request_id": "req-feedback-2", "thread_id": None},
            ],
        )

        fetch_calls: list[list[str]] = []

        def fake_fetch(self, ids):
            fetch_calls.append(list(ids))
            return [_fake_trace_no_session(rid) for rid in ids]

        with mock.patch(
            "src.trace_processing.trace_processor.TraceProcessor._get_traces_by_request_ids",
            fake_fetch,
        ):
            # First call — fetches both, LogGenerator returns empty, ledger
            # records their request_ids.
            trace_cache.ensure_trace_cache(self.log_dir, tracking_uri=self.tracking_uri)
            self.assertEqual(len(fetch_calls), 1)
            self.assertEqual(
                set(fetch_calls[0]), {"req-feedback-1", "req-feedback-2"}
            )

            # Second call — fingerprint would short-circuit; touch mtime
            # so we exercise the ledger path itself, not the sentinel.
            future = time.time() + 10
            os.utime(self.db_path, (future, future))
            trace_cache.ensure_trace_cache(self.log_dir, tracking_uri=self.tracking_uri)

            if len(fetch_calls) > 1:
                self.assertEqual(
                    fetch_calls[1], [],
                    "no-session ledger must skip refetching known feedback traces",
                )

        ledger_path = self.log_dir / trace_cache.NO_SESSION_LEDGER_FILENAME
        self.assertTrue(ledger_path.exists(), "no-session ledger must be written")
        recorded = trace_cache._read_no_session_ledger(ledger_path)
        self.assertEqual(
            recorded, {"req-feedback-1", "req-feedback-2"},
            f"ledger contents: {recorded}",
        )


class TestSchemaBumpClearsLedger(unittest.TestCase):
    """When a schema-version mismatch triggers quarantine, the
    no-session ledger from the old extractor should be dropped — its
    entries were classified under a different LogGenerator."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self._tmp.name)
        self.db_path = self.log_dir / "mlflow.db"
        self.tracking_uri = f"sqlite:///{self.db_path}"

    def tearDown(self):
        self._tmp.cleanup()

    def test_quarantine_clears_no_session_ledger(self):
        _seed_mlflow_sqlite(
            self.db_path,
            [{"request_id": "req-new", "thread_id": "case-new"}],
        )
        # Seed the cache under an older schema, plus a leftover ledger.
        old_schema = trace_cache._SCHEMA_VERSION - 1
        rows = [{
            "case_id": "case-stale",
            "identity:id": "id-stale",
            "time:timestamp": "2026-07-12T10:00:00.000",
            "time_finished": "2026-07-12T10:00:00.100",
            "concept:name": "call_llm",
            "concept:instance": "call_llm",
            "org:resource": "test",
            "duration": 100000000,
        }]
        pl.DataFrame(rows).write_csv(str(self.log_dir / trace_cache.CACHE_FILENAME))
        (self.log_dir / trace_cache.META_FILENAME).write_text(f"{old_schema}\n")
        ledger_path = self.log_dir / trace_cache.NO_SESSION_LEDGER_FILENAME
        ledger_path.write_text("stale-req-1\nstale-req-2\n")

        def fake_fetch(self, ids):
            return [_fake_trace(rid, "case-new") for rid in ids]

        def fake_generate(trace_dict):
            return pl.DataFrame([
                {
                    "case_id": "case-new",
                    "identity:id": "id-new",
                    "time:timestamp": "2026-07-12T11:00:00.000",
                    "time_finished": "2026-07-12T11:00:00.100",
                    "concept:name": "call_llm",
                    "concept:instance": "call_llm",
                    "org:resource": "test",
                    "duration": 100000000,
                }
            ])

        with mock.patch(
            "src.trace_processing.trace_processor.TraceProcessor._get_traces_by_request_ids",
            fake_fetch,
        ), mock.patch(
            "src.trace_processing.trace_processor.LogGenerator"
        ) as MockLogGen:
            MockLogGen.return_value.generate_event_log_df.side_effect = fake_generate
            trace_cache.ensure_trace_cache(self.log_dir, tracking_uri=self.tracking_uri)

        # After quarantine the ledger should have been dropped, then
        # rewritten (empty or non-empty depending on the resync) — the
        # stale entries must not still be present.
        recorded = trace_cache._read_no_session_ledger(ledger_path)
        self.assertNotIn("stale-req-1", recorded)
        self.assertNotIn("stale-req-2", recorded)


if __name__ == "__main__":
    unittest.main()
