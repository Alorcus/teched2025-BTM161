"""Unit tests for the Metrics Dashboard filter helpers.

These tests build synthetic polars frames and exercise the pure filter/
metadata helpers directly. No Panel server, no MLflow store, no dashboard
render. They cover:
  - _build_case_metadata's aggregation of per-case setup/scenario/timestamps
  - _apply_filters AND semantics across time / scenario / setup
  - empty-checkbox = pass-all
  - the "(unknown)" setup bucket (mapped to None)
  - _case_counts returning (contained, partial) with filters applied
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import polars as pl

from src.dashboard.metrics.metrics_page import (
    _apply_filters,
    _build_case_metadata,
    _case_counts,
    _same_filter,
)

_TS = "time:timestamp"


def _make_eventlog(rows: list[dict]) -> pl.DataFrame:
    """Build a synthetic event log frame. Each row's timestamp is a datetime
    so _build_case_metadata's group_by/agg receives the right dtype without
    us round-tripping through the string parser."""
    df = pl.DataFrame(rows)
    # Ensure the timestamp column is Datetime, even for an empty frame.
    if _TS in df.columns:
        df = df.with_columns(pl.col(_TS).cast(pl.Datetime))
    return df


def _row(case_id: str, ts: datetime, setup: str | None, scenario: int) -> dict:
    return {
        "case_id": case_id,
        _TS: ts,
        "case_setup": setup,
        "case_scenario_index": scenario,
    }


class CaseMetadataBuildTests(unittest.TestCase):
    """Test 1: _build_case_metadata aggregates per-case attributes correctly."""

    def test_one_row_per_case(self):
        t0 = datetime(2026, 7, 6, 10, 0, 0)
        events = _make_eventlog([
            _row("A", t0, "baseline", 0),
            _row("A", t0 + timedelta(seconds=30), "baseline", 0),
            _row("B", t0 + timedelta(minutes=5), "all_handovers", 1),
        ])
        cm = _build_case_metadata(events)
        self.assertEqual(cm.height, 2)
        self.assertEqual(set(cm["case_id"].to_list()), {"A", "B"})

    def test_first_last_timestamps(self):
        t0 = datetime(2026, 7, 6, 10, 0, 0)
        events = _make_eventlog([
            _row("A", t0, "baseline", 0),
            _row("A", t0 + timedelta(seconds=45), "baseline", 0),
            _row("A", t0 + timedelta(seconds=90), "baseline", 0),
        ])
        cm = _build_case_metadata(events).sort("case_id")
        self.assertEqual(cm["first_t"][0], t0)
        self.assertEqual(cm["last_t"][0], t0 + timedelta(seconds=90))

    def test_null_setup_survives_aggregation(self):
        t0 = datetime(2026, 7, 6, 10, 0, 0)
        events = _make_eventlog([
            _row("A", t0, None, -1),
            _row("A", t0 + timedelta(seconds=10), None, -1),
        ])
        cm = _build_case_metadata(events)
        self.assertIsNone(cm["case_setup"][0])
        self.assertEqual(cm["case_scenario_index"][0], -1)

    def test_missing_columns_fall_back(self):
        """Older caches (schema-v3) lack case_setup / case_scenario_index. The
        metadata builder should still emit rows with sensible defaults so the
        dashboard doesn't die on a stale cache."""
        t0 = datetime(2026, 7, 6, 10, 0, 0)
        events = _make_eventlog([
            {"case_id": "A", _TS: t0},
            {"case_id": "A", _TS: t0 + timedelta(seconds=10)},
        ])
        cm = _build_case_metadata(events)
        self.assertEqual(cm.height, 1)
        self.assertIsNone(cm["case_setup"][0])
        self.assertEqual(cm["case_scenario_index"][0], -1)

    def test_empty_input(self):
        cm = _build_case_metadata(pl.DataFrame())
        self.assertEqual(cm.height, 0)
        self.assertIn("case_id", cm.columns)


class ApplyFiltersTests(unittest.TestCase):
    """Test 2: _apply_filters AND semantics and empty-checkbox behaviour."""

    def setUp(self):
        t0 = datetime(2026, 7, 6, 10, 0, 0)
        self.t_early = t0
        self.t_late = t0 + timedelta(hours=1)
        # 5 cases: (id, setup, scenario, start_offset_min, dur_sec)
        cases = [
            ("c0", "baseline", 0, 0, 60),
            ("c1", "baseline", 1, 5, 60),
            ("c2", "all_handovers", 0, 10, 60),
            ("c3", "unconstrained", 3, 15, 60),
            ("c4", None, -1, 20, 60),  # untagged trace -> (unknown) bucket
        ]
        rows = []
        for cid, setup, scen, off_min, dur_s in cases:
            start = t0 + timedelta(minutes=off_min)
            rows.append(_row(cid, start, setup, scen))
            rows.append(_row(cid, start + timedelta(seconds=dur_s), setup, scen))
        self.cm = _build_case_metadata(_make_eventlog(rows))
        self.wide_start = t0 - timedelta(minutes=1)
        self.wide_end = t0 + timedelta(hours=2)

    def test_no_filters_passes_all(self):
        got = _apply_filters(self.cm, self.wide_start, self.wide_end, [], [])
        self.assertEqual(got.height, 5)

    def test_scenario_filter(self):
        got = _apply_filters(self.cm, self.wide_start, self.wide_end, [0], [])
        self.assertEqual(set(got["case_id"].to_list()), {"c0", "c2"})

    def test_scenario_multi_select(self):
        got = _apply_filters(self.cm, self.wide_start, self.wide_end, [0, 1], [])
        self.assertEqual(set(got["case_id"].to_list()), {"c0", "c1", "c2"})

    def test_unspecified_scenario_bucket(self):
        got = _apply_filters(self.cm, self.wide_start, self.wide_end, [-1], [])
        self.assertEqual(set(got["case_id"].to_list()), {"c4"})

    def test_setup_filter(self):
        got = _apply_filters(self.cm, self.wide_start, self.wide_end, [], ["baseline"])
        self.assertEqual(set(got["case_id"].to_list()), {"c0", "c1"})

    def test_unknown_setup_bucket(self):
        """None in the setups list is the '(unknown)' bucket — cases whose
        MLflow trace carried no `setup` tag."""
        got = _apply_filters(self.cm, self.wide_start, self.wide_end, [], [None])
        self.assertEqual(set(got["case_id"].to_list()), {"c4"})

    def test_setup_or_unknown(self):
        got = _apply_filters(
            self.cm, self.wide_start, self.wide_end, [], ["baseline", None]
        )
        self.assertEqual(set(got["case_id"].to_list()), {"c0", "c1", "c4"})

    def test_and_across_groups(self):
        """scenario=0 AND setup=baseline should keep only c0 (both match) —
        c2 has scenario 0 but different setup, c1 matches setup but different
        scenario."""
        got = _apply_filters(
            self.cm, self.wide_start, self.wide_end, [0], ["baseline"]
        )
        self.assertEqual(set(got["case_id"].to_list()), {"c0"})

    def test_time_fully_contained_only(self):
        """A case is included iff its entire span lies inside [start, end]."""
        t0 = datetime(2026, 7, 6, 10, 0, 0)
        # window that clips c0 (starts at t0, ends t0+60s) partially — c0's
        # last_t = t0+60s, so [t0+30s, t0+70s] does NOT fully contain c0.
        got = _apply_filters(
            self.cm,
            t0 + timedelta(seconds=30),
            t0 + timedelta(seconds=70),
            [], [],
        )
        # c0's first_t = t0 is BEFORE start, so it's not contained.
        self.assertNotIn("c0", got["case_id"].to_list())

    def test_empty_metadata_returns_empty(self):
        got = _apply_filters(pl.DataFrame(), self.wide_start, self.wide_end, [], [])
        self.assertEqual(got.height, 0)


class CaseCountsTests(unittest.TestCase):
    """Test 3: _case_counts returns (contained, partial) with filters applied."""

    def setUp(self):
        t0 = datetime(2026, 7, 6, 10, 0, 0)
        self.t0 = t0
        rows = [
            # c0 fully inside [t0, t0+5min]
            _row("c0", t0 + timedelta(seconds=10), "baseline", 0),
            _row("c0", t0 + timedelta(seconds=30), "baseline", 0),
            # c1 partial: starts inside window, ends after
            _row("c1", t0 + timedelta(minutes=4), "baseline", 1),
            _row("c1", t0 + timedelta(minutes=6), "baseline", 1),
        ]
        self.cm = _build_case_metadata(_make_eventlog(rows))

    def test_contained_and_partial_no_filter(self):
        contained, partial = _case_counts(
            self.cm,
            self.t0,
            self.t0 + timedelta(minutes=5),
            [], [],
        )
        self.assertEqual((contained, partial), (1, 1))

    def test_scenario_filter_narrows_both(self):
        """Applying scenario=1 excludes c0 entirely; c1 (scenario 1) is
        partial in the window. Contained=0, partial=1."""
        contained, partial = _case_counts(
            self.cm,
            self.t0,
            self.t0 + timedelta(minutes=5),
            [1], [],
        )
        self.assertEqual((contained, partial), (0, 1))


class SameFilterTests(unittest.TestCase):
    """Test 4: _same_filter compares staged vs applied filter dicts."""

    def test_identical(self):
        t0 = datetime(2026, 7, 6, 10, 0, 0)
        d = {"start": t0, "end": t0 + timedelta(hours=1), "scenarios": [1, 0], "setups": ["baseline"]}
        # Note reordered scenarios in one copy — sorted() comparison must ignore order.
        e = {"start": t0, "end": t0 + timedelta(hours=1), "scenarios": [0, 1], "setups": ["baseline"]}
        self.assertTrue(_same_filter(d, e))

    def test_different_time(self):
        t0 = datetime(2026, 7, 6, 10, 0, 0)
        d = {"start": t0, "end": t0 + timedelta(hours=1), "scenarios": [], "setups": []}
        e = {"start": t0, "end": t0 + timedelta(hours=2), "scenarios": [], "setups": []}
        self.assertFalse(_same_filter(d, e))

    def test_none_in_setups_ordering(self):
        t0 = datetime(2026, 7, 6, 10, 0, 0)
        d = {"start": t0, "end": t0, "scenarios": [], "setups": [None, "baseline"]}
        e = {"start": t0, "end": t0, "scenarios": [], "setups": ["baseline", None]}
        self.assertTrue(_same_filter(d, e))


if __name__ == "__main__":
    unittest.main()
