"""Unit tests for the MLflow trace-tag helper.

Exercises the pure `_tag_trace` function with mocked mlflow calls — no real
tracking store, no autolog, no LLM. Verifies the setup/scenario values are
stringified per the MLflow tag-storage contract and that scenario=None is
translated to the -1 sentinel used across the pipeline.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch, call

from src.conversation import _tag_trace


class TagTraceTests(unittest.TestCase):
    def test_tags_setup_and_scenario_as_strings(self):
        with patch("src.conversation.mlflow.set_trace_tag") as m:
            _tag_trace("tr-abc", "baseline", 0)
        m.assert_has_calls([
            call("tr-abc", "setup", "baseline"),
            call("tr-abc", "scenario_index", "0"),
        ])
        self.assertEqual(m.call_count, 2)

    def test_scenario_none_becomes_minus_one(self):
        """Custom prompts / Jupyter / random-then-unresolved cases arrive as
        None. The helper coerces to -1 so downstream buckets stay consistent."""
        with patch("src.conversation.mlflow.set_trace_tag") as m:
            _tag_trace("tr-xyz", "baseline", None)
        m.assert_has_calls([
            call("tr-xyz", "setup", "baseline"),
            call("tr-xyz", "scenario_index", "-1"),
        ])

    def test_int_scenario_stringified(self):
        with patch("src.conversation.mlflow.set_trace_tag") as m:
            _tag_trace("tr-1", "all_handovers", 3)
        m.assert_has_calls([
            call("tr-1", "setup", "all_handovers"),
            call("tr-1", "scenario_index", "3"),
        ])


if __name__ == "__main__":
    unittest.main()
