import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TIMEOUT_SECONDS = 600

_REQUIRED_KEYS = ("q1_supposed", "q2_actual", "q3_why_diff", "q4_next_time")
_SYNTHESIS_KEYS = (
    "what_worked", "what_broke", "agreements", "contradictions", "systemic_fix",
)
_TIMESTAMP_RE = re.compile(r"^\d{8}T\d{6}Z(?:_[0-9a-f]{8})?$")


class TestRetrospectiveE2E(unittest.TestCase):
    """Run a real simulation with --retrospective and verify the output file."""

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self._db_path = Path(self._tmp_dir) / "coffee_shop.db"
        self._retro_dir = Path(self._tmp_dir) / "retrospective_log"

    def test_retrospective_file_written(self):
        env = os.environ.copy()
        env["COFFEE_MACHINE_SEED"] = "100"
        env["COFFEE_SHOP_DB"] = str(self._db_path)
        env["COFFEE_SHOP_SETUP"] = "baseline"
        env["RETROSPECTIVE_LOG_DIR"] = str(self._retro_dir)

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "src.simulate",
                    "--scenario", "0",
                    "--traces", "1",
                    "--retrospective",
                ],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or b"").decode(errors="replace")
            stderr = (e.stderr or b"").decode(errors="replace")
            self.fail(
                f"Simulation timed out after {TIMEOUT_SECONDS}s\n"
                f"\n--- STDOUT (partial) ---\n{stdout}\n"
                f"\n--- STDERR (partial) ---\n{stderr}\n"
            )

        diagnostics = (
            f"\n--- EXIT CODE: {result.returncode} ---\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}\n"
        )
        self.assertEqual(result.returncode, 0, f"Simulation failed{diagnostics}")

        self.assertTrue(
            self._retro_dir.exists(),
            f"Retrospective directory was not created at {self._retro_dir}{diagnostics}",
        )

        files = sorted(self._retro_dir.glob("*.json"))
        self.assertEqual(
            len(files), 1,
            f"Expected exactly 1 retrospective file in {self._retro_dir}, got {len(files)}: "
            f"{[f.name for f in files]}{diagnostics}",
        )

        retro_path = files[0]
        # Filename is a UTC timestamp (with optional thread-id suffix).
        self.assertRegex(
            retro_path.stem, _TIMESTAMP_RE,
            f"Filename {retro_path.name} does not match timestamp pattern",
        )

        with open(retro_path, encoding="utf-8") as f:
            payload = json.load(f)

        self.assertIn("timestamp", payload, f"Missing timestamp in {payload!r}")
        self.assertIn("thread_id", payload, f"Missing thread_id in {payload!r}")
        # thread_id is still UUID-shaped, just no longer the filename.
        self.assertRegex(payload["thread_id"], r"^[0-9a-f-]{36}$")

        entries = payload.get("entries", [])
        self.assertGreater(
            len(entries), 0,
            f"Retrospective file has no entries{diagnostics}\nFile: {payload}",
        )

        agent_names = {e.get("agent_name") for e in entries}
        self.assertIn(
            "order_agent", agent_names,
            f"order_agent retrospective missing. Got: {agent_names}",
        )
        self.assertNotIn(
            "customer", agent_names,
            f"customer agent should not have a retrospective entry. Got: {agent_names}",
        )

        valid_entries = [e for e in entries if e.get("valid")]
        self.assertGreater(
            len(valid_entries), 0,
            f"No valid retrospective entries; raw entries: {entries}",
        )

        # Each valid entry has all four AAR keys with non-empty answers, and
        # peer_review present as a list.
        for entry in valid_entries:
            for key in _REQUIRED_KEYS:
                self.assertIn(key, entry, f"Entry {entry['agent_name']} missing {key}")
                block = entry[key]
                self.assertIsInstance(block, dict, f"{key} should be an object")
                self.assertIn("answer", block, f"{key} missing 'answer'")
                self.assertTrue(
                    isinstance(block["answer"], str) and block["answer"].strip(),
                    f"{entry['agent_name']}.{key}.answer must be non-empty string, got {block['answer']!r}",
                )
            self.assertIn(
                "peer_review", entry,
                f"Entry {entry['agent_name']} missing peer_review",
            )
            self.assertIsInstance(
                entry["peer_review"], list,
                f"peer_review for {entry['agent_name']} must be a list",
            )

        # At least one valid entry should have produced peer_review content.
        self.assertTrue(
            any(len(e["peer_review"]) > 0 for e in valid_entries),
            f"No agent produced any peer_review entries: {[e['peer_review'] for e in valid_entries]}",
        )

        # Synthesis pass produced a structured object.
        synthesis = payload.get("synthesis")
        self.assertIsNotNone(synthesis, f"Synthesis missing. Payload: {payload}")
        self.assertTrue(
            synthesis.get("valid"),
            f"Synthesis not valid. raw_response: {synthesis.get('raw_response')!r}",
        )
        for key in _SYNTHESIS_KEYS:
            self.assertIn(key, synthesis, f"Synthesis missing {key}")
            self.assertIsInstance(synthesis[key], dict, f"Synthesis.{key} should be an object")
            self.assertIn("summary", synthesis[key], f"Synthesis.{key} missing summary")


if __name__ == "__main__":
    unittest.main()
