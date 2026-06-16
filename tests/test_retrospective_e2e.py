import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TIMEOUT_SECONDS = 600

_REQUIRED_KEYS = ("q1_supposed", "q2_actual", "q3_why_diff", "q4_next_time")


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
        # Filename is the thread_id UUID — sanity-check it's a UUID-shaped name.
        self.assertRegex(retro_path.stem, r"^[0-9a-f-]{36}$")

        with open(retro_path, encoding="utf-8") as f:
            payload = json.load(f)

        self.assertIn("thread_id", payload, f"Missing thread_id in {payload!r}")
        self.assertEqual(payload["thread_id"], retro_path.stem)
        self.assertIn("entries", payload, f"Missing entries in {payload!r}")
        entries = payload["entries"]
        self.assertGreater(
            len(entries), 0,
            f"Retrospective file has no entries{diagnostics}\nFile: {payload}",
        )

        # At least one operator agent we know participated in scenario 0
        # (which always exercises the order_agent) should produce a valid entry.
        agent_names = {e.get("agent_name") for e in entries}
        self.assertIn(
            "order_agent", agent_names,
            f"order_agent retrospective missing. Got: {agent_names}",
        )

        # The customer agent must NOT have a retrospective entry — its voice is
        # captured separately via CustomerAgent.get_feedback().
        self.assertNotIn(
            "customer", agent_names,
            f"customer agent should not have a retrospective entry. Got: {agent_names}",
        )

        valid_entries = [e for e in entries if e.get("valid")]
        self.assertGreater(
            len(valid_entries), 0,
            f"No valid retrospective entries; raw entries: {entries}",
        )

        # Verify the structure of one valid entry: all four AAR keys present
        # with non-empty answers.
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


if __name__ == "__main__":
    unittest.main()
