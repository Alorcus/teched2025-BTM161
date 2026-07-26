"""End-to-end simulation of scenario 4 with the soft `assistant_message:on_menu_only`
guardrail active in `baseline`.

This test actually invokes the local Hyperspace AI LLM proxy (per project
convention) and drives a full customer-agent conversation. Scenario 4 —
"Ask for recommendation" — is the historically most reliable way to make the
order_agent hallucinate off-menu items (Hazelnut Latte, Caramel Macchiato,
etc.), so it is where the soft guardrail earns its keep.

The subprocess writes gateway decisions to `guardrail_log/events.jsonl`. After
the simulation, we look for at least one `assistant_message:on_menu_only` verdict —
either a DENY (guardrail actually caught a hallucination) or an ALLOW (the
model happened to stay on-menu but the guardrail still evaluated the turn).
Either outcome proves the guard is wired into the graph. What we assert:

  * The subprocess exits cleanly.
  * The JSONL contains at least one gateway_decision entry for the
    `assistant_message:on_menu_only` guardrail on `assistant_message`.

The subprocess must reach `localhost:6655`, so the bash tool invoking pytest
must disable the sandbox. Skipped automatically if `LLM_PROVIDER` is not set to
"anthropic" or if the proxy is unreachable.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_SECONDS = 600
LOG_RELATIVE = "guardrail_log/events.jsonl"


def _llm_proxy_reachable() -> bool:
    """Return True if either the LLM proxy is reachable or we're using ollama."""
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    if provider == "ollama":
        return True
    if provider != "anthropic":
        return False
    base_url = os.environ.get(
        "ANTHROPIC_BASE_URL", "http://localhost:6655/anthropic/"
    )
    if "localhost" not in base_url and "127.0.0.1" not in base_url:
        return True  # remote endpoint; can't easily probe, let the test try
    try:
        import socket
        host = "127.0.0.1"
        port = 6655
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


@unittest.skipUnless(
    _llm_proxy_reachable(),
    "LLM proxy unreachable — skipping real-LLM scenario 4 E2E test",
)
class TestOffMenuScenario4E2E(unittest.TestCase):
    """Runs scenario 4 on baseline with the real LLM and verifies the guardrail
    evaluated at least one assistant message."""

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self._db_path = Path(self._tmp_dir) / "coffee_shop.db"
        self._log_dir = Path(self._tmp_dir) / "guardrail_log"
        self._log_dir.mkdir()
        self._log_path = self._log_dir / "events.jsonl"

    def tearDown(self):
        for path in (self._db_path, self._log_path):
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    def _find_log(self) -> Path:
        """The simulate CLI writes to `./guardrail_log/events.jsonl` relative
        to the working directory. We use PROJECT_ROOT as cwd so the log
        naturally lands in the project root. We copy the pre-existing content
        away before running to isolate this run's decisions.
        """
        return PROJECT_ROOT / LOG_RELATIVE

    def test_scenario_4_triggers_off_menu_guardrail(self):
        real_log = self._find_log()
        backup = None
        if real_log.exists():
            backup = real_log.read_text(encoding="utf-8")
            real_log.write_text("", encoding="utf-8")

        try:
            env = os.environ.copy()
            env["COFFEE_SHOP_DB"] = str(self._db_path)
            env.setdefault("LLM_PROVIDER", "anthropic")

            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "src.simulate",
                        "--setup",
                        "baseline",
                        "--scenario",
                        "4",
                        "--traces",
                        "1",
                        "--log-level",
                        "info",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_SECONDS,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                self.fail(
                    f"scenario 4 simulation timed out after {TIMEOUT_SECONDS}s\n"
                    f"stdout: {(exc.stdout or b'').decode(errors='replace')}\n"
                    f"stderr: {(exc.stderr or b'').decode(errors='replace')}"
                )

            self.assertEqual(
                result.returncode,
                0,
                f"simulate exited with {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )

            self.assertTrue(real_log.exists(), "no guardrail log produced")
            entries = [
                json.loads(line)
                for line in real_log.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            off_menu_decisions = [
                entry
                for entry in entries
                if entry.get("event_type") == "gateway_decision"
                and entry.get("tool_name") == "assistant_message"
                and any(
                    verdict.get("guardrail_name") == "assistant_message:on_menu_only"
                    for verdict in entry.get("verdicts", [])
                )
            ]
            self.assertGreater(
                len(off_menu_decisions),
                0,
                f"expected at least one assistant_message:on_menu_only gateway decision "
                f"during scenario 4; got 0 out of {len(entries)} entries.\n"
                f"stdout tail:\n{result.stdout[-2000:]}",
            )

            deny_decisions = [
                entry
                for entry in off_menu_decisions
                if entry.get("final_decision") == "deny"
            ]
            print(
                f"\n[E2E] off_menu decisions: total={len(off_menu_decisions)}, "
                f"deny={len(deny_decisions)}"
            )
        finally:
            if backup is not None:
                real_log.write_text(backup, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
