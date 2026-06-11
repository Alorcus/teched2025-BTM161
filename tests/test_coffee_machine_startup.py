"""Test that start_coffee_machine() launches a working server with all endpoints."""

import time
import unittest

import requests

from src.agents.barista_agent import (
    COFFEE_MACHINE_URL,
    start_coffee_machine,
    stop_coffee_machine,
)


class TestCoffeeMachineStartup(unittest.TestCase):
    """Verify the coffee machine server starts and all endpoints are accessible."""

    @classmethod
    def setUpClass(cls):
        stop_coffee_machine()
        time.sleep(1)
        cls.started = start_coffee_machine()

    @classmethod
    def tearDownClass(cls):
        stop_coffee_machine()

    def test_server_started_successfully(self):
        self.assertTrue(self.started, "start_coffee_machine() returned False")

    def test_healthz(self):
        resp = requests.get(f"{COFFEE_MACHINE_URL}/healthz", timeout=3)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_queue_endpoint(self):
        resp = requests.get(f"{COFFEE_MACHINE_URL}/queue", timeout=3)
        self.assertEqual(resp.status_code, 200)
        queue = resp.json()["queue"]
        self.assertEqual(len(queue), 4)
        for item in queue:
            self.assertIn(item, ("SUCC", "FAIL"))

    def test_reseed_endpoint(self):
        resp = requests.post(
            f"{COFFEE_MACHINE_URL}/reseed", json={"seed": 42}, timeout=3
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "reseeded")

    def test_brew_endpoint(self):
        resp = requests.post(
            f"{COFFEE_MACHINE_URL}/brew",
            json={"drink": "espresso", "correlation_id": "test-001"},
            timeout=3,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("job_id", data)
        self.assertIn("eta_seconds", data)

    def test_job_status_endpoint(self):
        brew_resp = requests.post(
            f"{COFFEE_MACHINE_URL}/brew",
            json={"drink": "latte", "correlation_id": "test-002"},
            timeout=3,
        )
        job_id = brew_resp.json()["job_id"]

        resp = requests.get(f"{COFFEE_MACHINE_URL}/jobs/{job_id}", timeout=3)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(resp.json()["status"], ("brewing", "ready", "failed"))

    def test_clean_endpoint(self):
        resp = requests.post(f"{COFFEE_MACHINE_URL}/clean", json={}, timeout=3)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(resp.json()["status"], ("cleaned", "already_clean"))

    def test_docs_endpoint(self):
        """This is what is_machine_running() checks."""
        resp = requests.get(f"{COFFEE_MACHINE_URL}/docs", timeout=3)
        self.assertLess(resp.status_code, 500)


class TestCoffeeMachineShutdown(unittest.TestCase):
    """Verify stop_coffee_machine() actually terminates the server."""

    def test_stop_kills_server(self):
        started = start_coffee_machine()
        self.assertTrue(started)

        resp = requests.get(f"{COFFEE_MACHINE_URL}/healthz", timeout=3)
        self.assertEqual(resp.status_code, 200)

        stop_coffee_machine()

        # Poll until the server stops listening — fixed sleeps were flaky on
        # slow shutdowns (process teardown + OS port release can exceed 1 s).
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                requests.get(f"{COFFEE_MACHINE_URL}/healthz", timeout=1)
            except requests.ConnectionError:
                return
            time.sleep(0.1)

        self.fail("server still accepting connections 5s after stop_coffee_machine()")


if __name__ == "__main__":
    unittest.main()
