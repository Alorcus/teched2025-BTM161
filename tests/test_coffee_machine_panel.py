"""Tests for the CoffeeMachinePanel with HTTP-only interface via FakeClient."""
import unittest
from unittest.mock import patch

from src.dashboard.coffee_machine_panel import CoffeeMachinePanel, CoffeeMachineClient


class FakeClient:
    """Injectable test double for CoffeeMachineClient."""

    def __init__(self, queue=None):
        self._queue = queue or ["SUCC", "FAIL", "SUCC", "SUCC"]
        self.reseed_calls: list[int] = []

    def get_queue(self):
        return list(self._queue)

    def reseed(self, seed: int) -> bool:
        self.reseed_calls.append(seed)
        self._queue = ["SUCC", "SUCC", "SUCC", "SUCC"]
        return True

    def shift(self):
        """Simulate what the server does when a brew completes."""
        if self._queue:
            self._queue.pop(0)
            self._queue.append("SUCC")


class FakeClientReseedFails(FakeClient):
    def reseed(self, seed: int) -> bool:
        return False


class FakeClientOffline:
    """Simulates a server that is unreachable."""

    def get_queue(self):
        return None

    def reseed(self, seed: int) -> bool:
        return False


class TestPanelInitialState(unittest.TestCase):
    def test_initial_state(self):
        client = FakeClient(["SUCC", "FAIL", "SUCC", "SUCC"])
        panel = CoffeeMachinePanel(client=client)
        self.assertEqual(panel._last_result, "INIT")
        self.assertEqual(panel._queue, ["SUCC", "FAIL", "SUCC", "SUCC"])
        self.assertEqual(panel.state, "idle")

    def test_initial_state_server_offline(self):
        panel = CoffeeMachinePanel(client=FakeClientOffline())
        self.assertEqual(panel._queue, [])
        self.assertEqual(panel.peek_next_result(), "SUCC")


class TestPanelComplete(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient(["FAIL", "SUCC", "SUCC", "SUCC"])
        self.panel = CoffeeMachinePanel(client=self.client)

    def test_complete_success_updates_last_result(self):
        self.client.shift()
        self.panel.complete(True)
        self.assertEqual(self.panel._last_result, "SUCC")
        self.assertEqual(self.panel.state, "ready")

    def test_complete_failure_updates_last_result(self):
        self.client.shift()
        self.panel.complete(False)
        self.assertEqual(self.panel._last_result, "FAIL")
        self.assertEqual(self.panel.state, "failed")

    def test_complete_refreshes_queue_from_client(self):
        self.client.shift()
        self.panel.complete(True)
        self.assertEqual(len(self.panel._queue), 4)
        self.assertEqual(self.panel._queue[0], "SUCC")


class TestPanelRegenerate(unittest.TestCase):
    def test_regenerate_reseeds_and_refreshes(self):
        client = FakeClient(["FAIL", "SUCC", "FAIL", "SUCC"])
        panel = CoffeeMachinePanel(client=client)
        original_queue = list(panel._queue)

        panel.regenerate_queue()

        self.assertEqual(len(client.reseed_calls), 1)
        self.assertEqual(panel._queue, ["SUCC", "SUCC", "SUCC", "SUCC"])
        self.assertNotEqual(panel._queue, original_queue)

    def test_regenerate_keeps_last_result(self):
        client = FakeClient()
        panel = CoffeeMachinePanel(client=client)
        client.shift()
        panel.complete(True)
        self.assertEqual(panel._last_result, "SUCC")

        panel.regenerate_queue()
        self.assertEqual(panel._last_result, "SUCC")

    def test_regenerate_fails_keeps_old_queue(self):
        client = FakeClientReseedFails(["FAIL", "SUCC", "FAIL", "SUCC"])
        panel = CoffeeMachinePanel(client=client)
        original_queue = list(panel._queue)

        panel.regenerate_queue()
        self.assertEqual(panel._queue, original_queue)


class TestPanelBrewing(unittest.TestCase):
    def test_start_brewing_sets_state(self):
        panel = CoffeeMachinePanel(client=FakeClient())
        panel.start_brewing("latte")
        self.assertEqual(panel.state, "brewing")
        self.assertEqual(panel._drink, "latte")

    def test_mark_dirty_sets_state(self):
        panel = CoffeeMachinePanel(client=FakeClient())
        panel.mark_dirty()
        self.assertEqual(panel.state, "dirty")

    def test_reset_clears_state(self):
        panel = CoffeeMachinePanel(client=FakeClient())
        panel.start_brewing("espresso")
        panel.reset()
        self.assertEqual(panel.state, "idle")
        self.assertEqual(panel._drink, "")
        self.assertIsNone(panel._brew_start)


class TestPeekNextResult(unittest.TestCase):
    def test_peek_returns_first_queue_item(self):
        panel = CoffeeMachinePanel(client=FakeClient(["FAIL", "SUCC", "SUCC", "SUCC"]))
        self.assertEqual(panel.peek_next_result(), "FAIL")

    def test_peek_empty_queue_defaults_succ(self):
        panel = CoffeeMachinePanel(client=FakeClientOffline())
        self.assertEqual(panel.peek_next_result(), "SUCC")


class TestPanelLayout(unittest.TestCase):
    def test_panel_returns_column(self):
        panel = CoffeeMachinePanel(client=FakeClient())
        layout = panel.panel()
        self.assertEqual(layout.sizing_mode, "stretch_width")

    def test_regen_button_in_layout(self):
        panel = CoffeeMachinePanel(client=FakeClient())
        layout = panel.panel()
        queue_row = layout[1]
        self.assertIs(queue_row[1], panel._regen_button)

    def test_queue_pane_renders_badges(self):
        panel = CoffeeMachinePanel(client=FakeClient(["SUCC", "FAIL", "SUCC", "SUCC"]))
        html = panel._queue_pane.object
        self.assertIn("INIT", html)
        self.assertIn("▶", html)
        self.assertIn("SUCC", html)
        self.assertIn("FAIL", html)

    def test_button_click_triggers_regenerate(self):
        panel = CoffeeMachinePanel(client=FakeClient(["FAIL", "FAIL", "FAIL", "FAIL"]))
        panel.panel()
        html_before = panel._queue_pane.object
        panel._regen_button.clicks += 1
        html_after = panel._queue_pane.object
        self.assertNotEqual(html_before, html_after)


class TestBackendQueueState(unittest.TestCase):
    """Direct tests of the backend state module (server-side logic, not panel)."""

    def setUp(self):
        import random
        from services.coffee_machine.state import SEED
        import services.coffee_machine.state as state
        state.rng = random.Random(SEED)
        state.outcome_queue = [state._generate_outcome() for _ in range(4)]
        state.jobs = {}
        state.machine_dirty = False

    def test_queue_has_four_elements(self):
        from services.coffee_machine.state import get_queue
        q = get_queue()
        self.assertEqual(len(q), 4)
        for item in q:
            self.assertIn(item, ("SUCC", "FAIL"))

    def test_reseed_regenerates_queue(self):
        from services.coffee_machine.state import get_queue, reseed
        original = get_queue()
        reseed(42)
        new_queue = get_queue()
        self.assertEqual(len(new_queue), 4)
        self.assertNotEqual(new_queue, original)

    def test_queue_is_copy(self):
        from services.coffee_machine.state import get_queue
        q = get_queue()
        q[0] = "MODIFIED"
        self.assertNotEqual(get_queue()[0], "MODIFIED")

    def test_three_consecutive_brews_match_queue(self):
        from services.coffee_machine.state import get_queue, create_job
        for i in range(3):
            queue_before = get_queue()
            predicted = queue_before[0]
            job = create_job("espresso", f"corr-{i}")
            actual = "FAIL" if job["will_fail"] else "SUCC"
            self.assertEqual(actual, predicted)
            queue_after = get_queue()
            self.assertEqual(len(queue_after), 4)
            self.assertEqual(queue_after[:3], queue_before[1:])


if __name__ == "__main__":
    unittest.main()
