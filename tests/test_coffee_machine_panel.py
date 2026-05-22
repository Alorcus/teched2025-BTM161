"""Tests for the CoffeeMachinePanel and backend brew-outcome queue."""
import random
import unittest
from unittest.mock import patch, MagicMock

from services.coffee_machine.state import (
    create_job, get_queue, reseed,
    SEED, FAILURE_RATE,
)
from src.dashboard.coffee_machine_panel import CoffeeMachinePanel


def _reset_machine_state():
    """Reset the coffee machine module state for a clean test."""
    import services.coffee_machine.state as state
    state.rng = random.Random(SEED)
    state.outcome_queue = [state._generate_outcome() for _ in range(4)]
    state.jobs = {}
    state.machine_dirty = False


class TestBrewResultQueue(unittest.TestCase):
    """Verify queue initialization, shifting, and regeneration on the frontend panel."""

    def setUp(self):
        _reset_machine_state()
        self.panel = CoffeeMachinePanel()

    def test_initial_state(self):
        self.assertEqual(self.panel._last_result, "INIT")
        self.assertEqual(len(self.panel._queue), 4)
        for r in self.panel._queue:
            self.assertIn(r, ("SUCC", "FAIL"))

    def test_complete_shifts_queue(self):
        original_queue = list(self.panel._queue)
        create_job("coffee", "test-corr-1")
        self.panel.complete(True)
        self.assertEqual(self.panel._last_result, "SUCC")
        self.assertEqual(len(self.panel._queue), 4)
        self.assertEqual(self.panel._queue[:3], original_queue[1:])

    def test_complete_failure_shifts_queue(self):
        create_job("coffee", "test-corr-2")
        self.panel.complete(False)
        self.assertEqual(self.panel._last_result, "FAIL")
        self.assertEqual(len(self.panel._queue), 4)

    def test_regenerate_changes_queue(self):
        original_queue = list(self.panel._queue)
        self.panel.regenerate_queue()
        self.assertNotEqual(self.panel._queue, original_queue)
        self.assertEqual(len(self.panel._queue), 4)

    def test_regenerate_keeps_last_result(self):
        create_job("coffee", "test-corr-3")
        self.panel.complete(True)
        self.assertEqual(self.panel._last_result, "SUCC")
        self.panel.regenerate_queue()
        self.assertEqual(self.panel._last_result, "SUCC")


class TestQueueSyncWithBrewResults(unittest.TestCase):
    """Verify the queue predicts actual brew outcomes across 3 consecutive brews."""

    def setUp(self):
        _reset_machine_state()

    def test_three_consecutive_brews_match_queue(self):
        """Queue position 0 must match each brew outcome, and queue shifts correctly."""
        for i in range(3):
            queue_before = get_queue()
            predicted = queue_before[0]

            job = create_job("espresso", f"corr-{i}")

            actual = "FAIL" if job["will_fail"] else "SUCC"
            self.assertEqual(
                actual, predicted,
                f"Brew {i+1}: queue predicted {predicted} but got {actual}"
            )

            queue_after = get_queue()
            self.assertEqual(len(queue_after), 4)
            self.assertEqual(
                queue_after[:3], queue_before[1:],
                f"Brew {i+1}: queue did not shift correctly"
            )


class TestBackendQueueEndpoint(unittest.TestCase):
    """Verify the backend queue state functions."""

    def setUp(self):
        _reset_machine_state()

    def test_queue_has_four_elements(self):
        q = get_queue()
        self.assertEqual(len(q), 4)
        for item in q:
            self.assertIn(item, ("SUCC", "FAIL"))

    def test_reseed_regenerates_queue(self):
        original = get_queue()
        reseed(42)
        new_queue = get_queue()
        self.assertEqual(len(new_queue), 4)
        self.assertNotEqual(new_queue, original)

    def test_queue_is_copy(self):
        q = get_queue()
        q[0] = "MODIFIED"
        self.assertNotEqual(get_queue()[0], "MODIFIED")


class TestRegenerateTriggersBackendReseed(unittest.TestCase):
    """Verify that the regenerate button reseeds the backend and the frontend shows new queue."""

    def setUp(self):
        _reset_machine_state()

    def test_regenerate_changes_backend_queue_and_frontend_reflects_it(self):
        panel = CoffeeMachinePanel()
        original_queue = list(panel._queue)

        with patch("src.dashboard.coffee_machine_panel.random.randint", return_value=42):
            panel.regenerate_queue()

        self.assertEqual(len(panel._queue), 4)
        self.assertNotEqual(panel._queue, original_queue)
        self.assertEqual(panel._queue, get_queue())

    def test_button_click_triggers_regenerate_and_updates_html(self):
        """Simulate a Panel button click and verify the queue pane HTML changes."""
        panel = CoffeeMachinePanel()
        panel.panel()

        html_before = panel._queue_pane.object
        panel._regen_button.clicks += 1

        html_after = panel._queue_pane.object
        self.assertNotEqual(html_before, html_after)


class TestRegenerateButtonClickable(unittest.TestCase):
    """Verify the regenerate button fires and updates the queue."""

    def setUp(self):
        _reset_machine_state()
        self.panel = CoffeeMachinePanel()
        self.layout = self.panel.panel()

    def test_button_is_in_panel_layout(self):
        queue_row = self.layout[1]
        self.assertIs(queue_row[1], self.panel._regen_button)

    def test_queue_pane_renders_html(self):
        html = self.panel._queue_pane.object
        self.assertIn("INIT", html)
        self.assertIn("▶", html)


class TestQueueNotClipping(unittest.TestCase):
    """Verify that the queue row has no overflow/height issues that would cause clipping."""

    def setUp(self):
        _reset_machine_state()

    def test_frame_uses_stretch_width(self):
        panel = CoffeeMachinePanel()
        layout = panel.panel()
        self.assertEqual(layout.sizing_mode, "stretch_width")

    def test_queue_row_has_stretch_width(self):
        panel = CoffeeMachinePanel()
        layout = panel.panel()
        queue_row = layout[1]
        self.assertEqual(queue_row.sizing_mode, "stretch_width")

    def test_frame_has_border_style(self):
        panel = CoffeeMachinePanel()
        layout = panel.panel()
        self.assertIn("border", layout.styles)
        self.assertIn("solid", layout.styles["border"])

    def test_main_pane_no_border(self):
        panel = CoffeeMachinePanel()
        html = panel._pane.object
        self.assertNotIn("border:", html.replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
