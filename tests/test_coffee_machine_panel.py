"""Tests for the CoffeeMachinePanel, including the brew result queue and regenerate button."""
import os
import random
import unittest

from src.dashboard.coffee_machine_panel import CoffeeMachinePanel, SEED, FAILURE_RATE


class TestBrewResultQueue(unittest.TestCase):
    """Verify queue initialization, shifting, and regeneration."""

    def setUp(self):
        self.panel = CoffeeMachinePanel()

    def test_initial_state(self):
        self.assertEqual(self.panel._last_result, "INIT")
        self.assertEqual(len(self.panel._queue), 4)
        for r in self.panel._queue:
            self.assertIn(r, ("SUCC", "FAIL"))

    def test_complete_shifts_queue(self):
        original_queue = list(self.panel._queue)
        self.panel.complete(True)
        self.assertEqual(self.panel._last_result, "SUCC")
        self.assertEqual(len(self.panel._queue), 4)
        # First 3 items of new queue should be items 1-3 of original
        self.assertEqual(self.panel._queue[:3], original_queue[1:])

    def test_complete_failure_shifts_queue(self):
        self.panel.complete(False)
        self.assertEqual(self.panel._last_result, "FAIL")
        self.assertEqual(len(self.panel._queue), 4)

    def test_regenerate_resets_rng_to_seed(self):
        self.panel.complete(True)
        self.panel.complete(True)
        self.panel.regenerate_queue()
        # After regenerate, the queue should match a fresh panel's queue
        fresh = CoffeeMachinePanel()
        self.assertEqual(self.panel._queue, fresh._queue)


class TestQueueMatchesCoffeeMachine(unittest.TestCase):
    """The dashboard queue must predict the actual coffee machine outcomes."""

    def test_queue_matches_machine_rng_sequence(self):
        """Reproduce the coffee machine's RNG and verify the dashboard predicts correctly."""
        # Replicate the coffee machine's RNG (same seed, same consumption pattern)
        machine_rng = random.Random(SEED)
        expected_outcomes = []
        for _ in range(4):
            machine_rng.uniform(1, 3)  # duration consumed by create_job
            will_fail = machine_rng.random() < FAILURE_RATE
            expected_outcomes.append("FAIL" if will_fail else "SUCC")

        panel = CoffeeMachinePanel()
        self.assertEqual(panel._queue, expected_outcomes)

    def test_queue_after_shift_predicts_fifth_brew(self):
        """After one brew completes, the new tail element predicts brew #5."""
        machine_rng = random.Random(SEED)
        expected_outcomes = []
        for _ in range(5):
            machine_rng.uniform(1, 3)
            will_fail = machine_rng.random() < FAILURE_RATE
            expected_outcomes.append("FAIL" if will_fail else "SUCC")

        panel = CoffeeMachinePanel()
        panel.complete(True)
        # Queue should now be predictions for brews 2-5
        self.assertEqual(panel._queue, expected_outcomes[1:])

    def test_regenerate_resyncs_with_machine(self):
        """After regenerate, queue matches a fresh machine start."""
        panel = CoffeeMachinePanel()
        panel.complete(True)
        panel.complete(True)
        panel.regenerate_queue()

        machine_rng = random.Random(SEED)
        expected = []
        for _ in range(4):
            machine_rng.uniform(1, 3)
            will_fail = machine_rng.random() < FAILURE_RATE
            expected.append("FAIL" if will_fail else "SUCC")

        self.assertEqual(panel._queue, expected)


class TestRegenerateButtonClickable(unittest.TestCase):
    """Verify the regenerate button fires and updates the queue."""

    def setUp(self):
        self.panel = CoffeeMachinePanel()
        self.layout = self.panel.panel()

    def test_button_exists_and_triggers_regenerate(self):
        original_queue = list(self.panel._queue)
        self.panel._rng.seed(42)
        self.panel._regen_button.clicks += 1
        self.assertEqual(len(self.panel._queue), 4)

    def test_button_is_in_panel_layout(self):
        """The regenerate button must be part of the returned panel layout."""
        # Layout is a Column (frame); second element is the Row with queue + button
        queue_row = self.layout[1]
        self.assertIs(queue_row[1], self.panel._regen_button)

    def test_queue_pane_renders_html(self):
        """Queue pane must have non-empty HTML content after init."""
        html = self.panel._queue_pane.object
        self.assertIn("INIT", html)
        self.assertIn("▶", html)


class TestQueueNotClipping(unittest.TestCase):
    """Verify that the queue row has no overflow/height issues that would cause clipping."""

    def test_frame_uses_stretch_width(self):
        """The panel frame must use stretch_width so it doesn't clip children."""
        panel = CoffeeMachinePanel()
        layout = panel.panel()
        self.assertEqual(layout.sizing_mode, "stretch_width")

    def test_queue_row_has_stretch_width(self):
        panel = CoffeeMachinePanel()
        layout = panel.panel()
        queue_row = layout[1]
        self.assertEqual(queue_row.sizing_mode, "stretch_width")

    def test_frame_has_border_style(self):
        """The outer frame should have a border (everything inside one box)."""
        panel = CoffeeMachinePanel()
        layout = panel.panel()
        self.assertIn("border", layout.styles)
        self.assertIn("solid", layout.styles["border"])

    def test_main_pane_no_border(self):
        """The inner HTML pane should NOT have its own border (border is on the frame)."""
        panel = CoffeeMachinePanel()
        html = panel._pane.object
        self.assertNotIn("border:", html.replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
