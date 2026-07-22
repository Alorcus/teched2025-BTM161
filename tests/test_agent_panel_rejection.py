"""Tests for `AgentPanel.mark_last_ai_message_rejected`.

Guards the "strike through the rejected AI entry" UX behavior against
regressions — specifically that content-list AIMessage bodies (Anthropic's
tool-turn shape) get matched correctly, not just plain-string content.
"""
import unittest
from unittest.mock import MagicMock, patch

with patch("panel.pane.HTML") as _mock_html:
    _mock_html.return_value = MagicMock()
    from src.dashboard.interaction.agent_panel import AgentPanel


_TEST_CONFIG = {
    "name": "Order Agent",
    "icon": "🍽️",
    "color": "#123456",
    "bg_color": "#abcdef",
}


class TestMarkLastAiMessageRejected(unittest.TestCase):
    def _make_panel(self) -> AgentPanel:
        with patch("panel.pane.HTML") as _mock_html:
            _mock_html.return_value = MagicMock()
            return AgentPanel(agent_name="order_agent", config=_TEST_CONFIG)

    def test_marks_matching_ai_entry(self):
        panel = self._make_panel()
        panel.add_message("ai", "Try our hazelnut latte!")
        panel.add_message("user", "hmm, ok")

        result = panel.mark_last_ai_message_rejected("Try our hazelnut latte!")

        self.assertTrue(result)
        roles = [m["role"] for m in panel.messages]
        self.assertIn("rejected_ai", roles)
        rejected = next(m for m in panel.messages if m["role"] == "rejected_ai")
        self.assertEqual(rejected["content"], "Try our hazelnut latte!")

    def test_finds_most_recent_matching_entry_only(self):
        panel = self._make_panel()
        panel.add_message("ai", "First off-menu attempt")
        panel.add_message("user", "hmm")
        panel.add_message("ai", "Second off-menu attempt")

        result = panel.mark_last_ai_message_rejected("Second off-menu attempt")

        self.assertTrue(result)
        # First "ai" entry must remain untouched.
        roles = [m["role"] for m in panel.messages]
        self.assertEqual(roles.count("ai"), 1)
        self.assertEqual(roles.count("rejected_ai"), 1)

    def test_returns_false_when_no_match(self):
        panel = self._make_panel()
        panel.add_message("ai", "unrelated content")

        result = panel.mark_last_ai_message_rejected("different content")

        self.assertFalse(result)
        self.assertEqual(panel.messages[0]["role"], "ai")

    def test_returns_false_when_no_messages(self):
        panel = self._make_panel()
        self.assertFalse(panel.mark_last_ai_message_rejected("anything"))

    def test_skips_non_ai_roles(self):
        panel = self._make_panel()
        panel.add_message("tool_call", "process_order(...)")
        panel.add_message("user", "hi")

        # Even if a non-ai entry happens to have matching content, only the
        # last matching "ai" role should be affected.
        self.assertFalse(panel.mark_last_ai_message_rejected("hi"))


if __name__ == "__main__":
    unittest.main()
