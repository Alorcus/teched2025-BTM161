"""Tests for the Customer Context pane's render helper.

The pane is a strict mirror of `CustomerAgent.history` — customer turns, agent
turns rendered as "Staff", and `inject_experience()` notes rendered as
italic `[Experience]` rows. These tests lock the source-of-truth contract:
same input → same rows, in order, with content HTML-escaped and no truncation.
"""
import unittest

from src.dashboard.interaction.interaction_page import (
    CUSTOMER_CONTEXT_EMPTY_HTML,
    _render_customer_context,
)


class TestRenderCustomerContext(unittest.TestCase):
    def test_empty_history_uses_placeholder(self):
        self.assertEqual(_render_customer_context([]), CUSTOMER_CONTEXT_EMPTY_HTML)

    def test_all_three_roles_rendered_in_order(self):
        history = [
            ("customer", "Hi, I'd like an espresso."),
            ("agent", "Coming right up!"),
            (
                "system_note",
                "You received your coffee but it tastes slightly off.",
            ),
            ("customer", "This is metallic — I'd like a refund."),
        ]

        html = _render_customer_context(history)

        customer_pos = html.find("Hi, I&#x27;d like an espresso.")
        staff_pos = html.find("Coming right up!")
        experience_pos = html.find("slightly off.")
        refund_pos = html.find("metallic")

        self.assertNotEqual(customer_pos, -1)
        self.assertNotEqual(staff_pos, -1)
        self.assertNotEqual(experience_pos, -1)
        self.assertNotEqual(refund_pos, -1)
        self.assertLess(customer_pos, staff_pos)
        self.assertLess(staff_pos, experience_pos)
        self.assertLess(experience_pos, refund_pos)

        self.assertIn("<b>Customer</b>", html)
        self.assertIn("<b>Staff</b>", html)
        self.assertIn("<b>[Experience]</b>", html)
        self.assertIn("font-style:italic", html)

    def test_content_is_html_escaped(self):
        history = [("customer", "<script>alert('xss')</script>")]

        html = _render_customer_context(history)

        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_no_truncation_of_long_content(self):
        long_text = "espresso " * 200
        history = [("agent", long_text.strip())]

        html = _render_customer_context(history)

        self.assertIn(long_text.strip(), html)
        self.assertNotIn("...", html)

    def test_newlines_in_content_become_br(self):
        history = [("customer", "line one\nline two")]

        html = _render_customer_context(history)

        self.assertIn("line one<br>line two", html)

    def test_unknown_role_still_rendered(self):
        """History values are trusted (they come from CustomerAgent itself),
        but a defensive default keeps a typo from silently dropping rows."""
        history = [("mystery_role", "hello")]

        html = _render_customer_context(history)

        self.assertIn("hello", html)
        self.assertIn("mystery_role", html)


if __name__ == "__main__":
    unittest.main()
