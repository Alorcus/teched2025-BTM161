import unittest

from src.conversation import _build_retrospective_views


class TestBuildRetrospectiveViews(unittest.TestCase):
    """Verify per-agent transcript partitioning by handoff boundary."""

    def test_customer_message_attached_only_to_active_agent(self):
        """A customer message lands in the active agent's view, not in
        a downstream agent's view."""
        transcript = [
            ("customer", "Hi, I'd like a coffee"),
            ("order_agent", "What size?"),
            ("customer", "Medium"),
            ("order_agent", "$3.50"),
            ("inventory_agent", "Stock OK"),
        ]

        agents, views = _build_retrospective_views(transcript, supervisor_log_path=None)

        self.assertIn("order_agent", views)
        self.assertIn("inventory_agent", views)

        order_view = views["order_agent"]
        inventory_view = views["inventory_agent"]

        self.assertIn("Customer: Hi, I'd like a coffee", order_view)
        self.assertIn("Customer: Medium", order_view)

        # The leak: "Medium" must not appear in inventory_agent's view because
        # it was answering order_agent's question, before inventory_agent took
        # the conversation.
        self.assertNotIn("Customer: Medium", inventory_view)
        self.assertNotIn("Customer: Hi, I'd like a coffee", inventory_view)

    def test_customer_message_during_agent_window_attaches_to_that_agent(self):
        """A customer message between two of the same agent's turns belongs
        to that agent only."""
        transcript = [
            ("customer", "I want a latte"),
            ("order_agent", "Size?"),
            ("customer", "Large"),
            ("inventory_agent", "Checking stock"),
            ("customer", "Hello?"),
            ("inventory_agent", "Stock OK"),
            ("barista_agent", "Brewing"),
        ]

        _, views = _build_retrospective_views(transcript, supervisor_log_path=None)

        self.assertIn("Customer: I want a latte", views["order_agent"])
        self.assertIn("Customer: Large", views["order_agent"])
        self.assertNotIn("Customer: Hello?", views["order_agent"])

        self.assertIn("Customer: Hello?", views["inventory_agent"])
        self.assertNotIn("Customer: I want a latte", views["inventory_agent"])
        self.assertNotIn("Customer: Large", views["inventory_agent"])

        # barista_agent never had a customer turn during its window.
        self.assertIn("barista_agent", views)
        self.assertNotIn("Customer:", views["barista_agent"])

    def test_pre_agent_customer_message_attaches_to_first_agent(self):
        """Customer messages that arrive before any operator speaks attach
        to whichever operator speaks first."""
        transcript = [
            ("customer", "First message"),
            ("customer", "Second message"),
            ("order_agent", "Hello, how can I help?"),
        ]

        _, views = _build_retrospective_views(transcript, supervisor_log_path=None)

        self.assertIn("Customer: First message", views["order_agent"])
        self.assertIn("Customer: Second message", views["order_agent"])

    def test_agent_with_no_turns_absent_from_views(self):
        transcript = [
            ("customer", "Hi"),
            ("order_agent", "Hello"),
        ]

        agents, views = _build_retrospective_views(transcript, supervisor_log_path=None)

        self.assertEqual(agents, ["order_agent"])
        self.assertNotIn("inventory_agent", views)
        self.assertNotIn("barista_agent", views)
        self.assertNotIn("customer_service_agent", views)

    def test_handoff_back_to_earlier_agent(self):
        """If control returns to a prior agent, customer messages in that
        second window go to that agent."""
        transcript = [
            ("customer", "Coffee please"),
            ("order_agent", "Size?"),
            ("customer", "Small"),
            ("inventory_agent", "Out of small cups, can we substitute?"),
            ("customer", "Yes"),
            ("order_agent", "Confirmed substitution"),
        ]

        _, views = _build_retrospective_views(transcript, supervisor_log_path=None)

        # "Yes" was the customer answering inventory_agent — should appear there.
        self.assertIn("Customer: Yes", views["inventory_agent"])
        self.assertNotIn("Customer: Yes", views["order_agent"])

        # "Small" was answering order_agent.
        self.assertIn("Customer: Small", views["order_agent"])
        self.assertNotIn("Customer: Small", views["inventory_agent"])


if __name__ == "__main__":
    unittest.main()
