"""Unit tests for the Trace Table panel.

These tests do not require an LLM, the database, or the dashboard server.
They feed synthetic DashboardEvents into TraceTablePanel.handle_event /
flush() and assert on the resulting in-memory rows + rendered HTML.
"""
from __future__ import annotations

import time
import unittest

from src.dashboard.interaction.event_bus import DashboardEvent, EventType
from src.dashboard.trace_table_panel import COLUMN_KEYS, TraceTablePanel


def _ev(et, agent=None, content="", tool_name=None, tool_args=None,
        tool_result=None, target_agent=None, supervisor_line=None,
        handoff_context=None, rejection_reason=None):
    return DashboardEvent(
        event_type=et,
        agent_name=agent or "",
        timestamp=time.time(),
        content=content,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=tool_result,
        handoff_context=handoff_context,
        target_agent=target_agent,
        supervisor_line=supervisor_line,
        rejection_reason=rejection_reason,
    )


class TraceTablePanelTests(unittest.TestCase):
    def test_only_row_creators_produce_rows(self):
        p = TraceTablePanel()
        p.handle_event(_ev(EventType.CONVERSATION_START, agent="system"))
        p.handle_event(_ev(EventType.CUSTOMER_MESSAGE, agent="customer", content="hi"))
        p.handle_event(_ev(EventType.AGENT_THINKING, agent="order_agent", content="thinking"))
        p.handle_event(_ev(EventType.AGENT_MESSAGE, agent="order_agent", content="ok"))
        p.handle_event(_ev(EventType.TOOL_CALL, agent="order_agent",
                           tool_name="check_inventory", tool_args={"item": "latte"}))
        p.handle_event(_ev(EventType.TOOL_RESULT, agent="order_agent",
                           tool_name="check_inventory", tool_result="ok"))
        # HANDOFF is intentionally NOT a row-creator (the transfer_to_* TOOL_CALL
        # row already represents the transition).
        p.handle_event(_ev(EventType.HANDOFF, agent="order_agent",
                           target_agent="inventory_agent",
                           handoff_context={"from_agent": "order_agent"}))
        p.handle_event(_ev(EventType.LOG_MESSAGE, agent="system", content="debug"))
        p.handle_event(_ev(EventType.USER_VISIBLE, agent="order_agent", content="hi"))
        p.handle_event(_ev(EventType.CONVERSATION_END, agent="system"))
        p.flush()

        # Row-creators: CUSTOMER_MESSAGE, AGENT_MESSAGE, TOOL_CALL, TOOL_RESULT
        self.assertEqual(len(p.rows), 4)
        self.assertEqual(
            [r["event_type"] for r in p.rows],
            ["CUSTOMER_MESSAGE", "AGENT_MESSAGE", "TOOL_CALL", "TOOL_RESULT"],
        )

    def test_column_ownership(self):
        p = TraceTablePanel()
        p.handle_event(_ev(EventType.CUSTOMER_MESSAGE, agent="customer", content="latte please"))
        p.handle_event(_ev(EventType.AGENT_MESSAGE, agent="order_agent", content="creating order"))
        p.handle_event(_ev(EventType.TOOL_CALL, agent="inventory_agent",
                           tool_name="check_stock", tool_args={"sku": "latte"}))
        p.handle_event(_ev(EventType.AGENT_MESSAGE, agent="barista_agent", content="brewing"))
        p.handle_event(_ev(EventType.AGENT_MESSAGE, agent="customer_service_agent", content="ready"))
        p.flush()

        self.assertEqual(
            [r["agent"] for r in p.rows],
            ["customer", "order_agent", "inventory_agent",
             "barista_agent", "customer_service_agent"],
        )
        for r in p.rows:
            self.assertIn(r["agent"], COLUMN_KEYS)

    def test_supervisor_lines_classified_in_html(self):
        p = TraceTablePanel()
        events = [
            _ev(EventType.CUSTOMER_MESSAGE, agent="customer", content="one latte"),
            _ev(EventType.AGENT_MESSAGE, agent="order_agent", content="creating order",
                supervisor_line="Execution:A02:create_order"),
            _ev(EventType.TOOL_CALL, agent="order_agent",
                tool_name="transfer_to_inventory_agent", tool_args={},
                supervisor_line="Termination:A02:create_order:via_handoff_to_inventory_agent"),
            _ev(EventType.TOOL_RESULT, agent="inventory_agent",
                tool_name="check_inventory", tool_result="in stock",
                supervisor_line=None),
            _ev(EventType.AGENT_MESSAGE, agent="inventory_agent", content="confirmed",
                supervisor_line="Violation:llm_unknown_activity_A99"),
        ]
        for e in events:
            p.handle_event(e)
        p.flush()
        h = p.panel().object

        # Header columns present.
        for label in ["Order Agent", "Inventory Agent", "Barista Agent",
                      "Customer Service", "Customer", "Process Supervisor"]:
            self.assertIn(label, h)

        # Ownership classes appear (we use class="owned" + a CSS var, not
        # `owned-{key}`, so verify the rendered HTML has owned cells).
        self.assertIn('class="owned"', h)

        # Supervisor cells classified by prefix.
        self.assertIn('class="supervisor execution"', h)
        self.assertIn("A02:create_order", h)
        self.assertIn('class="supervisor termination"', h)
        self.assertIn("via_handoff_to_inventory_agent", h)
        self.assertIn('class="supervisor violation"', h)
        self.assertIn("llm_unknown_activity_A99", h)

        # None supervisor line renders as em-dash inside a dash cell.
        self.assertIn('class="supervisor dash"', h)
        self.assertIn("&mdash;", h)

        # Tool call rendering preserves the tool name.
        self.assertIn("transfer_to_inventory_agent", h)

    def test_html_escape_prevents_xss(self):
        p = TraceTablePanel()
        p.handle_event(_ev(EventType.AGENT_MESSAGE, agent="order_agent",
                           content='<script>alert(1)</script> & "x"'))
        p.handle_event(_ev(EventType.TOOL_CALL, agent="order_agent",
                           tool_name="<img onerror=x>", tool_args={"k": "<v>"}))
        p.flush()
        h = p.panel().object

        # No raw script or img tag for the user-supplied content.
        self.assertNotIn("<script>alert(1)</script>", h)
        self.assertIn("&lt;script&gt;", h)
        self.assertIn("&amp;", h)
        # Raw <img onerror=x> from tool_name must be escaped.
        self.assertNotIn("<img onerror=x>", h)
        self.assertIn("&lt;img onerror=x&gt;", h)

    def test_reset_clears_rows(self):
        p = TraceTablePanel()
        p.handle_event(_ev(EventType.AGENT_MESSAGE, agent="order_agent", content="x"))
        p.handle_event(_ev(EventType.AGENT_MESSAGE, agent="barista_agent", content="y"))
        p.flush()
        self.assertEqual(len(p.rows), 2)

        p.reset()
        self.assertEqual(p.rows, [])
        h = p.panel().object
        # Empty-state hint should be visible after reset.
        self.assertIn("No messages yet", h)

    def test_unknown_agent_row_dropped(self):
        p = TraceTablePanel()
        p.handle_event(_ev(EventType.AGENT_MESSAGE, agent="ghost_agent", content="boo"))
        p.flush()
        self.assertEqual(p.rows, [])

    def test_one_owned_cell_per_row_invariant(self):
        """No vertical overlap: exactly one of the agent columns is owned per row."""
        p = TraceTablePanel()
        p.handle_event(_ev(EventType.AGENT_MESSAGE, agent="order_agent", content="a"))
        p.handle_event(_ev(EventType.AGENT_MESSAGE, agent="barista_agent", content="b"))
        p.handle_event(_ev(EventType.CUSTOMER_MESSAGE, agent="customer", content="c"))
        p.flush()
        h = p.panel().object

        body = h.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
        # One <tr> per row.
        rows = [chunk for chunk in body.split("<tr>") if chunk.strip()]
        self.assertEqual(len(rows), 3)
        for r in rows:
            owned_count = r.count('class="owned"')
            self.assertEqual(owned_count, 1, msg=f"row had {owned_count} owned cells: {r[:200]}")

    def test_handle_event_without_flush_does_not_render(self):
        """handle_event accumulates; only flush() updates the pane object."""
        p = TraceTablePanel()
        baseline = p.panel().object
        p.handle_event(_ev(EventType.AGENT_MESSAGE, agent="order_agent", content="x"))
        # rows is updated, but the pane still shows the empty state.
        self.assertEqual(len(p.rows), 1)
        self.assertEqual(p.panel().object, baseline)
        p.flush()
        self.assertNotEqual(p.panel().object, baseline)


class DashboardEventSupervisorFieldTests(unittest.TestCase):
    """Sanity-check that DashboardEvent now carries the supervisor_line field."""

    def test_default_is_none(self):
        ev = DashboardEvent(event_type=EventType.AGENT_MESSAGE, agent_name="x")
        self.assertIsNone(ev.supervisor_line)

    def test_field_round_trips(self):
        ev = DashboardEvent(
            event_type=EventType.AGENT_MESSAGE,
            agent_name="order_agent",
            content="hi",
            supervisor_line="Execution:A01:identify_customer_request",
        )
        self.assertEqual(ev.supervisor_line, "Execution:A01:identify_customer_request")


class RejectedRowTests(unittest.TestCase):
    """AGENT_MESSAGE_REJECTED is a row-creator and renders as a 'rejected'
    kind with the supervisor critique inlined."""

    def test_rejected_event_creates_row(self):
        p = TraceTablePanel()
        p.handle_event(_ev(
            EventType.AGENT_MESSAGE_REJECTED,
            agent="order_agent",
            content="off-topic chitchat",
            supervisor_line="Violation:llm_unknown_activity_A99",
            rejection_reason="You should call process_order instead.",
        ))
        p.flush()

        self.assertEqual(len(p.rows), 1)
        self.assertEqual(p.rows[0]["kind"], "rejected")
        self.assertEqual(p.rows[0]["event_type"], "AGENT_MESSAGE_REJECTED")

        html = p.panel().object
        self.assertIn("owned rejected", html)  # CSS class set
        self.assertIn("off-topic chitchat", html)
        self.assertIn("supervisor:", html)  # critique caption
        self.assertIn("call process_order instead", html)
        # supervisor cell still classifies the line as Violation.
        self.assertIn("supervisor violation", html)


if __name__ == "__main__":
    unittest.main()
