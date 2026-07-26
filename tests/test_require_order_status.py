"""Order-lifecycle enforcement via the `require_order_status` gateway guardrail.

This replaces the former OrderStateMachine: each state-changing tool may only be
called from the order statuses listed in its guardrail's `allowed` set. These tests
exercise the predicate directly and via a YAML-built Catalog, and cover the
preconditions that used to be inline in the tools (start_preparation / update_stock).
"""

import tempfile
import unittest
from pathlib import Path

from src.agents.order_store import init_db, reset_inventory, save_order
from src.agents.shared_components import Order, OrderItem, OrderStatus
from src.control_plane.catalog import Catalog
from src.control_plane.predicates import (
    require_order_status_predicate,
    PREDICATE_REGISTRY,
)
from src.control_plane.types import Effect, GuardrailContext


def _order_with_status(status: OrderStatus) -> str:
    order = Order(
        customer="GateTest",
        status=status,
        total=4.0,
        items=[OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[])],
    )
    save_order(order)
    return order.order_id_str


def _ctx(order_id: str, tool_name: str = "offer_refund") -> GuardrailContext:
    return GuardrailContext(
        agent_id="customer_service_agent",
        tool_name=tool_name,
        tool_args={"order_id": order_id},
        state={},
        allowed_handovers=[],
    )


class TestRequireOrderStatusPredicate(unittest.TestCase):
    def setUp(self):
        init_db()
        reset_inventory()

    def test_registered(self):
        self.assertIn("require_order_status", PREDICATE_REGISTRY)

    def test_allows_when_status_in_allowed(self):
        order_id = _order_with_status(OrderStatus.COMPLETED)
        predicate = require_order_status_predicate(["completed"])
        self.assertEqual(predicate(_ctx(order_id)).effect, Effect.ALLOW)

    def test_denies_when_status_not_allowed(self):
        order_id = _order_with_status(OrderStatus.PENDING)
        predicate = require_order_status_predicate(["completed"])
        verdict = predicate(_ctx(order_id))
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("pending", verdict.reason_for_llm)
        self.assertIn("completed", verdict.reason_for_llm)

    def test_flag_effect_when_configured(self):
        order_id = _order_with_status(OrderStatus.PENDING)
        predicate = require_order_status_predicate(["completed"], effect="flag")
        self.assertEqual(predicate(_ctx(order_id)).effect, Effect.FLAG)

    def test_unresolvable_order_denies(self):
        predicate = require_order_status_predicate(["completed"])
        self.assertEqual(predicate(_ctx("ORD999999")).effect, Effect.DENY)

    def test_missing_order_id_denies(self):
        predicate = require_order_status_predicate(["completed"])
        ctx = GuardrailContext(
            agent_id="x",
            tool_name="offer_refund",
            tool_args={},
            state={},
            allowed_handovers=[],
        )
        self.assertEqual(predicate(ctx).effect, Effect.DENY)

    def test_unresolvable_respects_flag_effect(self):
        predicate = require_order_status_predicate(["completed"], effect="flag")
        self.assertEqual(predicate(_ctx("ORD999999")).effect, Effect.FLAG)

    def test_update_stock_precondition(self):
        """Formerly the inline gate in update_stock: only inventory_confirmed."""
        predicate = require_order_status_predicate(["inventory_confirmed"])
        confirmed = _order_with_status(OrderStatus.INVENTORY_CONFIRMED)
        pending = _order_with_status(OrderStatus.PENDING)
        completed = _order_with_status(OrderStatus.COMPLETED)
        self.assertEqual(
            predicate(_ctx(confirmed, "update_stock")).effect, Effect.ALLOW
        )
        self.assertEqual(predicate(_ctx(pending, "update_stock")).effect, Effect.DENY)
        self.assertEqual(predicate(_ctx(completed, "update_stock")).effect, Effect.DENY)

    def test_start_preparation_precondition(self):
        """Formerly the inline gate in start_preparation: confirmed or a retry status."""
        predicate = require_order_status_predicate(
            ["inventory_confirmed", "in_preparation", "preparation_error"]
        )
        for allowed_status in (
            OrderStatus.INVENTORY_CONFIRMED,
            OrderStatus.IN_PREPARATION,
            OrderStatus.PREPARATION_ERROR,
        ):
            oid = _order_with_status(allowed_status)
            self.assertEqual(
                predicate(_ctx(oid, "start_preparation")).effect, Effect.ALLOW
            )
        pending = _order_with_status(OrderStatus.PENDING)
        self.assertEqual(
            predicate(_ctx(pending, "start_preparation")).effect, Effect.DENY
        )


class TestRequireOrderStatusViaCatalog(unittest.TestCase):
    """The full YAML path: predicate_args carrying `allowed` and `effect`."""

    def setUp(self):
        init_db()
        reset_inventory()

    def _catalog(self, effect: str) -> Catalog:
        yaml_text = f"""\
guardrails:
  - id: offer_refund:order_status
    type: hard
    tools: [offer_refund]
    effect: {effect}
    predicate: require_order_status
    predicate_args:
      allowed: [completed]
      effect: {effect}
"""
        d = tempfile.mkdtemp()
        setup = Path(d) / "baseline"
        (setup / "guardrails").mkdir(parents=True)
        (setup / "guidelines").mkdir(parents=True)
        (setup / "guardrails" / "coffee_shop.yaml").write_text(
            yaml_text, encoding="utf-8"
        )
        return Catalog(setup)

    def test_deny_path(self):
        [gr] = self._catalog("deny").guardrails(["offer_refund:order_status"])
        completed = _order_with_status(OrderStatus.COMPLETED)
        pending = _order_with_status(OrderStatus.PENDING)
        self.assertEqual(gr.eval(_ctx(completed)).effect, Effect.ALLOW)
        self.assertEqual(gr.eval(_ctx(pending)).effect, Effect.DENY)

    def test_flag_path(self):
        [gr] = self._catalog("flag").guardrails(["offer_refund:order_status"])
        pending = _order_with_status(OrderStatus.PENDING)
        self.assertEqual(gr.eval(_ctx(pending)).effect, Effect.FLAG)


if __name__ == "__main__":
    unittest.main()
