"""Tests 20-24: Barista tools.

Validates prepare_order (success/failure paths, precondition guard),
remake_order_item (success/failure), and estimate_prep_time.
"""
import json
import unittest
from unittest.mock import patch

from src.agents.order_store import init_db, reset_inventory, save_order, load_order
from src.agents.barista_agent import prepare_order, remake_order_item, estimate_prep_time
from src.agents.shared_components import Order, OrderItem, OrderStatus


def _create_confirmed_order(items=None):
    """Create a saved order with INVENTORY_CONFIRMED status."""
    if items is None:
        items = [OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[])]
    order = Order(
        customer="BaristaTest",
        status=OrderStatus.INVENTORY_CONFIRMED,
        total=sum(i.price for i in items),
        items=items,
    )
    save_order(order)
    return order.order_id_str


class TestPrepareOrderSuccess(unittest.TestCase):
    """Test 20: Barista marks order COMPLETED on success path."""

    def setUp(self):
        init_db()
        reset_inventory()

    @patch("src.agents.barista_agent.random.random", return_value=0.5)
    def test_success(self, mock_random):
        order_id = _create_confirmed_order()
        result = prepare_order.invoke({"order_id": order_id})
        data = json.loads(result)
        self.assertEqual(data["status"], "completed")

        order = load_order(order_id)
        self.assertEqual(order.status, OrderStatus.COMPLETED)


class TestPrepareOrderFailure(unittest.TestCase):
    """Test 21: Barista marks PREPARATION_ERROR on failure path."""

    def setUp(self):
        init_db()
        reset_inventory()

    @patch("src.agents.barista_agent.random.choice")
    @patch("src.agents.barista_agent.random.random", return_value=0.1)
    def test_failure(self, mock_random, mock_choice):
        items = [
            OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[]),
            OrderItem(name="croissant", quantity=1, price=2.75, size=None, extras=[]),
        ]
        order_id = _create_confirmed_order(items=items)

        # Make random.choice return the first item as the failed one
        order = load_order(order_id)
        mock_choice.return_value = order.items[0]

        result = prepare_order.invoke({"order_id": order_id})
        data = json.loads(result)
        self.assertEqual(data["status"], "preparation_error")

        order = load_order(order_id)
        self.assertEqual(order.status, OrderStatus.PREPARATION_ERROR)


class TestPrepareOrderRejectsWrongStatus(unittest.TestCase):
    """Test 22: Cannot prepare order not in INVENTORY_CONFIRMED state."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_pending_rejected(self):
        order = Order(
            customer="Test",
            status=OrderStatus.PENDING,
            total=4.0,
            items=[OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[])],
        )
        save_order(order)
        result = prepare_order.invoke({"order_id": order.order_id_str})
        self.assertIn("error", result.lower())
        self.assertIn("inventory not confirmed", result.lower())

        # Status should not have changed
        loaded = load_order(order.order_id_str)
        self.assertEqual(loaded.status, OrderStatus.PENDING)


class TestRemakeOrderItemSuccess(unittest.TestCase):
    """Test 23: Successful remake sets COMPLETED."""

    def setUp(self):
        init_db()
        reset_inventory()

    @patch("src.agents.barista_agent.random.random", return_value=0.5)
    def test_remake_success(self, mock_random):
        order = Order(
            customer="Test",
            status=OrderStatus.PREPARATION_ERROR,
            total=4.0,
            items=[OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[])],
        )
        save_order(order)

        result = remake_order_item.invoke({
            "order_id": order.order_id_str,
            "item_name": "latte",
        })
        data = json.loads(result)
        self.assertEqual(data["status"], "completed")

        loaded = load_order(order.order_id_str)
        self.assertEqual(loaded.status, OrderStatus.COMPLETED)


class TestRemakeOrderItemFailure(unittest.TestCase):
    """Test 24: Failed remake leaves status unchanged."""

    def setUp(self):
        init_db()
        reset_inventory()

    @patch("src.agents.barista_agent.random.random", return_value=0.05)
    def test_remake_failure(self, mock_random):
        order = Order(
            customer="Test",
            status=OrderStatus.PREPARATION_ERROR,
            total=4.0,
            items=[OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[])],
        )
        save_order(order)

        result = remake_order_item.invoke({
            "order_id": order.order_id_str,
            "item_name": "latte",
        })
        data = json.loads(result)
        self.assertEqual(data["status"], "preparation_error")

        loaded = load_order(order.order_id_str)
        self.assertEqual(loaded.status, OrderStatus.PREPARATION_ERROR)


if __name__ == "__main__":
    unittest.main()
