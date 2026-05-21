"""Tests 25-28: Customer service tools.

Validates offer_refund, offer_partial_refund, get_alternatives, and estimate_prep_time.
"""
import json
import unittest

from src.agents.order_store import (
    init_db, reset_inventory, save_order, load_order, set_item_stock,
)
from src.agents.customer_service_agent import offer_refund, offer_partial_refund
from src.agents.inventory_agent import get_alternatives
from src.agents.barista_agent import estimate_prep_time
from src.agents.shared_components import Order, OrderItem, OrderStatus


def _create_order_with_total(total, status=OrderStatus.PENDING):
    """Create and save an order with a specific total."""
    order = Order(
        customer="ServiceTest",
        status=status,
        total=total,
        items=[OrderItem(name="latte", quantity=1, price=total, size=None, extras=[])],
    )
    save_order(order)
    return order.order_id_str


class TestOfferRefund(unittest.TestCase):
    """Test 25: Full refund zeroes total, sets REFUNDED."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_full_refund(self):
        order_id = _create_order_with_total(10.50)
        result = offer_refund.invoke({"order_id": order_id})
        data = json.loads(result)

        self.assertAlmostEqual(data["refund_amount"], 10.50, places=2)

        order = load_order(order_id)
        self.assertEqual(order.status, OrderStatus.REFUNDED)
        self.assertAlmostEqual(order.total, 0.0, places=2)


class TestOfferPartialRefund(unittest.TestCase):
    """Test 26: Partial refund deducts correct percentage."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_50_percent_refund(self):
        order_id = _create_order_with_total(20.00)
        result = offer_partial_refund.invoke({
            "order_id": order_id,
            "refund_percent": 50,
        })
        data = json.loads(result)

        self.assertAlmostEqual(data["refund_amount"], 10.00, places=2)
        self.assertAlmostEqual(data["original_total"], 20.00, places=2)
        self.assertAlmostEqual(data["new_total"], 10.00, places=2)

        order = load_order(order_id)
        self.assertAlmostEqual(order.total, 10.00, places=2)

    def test_25_percent_refund(self):
        order_id = _create_order_with_total(8.00)
        result = offer_partial_refund.invoke({
            "order_id": order_id,
            "refund_percent": 25,
        })
        data = json.loads(result)

        self.assertAlmostEqual(data["refund_amount"], 2.00, places=2)
        self.assertAlmostEqual(data["new_total"], 6.00, places=2)


class TestGetAlternativesReturnsSameCategory(unittest.TestCase):
    """Test 27: Alternatives are in-stock, same-category items."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_coffee_alternatives(self):
        # Set latte stock to 0 so it doesn't appear as alternative to itself
        set_item_stock("latte", 0)
        result = get_alternatives.invoke({"item_name": "latte"})
        data = json.loads(result)

        self.assertEqual(data["category"], "coffee")
        # Should include other coffee items with stock > 0
        alt_names = [a.lower() for a in data["alternatives"]]
        alt_text = " ".join(alt_names)
        self.assertIn("espresso", alt_text)
        self.assertIn("cappuccino", alt_text)
        self.assertIn("americano", alt_text)
        # Should NOT include pastries or food
        self.assertNotIn("croissant", alt_text)
        self.assertNotIn("muffin", alt_text)

    def test_pastry_alternatives(self):
        set_item_stock("croissant", 0)
        result = get_alternatives.invoke({"item_name": "croissant"})
        data = json.loads(result)

        self.assertEqual(data["category"], "pastry")
        alt_text = " ".join(data["alternatives"]).lower()
        self.assertIn("muffin", alt_text)
        self.assertNotIn("latte", alt_text)


class TestEstimatePrepTime(unittest.TestCase):
    """Test 28: Time estimate formula: 2 + 1.5 * (n-1) minutes."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_single_item(self):
        order = Order(
            customer="Test",
            status=OrderStatus.INVENTORY_CONFIRMED,
            total=4.0,
            items=[OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[])],
        )
        save_order(order)
        result = estimate_prep_time.invoke({"order_id": order.order_id_str})
        # 1 item: base 2 + (1-1)*1.5 = 2.0 min
        self.assertIn("2.0", result)

    def test_three_items(self):
        order = Order(
            customer="Test",
            status=OrderStatus.INVENTORY_CONFIRMED,
            total=10.0,
            items=[
                OrderItem(name="latte", quantity=2, price=8.0, size=None, extras=[]),
                OrderItem(name="croissant", quantity=1, price=2.75, size=None, extras=[]),
            ],
        )
        save_order(order)
        result = estimate_prep_time.invoke({"order_id": order.order_id_str})
        # total_items = 2 + 1 = 3; time = 2 + (3-1)*1.5 = 5.0 min
        self.assertIn("5.0", result)

    def test_five_items(self):
        order = Order(
            customer="Test",
            status=OrderStatus.INVENTORY_CONFIRMED,
            total=20.0,
            items=[
                OrderItem(name="espresso", quantity=5, price=12.5, size=None, extras=[]),
            ],
        )
        save_order(order)
        result = estimate_prep_time.invoke({"order_id": order.order_id_str})
        # total_items = 5; time = 2 + (5-1)*1.5 = 8.0 min
        self.assertIn("8.0", result)


if __name__ == "__main__":
    unittest.main()
