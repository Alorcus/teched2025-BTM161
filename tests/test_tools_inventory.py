"""Tests 15-19: Inventory tools.

Validates check_inventory (all available / partial unavailable),
update_stock (decrement, precondition guard, race condition).
"""
import json
import threading
import unittest

from src.agents.order_store import (
    init_db, reset_inventory, save_order, load_order,
    set_item_stock, check_and_update_stock,
)
from src.agents.inventory_agent import check_inventory, update_stock
from src.agents.shared_components import (
    Order, OrderItem, OrderStatus, MENU,
)


def _create_test_order(status=OrderStatus.PENDING, items=None):
    """Create and save a test order, return its order_id_str."""
    if items is None:
        items = [OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[])]
    order = Order(
        customer="TestCustomer",
        status=status,
        total=sum(i.price for i in items),
        items=items,
    )
    save_order(order)
    return order.order_id_str


class TestCheckInventoryAllAvailable(unittest.TestCase):
    """Test 15: Marks order INVENTORY_CONFIRMED when stock sufficient."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_all_available(self):
        order_id = _create_test_order(items=[
            OrderItem(name="latte", quantity=2, price=8.0, size=None, extras=[]),
        ])
        result = check_inventory.invoke({"order_id": order_id})
        data = json.loads(result)
        self.assertTrue(data["all_available"])
        self.assertEqual(data["status"], "inventory_confirmed")

        # Verify DB state
        order = load_order(order_id)
        self.assertEqual(order.status, OrderStatus.INVENTORY_CONFIRMED)


class TestCheckInventoryPartialUnavailable(unittest.TestCase):
    """Test 16: Marks order INVENTORY_ISSUES when any item out of stock."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_out_of_stock_item(self):
        set_item_stock("muffin", 0)
        order_id = _create_test_order(items=[
            OrderItem(name="muffin", quantity=1, price=3.25, size=None, extras=[]),
            OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[]),
        ])
        result = check_inventory.invoke({"order_id": order_id})
        data = json.loads(result)
        self.assertFalse(data["all_available"])
        self.assertEqual(data["status"], "inventory_issues")

        order = load_order(order_id)
        self.assertEqual(order.status, OrderStatus.INVENTORY_ISSUES)

    def test_partial_stock(self):
        set_item_stock("sandwich", 1)
        order_id = _create_test_order(items=[
            OrderItem(name="sandwich", quantity=3, price=19.5, size=None, extras=[]),
        ])
        result = check_inventory.invoke({"order_id": order_id})
        data = json.loads(result)
        self.assertFalse(data["all_available"])


class TestUpdateStockDecrementsCorrectly(unittest.TestCase):
    """Test 17: Stock levels decrease by order quantities."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_stock_decremented(self):
        order_id = _create_test_order(
            status=OrderStatus.INVENTORY_CONFIRMED,
            items=[
                OrderItem(name="espresso", quantity=3, price=7.5, size=None, extras=[]),
            ],
        )
        original_stock = MENU["espresso"].stock  # 20

        result = update_stock.invoke({"order_id": order_id})
        data = json.loads(result)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["items_updated"], 1)

        # Verify DB
        from src.agents.order_store import get_all_inventory
        inventory = get_all_inventory()
        self.assertEqual(inventory["espresso"].stock, original_stock - 3)


class TestUpdateStockRejectsNonConfirmedOrder(unittest.TestCase):
    """Test 18: Refuses to update stock if order isn't INVENTORY_CONFIRMED."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_pending_order_rejected(self):
        order_id = _create_test_order(status=OrderStatus.PENDING)
        result = update_stock.invoke({"order_id": order_id})
        self.assertIn("error", result.lower())
        self.assertIn("not 'inventory_confirmed'", result.lower())

    def test_completed_order_rejected(self):
        order_id = _create_test_order(status=OrderStatus.COMPLETED)
        result = update_stock.invoke({"order_id": order_id})
        self.assertIn("error", result.lower())


class TestUpdateStockRaceCondition(unittest.TestCase):
    """Test 19: Concurrent stock updates don't over-decrement."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_concurrent_updates(self):
        # Set sandwich stock to exactly 5
        set_item_stock("sandwich", 5)

        # Create two orders each wanting 4 sandwiches (total 8 > 5 available)
        order1 = Order(
            customer="Thread1",
            status=OrderStatus.INVENTORY_CONFIRMED,
            total=26.0,
            items=[OrderItem(name="sandwich", quantity=4, price=26.0, size=None, extras=[])],
        )
        order2 = Order(
            customer="Thread2",
            status=OrderStatus.INVENTORY_CONFIRMED,
            total=26.0,
            items=[OrderItem(name="sandwich", quantity=4, price=26.0, size=None, extras=[])],
        )
        save_order(order1)
        save_order(order2)

        results = [None, None]
        errors = [None, None]

        def deduct(order, idx):
            try:
                results[idx] = check_and_update_stock(order)
            except (ValueError, KeyError) as e:
                errors[idx] = e

        t1 = threading.Thread(target=deduct, args=(order1, 0))
        t2 = threading.Thread(target=deduct, args=(order2, 1))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one should succeed, one should fail
        successes = sum(1 for r in results if r is not None)
        failures = sum(1 for e in errors if e is not None)
        self.assertEqual(successes, 1, "Exactly one thread should succeed")
        self.assertEqual(failures, 1, "Exactly one thread should fail with ValueError")

        # Final stock must be >= 0
        from src.agents.order_store import get_all_inventory
        inventory = get_all_inventory()
        self.assertGreaterEqual(inventory["sandwich"].stock, 0)
        self.assertEqual(inventory["sandwich"].stock, 1)  # 5 - 4 = 1


if __name__ == "__main__":
    unittest.main()
