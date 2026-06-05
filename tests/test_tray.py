"""Tests for the tray mechanic.

Validates:
- Tray data model (place, get, clear, isolation between orders)
- place_on_tray tool (category lookup, contamination carry-over, order status checks)
- check_tray tool (returns correct contents)
- Tray consumption flow (contamination detection, order marked COMPLETED, tray cleared)
"""
import json
import unittest
from unittest.mock import MagicMock, patch

from src.agents.order_store import init_db, reset_inventory, save_order, load_order
from src.agents.shared_components import Order, OrderItem, OrderStatus
from src.agents.tray import (
    _trays, TrayEntry, place_on_tray as tray_place,
    get_tray, clear_tray, tray_as_list,
)
from src.agents.tray_tools import place_on_tray, check_tray
from src.agents.barista_agent import ORDER_STATUS_CACHE


def _create_order(status=OrderStatus.PENDING, items=None):
    if items is None:
        items = [OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[])]
    order = Order(customer="Test", status=status, total=sum(i.price for i in items), items=items)
    save_order(order)
    return order.order_id_str


class TestTrayDataModel(unittest.TestCase):
    """Low-level tray store operations."""

    def setUp(self):
        _trays.clear()

    def test_place_and_get(self):
        result = tray_place("ORD0001", "latte", 2, "coffee")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["item_name"], "latte")
        self.assertEqual(result[0]["quantity"], 2)
        self.assertEqual(result[0]["category"], "coffee")
        self.assertFalse(result[0]["contaminated"])

        entries = get_tray("ORD0001")
        self.assertEqual(len(entries), 1)
        self.assertIsInstance(entries[0], TrayEntry)

    def test_multiple_items(self):
        tray_place("ORD0001", "latte", 1, "coffee")
        tray_place("ORD0001", "croissant", 2, "pastry")
        entries = get_tray("ORD0001")
        self.assertEqual(len(entries), 2)

    def test_order_isolation(self):
        tray_place("ORD0001", "latte", 1, "coffee")
        tray_place("ORD0002", "espresso", 1, "coffee")
        self.assertEqual(len(get_tray("ORD0001")), 1)
        self.assertEqual(len(get_tray("ORD0002")), 1)
        self.assertEqual(get_tray("ORD0001")[0].item_name, "latte")
        self.assertEqual(get_tray("ORD0002")[0].item_name, "espresso")

    def test_clear_tray(self):
        tray_place("ORD0001", "latte", 1, "coffee")
        tray_place("ORD0001", "muffin", 1, "pastry")
        clear_tray("ORD0001")
        self.assertEqual(get_tray("ORD0001"), [])

    def test_clear_nonexistent(self):
        clear_tray("ORD9999")  # should not raise

    def test_get_empty(self):
        self.assertEqual(get_tray("ORD9999"), [])

    def test_tray_as_list(self):
        tray_place("ORD0001", "latte", 1, "coffee", contaminated=True)
        result = tray_as_list("ORD0001")
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], dict)
        self.assertTrue(result[0]["contaminated"])

    def test_contaminated_flag(self):
        tray_place("ORD0001", "latte", 1, "coffee", contaminated=True)
        entry = get_tray("ORD0001")[0]
        self.assertTrue(entry.contaminated)


class TestPlaceOnTrayTool(unittest.TestCase):
    """Tests for the place_on_tray LangChain tool."""

    def setUp(self):
        init_db()
        reset_inventory()
        _trays.clear()
        ORDER_STATUS_CACHE.clear()

    def test_place_food_item_success(self):
        order_id = _create_order(status=OrderStatus.INVENTORY_CONFIRMED, items=[
            OrderItem(name="croissant", quantity=2, price=5.50, size=None, extras=[]),
        ])
        result = place_on_tray.invoke({"order_id": order_id, "item_name": "croissant", "quantity": 2})
        data = json.loads(result)
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["tray"]), 1)
        self.assertEqual(data["tray"][0]["item_name"], "croissant")
        self.assertEqual(data["tray"][0]["category"], "pastry")
        self.assertFalse(data["tray"][0]["contaminated"])

    def test_place_food_wrong_status(self):
        order_id = _create_order(status=OrderStatus.PENDING, items=[
            OrderItem(name="croissant", quantity=1, price=2.75, size=None, extras=[]),
        ])
        result = place_on_tray.invoke({"order_id": order_id, "item_name": "croissant", "quantity": 1})
        data = json.loads(result)
        self.assertEqual(data["status"], "error")
        self.assertIn("Cannot place items", data["message"])

    def test_place_unknown_item(self):
        result = place_on_tray.invoke({"order_id": "ORD0001", "item_name": "unicorn_frappuccino", "quantity": 1})
        data = json.loads(result)
        self.assertEqual(data["status"], "error")
        self.assertIn("Unknown item", data["message"])

    def test_place_coffee_clean(self):
        order_id = _create_order(status=OrderStatus.IN_PREPARATION, items=[
            OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[]),
        ])
        ORDER_STATUS_CACHE[order_id] = {"status": "ready", "last_brew_contaminated": False}
        result = place_on_tray.invoke({"order_id": order_id, "item_name": "latte", "quantity": 1})
        data = json.loads(result)
        self.assertEqual(data["status"], "success")
        self.assertFalse(data["tray"][0]["contaminated"])

    def test_place_coffee_contaminated(self):
        order_id = _create_order(status=OrderStatus.IN_PREPARATION, items=[
            OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[]),
        ])
        ORDER_STATUS_CACHE[order_id] = {"status": "ready", "last_brew_contaminated": True}
        result = place_on_tray.invoke({"order_id": order_id, "item_name": "latte", "quantity": 1})
        data = json.loads(result)
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["tray"][0]["contaminated"])

    def test_place_food_in_preparation_status(self):
        """Food can be placed when order is IN_PREPARATION (barista has started)."""
        order_id = _create_order(status=OrderStatus.IN_PREPARATION, items=[
            OrderItem(name="muffin", quantity=1, price=3.25, size=None, extras=[]),
        ])
        result = place_on_tray.invoke({"order_id": order_id, "item_name": "muffin", "quantity": 1})
        data = json.loads(result)
        self.assertEqual(data["status"], "success")

    def test_place_multiple_items_accumulates(self):
        order_id = _create_order(status=OrderStatus.INVENTORY_CONFIRMED, items=[
            OrderItem(name="croissant", quantity=1, price=2.75, size=None, extras=[]),
            OrderItem(name="muffin", quantity=1, price=3.25, size=None, extras=[]),
        ])
        place_on_tray.invoke({"order_id": order_id, "item_name": "croissant", "quantity": 1})
        result = place_on_tray.invoke({"order_id": order_id, "item_name": "muffin", "quantity": 1})
        data = json.loads(result)
        self.assertEqual(len(data["tray"]), 2)


class TestCheckTrayTool(unittest.TestCase):
    """Tests for the check_tray LangChain tool."""

    def setUp(self):
        _trays.clear()

    def test_empty_tray(self):
        result = check_tray.invoke({"order_id": "ORD0001"})
        data = json.loads(result)
        self.assertEqual(data["item_count"], 0)
        self.assertEqual(data["tray"], [])

    def test_tray_with_items(self):
        tray_place("ORD0001", "latte", 2, "coffee")
        tray_place("ORD0001", "croissant", 1, "pastry")
        result = check_tray.invoke({"order_id": "ORD0001"})
        data = json.loads(result)
        self.assertEqual(data["item_count"], 2)
        self.assertEqual(data["order_id"], "ORD0001")
        self.assertEqual(data["tray"][0]["item_name"], "latte")
        self.assertEqual(data["tray"][1]["item_name"], "croissant")


class TestTrayConsumption(unittest.TestCase):
    """Tests for the tray consumption flow in ConversationRunner._consume_tray."""

    def setUp(self):
        init_db()
        reset_inventory()
        _trays.clear()
        ORDER_STATUS_CACHE.clear()

    def test_consume_marks_order_completed(self):
        order_id = _create_order(status=OrderStatus.IN_PREPARATION, items=[
            OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[]),
        ])
        tray_place(order_id, "latte", 1, "coffee", contaminated=False)

        from src.dashboard.interaction.conversation_runner import ConversationRunner
        runner = ConversationRunner.__new__(ConversationRunner)
        runner._current_order_id = order_id
        runner.shop = MagicMock()
        runner.event_bus = MagicMock()

        runner._consume_tray()

        order = load_order(order_id)
        self.assertEqual(order.status, OrderStatus.COMPLETED)
        self.assertEqual(get_tray(order_id), [])

    def test_consume_injects_contamination_experience(self):
        order_id = _create_order(status=OrderStatus.IN_PREPARATION, items=[
            OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[]),
        ])
        tray_place(order_id, "latte", 1, "coffee", contaminated=True)

        from src.dashboard.interaction.conversation_runner import ConversationRunner
        runner = ConversationRunner.__new__(ConversationRunner)
        runner._current_order_id = order_id
        runner.shop = MagicMock()
        runner.event_bus = MagicMock()

        runner._consume_tray()

        runner.shop.customer_agent.inject_experience.assert_called_once()
        call_args = runner.shop.customer_agent.inject_experience.call_args[0][0]
        self.assertIn("metallic", call_args.lower())

    def test_consume_no_order_id(self):
        """No crash when order_id is None."""
        from src.dashboard.interaction.conversation_runner import ConversationRunner
        runner = ConversationRunner.__new__(ConversationRunner)
        runner._current_order_id = None
        runner.shop = MagicMock()
        runner.event_bus = MagicMock()
        runner._consume_tray()  # should not raise

    def test_consume_empty_tray(self):
        """No state changes when tray is empty."""
        order_id = _create_order(status=OrderStatus.IN_PREPARATION, items=[
            OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[]),
        ])

        from src.dashboard.interaction.conversation_runner import ConversationRunner
        runner = ConversationRunner.__new__(ConversationRunner)
        runner._current_order_id = order_id
        runner.shop = MagicMock()
        runner.event_bus = MagicMock()

        runner._consume_tray()

        order = load_order(order_id)
        self.assertEqual(order.status, OrderStatus.IN_PREPARATION)
        runner.shop.customer_agent.inject_experience.assert_not_called()

    def test_consume_mixed_clean_and_contaminated(self):
        """Only contaminated items trigger the experience injection."""
        order_id = _create_order(status=OrderStatus.IN_PREPARATION, items=[
            OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[]),
            OrderItem(name="croissant", quantity=1, price=2.75, size=None, extras=[]),
        ])
        tray_place(order_id, "croissant", 1, "pastry", contaminated=False)
        tray_place(order_id, "latte", 1, "coffee", contaminated=True)

        from src.dashboard.interaction.conversation_runner import ConversationRunner
        runner = ConversationRunner.__new__(ConversationRunner)
        runner._current_order_id = order_id
        runner.shop = MagicMock()
        runner.event_bus = MagicMock()

        runner._consume_tray()

        runner.shop.customer_agent.inject_experience.assert_called_once()
        order = load_order(order_id)
        self.assertEqual(order.status, OrderStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
