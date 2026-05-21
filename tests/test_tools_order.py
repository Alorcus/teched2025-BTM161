"""Tests 9-14: Order processing tools.

Validates process_order (valid/invalid items, invalid extras),
pricing (size modifiers, extras), and calculate_total with discount.
"""
import json
import unittest

from src.agents.order_store import init_db, reset_inventory
from src.agents.order_agent import process_order, calculate_total
from src.agents.shared_components import Order, OrderStatus, Size, MENU


class TestProcessOrderValidItems(unittest.TestCase):
    """Test 9: Order creation with valid menu items."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_creates_order_with_correct_data(self):
        result = process_order.invoke({
            "order": [
                {"name": "latte", "quantity": 2, "size": None, "extras": []},
                {"name": "croissant", "quantity": 1, "size": None, "extras": []},
            ],
            "customer": "Alice",
        })
        data = json.loads(result)
        self.assertIn("order_id", data)
        self.assertTrue(data["order_id"].startswith("ORD"))
        self.assertIn("summary", data)
        self.assertIn("Alice", data["summary"])
        self.assertIn("Latte", data["summary"])
        self.assertIn("Croissant", data["summary"])
        # Verify pricing: 2 lattes = $8.00, 1 croissant = $2.75 => $10.75
        self.assertIn("10.75", data["summary"])


class TestProcessOrderInvalidItem(unittest.TestCase):
    """Test 10: Rejects items not on the menu."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_returns_error_for_unknown_item(self):
        result = process_order.invoke({
            "order": [{"name": "cheesecake", "quantity": 1}],
            "customer": "Bob",
        })
        self.assertIn("cheesecake", result.lower())
        self.assertIn("not on menu", result.lower())


class TestProcessOrderInvalidExtras(unittest.TestCase):
    """Test 11: Rejects extras not in ALLOWED_EXTRAS."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_returns_error_for_invalid_extras(self):
        result = process_order.invoke({
            "order": [{"name": "latte", "quantity": 1, "extras": ["gold flakes"]}],
            "customer": "Carol",
        })
        self.assertIn("gold flakes", result.lower())
        self.assertIn("unknown extras", result.lower())


class TestPricingSizeModifiers(unittest.TestCase):
    """Test 12: Size adjustments: small (-$0.50), large (+$0.75)."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_small_reduces_price(self):
        result = process_order.invoke({
            "order": [{"name": "latte", "quantity": 1, "size": "small", "extras": []}],
            "customer": "Dave",
        })
        data = json.loads(result)
        # Latte base $4.00 - $0.50 = $3.50
        self.assertIn("3.50", data["summary"])

    def test_large_increases_price(self):
        result = process_order.invoke({
            "order": [{"name": "latte", "quantity": 1, "size": "large", "extras": []}],
            "customer": "Eve",
        })
        data = json.loads(result)
        # Latte base $4.00 + $0.75 = $4.75
        self.assertIn("4.75", data["summary"])

    def test_size_applies_per_quantity(self):
        result = process_order.invoke({
            "order": [{"name": "espresso", "quantity": 2, "size": "large", "extras": []}],
            "customer": "Frank",
        })
        data = json.loads(result)
        # Espresso $2.50 * 2 = $5.00 + $0.75 * 2 = $6.50
        self.assertIn("6.50", data["summary"])


class TestPricingExtrasModifiers(unittest.TestCase):
    """Test 13: Non-temperature extras add $0.50 each per quantity."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_paid_extras_add_cost(self):
        result = process_order.invoke({
            "order": [{"name": "latte", "quantity": 1, "extras": ["soy milk", "vanilla syrup"]}],
            "customer": "Grace",
        })
        data = json.loads(result)
        # $4.00 + 2 * $0.50 = $5.00
        self.assertIn("5.00", data["summary"])

    def test_temperature_extras_are_free(self):
        result = process_order.invoke({
            "order": [{"name": "latte", "quantity": 1, "extras": ["iced"]}],
            "customer": "Heidi",
        })
        data = json.loads(result)
        # $4.00 + $0 (iced is free) = $4.00
        self.assertIn("4.00", data["summary"])

    def test_extras_multiply_by_quantity(self):
        result = process_order.invoke({
            "order": [{"name": "espresso", "quantity": 3, "extras": ["extra shot"]}],
            "customer": "Ivan",
        })
        data = json.loads(result)
        # ($2.50 + $0.50) * 3 = $9.00
        self.assertIn("9.00", data["summary"])


class TestCalculateTotalWithDiscount(unittest.TestCase):
    """Test 14: Discount percentage reduces total correctly."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_discount_applied(self):
        # Create an order first
        result = process_order.invoke({
            "order": [{"name": "latte", "quantity": 2}],
            "customer": "Julia",
        })
        data = json.loads(result)
        order_id = data["order_id"]

        # Apply 20% discount
        discount_result = calculate_total.invoke({
            "order_id": order_id,
            "discount_percent": 20,
        })
        discount_data = json.loads(discount_result)
        # Original: $8.00, discount: $1.60, final: $6.40
        self.assertAlmostEqual(discount_data["total"], 6.40, places=2)
        self.assertAlmostEqual(discount_data["discount"], 1.60, places=2)

    def test_zero_discount_unchanged(self):
        result = process_order.invoke({
            "order": [{"name": "americano", "quantity": 1}],
            "customer": "Karl",
        })
        data = json.loads(result)
        order_id = data["order_id"]

        discount_result = calculate_total.invoke({
            "order_id": order_id,
            "discount_percent": 0,
        })
        discount_data = json.loads(discount_result)
        self.assertAlmostEqual(discount_data["total"], 3.00, places=2)
        self.assertAlmostEqual(discount_data["discount"], 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
