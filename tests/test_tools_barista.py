"""Tests 20-24: Barista tools.

Validates start_preparation (fires brew), end_preparation (polls for result),
retry paths, precondition guard, and estimate_prep_time.
"""
import json
import unittest
from unittest.mock import patch, MagicMock

from src.agents.order_store import init_db, reset_inventory, save_order, load_order
from src.agents.barista_agent import (
    start_preparation, end_preparation, estimate_prep_time,
    ORDER_STATUS_CACHE, ORDER_JOB_MAP,
)
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


def _mock_brew_response(job_id="job-123", eta=2.0):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"job_id": job_id, "eta_seconds": eta}
    return resp


def _mock_status_response(status):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"job_id": "job-123", "status": status}
    return resp


class TestStartPreparationSuccess(unittest.TestCase):
    """Test 20: start_preparation fires brew and returns brewing status."""

    def setUp(self):
        init_db()
        reset_inventory()
        ORDER_STATUS_CACHE.clear()
        ORDER_JOB_MAP.clear()

    @patch("src.agents.barista_agent.safe_post")
    @patch("src.agents.barista_agent.is_machine_running", return_value=True)
    @patch("src.agents.barista_agent.time.sleep")
    def test_start_returns_brewing(self, mock_sleep, mock_running, mock_post):
        order_id = _create_confirmed_order()
        mock_post.return_value = _mock_brew_response()

        result = start_preparation.invoke({"order_id": order_id})
        data = json.loads(result)
        self.assertEqual(data["status"], "brewing")
        self.assertIn("eta_seconds", data)

        order = load_order(order_id)
        self.assertEqual(order.status, OrderStatus.IN_PREPARATION)


class TestEndPreparationSuccess(unittest.TestCase):
    """Test 20b: end_preparation marks order COMPLETED on success path."""

    def setUp(self):
        init_db()
        reset_inventory()
        ORDER_STATUS_CACHE.clear()
        ORDER_JOB_MAP.clear()

    @patch("src.agents.barista_agent.safe_get")
    @patch("src.agents.barista_agent.safe_post")
    @patch("src.agents.barista_agent.is_machine_running", return_value=True)
    @patch("src.agents.barista_agent.time.sleep")
    def test_end_success(self, mock_sleep, mock_running, mock_post, mock_get):
        order_id = _create_confirmed_order()
        mock_post.return_value = _mock_brew_response()
        mock_get.return_value = _mock_status_response("ready")

        start_preparation.invoke({"order_id": order_id})
        result = end_preparation.invoke({"order_id": order_id})
        data = json.loads(result)
        self.assertEqual(data["status"], "ready")

        order = load_order(order_id)
        self.assertEqual(order.status, OrderStatus.COMPLETED)


class TestEndPreparationFailure(unittest.TestCase):
    """Test 21: end_preparation marks PREPARATION_ERROR on failure path."""

    def setUp(self):
        init_db()
        reset_inventory()
        ORDER_STATUS_CACHE.clear()
        ORDER_JOB_MAP.clear()

    @patch("src.agents.barista_agent.safe_get")
    @patch("src.agents.barista_agent.safe_post")
    @patch("src.agents.barista_agent.is_machine_running", return_value=True)
    @patch("src.agents.barista_agent.time.sleep")
    def test_failure(self, mock_sleep, mock_running, mock_post, mock_get):
        order_id = _create_confirmed_order()
        mock_post.return_value = _mock_brew_response()
        mock_get.return_value = _mock_status_response("failed")

        start_preparation.invoke({"order_id": order_id})
        result = end_preparation.invoke({"order_id": order_id})
        data = json.loads(result)
        self.assertEqual(data["status"], "failed")

        order = load_order(order_id)
        self.assertEqual(order.status, OrderStatus.PREPARATION_ERROR)


class TestStartPreparationRejectsWrongStatus(unittest.TestCase):
    """Test 22: Cannot prepare order not in INVENTORY_CONFIRMED state."""

    def setUp(self):
        init_db()
        reset_inventory()
        ORDER_STATUS_CACHE.clear()
        ORDER_JOB_MAP.clear()

    @patch("src.agents.barista_agent.is_machine_running", return_value=True)
    def test_pending_rejected(self, mock_running):
        order = Order(
            customer="Test",
            status=OrderStatus.PENDING,
            total=4.0,
            items=[OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[])],
        )
        save_order(order)
        result = start_preparation.invoke({"order_id": order.order_id_str})
        data = json.loads(result)
        self.assertEqual(data["status"], "error")
        self.assertIn("Cannot prepare", data["message"])

        loaded = load_order(order.order_id_str)
        self.assertEqual(loaded.status, OrderStatus.PENDING)


class TestStartPreparationRetryAfterFailure(unittest.TestCase):
    """Test 23: Retry succeeds after PREPARATION_ERROR with attempt_count > 0."""

    def setUp(self):
        init_db()
        reset_inventory()
        ORDER_STATUS_CACHE.clear()
        ORDER_JOB_MAP.clear()

    @patch("src.agents.barista_agent.safe_get")
    @patch("src.agents.barista_agent.safe_post")
    @patch("src.agents.barista_agent.is_machine_running", return_value=True)
    @patch("src.agents.barista_agent.time.sleep")
    def test_retry_allowed(self, mock_sleep, mock_running, mock_post, mock_get):
        order = Order(
            customer="Test",
            status=OrderStatus.PREPARATION_ERROR,
            total=4.0,
            items=[OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[])],
        )
        save_order(order)

        # Simulate a previous attempt
        ORDER_STATUS_CACHE[order.order_id_str] = {"attempt_count": 1}

        mock_post.return_value = _mock_brew_response()
        mock_get.return_value = _mock_status_response("ready")

        result = start_preparation.invoke({"order_id": order.order_id_str})
        data = json.loads(result)
        self.assertEqual(data["status"], "brewing")

        result = end_preparation.invoke({"order_id": order.order_id_str})
        data = json.loads(result)
        self.assertEqual(data["status"], "ready")

        loaded = load_order(order.order_id_str)
        self.assertEqual(loaded.status, OrderStatus.COMPLETED)


class TestEndPreparationWithoutStart(unittest.TestCase):
    """Test: end_preparation fails if no brewing is active."""

    def setUp(self):
        init_db()
        reset_inventory()
        ORDER_STATUS_CACHE.clear()
        ORDER_JOB_MAP.clear()

    def test_no_active_brewing(self):
        order_id = _create_confirmed_order()
        result = end_preparation.invoke({"order_id": order_id})
        data = json.loads(result)
        self.assertEqual(data["status"], "error")
        self.assertIn("No active brewing", data["message"])


class TestEstimatePrepTime(unittest.TestCase):
    """Test 24: estimate_prep_time returns correct estimate."""

    def setUp(self):
        init_db()
        reset_inventory()
        ORDER_STATUS_CACHE.clear()

    def test_estimate(self):
        items = [
            OrderItem(name="latte", quantity=2, price=4.0, size=None, extras=[]),
            OrderItem(name="croissant", quantity=1, price=2.75, size=None, extras=[]),
        ]
        order_id = _create_confirmed_order(items=items)

        result = estimate_prep_time.invoke({"order_id": order_id})
        data = json.loads(result)
        self.assertEqual(data["status"], "info")
        self.assertIn("estimated_minutes", data)
        # 3 items total: 2 + (3-1)*1.5 = 5.0 minutes
        self.assertAlmostEqual(data["estimated_minutes"], 5.0)


if __name__ == "__main__":
    unittest.main()
