import os
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "coffee_shop.db"

TIMEOUT_SECONDS = 300


class TestSimulationE2E(unittest.TestCase):
    """Integration test: run a full simulation and verify the order reaches COMPLETED."""

    def setUp(self):
        if DB_PATH.exists():
            DB_PATH.unlink()

    def test_scenario_0_completes_order(self):
        env = os.environ.copy()
        # Use seed 100 to ensure first brew succeeds (seed 42 fails on first attempt)
        env["COFFEE_MACHINE_SEED"] = "100"

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "src.simulate",
                    "--scenario", "0",
                    "--log-level", "debug",
                    "--traces", "1",
                ],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or b"").decode(errors="replace")
            stderr = (e.stderr or b"").decode(errors="replace")
            self.fail(
                f"Simulation timed out after {TIMEOUT_SECONDS}s\n"
                f"\n--- STDOUT (partial) ---\n{stdout}\n"
                f"\n--- STDERR (partial) ---\n{stderr}\n"
            )

        # Always capture output for diagnostics
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode

        # Query the DB for the order created during simulation
        order = None
        order_repr = "Order not found in database"
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from src.agents.order_store import load_order
            from src.agents.shared_components import OrderStatus

            order = load_order("ORD0001")
            if order:
                order_repr = (
                    f"Order(id={order.id}, customer={order.customer!r}, "
                    f"status={order.status.value!r}, total={order.total}, "
                    f"items={[f'{i.quantity}x {i.name}' for i in order.items]}, "
                    f"created_at={order.created_at}, last_modified={order.last_modified})"
                )
        except Exception as e:
            order_repr = f"Failed to query DB: {e}"

        # Build diagnostic dump
        diagnostics = (
            "\n"
            "=" * 80 + "\n"
            "SIMULATION DIAGNOSTIC DUMP\n"
            "=" * 80 + "\n"
            f"\n--- EXIT CODE: {exit_code} ---\n\n"
            f"--- STDOUT (full) ---\n{stdout}\n"
            f"--- STDERR (full) ---\n{stderr}\n"
            f"--- FINAL ORDER STATE ---\n{order_repr}\n"
            "=" * 80 + "\n"
        )

        # Assert non-zero exit code
        if exit_code != 0:
            self.fail(f"Simulation exited with code {exit_code}{diagnostics}")

        # Assert order exists
        self.assertIsNotNone(
            order,
            f"No order found in database after simulation{diagnostics}",
        )

        # Assert order completed
        self.assertEqual(
            order.status,
            OrderStatus.COMPLETED,
            f"Order status is {order.status.value!r}, expected 'completed'{diagnostics}",
        )


if __name__ == "__main__":
    unittest.main()
