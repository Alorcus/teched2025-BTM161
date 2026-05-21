import requests
import json
import subprocess
import socket
import time
import threading
from typing import Dict
import logging
from pathlib import Path

logger = logging.getLogger("coffee_shop.barista_agent")

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from src.llm import bind_tools_sequential

from .shared_components import (
    OrderIdSchema,
    OrderStatus,
    transfer_to_agent,
)
from .order_store import load_order, save_order, get_order
from .context_isolation import create_context_isolation_hook


COFFEE_MACHINE_URL = "http://127.0.0.1:8001"
REQUEST_TIMEOUT = 5
COFFEE_MACHINE_PATH = Path(__file__).parent.parent / "services" / "coffee-machine"
COFFEE_MACHINE_PORT = 8001
COFFEE_MACHINE_PROCESS = None
_MACHINE_LOCK = threading.Lock()

# Persistent state for machine jobs
ORDER_JOB_MAP: Dict[str, str] = {}
ORDER_STATUS_CACHE: Dict[str, dict] = {}


def is_machine_running() -> bool:
    """Check if coffee machine is responsive."""
    try:
        response = safe_get(f"{COFFEE_MACHINE_URL}/docs")  # FastAPI docs endpoint
        return response is not None and response.status_code < 500
    except:
        return False


def check_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False
        except socket.error:
            return True


def start_coffee_machine() -> bool:
    """Start the coffee machine uvicorn server as a subprocess."""
    global COFFEE_MACHINE_PROCESS

    with _MACHINE_LOCK:
        # Check if already running
        if is_machine_running():
            return True

        # Check if port is in use but machine not responding (stuck process)
        if check_port_in_use(COFFEE_MACHINE_PORT):
            logger.warning(
                f"Port {COFFEE_MACHINE_PORT} is in use but machine not responding"
            )
            return False

        try:
            COFFEE_MACHINE_PROCESS = subprocess.Popen(
                [
                    "poetry",
                    "run",
                    "uvicorn",
                    "main:app",
                    "--reload",
                    "--port",
                    str(COFFEE_MACHINE_PORT),
                    "--host",
                    "127.0.0.1",
                ],
                cwd=str(COFFEE_MACHINE_PATH),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP")
                else 0,
            )

            for _ in range(10):
                time.sleep(1)
                if is_machine_running():
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to start coffee machine: {e}")
            return False


def stop_coffee_machine():
    """Stop the coffee machine subprocess (optional, for cleanup)."""
    global COFFEE_MACHINE_PROCESS
    with _MACHINE_LOCK:
        if COFFEE_MACHINE_PROCESS:
            COFFEE_MACHINE_PROCESS.terminate()
            try:
                COFFEE_MACHINE_PROCESS.wait(timeout=5)
            except subprocess.TimeoutExpired:
                COFFEE_MACHINE_PROCESS.kill()
            COFFEE_MACHINE_PROCESS = None


# ----------------------------
# SAFE HTTP HELPERS
# ----------------------------
def safe_post(url, payload):
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"[CoffeeMachine] POST failed: {e}")
        return None


def safe_get(url):
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"[CoffeeMachine] GET failed: {e}")
        return None


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def tool_response(status, message, order_id: str, extra=None):
    payload = {
        "status": status,
        "message": message,
        "order_id": order_id,
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload)


# ----------------------------
# MACHINE TOOLS
# ----------------------------
@tool(args_schema=OrderIdSchema)
def start_preparation(order_id: str) -> str:
    """Start coffee preparation and automatically wait for completion."""

    if not is_machine_running():
        # Try to start the coffee machine
        if not start_coffee_machine():
            return tool_response(
                "error",
                "❌ Coffee machine is not available. Please try again in a moment or contact customer service.",
                order_id,
            )

        # Extra wait for the machine to fully initialize
        time.sleep(2)

    order = load_order(order_id)
    if not order:
        return tool_response("error", f"Order {order_id} not found", order_id)

    # Allow preparation if inventory is confirmed OR if we're retrying after a failure
    is_retry = ORDER_STATUS_CACHE.get(order_id, {}).get("attempt_count", 0) > 0
    is_inventory_confirmed = order.status == OrderStatus.INVENTORY_CONFIRMED
    is_retryable = order.status in (
        OrderStatus.IN_PREPARATION,
        OrderStatus.PREPARATION_ERROR,
    )

    if not is_inventory_confirmed and not (is_retryable and is_retry):
        return tool_response(
            "error",
            f"Cannot prepare order {order_id}. Current status: {order.status}",
            order_id,
        )

    # Start brewing — use the first item's name as the drink type
    drink_name = order.items[0].name if order.items else "coffee"
    response = safe_post(
        f"{COFFEE_MACHINE_URL}/brew", {"drink": drink_name, "correlation_id": order_id}
    )

    if response is None:
        return tool_response("error", "Coffee machine unreachable", order_id)

    if response.status_code != 200:
        return tool_response("error", f"Machine error", order_id)

    try:
        data = response.json()
    except Exception:
        return tool_response("error", "Invalid response", order_id)

    job_id = data.get("job_id")
    if not job_id:
        return tool_response("error", "No job_id returned", order_id)

    order.status = OrderStatus.IN_PREPARATION
    save_order(order)

    # Increment attempt count
    attempt_count = ORDER_STATUS_CACHE.get(order_id, {}).get("attempt_count", 0) + 1

    ORDER_JOB_MAP[order_id] = job_id
    ORDER_STATUS_CACHE[order_id] = {
        "job_id": job_id,
        "status": "brewing",
        "started_at": time.time(),
        "attempt_count": attempt_count,
    }

    # Get ETA
    eta_seconds = data.get("eta_seconds", 15)

    # Send initial response (this will be shown to customer)
    initial_message = f"☕ Brewing started! This will take about {eta_seconds:.0f} seconds. I'll let you know when it's ready."

    # POLLING LOOP - wait for completion
    max_wait = eta_seconds + 5  # Wait a bit longer than ETA
    poll_interval = 2  # Check every 2 seconds
    waited = 0

    while waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval

        # Check status
        status_response = safe_get(f"{COFFEE_MACHINE_URL}/jobs/{job_id}")
        if status_response and status_response.status_code == 200:
            try:
                job = status_response.json()
                status = job.get("status", "unknown")

                if status == "ready":
                    # Success!
                    order.status = OrderStatus.COMPLETED
                    save_order(order)
                    if order_id in ORDER_JOB_MAP:
                        del ORDER_JOB_MAP[order_id]
                    if order_id in ORDER_STATUS_CACHE:
                        del ORDER_STATUS_CACHE[order_id]

                    return tool_response(
                        "ready",
                        f"✅ Your coffee is ready! ☕",
                        order_id,
                        {"attempt": attempt_count},
                    )

                elif status == "failed":
                    # Failed
                    order.status = OrderStatus.PREPARATION_ERROR
                    save_order(order)
                    if order_id in ORDER_JOB_MAP:
                        del ORDER_JOB_MAP[order_id]

                    return tool_response(
                        "failed",
                        f"❌ Brewing failed on attempt #{attempt_count}.",
                        order_id,
                        {"attempt": attempt_count},
                    )

            except Exception as e:
                logger.error(f"Status check error: {e}")

    # Timeout — mark as failed so retries are possible
    order.status = OrderStatus.PREPARATION_ERROR
    save_order(order)
    return tool_response("error", "Brewing timed out. Please try again.", order_id)


@tool(args_schema=OrderIdSchema)
def estimate_prep_time(order_id: str) -> str:
    """Estimate preparation time for an order."""
    order = load_order(order_id)
    if not order:
        return tool_response("error", f"Order not found", order_id)

    total_items = sum(item.quantity for item in order.items)
    base_time = 2
    time_per_item = 1.5
    estimated_time = base_time + max(0, total_items - 1) * time_per_item

    if order_id in ORDER_STATUS_CACHE:
        started_at = ORDER_STATUS_CACHE[order_id].get("started_at")
        if started_at:
            elapsed = time.time() - started_at
            if elapsed < estimated_time * 60:
                remaining = max(0, (estimated_time * 60) - elapsed)
                return tool_response(
                    "info",
                    f"⏱️ About {remaining:.0f} seconds remaining.",
                    order_id,
                    {"remaining_seconds": remaining},
                )

    return tool_response(
        "info",
        f"⏱️ Estimated time: {estimated_time:.1f} minutes",
        order_id,
        {"estimated_minutes": estimated_time},
    )


DEFAULT_PROMPT = """You are a barista agent responsible for coffee preparation.

WORKFLOW:
1. Call start_preparation(order_id) - This starts brewing AND automatically waits for completion
   - It will take a few seconds (the coffee needs time to brew)
   - You will see "Brewing started..." then the tool will wait
   - It returns either "ready" or "failed"

2. Based on the result:
   - If "ready" → Tell the customer: "✅ Your coffee is ready!"
   - If "failed" → Ask the customer: "❌ Brewing failed on attempt #{attempt}. Would you like me to try again or transfer you to customer service?"

3. If customer wants to retry:
   - Call start_preparation(order_id) again (the attempt count will auto-increment)

4. If customer wants customer service:
   - Call transfer_to_agent(customer_service_agent,context_summary, expectation)
   - context_summary: summarize what happened (e.g. "Brewing failed twice for order X")
   - expectation: what should customer service do (e.g. "Help the customer with alternatives")

IMPORTANT NOTES:
- The start_preparation tool handles all the waiting and checking automatically
- You don't need to call any other status checking tools
- Be honest about failures and give customers clear choices
- Don't call start_preparation without asking the customer if he wants to try

Remember: Coffee takes time to brew. Be patient and keep the customer informed!
"""

DEFAULT_TOOLS = [start_preparation, estimate_prep_time, get_order, transfer_to_agent]
DEFAULT_TOOL_NAMES = [t.name for t in DEFAULT_TOOLS]


def create_barista_agent(chat_llm, prompt=None):
    """Create and return the barista agent."""

    if not prompt:
        prompt = DEFAULT_PROMPT

    tools = list(DEFAULT_TOOLS)

    llm_with_tools = bind_tools_sequential(chat_llm, tools)

    return create_react_agent(
        model=llm_with_tools,
        name="barista_agent",
        tools=tools,
        prompt=prompt,
        pre_model_hook=create_context_isolation_hook("barista_agent"),
    )
