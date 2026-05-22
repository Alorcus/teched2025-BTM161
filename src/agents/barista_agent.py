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
COFFEE_MACHINE_PATH = Path(__file__).resolve().parents[2] / "services" / "coffee_machine"
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
    logger.debug("start_preparation called for %s", order_id)

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
        "eta_seconds": data.get("eta_seconds", 15),
    }

    eta_seconds = ORDER_STATUS_CACHE[order_id]["eta_seconds"]

    return tool_response(
        "brewing",
        f"☕ Brewing started! This will take about {eta_seconds:.0f} seconds. Call end_preparation when ready to check.",
        order_id,
        {"attempt": attempt_count, "eta_seconds": eta_seconds},
    )


@tool(args_schema=OrderIdSchema)
def end_preparation(order_id: str) -> str:
    """Wait for coffee preparation to complete and return the final result."""
    logger.debug("end_preparation called for %s", order_id)

    cache = ORDER_STATUS_CACHE.get(order_id)
    if not cache or cache.get("status") != "brewing":
        return tool_response(
            "error",
            f"No active brewing found for order {order_id}. Call start_preparation first.",
            order_id,
        )

    job_id = cache.get("job_id")
    if not job_id:
        return tool_response("error", "No job_id found for this order.", order_id)

    order = load_order(order_id)
    if not order:
        return tool_response("error", f"Order {order_id} not found", order_id)

    attempt_count = cache.get("attempt_count", 1)
    started_at = cache.get("started_at", time.time())
    elapsed = time.time() - started_at

    eta_seconds = cache.get("eta_seconds", 15)
    max_wait = max(eta_seconds + 5 - elapsed, 5)
    poll_interval = 2
    waited = 0

    while waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval

        status_response = safe_get(f"{COFFEE_MACHINE_URL}/jobs/{job_id}")
        if status_response and status_response.status_code == 200:
            try:
                job = status_response.json()
                status = job.get("status", "unknown")

                if status == "ready":
                    is_contaminated = job.get("contaminated", False)
                    order.status = OrderStatus.COMPLETED
                    save_order(order)
                    if order_id in ORDER_JOB_MAP:
                        del ORDER_JOB_MAP[order_id]
                    if order_id in ORDER_STATUS_CACHE:
                        del ORDER_STATUS_CACHE[order_id]

                    if is_contaminated:
                        return tool_response(
                            "contaminated",
                            f"⚠️ Coffee is ready but the machine was dirty — the drink may be contaminated.",
                            order_id,
                            {"attempt": attempt_count, "contaminated": True},
                        )

                    return tool_response(
                        "ready",
                        f"✅ Your coffee is ready! ☕",
                        order_id,
                        {"attempt": attempt_count},
                    )

                elif status == "failed":
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

    order.status = OrderStatus.PREPARATION_ERROR
    save_order(order)
    return tool_response("error", "Brewing timed out. Please try again.", order_id)


@tool(args_schema=OrderIdSchema)
def estimate_prep_time(order_id: str) -> str:
    """Estimate preparation time for an order."""
    logger.debug("estimate_prep_time called for %s", order_id)
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


@tool
def clean_machine() -> str:
    """Clean the coffee machine after a brew failure to prevent contamination."""
    logger.debug("clean_machine called")

    if not is_machine_running():
        return json.dumps({"status": "error", "message": "Coffee machine is not available."})

    response = safe_post(f"{COFFEE_MACHINE_URL}/clean", {})
    if response is None:
        return json.dumps({"status": "error", "message": "Coffee machine unreachable."})

    try:
        data = response.json()
        return json.dumps(data)
    except Exception:
        return json.dumps({"status": "error", "message": "Invalid response from machine."})


DEFAULT_PROMPT = """You are a barista agent responsible for coffee preparation.

WORKFLOW:
1. Call start_preparation(order_id) - This starts brewing and returns immediately with the ETA
2. Call end_preparation(order_id) - This waits for the brew to finish and returns the result
   - It returns either "ready", "contaminated", or "failed"

3. Based on the result:
   - If "ready" → Tell the customer: "✅ Your coffee is ready!"
   - If "contaminated" → The coffee was brewed on a dirty machine. Tell the customer: "⚠️ I need to remake your coffee — the machine wasn't clean." Then call clean_machine(), then retry.
   - If "failed" → The machine broke. Call clean_machine() IMMEDIATELY, then ask: "❌ Brewing failed on attempt #{attempt}. Would you like me to try again or transfer you to customer service?"

4. If customer wants to retry:
   - Call start_preparation(order_id) again (the attempt count will auto-increment)
   - Then call end_preparation(order_id) to wait for the result

5. If customer wants customer service:
   - Call transfer_to_agent(customer_service_agent,context_summary, expectation)
   - context_summary: summarize what happened (e.g. "Brewing failed twice for order X")
   - expectation: what should customer service do (e.g. "Help the customer with alternatives")

IMPORTANT NOTES:
- Always call end_preparation after start_preparation to get the final result
- After a brew failure, you MUST call clean_machine() before retrying. If you skip cleaning, the coffee will be contaminated.
- Be honest about failures and give customers clear choices
- Don't call start_preparation without asking the customer if he wants to try

Remember: Coffee takes time to brew. Be patient and keep the customer informed!
"""

DEFAULT_TOOLS = [start_preparation, end_preparation, estimate_prep_time, clean_machine, get_order, transfer_to_agent]
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
