import requests
import json
import os
import signal
import subprocess
import socket
import sys
import time
import threading
from typing import Dict
import logging
from pathlib import Path

logger = logging.getLogger("coffee_shop.barista_agent")

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from .shared_components import (
    OrderIdSchema,
    OrderStatus,
)
from .order_store import load_order
from .order_state_machine import state_machine, InvalidTransitionError


def _thread_id(config: RunnableConfig | None) -> str | None:
    """Pull the LangGraph thread_id (== MLflow case_id) out of an injected config.

    Production tools are invoked through `tool_node.invoke(state, config=config)`
    so the config is always present. Tests that call `.invoke({...})` directly
    skip the config; callers handle the None case explicitly.
    """
    if config is None:
        return None
    return (config.get("configurable") or {}).get("thread_id")


COFFEE_MACHINE_URL = "http://127.0.0.1:8001"
REQUEST_TIMEOUT = 5
COFFEE_MACHINE_PATH = Path(__file__).resolve().parents[2]
COFFEE_MACHINE_PORT = 8001
COFFEE_MACHINE_PROCESS = None
_MACHINE_LOCK = threading.Lock()

ORDER_JOB_MAP: Dict[str, str] = {}
ORDER_STATUS_CACHE: Dict[str, dict] = {}


def is_machine_running() -> bool:
    """Check if coffee machine is responsive."""
    try:
        response = safe_get(f"{COFFEE_MACHINE_URL}/docs")
        return response is not None and response.status_code < 500
    except:
        return False


def check_port_in_use(port: int) -> bool:
    """Check if a port has an active listener."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except (socket.error, OSError):
            return False


def start_coffee_machine() -> bool:
    """Start the coffee machine uvicorn server as a subprocess."""
    global COFFEE_MACHINE_PROCESS

    with _MACHINE_LOCK:
        if is_machine_running():
            return True

        if check_port_in_use(COFFEE_MACHINE_PORT):
            logger.warning(
                f"Port {COFFEE_MACHINE_PORT} is in use but machine not responding"
            )
            return False

        try:
            # Run the server in its own process group / session so we can
            # signal the whole group on shutdown. We launch via `poetry run
            # uvicorn`, so a plain terminate() would only signal the poetry
            # wrapper and leave the uvicorn grandchild listening on the port.
            popen_kwargs: dict = {
                "cwd": str(COFFEE_MACHINE_PATH),
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            }
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True

            COFFEE_MACHINE_PROCESS = subprocess.Popen(
                [
                    "poetry",
                    "run",
                    "uvicorn",
                    "services.coffee_machine.main:app",
                    "--port",
                    str(COFFEE_MACHINE_PORT),
                    "--host",
                    "127.0.0.1",
                ],
                **popen_kwargs,
            )

            for _ in range(10):
                time.sleep(1)
                if is_machine_running():
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to start coffee machine: {e}")
            return False


def _kill_process_on_port(port: int) -> None:
    """Kill any process listening on `port`. Best-effort; silent on failure.

    Catches the cross-run leak where COFFEE_MACHINE_PROCESS is None (a previous
    Python process started the server) but a uvicorn is still bound to the port.
    """
    if sys.platform == "win32":
        return
    try:
        out = subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return


def stop_coffee_machine():
    """Stop the coffee machine subprocess (optional, for cleanup)."""
    global COFFEE_MACHINE_PROCESS
    with _MACHINE_LOCK:
        if COFFEE_MACHINE_PROCESS:
            proc = COFFEE_MACHINE_PROCESS
            # Signal the whole process group so the uvicorn grandchild dies
            # together with the `poetry run` wrapper.
            try:
                if sys.platform == "win32":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                # Group already gone; fall through to wait().
                pass

            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    if sys.platform == "win32":
                        proc.kill()
                    else:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                proc.wait(timeout=2)
            COFFEE_MACHINE_PROCESS = None

        # Cross-run safety: if the port is still bound (e.g. a leak from a
        # previous Python process), nuke whoever's holding it.
        if check_port_in_use(COFFEE_MACHINE_PORT):
            _kill_process_on_port(COFFEE_MACHINE_PORT)


def safe_post(url, payload):
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        return response
    except requests.exceptions.ConnectionError as e:
        logger.debug(f"[CoffeeMachine] POST failed (connection): {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"[CoffeeMachine] POST failed: {e}")
        return None


def safe_get(url):
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        return response
    except requests.exceptions.ConnectionError as e:
        logger.debug(f"[CoffeeMachine] GET failed (connection): {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"[CoffeeMachine] GET failed: {e}")
        return None


def tool_response(status, message, order_id: str, extra=None):
    payload = {
        "status": status,
        "message": message,
        "order_id": order_id,
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload)


@tool(args_schema=OrderIdSchema)
def start_preparation(order_id: str, config: RunnableConfig = None) -> str:
    """Start coffee preparation and automatically wait for completion."""
    logger.debug("start_preparation called for %s", order_id)

    if not is_machine_running():
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

    drink_name = order.items[0].name if order.items else "coffee"
    # The coffee machine writes events to its CSV keyed by `correlation_id` —
    # which the trace processor merges into the export only if it matches a
    # LangGraph thread_id (the MLflow case_id). Pass thread_id, not order_id.
    thread_id = _thread_id(config)
    if thread_id is None:
        logger.warning(
            "start_preparation invoked without thread_id in config; "
            "coffee-machine rows for order %s will not merge into the export.",
            order_id,
        )
        correlation_id = order_id
    else:
        correlation_id = thread_id
    response = safe_post(
        f"{COFFEE_MACHINE_URL}/brew", {"drink": drink_name, "correlation_id": correlation_id}
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

    try:
        order = state_machine.transition(order, OrderStatus.IN_PREPARATION, context="prepare_order: starting")
    except InvalidTransitionError as e:
        return json.dumps({"order_id": order_id, "error": f"Cannot start preparation: {e}"})

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
                    if order_id in ORDER_JOB_MAP:
                        del ORDER_JOB_MAP[order_id]
                    ORDER_STATUS_CACHE[order_id] = {
                        **ORDER_STATUS_CACHE.get(order_id, {}),
                        "status": "ready",
                        "last_brew_contaminated": is_contaminated,
                    }

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
                    try:
                        order = state_machine.transition(order, OrderStatus.PREPARATION_ERROR, context=f"brewing failed on attempt #{attempt_count}")
                    except InvalidTransitionError as e:
                        logger.warning(f"Cannot transition to PREPARATION_ERROR: {e}")
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

    try:
        state_machine.transition(order, OrderStatus.PREPARATION_ERROR, context="brewing timed out")
    except InvalidTransitionError as e:
        logger.warning(f"Cannot transition to PREPARATION_ERROR: {e}")
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
def clean_machine(config: RunnableConfig = None) -> str:
    """Clean the coffee machine after a brew failure to prevent contamination."""
    logger.debug("clean_machine called")

    if not is_machine_running():
        return json.dumps({"status": "error", "message": "Coffee machine is not available."})

    thread_id = _thread_id(config)
    if thread_id is None:
        logger.warning(
            "clean_machine invoked without thread_id in config; "
            "clean event will not merge into the export."
        )
        correlation_id = "unknown"
    else:
        correlation_id = thread_id
    response = safe_post(f"{COFFEE_MACHINE_URL}/clean", {"correlation_id": correlation_id})
    if response is None:
        return json.dumps({"status": "error", "message": "Coffee machine unreachable."})

    try:
        data = response.json()
        return json.dumps(data)
    except Exception:
        return json.dumps({"status": "error", "message": "Invalid response from machine."})


