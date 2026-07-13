import logging
import argparse
import os
import signal
import sys
import time

import panel as pn
import psutil

from src.setups import list_setups, resolve_setup_name, setup_dir

from .interaction import create_observatory_dashboard
from .metrics import create_metrics_dashboard
from .trace_app import create_trace_dashboard


DASHBOARD_PORT = 5006

logger = logging.getLogger("coffee_shop.dashboard.app")


def _reclaim_port_if_orphaned(port: int) -> None:
    """If `port` is held by a leaked previous dashboard owned by the current
    user, kill it. If held by an unrelated process, raise SystemExit with a
    clear message instead of letting bind() fail with a cryptic OSError 98.
    """
    holders: list[psutil.Process] = []
    try:
        tcp_connections = psutil.net_connections(kind="tcp")
    except psutil.AccessDenied:
        return
    for conn in tcp_connections:
        if (
            conn.status == psutil.CONN_LISTEN
            and conn.laddr
            and conn.laddr.port == port
            and conn.pid is not None
        ):
            try:
                holders.append(psutil.Process(conn.pid))
            except psutil.NoSuchProcess:
                pass

    if not holders:
        return

    current_uid = os.getuid() if hasattr(os, "getuid") else None
    for proc in holders:
        try:
            cmdline = " ".join(proc.cmdline())
            owner_uid = proc.uids().real if hasattr(proc, "uids") else None
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

        is_ours = current_uid is None or owner_uid == current_uid
        looks_like_dashboard = "dashboard" in cmdline and "python" in cmdline.lower()

        if is_ours and looks_like_dashboard:
            logger.warning(
                "Found orphaned dashboard pid=%s on port %s; killing it.",
                proc.pid,
                port,
                proc.pid,
                port,
            )
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
            except psutil.NoSuchProcess:
                pass
        else:
            raise SystemExit(
                f"Port {port} is held by an unrelated process "
                f"(pid={proc.pid}, cmd={cmdline!r}). "
                f"Free the port (e.g. `kill {proc.pid}`) before starting the dashboard."
            )

    # Give the kernel a moment to release the socket.
    for _ in range(20):
        if not any(
            c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port
            for c in psutil.net_connections(kind="tcp")
        ):
            return
        time.sleep(0.1)


def main():
    """Start the multi-page Panel dashboard server."""
    parser = argparse.ArgumentParser(
        description="Coffee Shop Agent Observatory dashboard"
    )
    parser = argparse.ArgumentParser(
        description="Coffee Shop Agent Observatory dashboard"
    )
    parser.add_argument(
        "--setup",
        type=str,
        default=None,
        help="Name of the setup under config/setups/ to load.",
    )
    parser.add_argument(
        "--list-setups",
        action="store_true",
        help="List available setups under config/setups/ and exit.",
    )
    args = parser.parse_args()

    if args.list_setups:
        names = list_setups()
        if not names:
            print("(no setups found in config/setups/)")
        else:
            for name in names:
                print(name)
        return 0

    setup_name = resolve_setup_name(args.setup)
    setup_dir(setup_name)  # fail fast if the setup is missing or malformed

    logging.getLogger("bokeh.server.views.static_handler").setLevel(logging.WARNING)
    logging.getLogger("tornado.access").setLevel(logging.WARNING)

    _reclaim_port_if_orphaned(DASHBOARD_PORT)

    # Multi-page routing
    routes = {
        "/": lambda: create_observatory_dashboard(setup_name),
        "/metrics": create_metrics_dashboard,
        "/trace": create_trace_dashboard,
    }

    pn.serve(
        routes,
        port=DASHBOARD_PORT,
        show=False,
        title=f"Coffee Shop Agent Observatory — {setup_name}",
    )
    print(
        f"Dashboard running at http://localhost:{DASHBOARD_PORT} (setup: {setup_name})"
    )
    print(f"  - Interaction: http://localhost:{DASHBOARD_PORT}/")
    print(f"  - Metrics:     http://localhost:{DASHBOARD_PORT}/metrics")
    print(f"  - Trace:       http://localhost:{DASHBOARD_PORT}/trace")
    return 0


if __name__ == "__main__":
    sys.exit(main())
