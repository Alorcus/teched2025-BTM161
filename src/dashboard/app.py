import logging
import argparse
import sys

import panel as pn

from src.setups import list_setups, resolve_setup_name, setup_dir

from .interaction import create_observatory_dashboard
from .metrics import create_metrics_dashboard
from .trace_app import create_trace_dashboard


def main():
    """Start the multi-page Panel dashboard server."""
    parser = argparse.ArgumentParser(description="Coffee Shop Agent Observatory dashboard")
    parser.add_argument(
        "--setup", type=str, default=None,
        help="Name of the setup under config/setups/ to load. The COFFEE_SHOP_SETUP env var supersedes this flag.",
    )
    parser.add_argument(
        "--list-setups", action="store_true",
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

    # Multi-page routing
    routes = {
        '/': lambda: create_observatory_dashboard(setup_name),
        '/metrics': create_metrics_dashboard,
        '/trace': create_trace_dashboard,
    }

    pn.serve(
        routes,
        port=5006,
        show=False,
        title=f"Coffee Shop Agent Observatory — {setup_name}",
    )
    print(f"Dashboard running at http://localhost:5006 (setup: {setup_name})")
    print("  - Observatory: http://localhost:5006/")
    print("  - Metrics:     http://localhost:5006/metrics")
    print("  - Trace:       http://localhost:5006/trace")
    return 0


if __name__ == "__main__":
    sys.exit(main())
