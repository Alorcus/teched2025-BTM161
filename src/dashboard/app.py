import logging

import panel as pn

from .observatory import create_observatory_dashboard
from .metrics_panel import create_metrics_dashboard


def main():
    """Start the multi-page Panel dashboard server."""
    logging.getLogger("bokeh.server.views.static_handler").setLevel(logging.WARNING)
    logging.getLogger("tornado.access").setLevel(logging.WARNING)

    # Multi-page routing
    routes = {
        '/': create_observatory_dashboard,
        '/metrics': create_metrics_dashboard,
    }

    pn.serve(
        routes,
        port=5006,
        show=False,
        title="Coffee Shop Agent Observatory",
    )
    print("Dashboard running at http://localhost:5006")
    print("  - Observatory: http://localhost:5006/")
    print("  - Metrics:     http://localhost:5006/metrics")


if __name__ == "__main__":
    main()
