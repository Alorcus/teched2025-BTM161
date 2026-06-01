"""Interaction Observatory page — live multi-agent dashboard.

Modules:
- ``observatory``         — top-level page composition (entry point)
- ``agent_panel``         — single-agent live trace pane
- ``conversation_runner`` — drives the swarm; emits dashboard events
- ``event_bus``           — pub/sub used by the runner and panels
- ``log_saver``           — captures events into the OCEL CSV format
- ``stock_panel``         — inventory snapshot
- ``coffee_machine_panel``— brew machine status + animation
- ``tray_panel``          — what's currently on the customer's tray
"""

from .observatory import create_observatory_dashboard

__all__ = ["create_observatory_dashboard"]
