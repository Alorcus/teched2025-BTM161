"""Metrics Observatory page — split into one module per section.

Each section class takes a loaded ``ObjectCentricEventlog`` and exposes a
``panel() -> pn.viewable.Viewable`` method (mirroring the panel-class
pattern used in the Interaction Observatory: ``StockPanel``, ``TrayPanel``,
etc.). The page itself is composed in ``page.create_metrics_dashboard``.
"""

from .overview_section import OverviewSection
from .system_metrics import SystemMetricsSection
from .time_metrics import TimeMetricsSection
from .page import create_metrics_dashboard

__all__ = [
    "OverviewSection",
    "SystemMetricsSection",
    "TimeMetricsSection",
    "create_metrics_dashboard",
]
