"""System Metrics section — system-level views of the agent ecosystem.

Currently: an Agent Workload bar chart. Future additions (handover matrix,
tool usage, etc.) belong here.
"""

from __future__ import annotations

import panel as pn
import plotly.express as px

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

from .eventlog_helpers import agent_event_counts
from .ui import AGENT_COLORS, section_header, subsection_header


class SystemMetricsSection:
    """Charts that describe the overall agent system."""

    def __init__(self, ocel: ObjectCentricEventlog):
        self._ocel = ocel
        self._pane = self._build()

    def panel(self) -> pn.viewable.Viewable:
        return self._pane

    def _build(self) -> pn.Column:
        column = section_header("System Metrics")
        column.append(subsection_header("Agent Workload"))
        column.append(self._agent_workload_chart())
        return column

    def _agent_workload_chart(self) -> pn.viewable.Viewable:
        agent_counts = agent_event_counts(self._ocel)
        if not agent_counts.height:
            return pn.pane.Alert(
                "No agent–event relationships found in this log.",
                alert_type="info",
            )
        fig = px.bar(
            agent_counts.to_pandas(),
            x="agent", y="event_count",
            color="agent", color_discrete_map=AGENT_COLORS,
            labels={"agent": "Agent", "event_count": "Events Handled"},
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(l=30, r=10, t=5, b=25),
            height=180,
            font=dict(size=10),
        )
        return pn.pane.Plotly(fig, height=180, sizing_mode="stretch_width")
