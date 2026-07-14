import panel as pn
import plotly.express as px

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

from .eventlog_helpers import agent_event_counts
from .styling_helpers import AGENT_COLORS, COLOR_SCHEME, section_header, subsection_header


class SystemMetricsSection:
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
            plot_bgcolor=COLOR_SCHEME["off-white"],
        )
        return pn.pane.Plotly(fig, height=180, sizing_mode="stretch_width")
