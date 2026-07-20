import panel as pn
import plotly.express as px
import polars as pl

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

from .eventlog_helpers import (
    activity_divergence,
    agents_per_case,
    handover_counts_per_case,
    tool_call_fanout_per_case,
)
from .styling_helpers import COLOR_SCHEME, kpi_row, section_header, subsection_header, subtitled_kpi_card


class ObjectCardinalitySection:
    """Object-centric complexity metrics: how many objects, of which types,
    a case pulls in, and how much repetition/rework that produces.

    Complements the OC-DFG / Petri net / event-object graphs in
    VisualizationSection — those show the *shape* of object interactions,
    this section turns that shape into concrete numbers and rankings.
    """

    def __init__(self, ocel: ObjectCentricEventlog):
        self._ocel = ocel
        self._pane = self._build()

    def panel(self) -> pn.viewable.Viewable:
        return self._pane

    def _build(self) -> pn.Column:
        column = section_header("Conversation Composition")
        column.append(self._summary_cards())
        column.append(subsection_header("Agents per Conversation", top_margin=10))
        column.append(self._agents_per_case_chart())
        column.append(subsection_header("Most-Repeated Activities", top_margin=10))
        column.append(self._activity_divergence_chart())
        return column

    # ---- KPI row ------------------------------------------------------

    def _summary_cards(self) -> pn.viewable.Viewable:
        agents = agents_per_case(self._ocel)
        handovers = handover_counts_per_case(self._ocel)
        tool_calls = tool_call_fanout_per_case(self._ocel)

        agents_avg = f"{agents['agent_count'].mean():.1f}" if agents.height else "—"
        handovers_avg = f"{handovers['handover_count'].mean():.1f}" if handovers.height else "0.0"
        tool_calls_avg = f"{tool_calls['tool_call_count'].mean():.1f}" if tool_calls.height else "—"

        cards_html = "".join([
            subtitled_kpi_card(
                "Agents per Conversation", "Distinct agent objects touching a single conversation.", agents_avg,
            ),
            subtitled_kpi_card(
                "Handovers per Conversation", "Agent-to-agent transitions within a conversation.", handovers_avg,
            ),
            subtitled_kpi_card(
                "Tool Calls per Conversation", "Distinct tool_call objects executed per conversation.", tool_calls_avg,
            ),
        ])
        return kpi_row(cards_html, columns=4, top_padding=12)

    # ---- Agents-per-case distribution ---------------------------------

    def _agents_per_case_chart(self) -> pn.viewable.Viewable:
        agents = agents_per_case(self._ocel)
        if not agents.height:
            return pn.pane.Alert("No agent–conversation relationships found in this log.", alert_type="info")

        dist = (
            agents.group_by("agent_count")
            .agg(pl.len().alias("case_count"))
            .sort("agent_count")
            .with_columns(pl.col("agent_count").cast(pl.Utf8))
        )
        fig = px.bar(
            dist.to_pandas(),
            x="agent_count", y="case_count",
            labels={"agent_count": "Distinct Agents in Conversation", "case_count": "Conversations"},
            color_discrete_sequence=[COLOR_SCHEME["orange"]],
        )
        fig.update_layout(
            margin=dict(l=30, r=10, t=5, b=25),
            height=180,
            font=dict(size=10),
            plot_bgcolor=COLOR_SCHEME["off-white"],
        )
        return pn.pane.Plotly(fig, height=180, sizing_mode="stretch_width")

    # ---- Activity divergence -------------------------------------------

    def _activity_divergence_chart(self) -> pn.viewable.Viewable:
        divergence = activity_divergence(self._ocel)
        if not divergence.height:
            return pn.pane.Alert(
                "No activity repeats more than once per conversation on average.", alert_type="info",
            )

        top = divergence.head(15)
        n = top.height
        # Anchor the axis at 1 — activity_divergence filters to avg_per_case > 1,
        # so bar length reads as "excess repeats beyond the first occurrence".
        max_val = float(top["avg_per_case"].max())
        height = max(140, 22 * n + 40)

        fig = px.bar(
            top.to_pandas(),
            x="avg_per_case", y="ocel_type", orientation="h",
            labels={"ocel_type": "Activity", "avg_per_case": "Avg Occurrences per Conversation"},
            hover_data={"max_per_case": True},
            color_discrete_sequence=[COLOR_SCHEME["red"]],
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis=dict(range=[1, max_val * 1.05]),
            margin=dict(l=150, r=10, t=5, b=25),
            height=height,
            font=dict(size=10),
            plot_bgcolor=COLOR_SCHEME["off-white"],
        )
        return pn.pane.Plotly(fig, height=height, sizing_mode="stretch_width")