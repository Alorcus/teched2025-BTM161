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

_TOP_N_HANDOVER_CASES = 10


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
        column = section_header("Object Cardinality & Divergence")
        column.append(self._summary_cards())
        column.append(subsection_header("Agents per Case", top_margin=10))
        column.append(self._agents_per_case_chart())
        column.append(subsection_header("Most-Repeated Activities", top_margin=10))
        column.append(self._activity_divergence_chart())
        column.append(subsection_header("Highest-Handover Cases", top_margin=10))
        column.append(self._top_handover_cases_chart())
        return column

    # ---- KPI row ------------------------------------------------------

    def _summary_cards(self) -> pn.viewable.Viewable:
        agents = agents_per_case(self._ocel)
        handovers = handover_counts_per_case(self._ocel)
        tool_calls = tool_call_fanout_per_case(self._ocel)

        agents_avg = f"{agents['agent_count'].mean():.1f}" if agents.height else "—"
        handovers_avg = f"{handovers['handover_count'].mean():.1f}" if handovers.height else "0.0"
        tool_calls_avg = f"{tool_calls['tool_call_count'].mean():.1f}" if tool_calls.height else "—"
        if tool_calls.height:
            total_calls = tool_calls["tool_call_count"].sum()
            total_friction = tool_calls["flagged_count"].sum() + tool_calls["denied_count"].sum()
            friction_pct = f"{(100 * total_friction / total_calls):.0f}%" if total_calls else "—"
        else:
            friction_pct = "—"

        cards_html = "".join([
            subtitled_kpi_card(
                "Agents per Case", "Distinct agent objects touching a single case.", agents_avg,
            ),
            subtitled_kpi_card(
                "Handovers per Case", "Agent-to-agent transitions within a case.", handovers_avg,
            ),
            subtitled_kpi_card(
                "Tool Calls per Case", "Distinct tool_call objects executed per case.", tool_calls_avg,
            ),
            subtitled_kpi_card(
                "Guardrail Friction", "Share of tool calls flagged or denied.", friction_pct,
            ),
        ])
        return kpi_row(cards_html, columns=4, top_padding=12)

    # ---- Agents-per-case distribution ---------------------------------

    def _agents_per_case_chart(self) -> pn.viewable.Viewable:
        agents = agents_per_case(self._ocel)
        if not agents.height:
            return pn.pane.Alert("No agent–case relationships found in this log.", alert_type="info")

        dist = (
            agents.group_by("agent_count")
            .agg(pl.len().alias("case_count"))
            .sort("agent_count")
            .with_columns(pl.col("agent_count").cast(pl.Utf8))
        )
        fig = px.bar(
            dist.to_pandas(),
            x="agent_count", y="case_count",
            labels={"agent_count": "Distinct Agents in Case", "case_count": "Cases"},
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
                "No activity repeats more than once per case on average.", alert_type="info",
            )

        fig = px.bar(
            divergence.to_pandas().head(15),
            x="avg_per_case", y="ocel_type", orientation="h",
            color="avg_per_case", color_continuous_scale="Reds",
            labels={"ocel_type": "Activity", "avg_per_case": "Avg Occurrences per Case"},
            hover_data={"max_per_case": True},
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            margin=dict(l=150, r=10, t=5, b=25),
            height=210,
            font=dict(size=10),
            plot_bgcolor=COLOR_SCHEME["off-white"],
        )
        return pn.pane.Plotly(fig, height=210, sizing_mode="stretch_width")

    # ---- Top handover cases --------------------------------------------

    def _top_handover_cases_chart(self) -> pn.viewable.Viewable:
        handovers = handover_counts_per_case(self._ocel)
        if not handovers.height:
            return pn.pane.Alert("No agent handovers found in this log.", alert_type="info")

        top = handovers.head(_TOP_N_HANDOVER_CASES).with_columns(
            pl.col("case_id").str.slice(-12).alias("case_label")
        )
        fig = px.bar(
            top.to_pandas(),
            x="handover_count", y="case_label", orientation="h",
            color_discrete_sequence=[COLOR_SCHEME["dark_red"]],
            labels={"case_label": "Case", "handover_count": "Handovers"},
            hover_data={"case_id": True},
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=110, r=10, t=5, b=25),
            height=210,
            font=dict(size=10),
            plot_bgcolor=COLOR_SCHEME["off-white"],
        )
        return pn.pane.Plotly(fig, height=210, sizing_mode="stretch_width")