import panel as pn
import plotly.express as px
import polars as pl

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

from .eventlog_helpers import flat_event_table, per_order_durations
from .styling_helpers import COLOR_SCHEME, kpi_row, per_order_kpi_card, section_header, subsection_header


# Per-order KPI configuration. (label, subtitle, column-in-per_order_durations, unit)
_PER_ORDER_CARDS: list[tuple[str, str, str, str]] = [
    (
        "Total Order Time",
        "From the customer's first message until the conversation ends.",
        "full_duration_s",
        "orders",
    ),
    (
        "Fulfillment Time",
        "From order placement until the last activity in the case.",
        "pipeline_duration_s",
        "orders",
    ),
    (
        "Time to Tray",
        "From order placement until every item is on the customer's tray.",
        "confirm_to_tray_s",
        "orders",
    ),
    (
        "Customer Service Resolution",
        "From handover to customer service until control returns to another agent.",
        "cs_resolution_s",
        "incidents",
    ),
]


class TimeMetricsSection:
    def __init__(self, ocel: ObjectCentricEventlog):
        self._ocel = ocel
        self._pane = self._build()

    def panel(self) -> pn.viewable.Viewable:
        return self._pane

    def _build(self) -> pn.Column:
        column = section_header("Time Metrics")
        column.append(self._per_order_cards())
        column.append(subsection_header("Average Activity Duration", top_margin=10))
        column.append(self._avg_activity_chart())
        return column

    def _per_order_cards(self) -> pn.viewable.Viewable:
        order_durations = per_order_durations(self._ocel)
        if order_durations.is_empty():
            return pn.pane.Alert("No per-order data in this log.", alert_type="info")

        cards_html = "".join(
            per_order_kpi_card(title, subtitle, order_durations[col], unit=unit)
            for title, subtitle, col, unit in _PER_ORDER_CARDS
        )
        return kpi_row(cards_html, columns=4, top_padding=12)

    def _avg_activity_chart(self) -> pn.viewable.Viewable:
        events_flat = flat_event_table(self._ocel)
        non_handover = events_flat.filter(~pl.col("ocel_type").str.contains("_handover_"))
        duration_stats = (
            non_handover.drop_nulls("duration")
            .group_by("ocel_type")
            .agg(
                pl.mean("duration").alias("avg_duration"),
                pl.median("duration").alias("median_duration"),
                pl.max("duration").alias("max_duration"),
            )
            .sort("avg_duration", descending=True)
        )
        if not duration_stats.height:
            return pn.pane.Alert("No duration data in this log.", alert_type="info")

        fig = px.bar(
            duration_stats.to_pandas().head(15),
            x="avg_duration", y="ocel_type", orientation="h",
            color="avg_duration", color_continuous_scale="Oranges",
            labels={"ocel_type": "Activity", "avg_duration": "Avg Duration (s)"},
            hover_data={"median_duration": True, "max_duration": True},
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
