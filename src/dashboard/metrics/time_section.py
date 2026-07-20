import panel as pn

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

from .eventlog_helpers import per_order_durations
from .styling_helpers import kpi_row, per_order_kpi_card, section_header


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
        "From order placement until the last activity in the order.",
        "pipeline_duration_s",
        "orders",
    ),
    (
        "Time to Tray",
        "From order placement until every item is on the customer's tray.",
        "confirm_to_tray_s",
        "orders",
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
        return column

    def _per_order_cards(self) -> pn.viewable.Viewable:
        order_durations = per_order_durations(self._ocel)
        if order_durations.is_empty():
            return pn.pane.Alert("No per-order data in this log.", alert_type="info")

        cards_html = "".join(
            per_order_kpi_card(title, subtitle, order_durations[col], unit=unit)
            for title, subtitle, col, unit in _PER_ORDER_CARDS
        )
        return kpi_row(cards_html, columns=3, top_padding=12)
