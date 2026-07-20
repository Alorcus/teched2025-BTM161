from pathlib import Path

import panel as pn
import plotly.express as px
import polars as pl

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

from .eventlog_helpers import agent_event_counts, flat_event_table, per_order_durations
from .styling_helpers import (
    AGENT_COLORS,
    COLOR_SCHEME,
    kpi_card,
    kpi_row,
    per_order_kpi_card,
    subsection_header,
)


# Per-order KPI configuration (moved from the retired TimeMetricsSection).
# (label, subtitle, column-in-per_order_durations, unit)
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


class OverviewSection:
    """Unheaded intro strip: log-level KPIs, per-order times, tokens, and
    the agent workload chart. Reads as the 'at a glance' band above the
    analytical sections."""

    def __init__(self, ocel: ObjectCentricEventlog, log_path: Path):
        self._ocel = ocel
        self._log_path = log_path
        self._events_flat = flat_event_table(self._ocel)
        self._pane = pn.Column(
            self._build_header(),
            self._build_overview_kpi_row(),
            self._build_time_kpi_row(),
            self._build_tokens_kpi_row(),
            subsection_header("Agent Workload", top_margin=10),
            self._build_agent_workload_chart(),
            sizing_mode="stretch_width",
        )

    def panel(self) -> pn.viewable.Viewable:
        return self._pane

    # ---- Log meta line ------------------------------------------------

    def _build_header(self) -> pn.pane.HTML:
        return pn.pane.HTML(
            f'<div style="font-size:11px;color:#666;margin-bottom:4px;padding:2px 0;">'
            f'<b>Log:</b> {self._log_path.name}  ·  '
            f'{self._ocel.events.height:,} events  ·  '
            f'{self._ocel.objects.height:,} objects</div>',
            sizing_mode="stretch_width",
        )

    # ---- Overview KPI row (6 cards) -----------------------------------

    def _build_overview_kpi_row(self) -> pn.pane.HTML:
        is_handover = self._events_flat["ocel_type"].str.contains("_handover_")
        non_handover = self._events_flat.filter(~is_handover)
        handover = self._events_flat.filter(is_handover)

        conversation_count = self._ocel.objects.filter(
            pl.col("ocel_type") == "user"
        ).height
        agent_count = self._ocel.objects.filter(
            pl.col("ocel_type") == "agent"
        ).height

        # Avg messages per conversation: same computation the distribution
        # chart in ConversationCompositionSection uses — we only need the
        # mean here, so recomputing is cheap and keeps this class free of
        # any dependency on the composition section.
        avg_messages = self._avg_messages_per_conversation(conversation_count)

        cards = [
            ("Total Events", f"{self._ocel.events.height:,}"),
            ("Number of Conversations", f"{conversation_count:,}"),
            ("Unique Event Types", f"{non_handover['ocel_type'].n_unique()}"),
            ("Agent Handovers", f"{handover.height:,}"),
            ("Avg Messages / Conversation",
             f"{avg_messages:.1f}" if avg_messages is not None else "—"),
            ("Total Agents", f"{agent_count:,}"),
        ]
        cards_html = "".join(kpi_card(title, value) for title, value in cards)
        return kpi_row(cards_html, columns=6)

    def _avg_messages_per_conversation(self, conversation_count: int) -> float | None:
        if conversation_count == 0:
            return None
        agent_types = {"order_agent", "inventory_agent",
                       "barista_agent", "customer_service_agent"}
        agent_messages = (
            self._ocel.event_object
            .filter(pl.col("ocel_qualifier").is_in(agent_types))
            .select("ocel_event_id")
            .unique()
        )
        message_events = (
            self._events_flat.filter(pl.col("ocel_type") != "call_llm")
            .join(agent_messages, left_on="ocel_id", right_on="ocel_event_id", how="inner")
        )
        return message_events.height / conversation_count

    # ---- Time KPI row (3 cards) ---------------------------------------

    def _build_time_kpi_row(self) -> pn.viewable.Viewable:
        order_durations = per_order_durations(self._ocel)
        if order_durations.is_empty():
            return pn.pane.Alert("No per-order data in this log.", alert_type="info")

        cards_html = "".join(
            per_order_kpi_card(title, subtitle, order_durations[col], unit=unit)
            for title, subtitle, col, unit in _PER_ORDER_CARDS
        )
        return kpi_row(cards_html, columns=3, top_padding=8)

    # ---- Tokens KPI row (2 cards) -------------------------------------

    def _build_tokens_kpi_row(self) -> pn.pane.HTML:
        token_events = self._events_flat.filter(
            pl.col("input_tokens").is_not_null() & pl.col("response_tokens").is_not_null()
        )
        input_tokens = int(token_events["input_tokens"].sum()) if token_events.height else 0
        response_tokens = int(token_events["response_tokens"].sum()) if token_events.height else 0
        cards_html = "".join([
            kpi_card("Input Tokens", f"{input_tokens:,}" if input_tokens > 0 else "—"),
            kpi_card("Response Tokens", f"{response_tokens:,}" if response_tokens > 0 else "—"),
        ])
        return kpi_row(cards_html, columns=2, top_padding=8)

    # ---- Agent workload chart -----------------------------------------

    def _build_agent_workload_chart(self) -> pn.viewable.Viewable:
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
