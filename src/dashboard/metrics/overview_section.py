from pathlib import Path

import panel as pn
import plotly.express as px
import polars as pl

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

from .eventlog_helpers import flat_event_table
from .styling_helpers import COLOR_SCHEME, small_kpi_card, subsection_header


class OverviewSection:
    def __init__(self, ocel: ObjectCentricEventlog, log_path: Path):
        self._ocel = ocel
        self._log_path = log_path
        self._messages_per_conversation = self._compute_messages_per_conversation()
        self._pane = pn.Column(
            self._build_header(),
            self._build_kpi_row(),
            subsection_header("Messages per Conversation"),
            self._build_messages_distribution_chart(),
            sizing_mode="stretch_width",
        )

    def panel(self) -> pn.viewable.Viewable:
        return self._pane

    def _build_header(self) -> pn.pane.HTML:
        return pn.pane.HTML(
            f'<div style="font-size:11px;color:#666;margin-bottom:4px;padding:2px 0;">'
            f'<b>Log:</b> {self._log_path.name}  ·  '
            f'{self._ocel.events.height:,} events  ·  '
            f'{self._ocel.objects.height:,} objects</div>',
            sizing_mode="stretch_width",
        )

    def _build_kpi_row(self) -> pn.pane.HTML:
        events_flat = flat_event_table(self._ocel)
        is_handover = events_flat["ocel_type"].str.contains("_handover_")
        non_handover = events_flat.filter(~is_handover)
        handover = events_flat.filter(is_handover)
        token_events = events_flat.filter(
            pl.col("input_tokens").is_not_null() & pl.col("response_tokens").is_not_null()
        )
        input_tokens = int(token_events["input_tokens"].sum()) if token_events.height else 0
        response_tokens = int(token_events["response_tokens"].sum()) if token_events.height else 0
        avg_messages = (
            float(self._messages_per_conversation.mean())
            if self._messages_per_conversation.len()
            else None
        )

        cards = [
            ("Total Events", f"{self._ocel.events.height:,}"),
            ("Unique Event Types", f"{non_handover['ocel_type'].n_unique()}"),
            ("Agent Handovers", f"{handover.height:,}"),
            ("Avg Messages / Conversation",
             f"{avg_messages:.1f}" if avg_messages is not None else "—"),
            ("Input Tokens", f"{input_tokens:,}" if input_tokens > 0 else "—"),
            ("Response Tokens", f"{response_tokens:,}" if response_tokens > 0 else "—"),
        ]
        cards_html = "".join(small_kpi_card(label, value) for label, value in cards)
        return pn.pane.HTML(
            f'<div style="padding:2px 0;display:flex;flex-wrap:wrap;">{cards_html}</div>',
            sizing_mode="stretch_width",
        )

    def _compute_messages_per_conversation(self) -> pl.Series:
        """Message count per conversation (one row per user object).

        A message is a tool call, handover, or agent reply to the user (i.e. any
        event qualified by an agent object, excluding the internal `call_llm`
        span). User prompts, user feedback, and coffee-machine events are
        excluded by virtue of their non-agent qualifier. Conversations with
        zero messages are included as zeros so the distribution reflects the
        full sample.

        Agent object ids follow the pattern ``<user-uuid>_<agent-name>``; the
        user uuid is used as the conversation key.
        """
        agent_types = {"order_agent", "inventory_agent",
                       "barista_agent", "customer_service_agent"}
        conversations = self._ocel.objects.filter(pl.col("ocel_type") == "user")
        if conversations.is_empty():
            return pl.Series("messages", [], dtype=pl.Int64)

        events_flat = flat_event_table(self._ocel)
        agent_messages = (
            self._ocel.event_object
            .filter(pl.col("ocel_qualifier").is_in(agent_types))
            .with_columns(
                pl.col("ocel_object_id").str.split("_").list.first().alias("user_id")
            )
            .select("ocel_event_id", "user_id")
            .unique()
        )
        counts = (
            events_flat.filter(pl.col("ocel_type") != "call_llm")
            .join(agent_messages, left_on="ocel_id", right_on="ocel_event_id", how="inner")
            .group_by("user_id")
            .agg(pl.len().alias("messages"))
        )
        per_conv = (
            conversations.select(pl.col("ocel_id").alias("user_id"))
            .join(counts, on="user_id", how="left")
            .with_columns(pl.col("messages").fill_null(0))
        )
        return per_conv["messages"]

    def _build_messages_distribution_chart(self) -> pn.viewable.Viewable:
        counts = self._messages_per_conversation
        n = counts.len()
        if n == 0:
            return pn.pane.Alert("No conversations in this log.", alert_type="info")

        fig = px.box(
            counts.to_pandas().to_frame(name="messages"),
            x="messages",
            points="all",
            hover_data={"messages": True},
        )
        fig.update_traces(
            marker=dict(color=COLOR_SCHEME["orange"], size=6, opacity=0.75),
            line=dict(color=COLOR_SCHEME["brown"]),
            fillcolor=COLOR_SCHEME["beige"],
            jitter=0.4,
            pointpos=0,
        )
        median = int(counts.median()) if n else 0
        fig.update_layout(
            xaxis_title=f"Messages per conversation  ·  n={n}  ·  median={median}",
            yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=5, b=30),
            height=140,
            font=dict(size=10),
            plot_bgcolor=COLOR_SCHEME["off-white"],
            showlegend=False,
        )
        return pn.pane.Plotly(fig, height=140, sizing_mode="stretch_width")
