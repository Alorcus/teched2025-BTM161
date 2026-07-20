import panel as pn
import plotly.express as px
import polars as pl

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

from .eventlog_helpers import (
    activity_divergence,
    agents_per_case,
    flat_event_table,
    handover_counts_per_case,
    tool_call_fanout_per_case,
)
from .styling_helpers import COLOR_SCHEME, kpi_row, section_header, subsection_header, subtitled_kpi_card


class ConversationCompositionSection:
    """Per-conversation shape metrics: how many objects (of which types) a
    conversation pulls in, and how much repetition/rework that produces.

    Complements the OC-DFG / Petri net / event-object graphs in
    VisualizationSection — those show the *shape* of object interactions,
    this section turns that shape into concrete numbers and rankings.
    """

    def __init__(self, ocel: ObjectCentricEventlog):
        self._ocel = ocel
        self._messages_per_conversation = self._compute_messages_per_conversation()
        self._pane = self._build()

    def panel(self) -> pn.viewable.Viewable:
        return self._pane

    def _build(self) -> pn.Column:
        column = section_header("Conversation Composition")
        column.append(self._summary_cards())
        column.append(subsection_header("Agents per Conversation — Distribution", top_margin=10))
        column.append(self._agents_per_case_chart())
        column.append(subsection_header("Messages per Conversation", top_margin=10))
        column.append(self._messages_distribution_chart())
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
        return kpi_row(cards_html, columns=3, top_padding=12)

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

    # ---- Messages-per-conversation distribution -----------------------

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

    def _messages_distribution_chart(self) -> pn.viewable.Viewable:
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
