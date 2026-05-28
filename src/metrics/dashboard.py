"""
Agentic Behavior Dashboard — OCEL 2.0
Scans ./event_logs/ for raw CSV event logs, converts the selected one to an
ObjectCentricEventlog on the fly, and works directly against its properties:
  ocel.events, ocel.objects, ocel.event_object, ocel.object_object,
  ocel.event_tables, ocel.object_tables, ocel.event_map_type, ocel.object_map_type
"""

from __future__ import annotations

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Agentic Behavior Dashboard", layout="wide")

LOG_DIR = Path("generated_event_log")

AGENT_COLORS = {
    "order_agent":            "#6366f1",
    "barista_agent":          "#f59e0b",
    "inventory_agent":        "#10b981",
    "customer_service_agent": "#ef4444",
    "user":                   "#64748b",
}

# ── Loader: CSV → ObjectCentricEventlog ───────────────────────────────────────

@st.cache_resource(show_spinner="Converting event log to OCEL …")
def load_ocel(path: Path) -> ObjectCentricEventlog:
    return ObjectCentricEventlog.from_eventlog(str(path))

# ── Derived-data helpers ──────────────────────────────────────────────────────

def flat_event_table(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Union all per-type event tables into one flat DataFrame."""
    TARGET_SCHEMA: dict[str, pl.PolarsDataType] = {
        "ocel_id":         pl.Utf8,
        "ocel_time":       pl.Datetime,
        "duration":        pl.Float64,
        "input_tokens":    pl.Float64,
        "response_tokens": pl.Float64,
        "model":           pl.Utf8,
    }
    frames = []
    for tbl_name, df in ocel.event_tables.items():
        event_type = tbl_name.removeprefix("event_")
        # Add any missing columns as typed nulls
        for col, dtype in TARGET_SCHEMA.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        # Cast existing columns to the required type
        df = df.with_columns(
            [pl.col(c).cast(t, strict=False) for c, t in TARGET_SCHEMA.items()]
        ).with_columns(pl.lit(event_type).alias("ocel_type"))
        frames.append(df.select(["ocel_id", "ocel_type", "ocel_time",
                                  "duration", "input_tokens", "response_tokens", "model"]))
    if not frames:
        return pl.DataFrame(schema={**TARGET_SCHEMA, "ocel_type": pl.Utf8})
    return pl.concat(frames)


def agent_event_counts(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    agent_objects = ocel.objects.filter(pl.col("ocel_type").str.contains("agent"))
    return (
        ocel.event_object
        .join(agent_objects, left_on="ocel_object_id", right_on="ocel_id", how="inner")
        .group_by("ocel_type")
        .agg(pl.len().alias("event_count"))
        .sort("event_count", descending=True)
        .rename({"ocel_type": "agent"})
    )


def handover_matrix(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    handover_events = ocel.events.filter(pl.col("ocel_type").str.contains("_handover_"))
    if handover_events.is_empty():
        return pl.DataFrame(schema={"source": str, "target": str, "count": pl.UInt32})

    def _split(s: str):
        parts = s.split("_handover_")
        return (
            parts[0].replace("_", " ").title(),
            parts[1].replace("_", " ").title(),
        ) if len(parts) == 2 else (s, "")

    pairs = [_split(t) for t in handover_events["ocel_type"].to_list()]
    return (
        pl.DataFrame({"source": [p[0] for p in pairs], "target": [p[1] for p in pairs]})
        .group_by(["source", "target"])
        .agg(pl.len().alias("count"))
    )

# ── Sidebar: file selector ────────────────────────────────────────────────────

st.sidebar.title("📂 Event Log")
LOG_DIR.mkdir(parents=True, exist_ok=True)
csv_files = sorted(LOG_DIR.glob("*.csv"))

if not csv_files:
    st.warning(
        f"No CSV event logs found in **{LOG_DIR.resolve()}**. "
        "Drop your raw event log CSVs there and refresh."
    )
    st.stop()

selected_name = st.sidebar.selectbox(
    "Select event log",
    options=[f.name for f in csv_files],
    index=0,
)

ocel: ObjectCentricEventlog = load_ocel(LOG_DIR / selected_name)

# ── Build derived tables ──────────────────────────────────────────────────────

events_flat   = flat_event_table(ocel)
is_handover   = events_flat["ocel_type"].str.contains("_handover_")
core_events   = events_flat.filter(~is_handover)
handover_evts = events_flat.filter(is_handover)
h_matrix      = handover_matrix(ocel)
agent_counts  = agent_event_counts(ocel)

token_events  = events_flat.filter(
    pl.col("input_tokens").is_not_null() & pl.col("response_tokens").is_not_null()
)
activity_freq = (
    core_events.group_by("ocel_type")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)
duration_stats = (
    core_events.drop_nulls("duration")
    .group_by("ocel_type")
    .agg(
        pl.mean("duration").alias("avg_duration"),
        pl.median("duration").alias("median_duration"),
        pl.max("duration").alias("max_duration"),
    )
    .sort("avg_duration", descending=True)
)
obj_type_counts = (
    ocel.objects.group_by("ocel_type")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🤖 Agent Behavior Dashboard")
st.caption(
    f"Log: `{selected_name}`  ·  "
    f"{ocel.events.height:,} events  ·  "
    f"{ocel.objects.height:,} objects"
)

# ── KPIs ──────────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Events",      f"{ocel.events.height:,}")
k2.metric("Unique Activities", str(core_events["ocel_type"].n_unique()))
k3.metric("Handovers",         f"{handover_evts.height:,}")
k4.metric("Input Tokens",
          f"{int(token_events['input_tokens'].sum()):,}" if token_events.height else "—")
k5.metric("Response Tokens",
          f"{int(token_events['response_tokens'].sum()):,}" if token_events.height else "—")

st.divider()

# ── Row 1: Activity frequency + Agent workload ────────────────────────────────

col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Activity Frequency")
    fig = px.bar(
        activity_freq.to_pandas(),
        x="count", y="ocel_type", orientation="h",
        color="count", color_continuous_scale="Blues",
        labels={"ocel_type": "Activity", "count": "Occurrences"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      coloraxis_showscale=False,
                      margin=dict(l=0, r=10, t=10, b=0), height=380)
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("Agent Workload")
    if agent_counts.height:
        fig = px.bar(
            agent_counts.to_pandas(),
            x="agent", y="event_count",
            color="agent", color_discrete_map=AGENT_COLORS,
            labels={"agent": "Agent", "event_count": "Events Handled"},
        )
        fig.update_layout(showlegend=False,
                          margin=dict(l=0, r=10, t=10, b=0), height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No agent–event relationships found in this log.")

# ── Row 2: Duration + Token scatter ──────────────────────────────────────────

st.divider()
col_l2, col_r2 = st.columns(2)

with col_l2:
    st.subheader("Average Duration by Activity (s)")
    if duration_stats.height:
        fig = px.bar(
            duration_stats.to_pandas().head(15),
            x="avg_duration", y="ocel_type", orientation="h",
            color="avg_duration", color_continuous_scale="Oranges",
            labels={"ocel_type": "Activity", "avg_duration": "Avg Duration (s)"},
            hover_data={"median_duration": True, "max_duration": True},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                          coloraxis_showscale=False,
                          margin=dict(l=0, r=10, t=10, b=0), height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No duration data in this log.")

with col_r2:
    st.subheader("Token Usage per LLM Call")
    if token_events.height:
        fig = px.scatter(
            token_events.select(["ocel_type", "input_tokens", "response_tokens"]).to_pandas(),
            x="input_tokens", y="response_tokens",
            color="ocel_type", opacity=0.75,
            labels={"input_tokens": "Input Tokens",
                    "response_tokens": "Response Tokens",
                    "ocel_type": "Event Type"},
        )
        fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No token data in this log.")

# ── Row 3: Handover Sankey + Timeline ────────────────────────────────────────

st.divider()
col_l3, col_r3 = st.columns(2)

with col_l3:
    st.subheader("Agent Handover Flow")
    if h_matrix.height:
        nodes = list(dict.fromkeys(
            h_matrix["source"].to_list() + h_matrix["target"].to_list()
        ))
        node_idx    = {n: i for i, n in enumerate(nodes)}
        node_colors = [AGENT_COLORS.get(n.lower().replace(" ", "_"), "#94a3b8")
                       for n in nodes]
        fig = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=20, label=nodes, color=node_colors),
            link=dict(
                source=[node_idx[s] for s in h_matrix["source"].to_list()],
                target=[node_idx[t] for t in h_matrix["target"].to_list()],
                value=h_matrix["count"].to_list(),
                color="rgba(150,150,150,0.3)",
            ),
        ))
        fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No handover events found in this log.")

with col_r3:
    st.subheader("Event Timeline")
    tl = events_flat.drop_nulls("ocel_time").sort("ocel_time")
    if tl.height:
        tl_pd = tl.to_pandas()
        tl_pd["minute"] = tl_pd["ocel_time"].dt.floor("min")
        density = tl_pd.groupby("minute").size().reset_index(name="count")
        fig = px.area(density, x="minute", y="count",
                      labels={"minute": "Time", "count": "Events"},
                      color_discrete_sequence=["#6366f1"])
        fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timestamp data in this log.")

# ── Row 4: Object distribution + Event explorer ───────────────────────────────

st.divider()
col_l4, col_r4 = st.columns([1, 2])

with col_l4:
    st.subheader("Object Type Distribution")
    fig = px.pie(
        obj_type_counts.to_pandas(),
        names="ocel_type", values="count", hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=340)
    st.plotly_chart(fig, use_container_width=True)

with col_r4:
    st.subheader("Event Explorer")
    selected_table = st.selectbox(
        "Event table",
        options=list(ocel.event_tables.keys()),
        format_func=lambda k: k.removeprefix("event_"),
    )
    type_filter = st.multiselect(
        "Cross-type filter",
        options=sorted(events_flat["ocel_type"].unique().to_list()),
        default=[],
        placeholder="All event types",
    )
    if type_filter:
        show_df = events_flat.filter(pl.col("ocel_type").is_in(type_filter))
        st.dataframe(show_df.sort("ocel_time").to_pandas(),
                     use_container_width=True, height=300, hide_index=True)
    else:
        st.dataframe(ocel.event_tables[selected_table].sort("ocel_time").to_pandas(),
                     use_container_width=True, height=300, hide_index=True)

# ── Footer: raw OCEL table inspector ─────────────────────────────────────────

with st.expander("🔍 Raw OCEL tables"):
    tab_ev, tab_ob, tab_eo, tab_oo = st.tabs(
        ["events", "objects", "event_object", "object_object"]
    )
    with tab_ev:
        st.dataframe(ocel.events.to_pandas(), use_container_width=True, hide_index=True)
    with tab_ob:
        st.dataframe(ocel.objects.to_pandas(), use_container_width=True, hide_index=True)
    with tab_eo:
        st.dataframe(ocel.event_object.to_pandas(), use_container_width=True, hide_index=True)
    with tab_oo:
        st.dataframe(ocel.object_object.to_pandas(), use_container_width=True, hide_index=True)