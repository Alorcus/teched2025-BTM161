"""
Agentic Behavior Dashboard — OCEL 2.0
Reads exported JSON files from ./generated_ocel/ and visualises key metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Agentic Behavior Dashboard", layout="wide")

OCEL_DIR = Path("../../generated_ocel")

# ── Helpers ───────────────────────────────────────────────────────────────────

AGENT_COLORS = {
    "order_agent": "#6366f1",
    "barista_agent": "#f59e0b",
    "inventory_agent": "#10b981",
    "customer_service_agent": "#ef4444",
    "user": "#64748b",
}

def _attr_val(attrs: list[dict], name: str):
    for a in attrs:
        if a["name"] == name:
            return a["value"]
    return None


@st.cache_data(show_spinner="Parsing OCEL file …")
def load_ocel(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_events(ocel: dict) -> pl.DataFrame:
    rows = []
    for ev in ocel.get("events", []):
        attrs = ev.get("attributes", [])
        rows.append(
            {
                "event_id": ev["id"],
                "event_type": ev["type"].replace("event_", "", 1),
                "time": ev.get("time"),
                "duration": _attr_val(attrs, "duration"),
                "model": _attr_val(attrs, "model"),
                "input_tokens": _attr_val(attrs, "input_tokens"),
                "response_tokens": _attr_val(attrs, "response_tokens"),
                "n_relationships": len(ev.get("relationships", [])),
                "agents": [
                    r["objectId"]
                    for r in ev.get("relationships", [])
                    if "agent" in r.get("qualifier", "")
                ],
            }
        )
    df = pl.DataFrame(
        {
            "event_id": [r["event_id"] for r in rows],
            "event_type": [r["event_type"] for r in rows],
            "time": [r["time"] for r in rows],
            "duration": [r["duration"] for r in rows],
            "model": [r["model"] for r in rows],
            "input_tokens": [r["input_tokens"] for r in rows],
            "response_tokens": [r["response_tokens"] for r in rows],
            "n_relationships": [r["n_relationships"] for r in rows],
        }
    )
    df = df.with_columns(
        pl.col("time").str.to_datetime(time_unit="ms", strict=False),
        pl.col("duration").cast(pl.Float64, strict=False),
        pl.col("input_tokens").cast(pl.Int64, strict=False),
        pl.col("response_tokens").cast(pl.Int64, strict=False),
    )
    return df


def parse_event_object(ocel: dict) -> pl.DataFrame:
    """Flat table: event_id, object_id, qualifier"""
    rows = []
    for ev in ocel.get("events", []):
        for rel in ev.get("relationships", []):
            rows.append(
                {
                    "event_id": ev["id"],
                    "event_type": ev["type"].replace("event_", "", 1),
                    "object_id": rel["objectId"],
                    "qualifier": rel["qualifier"],
                }
            )
    if not rows:
        return pl.DataFrame(schema={"event_id": str, "event_type": str, "object_id": str, "qualifier": str})
    return pl.DataFrame(rows)


def parse_objects(ocel: dict) -> pl.DataFrame:
    rows = [{"object_id": o["id"], "object_type": o["type"]} for o in ocel.get("objects", [])]
    if not rows:
        return pl.DataFrame(schema={"object_id": str, "object_type": str})
    return pl.DataFrame(rows)


def agent_from_object_id(object_id: str) -> str:
    for agent in ["order_agent", "barista_agent", "inventory_agent", "customer_service_agent"]:
        if agent in object_id:
            return agent
    return "user"


# ── Sidebar: file selector ────────────────────────────────────────────────────

st.sidebar.title("📂 OCEL File")
OCEL_DIR.mkdir(parents=True, exist_ok=True)
ocel_files = sorted(OCEL_DIR.glob("*.json"))

if not ocel_files:
    st.warning(
        f"No OCEL JSON files found in **{OCEL_DIR.resolve()}**. "
        "Run the event log conversion first to generate files."
    )
    st.stop()

selected_name = st.sidebar.selectbox(
    "Select log",
    options=[f.name for f in ocel_files],
    index=0,
)
selected_path = OCEL_DIR / selected_name

ocel = load_ocel(selected_path)
events_df = parse_events(ocel)
eo_df = parse_event_object(ocel)
objects_df = parse_objects(ocel)

# ── Derived data ──────────────────────────────────────────────────────────────

# Handover events
handover_mask = events_df["event_type"].str.contains("_handover_")
handover_df = events_df.filter(handover_mask)

# Core (non-handover) events
core_df = events_df.filter(~handover_mask)

# Agent activity: join event_object with objects to tag each event with its agent
agent_events = (
    eo_df.filter(pl.col("qualifier").str.contains("agent"))
    .with_columns(
        pl.col("object_id").map_elements(agent_from_object_id, return_dtype=pl.Utf8).alias("agent")
    )
)

# Events per agent
agent_counts = (
    agent_events.group_by("agent")
    .agg(pl.len().alias("event_count"))
    .sort("event_count", descending=True)
)

# Activity frequency (core events only)
activity_freq = (
    core_df.group_by("event_type")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)

# Duration stats per event type (ms → s)
duration_df = (
    core_df.drop_nulls("duration")
    .group_by("event_type")
    .agg(
        pl.mean("duration").alias("avg_duration"),
        pl.median("duration").alias("median_duration"),
        pl.max("duration").alias("max_duration"),
    )
    .sort("avg_duration", descending=True)
)

# Token usage per event
token_df = events_df.drop_nulls(["input_tokens", "response_tokens"])

# Handover matrix
if handover_df.height > 0:
    def split_handover(s: str):
        parts = s.split("_handover_")
        return (parts[0].replace("_", " ").title(), parts[1].replace("_", " ").title()) if len(parts) == 2 else (s, "")

    sources, targets = zip(*[split_handover(t) for t in handover_df["event_type"].to_list()])
    handover_matrix_df = pl.DataFrame({"source": list(sources), "target": list(targets)})
    handover_matrix = (
        handover_matrix_df.group_by(["source", "target"])
        .agg(pl.len().alias("count"))
    )
else:
    handover_matrix = pl.DataFrame(schema={"source": str, "target": str, "count": pl.Int32})

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🤖 Agent Behavior Dashboard")
st.caption(f"OCEL file: `{selected_name}`")

# ── KPI Row ───────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)

n_cases = objects_df.filter(pl.col("object_type") == "user").height or objects_df.filter(
    pl.col("object_type").str.contains("agent")
).select("object_id").n_unique()

# Approximate cases from prompt objects
n_prompts = objects_df.filter(pl.col("object_type") == "prompt").height
n_handovers = handover_df.height
total_input_tokens = int(token_df["input_tokens"].sum()) if token_df.height > 0 else 0
total_resp_tokens = int(token_df["response_tokens"].sum()) if token_df.height > 0 else 0

k1.metric("Total Events", f"{events_df.height:,}")
k2.metric("Unique Activities", f"{core_df['event_type'].n_unique()}")
k3.metric("Handovers", f"{n_handovers:,}")
k4.metric("Input Tokens", f"{total_input_tokens:,}")
k5.metric("Response Tokens", f"{total_resp_tokens:,}")

st.divider()

# ── Row 1: Activity frequency + Agent workload ────────────────────────────────

col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Activity Frequency")
    fig_freq = px.bar(
        activity_freq.to_pandas(),
        x="count",
        y="event_type",
        orientation="h",
        color="count",
        color_continuous_scale="Blues",
        labels={"event_type": "Activity", "count": "Occurrences"},
    )
    fig_freq.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        margin=dict(l=0, r=10, t=10, b=0),
        height=380,
    )
    st.plotly_chart(fig_freq, use_container_width=True)

with col_r:
    st.subheader("Agent Workload")
    if agent_counts.height > 0:
        ac_pd = agent_counts.to_pandas()
        ac_pd["color"] = ac_pd["agent"].map(AGENT_COLORS).fillna("#94a3b8")
        fig_agents = px.bar(
            ac_pd,
            x="agent",
            y="event_count",
            color="agent",
            color_discrete_map=AGENT_COLORS,
            labels={"agent": "Agent", "event_count": "Events Handled"},
        )
        fig_agents.update_layout(showlegend=False, margin=dict(l=0, r=10, t=10, b=0), height=380)
        st.plotly_chart(fig_agents, use_container_width=True)
    else:
        st.info("No agent–event relationships found in this log.")

# ── Row 2: Duration heatmap + Token usage ────────────────────────────────────

st.divider()
col_l2, col_r2 = st.columns(2)

with col_l2:
    st.subheader("Average Duration by Activity (s)")
    if duration_df.height > 0:
        dur_pd = duration_df.to_pandas().head(15)
        fig_dur = px.bar(
            dur_pd,
            x="avg_duration",
            y="event_type",
            orientation="h",
            error_x=None,
            color="avg_duration",
            color_continuous_scale="Oranges",
            labels={"event_type": "Activity", "avg_duration": "Avg Duration (s)"},
            hover_data={"median_duration": True, "max_duration": True},
        )
        fig_dur.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            margin=dict(l=0, r=10, t=10, b=0),
            height=380,
        )
        st.plotly_chart(fig_dur, use_container_width=True)
    else:
        st.info("No duration data available in this log.")

with col_r2:
    st.subheader("Token Usage per LLM Call")
    if token_df.height > 0:
        tok_pd = token_df.select(["event_type", "input_tokens", "response_tokens"]).to_pandas()
        fig_tok = px.scatter(
            tok_pd,
            x="input_tokens",
            y="response_tokens",
            color="event_type",
            labels={
                "input_tokens": "Input Tokens",
                "response_tokens": "Response Tokens",
                "event_type": "Event Type",
            },
            opacity=0.75,
        )
        fig_tok.update_layout(margin=dict(l=0, r=10, t=10, b=0), height=380)
        st.plotly_chart(fig_tok, use_container_width=True)
    else:
        st.info("No token data available in this log.")

# ── Row 3: Handover Sankey + Event timeline ───────────────────────────────────

st.divider()
col_l3, col_r3 = st.columns(2)

with col_l3:
    st.subheader("Agent Handover Flow")
    if handover_matrix.height > 0:
        all_nodes = list(
            dict.fromkeys(
                handover_matrix["source"].to_list() + handover_matrix["target"].to_list()
            )
        )
        node_idx = {n: i for i, n in enumerate(all_nodes)}
        node_colors = [
            AGENT_COLORS.get(n.lower().replace(" ", "_"), "#94a3b8") for n in all_nodes
        ]
        fig_sankey = go.Figure(
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    label=all_nodes,
                    color=node_colors,
                ),
                link=dict(
                    source=[node_idx[s] for s in handover_matrix["source"].to_list()],
                    target=[node_idx[t] for t in handover_matrix["target"].to_list()],
                    value=handover_matrix["count"].to_list(),
                    color="rgba(150,150,150,0.3)",
                ),
            )
        )
        fig_sankey.update_layout(margin=dict(l=0, r=10, t=10, b=0), height=380)
        st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.info("No handover events found in this log.")

with col_r3:
    st.subheader("Event Timeline")
    tl_df = events_df.drop_nulls("time").sort("time")
    if tl_df.height > 0:
        tl_pd = tl_df.to_pandas()
        # Bucket events into 1-minute bins for density plot
        tl_pd["minute"] = tl_pd["time"].dt.floor("min")
        density = tl_pd.groupby("minute").size().reset_index(name="count")
        fig_tl = px.area(
            density,
            x="minute",
            y="count",
            labels={"minute": "Time", "count": "Events"},
            color_discrete_sequence=["#6366f1"],
        )
        fig_tl.update_layout(margin=dict(l=0, r=10, t=10, b=0), height=380)
        st.plotly_chart(fig_tl, use_container_width=True)
    else:
        st.info("No timestamp data available in this log.")

# ── Row 4: Object type distribution + Raw data explorer ──────────────────────

st.divider()
col_l4, col_r4 = st.columns([1, 2])

with col_l4:
    st.subheader("Object Type Distribution")
    obj_counts = (
        objects_df.group_by("object_type")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    fig_obj = px.pie(
        obj_counts.to_pandas(),
        names="object_type",
        values="count",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_obj.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=340)
    st.plotly_chart(fig_obj, use_container_width=True)

with col_r4:
    st.subheader("Event Explorer")
    type_filter = st.multiselect(
        "Filter by event type",
        options=sorted(events_df["event_type"].unique().to_list()),
        default=[],
        placeholder="All event types",
    )
    show_df = events_df if not type_filter else events_df.filter(pl.col("event_type").is_in(type_filter))
    st.dataframe(
        show_df.sort("time").to_pandas(),
        use_container_width=True,
        height=300,
        hide_index=True,
    )