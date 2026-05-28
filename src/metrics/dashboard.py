from __future__ import annotations

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog


COLOR_SCHEME = {
    "beige":"#EBDBCB",
    "yellow":"#FDCA40",
    "orange":"#D87F12",
    "red":"#8D0209",
    "dark_red":"#721A0D",
    "brown":"#563210",
}


st.set_page_config(page_title="Agentic Behavior Dashboard", layout="wide")
LOG_DIR = Path("generated_event_log")

AGENT_COLORS = {
    "order_agent": COLOR_SCHEME["yellow"],
    "barista_agent": COLOR_SCHEME["orange"],
    "inventory_agent": COLOR_SCHEME["red"],
    "customer_service_agent": COLOR_SCHEME["brown"],
} 

@st.cache_resource(show_spinner="Converting event log to OCEL …")
def load_ocel(path: Path) -> ObjectCentricEventlog:
    return ObjectCentricEventlog.from_eventlog(str(path))

# Helpers

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

# Helper Tables

ocel: ObjectCentricEventlog = load_ocel(LOG_DIR / selected_name)
events_flat   = flat_event_table(ocel)
is_handover   = events_flat["ocel_type"].str.contains("_handover_")
non_handover_events   = events_flat.filter(~is_handover)
handover_events = events_flat.filter(is_handover)
h_matrix      = handover_matrix(ocel)
agent_counts  = agent_event_counts(ocel)

token_events  = events_flat.filter(
    pl.col("input_tokens").is_not_null() & pl.col("response_tokens").is_not_null()
)
activity_freq = (
    non_handover_events.group_by("ocel_type")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)
duration_stats = (
    non_handover_events.drop_nulls("duration")
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

# Header

st.title("Agent Behavior Dashboard")
st.caption(
    f"Log: `{selected_name}`  ·  "
    f"{ocel.events.height:,} events  ·  "
    f"{ocel.objects.height:,} objects"
)

# General KPIs

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Events", f"{ocel.events.height:,}")
m2.metric("Unique Activities", str(non_handover_events["ocel_type"].n_unique()))
m3.metric("Handovers", f"{handover_events.height:,}")
m4.metric("Used Input Tokens",
          f"{int(token_events['input_tokens'].sum()):,}" if token_events.height else "—")
m5.metric("Used Response Tokens",
          f"{int(token_events['response_tokens'].sum()):,}" if token_events.height else "—")


# System Metrics Section

st.subheader("System Metrics ", divider="grey")

st.markdown("_Agent Workload_")
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


# Time Metrics Section

st.subheader("Time Metrics ", divider="grey")

st.markdown("_Average Activity Duration_")
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
