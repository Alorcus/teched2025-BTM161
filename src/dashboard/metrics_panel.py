"""
Metrics Dashboard Panel - migrated from Streamlit to Panel.
Displays analytics and visualizations for event logs.
"""
from pathlib import Path

import panel as pn
import plotly.express as px
import polars as pl

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog


COLOR_SCHEME = {
    "beige": "#EBDBCB",
    "yellow": "#FDCA40",
    "orange": "#D87F12",
    "red": "#8D0209",
    "dark_red": "#721A0D",
    "brown": "#563210",
}

AGENT_COLORS = {
    "order_agent": COLOR_SCHEME["yellow"],
    "barista_agent": COLOR_SCHEME["orange"],
    "inventory_agent": COLOR_SCHEME["red"],
    "customer_service_agent": COLOR_SCHEME["brown"],
}


def flat_event_table(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Union all per-type event tables into one flat DataFrame."""
    TARGET_SCHEMA: dict[str, pl.PolarsDataType] = {
        "ocel_id": pl.Utf8,
        "ocel_time": pl.Datetime,
        "duration": pl.Float64,
        "input_tokens": pl.Float64,
        "response_tokens": pl.Float64,
        "model": pl.Utf8,
    }
    frames = []
    for tbl_name, df in ocel.event_tables.items():
        event_type = tbl_name.removeprefix("event_")
        for col, dtype in TARGET_SCHEMA.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        df = df.with_columns(
            [pl.col(c).cast(t, strict=False) for c, t in TARGET_SCHEMA.items()]
        ).with_columns(pl.lit(event_type).alias("ocel_type"))
        frames.append(df.select(["ocel_id", "ocel_type", "ocel_time",
                                  "duration", "input_tokens", "response_tokens", "model"]))
    if not frames:
        return pl.DataFrame(schema={**TARGET_SCHEMA, "ocel_type": pl.Utf8})
    return pl.concat(frames)


def agent_event_counts(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Count events handled by each agent."""
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
    """Build handover matrix from agent-to-agent transitions."""
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


def create_metrics_dashboard():
    """Create the Metrics Dashboard page."""
    pn.extension('plotly', sizing_mode="stretch_both")

    LOG_DIR = Path("generated_event_log")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # File selector
    csv_files = sorted(LOG_DIR.glob("*.csv"), reverse=True)  # Most recent first

    if not csv_files:
        return pn.template.FastListTemplate(
            title="Coffee Shop Metrics",
            sidebar=[pn.pane.Alert(
                f"No CSV event logs found in **{LOG_DIR.resolve()}**. "
                "Run a conversation in the Observatory and save it first.",
                alert_type="warning"
            )],
            main=[],
            accent_base_color="#795548",
            header_background="#4E342E",
            theme="default",
        )

    file_selector = pn.widgets.Select(
        name="Select Event Log",
        options={f.name: f for f in csv_files},
        value=csv_files[0],
        sizing_mode="stretch_width",
        margin=(0, 0, 10, 0),
    )

    # Metrics content (reactive to file selector)
    @pn.depends(file_selector.param.value)
    def metrics_content(selected_file):
        if not selected_file:
            return pn.pane.Alert("No log selected", alert_type="warning")

        try:
            # Load OCEL
            ocel = ObjectCentricEventlog.from_eventlog(str(selected_file))
            events_flat = flat_event_table(ocel)
            is_handover = events_flat["ocel_type"].str.contains("_handover_")
            non_handover_events = events_flat.filter(~is_handover)
            handover_events = events_flat.filter(is_handover)
            agent_counts = agent_event_counts(ocel)

            token_events = events_flat.filter(
                pl.col("input_tokens").is_not_null() & pl.col("response_tokens").is_not_null()
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

            # Header
            header = pn.pane.Markdown(
                f"**Log:** `{selected_file.name}`  ·  "
                f"{ocel.events.height:,} events  ·  "
                f"{ocel.objects.height:,} objects",
                styles={"font-size": "14px", "color": "#666", "margin-bottom": "15px"}
            )

            # KPI Metrics
            kpi_grid = pn.GridSpec(ncols=5, nrows=1, sizing_mode='stretch_width')

            kpi_grid[0, 0] = pn.indicators.Number(
                name="Total Events",
                value=ocel.events.height,
                format="{value:,}",
                styles={'font-size': '16pt'}
            )

            kpi_grid[0, 1] = pn.indicators.Number(
                name="Unique Activities",
                value=non_handover_events["ocel_type"].n_unique(),
                format="{value}",
                styles={'font-size': '16pt'}
            )

            kpi_grid[0, 2] = pn.indicators.Number(
                name="Handovers",
                value=handover_events.height,
                format="{value:,}",
                styles={'font-size': '16pt'}
            )

            input_tokens_val = int(token_events['input_tokens'].sum()) if token_events.height else 0
            kpi_grid[0, 3] = pn.indicators.Number(
                name="Input Tokens",
                value=input_tokens_val,
                format="{value:,}" if input_tokens_val > 0 else "—",
                styles={'font-size': '16pt'}
            )

            response_tokens_val = int(token_events['response_tokens'].sum()) if token_events.height else 0
            kpi_grid[0, 4] = pn.indicators.Number(
                name="Response Tokens",
                value=response_tokens_val,
                format="{value:,}" if response_tokens_val > 0 else "—",
                styles={'font-size': '16pt'}
            )

            # Agent Workload Chart
            workload_section = pn.Column(
                pn.pane.Markdown("### System Metrics", styles={"margin-top": "20px"}),
                pn.layout.Divider(),
                pn.pane.Markdown("**Agent Workload**", styles={"margin-bottom": "10px"}),
            )

            if agent_counts.height:
                fig_workload = px.bar(
                    agent_counts.to_pandas(),
                    x="agent", y="event_count",
                    color="agent", color_discrete_map=AGENT_COLORS,
                    labels={"agent": "Agent", "event_count": "Events Handled"},
                )
                fig_workload.update_layout(
                    showlegend=False,
                    margin=dict(l=0, r=10, t=10, b=0),
                    height=380
                )
                workload_section.append(pn.pane.Plotly(fig_workload, sizing_mode="stretch_width"))
            else:
                workload_section.append(pn.pane.Alert("No agent–event relationships found in this log.", alert_type="info"))

            # Duration Chart
            duration_section = pn.Column(
                pn.pane.Markdown("### Time Metrics", styles={"margin-top": "20px"}),
                pn.layout.Divider(),
                pn.pane.Markdown("**Average Activity Duration**", styles={"margin-bottom": "10px"}),
            )

            if duration_stats.height:
                fig_duration = px.bar(
                    duration_stats.to_pandas().head(15),
                    x="avg_duration", y="ocel_type", orientation="h",
                    color="avg_duration", color_continuous_scale="Oranges",
                    labels={"ocel_type": "Activity", "avg_duration": "Avg Duration (s)"},
                    hover_data={"median_duration": True, "max_duration": True},
                )
                fig_duration.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=10, t=10, b=0),
                    height=380
                )
                duration_section.append(pn.pane.Plotly(fig_duration, sizing_mode="stretch_width"))
            else:
                duration_section.append(pn.pane.Alert("No duration data in this log.", alert_type="info"))

            return pn.Column(
                header,
                kpi_grid,
                workload_section,
                duration_section,
                sizing_mode="stretch_both"
            )

        except Exception as e:
            return pn.pane.Alert(f"Error loading log: {e}", alert_type="danger")

    # Sidebar
    file_count = pn.pane.HTML(
        f'<div style="font-size:12px;color:#999;margin-bottom:10px;">{len(csv_files)} event log(s) available</div>',
        sizing_mode="stretch_width"
    )

    sidebar = pn.Column(
        pn.pane.Markdown("## Metrics Dashboard"),
        file_count,
        file_selector,
        width=300,
    )

    # Navigation tabs
    nav_tabs = pn.Row(
        pn.pane.HTML(
            '<a href="/" style="display:inline-block;padding:8px 16px;background:#F5F5F5;color:#333;'
            'border-radius:4px;text-decoration:none;font-weight:500;border:1px solid #ddd;margin-right:10px;">'
            'Observatory</a>',
            sizing_mode="fixed"
        ),
        pn.pane.HTML(
            '<div style="display:inline-block;padding:8px 16px;background:#4E342E;color:white;'
            'border-radius:4px;font-weight:600;">Metrics</div>',
            sizing_mode="fixed"
        ),
        margin=(0, 0, 10, 0),
    )

    # Template
    template = pn.template.FastListTemplate(
        title="Coffee Shop Metrics",
        sidebar=[sidebar],
        main=[pn.Column(nav_tabs, metrics_content, sizing_mode="stretch_both")],
        accent_base_color="#795548",
        header_background="#4E342E",
        theme="default",
    )

    return template
