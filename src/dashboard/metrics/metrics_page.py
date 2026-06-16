from pathlib import Path

import panel as pn

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

from ..nav import header_nav
from .overview_section import OverviewSection
from .system_metrics_section import SystemMetricsSection
from .time_metrics_section import TimeMetricsSection
from .visualization_section import VisualizationSection


def create_metrics_dashboard():
    """Create the Metrics Observatory page."""
    pn.extension("plotly", sizing_mode="stretch_width")

    LOG_DIR = Path("generated_event_log")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(LOG_DIR.glob("*.csv"), reverse=True)  # most recent first
    if not csv_files:
        return _empty_template(LOG_DIR)

    file_selector = pn.widgets.Select(
        name="Select Event Log",
        options={f.name: f for f in csv_files},
        value=csv_files[0],
        sizing_mode="stretch_width",
        margin=(0, 0, 10, 0),
    )

    @pn.depends(file_selector.param.value)
    def metrics_content(selected_file):
        if not selected_file:
            return pn.pane.Alert("No log selected", alert_type="warning")
        try:
            ocel = ObjectCentricEventlog.from_eventlog(str(selected_file))
        except Exception as e:
            return pn.pane.Alert(f"Error loading log: {e}", alert_type="danger")

        return pn.Column(
            OverviewSection(ocel, selected_file).panel(),
            SystemMetricsSection(ocel).panel(),
            TimeMetricsSection(ocel).panel(),
            VisualizationSection(ocel).panel(),
            sizing_mode="stretch_width",
            styles={"padding": "4px 0"},
        )

    sidebar = pn.Column(
        pn.pane.HTML(
            '<div style="font-size:14px;font-weight:600;margin-bottom:6px;">Metrics Observatory</div>',
            sizing_mode="stretch_width",
        ),
        pn.pane.HTML(
            f'<div style="font-size:11px;color:#999;margin-bottom:6px;">'
            f"{len(csv_files)} event log(s) available</div>",
            sizing_mode="stretch_width",
        ),
        file_selector,
        width=276,
        styles={"padding": "10px 12px 10px 16px"},
    )

    nav_tabs = header_nav(active="/metrics")

    return pn.template.FastListTemplate(
        title="Coffee Shop Agent Observatory",
        sidebar=[sidebar],
        header=[nav_tabs],
        main=[metrics_content],
        accent_base_color="#795548",
        header_background="#4E342E",
        theme="default",
    )


def _empty_template(log_dir: Path) -> pn.template.FastListTemplate:
    return pn.template.FastListTemplate(
        title="Coffee Shop Metrics",
        sidebar=[
            pn.pane.Alert(
                f"No CSV event logs found in **{log_dir.resolve()}**. "
                "Run a conversation in the Observatory and save it first.",
                alert_type="warning",
            )
        ],
        header=[header_nav(active="/metrics")],
        main=[],
        accent_base_color="#795548",
        header_background="#4E342E",
        theme="default",
    )
