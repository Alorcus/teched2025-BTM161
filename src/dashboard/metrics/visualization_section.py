from __future__ import annotations

import logging
import uuid
from pathlib import Path

import panel as pn
import polars as pl

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog
from src.visualization.visualizer import Visualizer, VisualizationConfig, export_case_dfg
from .eventlog_helpers import (
    FEEDBACK_HIGH,
    FEEDBACK_LOW,
    case_feedback_scores,
    event_case_map,
    flat_event_table,
)
from .styling_helpers import COLOR_SCHEME, AGENT_COLORS, section_header, subsection_header


logger = logging.getLogger(__name__)

_CASE_DFG_UNAVAILABLE = "Case-centric DFG unavailable."

# Graph switcher: coffee-brown system from the dashboard palette
_GRAPH_SWITCH_STYLESHEET = """
.bk-btn-group .bk-btn-primary {
  background-color: #EBDBCB !important;
  border-color: #D9C4AD !important;
  color: #563210 !important;
}
.bk-btn-group .bk-btn-primary:hover {
  background-color: #E0CCB6 !important;
}
.bk-btn-group .bk-btn-primary.bk-active {
  background-color: #563210 !important;
  border-color: #563210 !important;
  color: #ffffff !important;
  font-weight: 600;
  box-shadow: none !important;
}
"""

# Feedback-class switcher in the orange system from the dashboard palette 
_FEEDBACK_SWITCH_STYLESHEET = """
.bk-btn-group .bk-btn {
  background-color: #F8E4CB !important;
  border: none !important;
  border-radius: 0;
  color: #A35F0C !important;
  font-size: 13px;
  padding: 3px 10px;
}
/* Joined group like the graph switcher: thin divider lines between
   buttons, rounding only on the outer corners. */
.bk-btn-group .bk-btn + .bk-btn {
  border-left: 1px solid #E8C9A0 !important;
}
.bk-btn-group .bk-btn:first-child {
  border-radius: 4px 0 0 4px;
}
.bk-btn-group .bk-btn:last-child {
  border-radius: 0 4px 4px 0;
}
.bk-btn-group .bk-btn:hover {
  background-color: #F2D7B2 !important;
}
.bk-btn-group .bk-btn.bk-active {
  background-color: #D87F12 !important;
  color: #ffffff !important;
  font-weight: 600;
  box-shadow: none !important;
}
"""

_COLOR_MAP = {
    "order_agent": AGENT_COLORS["order_agent"],
    "barista_agent": AGENT_COLORS["order_agent"],
    "inventory_agent": AGENT_COLORS["order_agent"],
    "customer_service_agent": AGENT_COLORS["order_agent"],
    "user": COLOR_SCHEME["dark_red"],
    "prompt": "#000000",
    "response": COLOR_SCHEME["beige"],
}


def _button_switcher(
    panels: dict[str, pn.viewable.Viewable],
    **selector_kwargs,
) -> pn.Column:
    """Row of buttons + content slot: a click swaps the displayed panel.

    The first key in `panels` is shown initially.
    """
    selector = pn.widgets.RadioButtonGroup(
        options=list(panels),
        value=next(iter(panels)),
        **selector_kwargs,
    )
    slot = pn.Column(panels[selector.value], sizing_mode="stretch_width")

    def _switch(event) -> None:
        slot[:] = [panels[event.new]]

    selector.param.watch(_switch, "value")
    return pn.Column(selector, slot, sizing_mode="stretch_width")


def _wrap_svg(svg_text: str) -> pn.viewable.Viewable:
    zoom = pn.widgets.FloatSlider(name="Zoom", start=0.2, end=3.0, value=1.0, step=0.1)

    @pn.depends(zoom)
    def view(scale):
        return pn.pane.HTML(
            f'<div style="overflow:auto;height:600px;">'
            f'<div style="transform:scale({scale});transform-origin:top left;display:inline-block;">'
            f"{svg_text}"
            f"</div></div>"
        )

    return pn.Column(zoom, view)


class VisualizationSection:
    def __init__(self, ocel: ObjectCentricEventlog):
        self._ocel = ocel
        self._pane = self._build()

    def panel(self) -> pn.viewable.Viewable:
        return self._pane

    def _build(self) -> pn.Column:
        column = section_header("Process Visualization")
        column.append(self._visualization_tabs())
        return column

    def _build_case_dfg_df(self) -> pl.DataFrame | None:
        """Build enriched flat event log for case-centric DFG.

        Filter out call_llm, agent_response and user_prompt activities, to make the DFG clearer.

        Returns a polars event log with pm4py columns (case:concept:name,
        concept:name, time:timestamp) plus optional case_feedback_score /
        case_feedback_class columns, or None on failure.
        """
        try:
            flat = flat_event_table(self._ocel)
            eo = event_case_map(self._ocel)
            # Display name: "customer_service_agent" -> "Customer Service", "barista_agent" -> "Barista"
            agent_display = (
                pl.col("agent_type")
                .str.replace(r"_agent$", "")
                .str.replace_all("_", " ")
                .str.to_titlecase()
            )
            flat_log = (
                flat
                .join(eo, left_on="ocel_id", right_on="ocel_event_id", how="inner")
                .filter(~pl.col("ocel_type").str.contains("_handover_"))
                .filter(~pl.col("ocel_type").is_in(["call_llm", "agent_response", "user_prompt"]))
                .with_columns(
                    activity=pl.when(pl.col("ocel_type") == "user_feedback")
                    .then(pl.lit("User: feedback"))
                    .otherwise(pl.format("{}: {}", agent_display, pl.col("ocel_type")))
                )
            )
            if flat_log.is_empty():
                return None

            df = flat_log.select(
                pl.col("case_id").alias("case:concept:name"),
                pl.col("activity").alias("concept:name"),
                pl.col("ocel_time").alias("time:timestamp"),
            )

            # Case-level feedback enrichment.
            feedback = case_feedback_scores(self._ocel)
            if not feedback.is_empty():
                df = df.join(
                    feedback.select(
                        "case_id",
                        pl.col("feedback_score").alias("case_feedback_score"),
                    ),
                    left_on="case:concept:name",
                    right_on="case_id",
                    how="left",
                ).with_columns(
                    case_feedback_class=pl.when(
                        pl.col("case_feedback_score") < FEEDBACK_LOW
                    )
                    .then(pl.lit("low"))
                    .when(pl.col("case_feedback_score") < FEEDBACK_HIGH)
                    .then(pl.lit("medium"))
                    .when(pl.col("case_feedback_score").is_not_null())
                    .then(pl.lit("high"))
                    .otherwise(pl.lit("unrated"))
                )

            return df
        except Exception:
            logger.exception("Case-centric DFG build failed")
            return None

    def _svg_from_df(self, df: pl.DataFrame, out_path: Path) -> str | None:
        """Export DFG SVG from a polars event log DataFrame, return SVG text or None."""
        try:
            cols = ["case:concept:name", "concept:name", "time:timestamp"]
            export_case_dfg(df.select(cols), out_path, export_format="svg")
            return out_path.read_text()
        except Exception:
            logger.exception("DFG export to %s failed", out_path.name)
            return None

    def _case_dfg_panel(self, export_name: str) -> pn.viewable.Viewable:
        """Return panel for case-centric DFG: 3 tabs by feedback class, or single DFG."""
        df = self._build_case_dfg_df()
        if df is None:
            return pn.pane.Alert(_CASE_DFG_UNAVAILABLE, alert_type="info")

        out_dir = Path("generated_visualizations")

        if "case_feedback_class" not in df.columns:
            svg = self._svg_from_df(df, out_dir / f"{export_name}-case-dfg.svg")
            return _wrap_svg(svg) if svg else pn.pane.Alert(_CASE_DFG_UNAVAILABLE, alert_type="info")

        panels: dict[str, pn.viewable.Viewable] = {}
        for cls in ["low", "medium", "high", "unrated"]:
            subset = df.filter(pl.col("case_feedback_class") == cls)
            label = f"{cls.title()} feedback" if cls != "unrated" else "Unrated"
            if subset.is_empty():
                panels[label] = pn.pane.Alert(f"No {cls} cases.", alert_type="info")
                continue
            svg = self._svg_from_df(subset, out_dir / f"{export_name}-case-dfg-{cls}.svg")
            content = _wrap_svg(svg) if svg else pn.pane.Alert(f"DFG for {cls} cases unavailable.", alert_type="info")
            panels[label] = content

        return _button_switcher(
            panels,
            button_type="default",
            stylesheets=[_FEEDBACK_SWITCH_STYLESHEET],
            width=440,
            margin=(0, 0, 6, 0),
        )

    def _visualization_tabs(self) -> pn.viewable.Viewable:
        try:
            if self._ocel.events.is_empty() or self._ocel.objects.is_empty():
                return pn.pane.Alert(
                    "Event log is empty or has no objects. Run a conversation first.",
                    alert_type="info",
                )

            # Generate unique export name to avoid collisions
            export_name = f"observatory_{uuid.uuid4().hex[:8]}"

            try:
                self._ocel.export_to_json(export_name)
            except Exception as e:
                return pn.pane.Alert(
                    f"Failed to export event log: {str(e)[:100]}",
                    alert_type="warning",
                )

            try:
                config = VisualizationConfig(
                    ocel_path=Path("generated_ocel") / f"{export_name}.json",
                    out_dir=Path("generated_visualizations"),
                    export_format="svg",
                    color_map=_COLOR_MAP,
                )
                visualizer = Visualizer(config)
                outputs = visualizer.run()

                ocdfg_svg = outputs["oc_dfg"].read_text()
                ocpn_svg = outputs["oc_pn"].read_text()

                # Graph panels switched via primary buttons (Apply-filters
                # style) instead of tabs; Event → Object Types is hidden.
                panels = {
                    "Case-Centric DFG": self._case_dfg_panel(export_name),
                    "Object-Centric DFG": _wrap_svg(ocdfg_svg),
                    "Object-Centric PN": _wrap_svg(ocpn_svg),
                }
                return _button_switcher(
                    panels,
                    button_type="primary",
                    stylesheets=[_GRAPH_SWITCH_STYLESHEET],
                    sizing_mode="stretch_width",
                    # Top margin keeps the switcher clear of the section
                    # divider, matching the spacing of other sections.
                    margin=(14, 0, 10, 0),
                )

            except Exception as e:
                import traceback

                error_msg = f"Visualization generation failed: {str(e)}\n{traceback.format_exc()}"
                print(error_msg, flush=True)
                return pn.pane.Alert(
                    f"Process visualization unavailable: {str(e)[:100]}",
                    alert_type="info",
                )

        except Exception as e:
            import traceback

            error_msg = (
                f"Error in visualization section: {str(e)}\n{traceback.format_exc()}"
            )
            print(error_msg, flush=True)
            return pn.pane.Alert(
                f"Visualization error: {str(e)[:100]}",
                alert_type="warning",
            )
