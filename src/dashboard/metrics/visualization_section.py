from __future__ import annotations

import logging
import uuid
from pathlib import Path

import pandas as pd
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

_COLOR_MAP = {
    "order_agent": AGENT_COLORS["order_agent"],
    "barista_agent": AGENT_COLORS["order_agent"],
    "inventory_agent": AGENT_COLORS["order_agent"],
    "customer_service_agent": AGENT_COLORS["order_agent"],
    "user": COLOR_SCHEME["dark_red"],
    "prompt": "#000000",
    "response": COLOR_SCHEME["beige"],
}


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

    def _build_case_dfg_df(self) -> pd.DataFrame | None:
        """Build enriched flat event log for case-centric DFG.

        Filter out call_llm, agent_response and user_prompt activities, to make the DFG clearer.

        Returns a pandas event log with pm4py columns (case:concept:name,
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

            df_pd = flat_log.select(
                pl.col("case_id").alias("case:concept:name"),
                pl.col("activity").alias("concept:name"),
                pl.col("ocel_time").alias("time:timestamp"),
            ).to_pandas()

            # Case-level feedback enrichment
            feedback = case_feedback_scores(self._ocel)
            if not feedback.is_empty():
                case_feedback = (
                    feedback.to_pandas().set_index("case_id")["feedback_score"]
                )
                df_pd["case_feedback_score"] = df_pd["case:concept:name"].map(case_feedback)
                df_pd["case_feedback_class"] = pd.cut(
                    df_pd["case_feedback_score"],
                    bins=[0.0, FEEDBACK_LOW, FEEDBACK_HIGH, 1.01],
                    right=False,
                    labels=["low", "medium", "high"],
                )

            return df_pd
        except Exception:
            logger.exception("Case-centric DFG build failed")
            return None

    def _svg_from_df(self, df_pd: pd.DataFrame, out_path: Path) -> str | None:
        """Export DFG SVG from a pandas event log DataFrame, return SVG text or None."""
        try:
            cols = ["case:concept:name", "concept:name", "time:timestamp"]
            export_case_dfg(df_pd[cols].copy(), out_path, export_format="svg")
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

        tabs = []
        for cls in ["low", "medium", "high"]:
            subset = df[df["case_feedback_class"] == cls]
            if subset.empty:
                tabs.append((f"{cls.title()} feedback", pn.pane.Alert(f"No cases with {cls} feedback.", alert_type="info")))
                continue
            svg = self._svg_from_df(subset, out_dir / f"{export_name}-case-dfg-{cls}.svg")
            content = _wrap_svg(svg) if svg else pn.pane.Alert(f"DFG for {cls} feedback unavailable.", alert_type="info")
            tabs.append((f"{cls.title()} feedback", content))

        return pn.Tabs(*tabs, sizing_mode="stretch_width")

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
                eto_svg = outputs["object_types"].read_text()

                # Create tabs with scrollable SVG visualizations
                tabs = pn.Tabs(
                    ("Case-Centric DFG", self._case_dfg_panel(export_name)),
                    ("Object-Centric DFG", _wrap_svg(ocdfg_svg)),
                    ("Object-Centric PN", _wrap_svg(ocpn_svg)),
                    ("Event → Object Types", _wrap_svg(eto_svg)),
                    active=0,
                    sizing_mode="stretch_width",
                )
                return tabs

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
