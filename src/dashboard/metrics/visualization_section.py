"""Visualization section — process mining charts and OCEL visualizations.

Integrates pm4py-based visualizations (Object-Centric Directly-Follows Graph,
Object-Centric Petri Net, Event-to-Object Type chart) into the Observatory dashboard
using the existing Visualizer class.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import panel as pn

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog
from src.visualization.visualizer import Visualizer, VisualizationConfig

from .ui import COLOR_SCHEME, section_header, subsection_header


# Color mapping for object types in visualizations (matches coffee shop domain)
_COLOR_MAP = {
    "order_agent": "#F1AEB6",  # pastel pink
    "barista_agent": "#A9DFBF",  # pastel green
    "inventory_agent": "#FAD7A0",  # pastel orange
    "customer_service_agent": "#D7BDE2",  # pastel purple
    "user": "#D5D8DC",  # pastel grey
    "prompt": "#B0A3E4",  # pastel teal
    "response": "#F9E79F",  # pastel yellow
}


class VisualizationSection:
    """Process mining visualizations: OCDFG, OCPN, and object-type charts."""

    def __init__(self, ocel: ObjectCentricEventlog):
        self._ocel = ocel
        self._pane = self._build()

    def panel(self) -> pn.viewable.Viewable:
        return self._pane

    def _build(self) -> pn.Column:
        column = section_header("Process Visualization")
        column.append(subsection_header("Object-Centric Process Models"))
        column.append(self._visualization_tabs())
        return column

    def _visualization_tabs(self) -> pn.viewable.Viewable:
        """Create tabs for different visualization types."""
        try:
            # Validate OCEL has required data
            if self._ocel.events.is_empty() or self._ocel.objects.is_empty():
                return pn.pane.Alert(
                    "Event log is empty or has no objects. Run a conversation first.",
                    alert_type="info",
                )

            # Generate unique export name to avoid collisions
            export_name = f"observatory_{uuid.uuid4().hex[:8]}"

            # Step 1: Export OCEL to JSON
            try:
                self._ocel.export_to_json(export_name)
            except Exception as e:
                return pn.pane.Alert(
                    f"Failed to export event log: {str(e)[:100]}",
                    alert_type="warning",
                )

            # Step 2: Create visualizations using existing Visualizer
            try:
                config = VisualizationConfig(
                    ocel_path=Path("generated_ocel") / f"{export_name}.json",
                    out_dir=Path("generated_visualizations"),
                    export_format="svg",
                    color_map=_COLOR_MAP,
                )
                visualizer = Visualizer(config)
                outputs = visualizer.run()

                # Read SVG files
                ocdfg_svg = outputs["oc_dfg"].read_text()
                ocpn_svg = outputs["oc_pn"].read_text()
                eto_svg = outputs["object_types"].read_text()

                # Create scrollable containers for each visualization
                def _wrap_svg(svg_text: str) -> pn.viewable.Viewable:
                    """Wrap SVG in a scrollable container."""
                    return pn.pane.HTML(
                        svg_text,
                        sizing_mode="stretch_width",
                        styles={
                            "overflow": "auto",
                            "max-height": "600px",
                            "border": "1px solid #e0e0e0",
                            "border-radius": "4px",
                            "padding": "8px",
                        },
                    )

                # Create tabs with scrollable SVG visualizations
                tabs = pn.Tabs(
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
