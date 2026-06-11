from pathlib import Path

import panel as pn
import polars as pl

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

from .eventlog_helpers import flat_event_table
from .styling_helpers import small_kpi_card


class OverviewSection:
    def __init__(self, ocel: ObjectCentricEventlog, log_path: Path):
        self._ocel = ocel
        self._log_path = log_path
        self._pane = pn.Column(
            self._build_header(),
            self._build_kpi_row(),
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

        cards = [
            ("Total Events", f"{self._ocel.events.height:,}"),
            ("Unique Event Types", f"{non_handover['ocel_type'].n_unique()}"),
            ("Agent Handovers", f"{handover.height:,}"),
            ("Input Tokens", f"{input_tokens:,}" if input_tokens > 0 else "—"),
            ("Response Tokens", f"{response_tokens:,}" if response_tokens > 0 else "—"),
        ]
        cards_html = "".join(small_kpi_card(label, value) for label, value in cards)
        return pn.pane.HTML(
            f'<div style="padding:2px 0;display:flex;flex-wrap:wrap;">{cards_html}</div>',
            sizing_mode="stretch_width",
        )
