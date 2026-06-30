from datetime import datetime, timedelta
from pathlib import Path

import panel as pn
import polars as pl

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

from ..nav import header_nav
from .feedback_section import FeedbackSection
from .overview_section import OverviewSection
from .system_metrics_section import SystemMetricsSection
from .time_metrics_section import TimeMetricsSection
from .trace_cache import CACHE_FILENAME, ensure_trace_cache
from .visualization_section import VisualizationSection


_TIMESTAMP_COL = "time:timestamp"


def create_metrics_dashboard():
    """Create the Metrics Dashboard page."""
    pn.extension("plotly", sizing_mode="stretch_width")

    LOG_DIR = Path("generated_event_log")
    cache_path = ensure_trace_cache(LOG_DIR)
    if cache_path is None:
        return _empty_template(LOG_DIR)

    combined = _load_combined_eventlog([cache_path])
    if combined.is_empty() or _TIMESTAMP_COL not in combined.columns:
        return _empty_template(LOG_DIR)

    ts_series = combined[_TIMESTAMP_COL].drop_nulls()
    if ts_series.is_empty():
        return _empty_template(LOG_DIR)

    full_start: datetime = ts_series.min()
    full_end: datetime = ts_series.max()
    # Slider's start/end must differ; if a single timestamp exists, pad by 1s.
    if full_end <= full_start:
        full_end = full_start + timedelta(seconds=1)

    time_slider = pn.widgets.DatetimeRangeSlider(
        name="",
        start=full_start,
        end=full_end,
        value=(full_start, full_end),
        step=1000,  # 1 second
        format="%Y-%m-%d %H:%M:%S",
        sizing_mode="stretch_width",
        margin=(0, 0, 4, 0),
    )

    slider_label = pn.pane.HTML(
        '<div style="font-size:12px;font-weight:600;color:#444;'
        'margin-bottom:2px;">Trace Timeframe</div>',
        sizing_mode="stretch_width",
        margin=(0, 0, 0, 0),
    )

    trace_count_label = pn.pane.HTML(
        _format_count_label(*_case_counts(combined, full_start, full_end)),
        sizing_mode="stretch_width",
        margin=(0, 0, 10, 0),
    )

    metrics_pane = pn.Column(
        _render_metrics(combined, full_start, full_end),
        sizing_mode="stretch_width",
        styles={"padding": "4px 0"},
    )

    def _on_range_change(event):
        start, end = event.new
        trace_count_label.object = _format_count_label(
            *_case_counts(combined, start, end)
        )

    def _on_range_commit(event):
        start, end = event.new
        metrics_pane[:] = [_render_metrics(combined, start, end)]

    time_slider.param.watch(_on_range_change, "value")
    time_slider.param.watch(_on_range_commit, "value_throttled")

    total_cases = (
        combined["case_id"].n_unique() if "case_id" in combined.columns else 0
    )
    sidebar = pn.Column(
        pn.pane.HTML(
            '<div style="font-size:14px;font-weight:600;margin-bottom:6px;">Metrics Dashboard</div>',
            sizing_mode="stretch_width",
        ),
        pn.pane.HTML(
            f'<div style="font-size:11px;color:#999;margin-bottom:6px;">'
            f"Across {total_cases} trace(s) in MLflow store</div>",
            sizing_mode="stretch_width",
        ),
        slider_label,
        time_slider,
        trace_count_label,
        width=320,
        styles={"padding": "10px 12px 10px 16px"},
    )

    nav_tabs = header_nav(active="/metrics")

    return pn.template.FastListTemplate(
        title="Coffee Shop Agent Observatory",
        sidebar=[sidebar],
        header=[nav_tabs],
        main=[metrics_pane],
        accent_base_color="#795548",
        header_background="#4E342E",
        theme="default",
    )


def _load_combined_eventlog(csv_files: list[Path]) -> pl.DataFrame:
    """Read every CSV in `csv_files` and return a single DataFrame with a
    parsed datetime `time:timestamp` column. Files that fail to read are
    skipped silently — the dashboard renders from whatever loaded.
    """
    frames: list[pl.DataFrame] = []
    for path in csv_files:
        try:
            df = pl.read_csv(str(path), infer_schema_length=10_000)
        except Exception:
            continue
        if _TIMESTAMP_COL not in df.columns:
            continue
        frames.append(df)
    if not frames:
        return pl.DataFrame()
    combined = pl.concat(frames, how="diagonal_relaxed")
    return combined.with_columns(
        pl.col(_TIMESTAMP_COL).str.to_datetime(strict=False)
    )


def _case_bounds(eventlog: pl.DataFrame) -> pl.DataFrame:
    """Return one row per `case_id` with `first_t` / `last_t` timestamps."""
    if "case_id" not in eventlog.columns:
        return pl.DataFrame(
            schema={"case_id": pl.Utf8, "first_t": pl.Datetime, "last_t": pl.Datetime}
        )
    return (
        eventlog.drop_nulls(_TIMESTAMP_COL)
        .group_by("case_id")
        .agg(
            pl.col(_TIMESTAMP_COL).min().alias("first_t"),
            pl.col(_TIMESTAMP_COL).max().alias("last_t"),
        )
    )


def _contained_case_ids(
    eventlog: pl.DataFrame, start: datetime, end: datetime
) -> pl.Series:
    """Cases whose entire span lies inside [start, end] — first event at or
    after `start` AND last event at or before `end`. Partially-overlapping
    cases are excluded so the rendered metrics see only whole traces."""
    bounds = _case_bounds(eventlog)
    return bounds.filter(
        (pl.col("first_t") >= start) & (pl.col("last_t") <= end)
    )["case_id"]


def _filter_by_range(
    eventlog: pl.DataFrame, start: datetime, end: datetime
) -> pl.DataFrame:
    if "case_id" not in eventlog.columns:
        return eventlog.clear()
    contained = _contained_case_ids(eventlog, start, end)
    return eventlog.filter(pl.col("case_id").is_in(contained))


def _case_counts(
    eventlog: pl.DataFrame, start: datetime, end: datetime
) -> tuple[int, int]:
    """Return (contained, partial) case counts for the window. A *partial*
    case is one that overlaps the window but is not fully contained — its
    conversation started before `start` or ended after `end`. These are
    excluded from the dashboard so durations stay meaningful."""
    bounds = _case_bounds(eventlog)
    contained = bounds.filter(
        (pl.col("first_t") >= start) & (pl.col("last_t") <= end)
    ).height
    overlapping = bounds.filter(
        (pl.col("first_t") <= end) & (pl.col("last_t") >= start)
    ).height
    return contained, overlapping - contained


def _format_count_label(contained: int, partial: int) -> str:
    plural = "trace" if contained == 1 else "traces"
    excluded_line = ""
    if partial > 0:
        excluded_plural = "trace" if partial == 1 else "traces"
        excluded_line = (
            f'<div style="font-size:11px;color:#999;padding-top:2px;">'
            f"{partial} partial {excluded_plural} excluded "
            f"(started before or ended after the window)</div>"
        )
    return (
        f'<div style="padding:4px 0;">'
        f'<div style="font-size:12px;color:#444;">'
        f"<b>{contained}</b> {plural} fully in selected timeframe</div>"
        f"{excluded_line}"
        f"</div>"
    )


def _render_metrics(
    eventlog: pl.DataFrame, start: datetime, end: datetime
) -> pn.viewable.Viewable:
    filtered = _filter_by_range(eventlog, start, end)
    if filtered.is_empty():
        return pn.pane.Alert(
            "No events in the selected timeframe.", alert_type="warning"
        )
    try:
        ocel = ObjectCentricEventlog.from_eventlog(filtered)
    except Exception as e:
        return pn.pane.Alert(f"Error processing events: {e}", alert_type="danger")

    range_label = _RangeLabel(start, end, filtered.height)
    return pn.Column(
        range_label.panel(),
        OverviewSection(ocel, range_label.fake_path()).panel(),
        FeedbackSection(ocel).panel(),
        SystemMetricsSection(ocel).panel(),
        TimeMetricsSection(ocel).panel(),
        VisualizationSection(ocel).panel(),
        sizing_mode="stretch_width",
        styles={"padding": "4px 0"},
    )


class _RangeLabel:
    """Lightweight stand-in so OverviewSection (which expects a Path with
    `.name`) can render the active timeframe instead of a filename."""

    def __init__(self, start: datetime, end: datetime, event_count: int):
        self._start = start
        self._end = end
        self._event_count = event_count

    def fake_path(self) -> Path:
        # OverviewSection only reads `.name`; pass a Path whose name describes
        # the filter window so the existing header line stays informative.
        label = f"{self._start:%Y-%m-%d %H:%M:%S} → {self._end:%Y-%m-%d %H:%M:%S}"
        return Path(label)

    def panel(self) -> pn.pane.HTML:
        return pn.pane.HTML(
            f'<div style="font-size:11px;color:#666;padding:2px 0 6px 0;">'
            f"<b>Window:</b> {self._start:%Y-%m-%d %H:%M:%S} → "
            f"{self._end:%Y-%m-%d %H:%M:%S}  ·  "
            f"{self._event_count:,} events in window</div>",
            sizing_mode="stretch_width",
        )


def _empty_template(log_dir: Path) -> pn.template.FastListTemplate:
    return pn.template.FastListTemplate(
        title="Coffee Shop Metrics",
        sidebar=[
            pn.pane.Alert(
                f"No MLflow traces found. Run a conversation in the "
                f"Interaction Observatory or via `poetry run simulate` "
                f"first. (Cache directory: **{log_dir.resolve()}**, "
                f"file: `{CACHE_FILENAME}`)",
                alert_type="warning",
            )
        ],
        header=[header_nav(active="/metrics")],
        main=[],
        accent_base_color="#795548",
        header_background="#4E342E",
        theme="default",
    )
