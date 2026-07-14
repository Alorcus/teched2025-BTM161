import panel as pn
import plotly.express as px
import polars as pl

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

from .eventlog_helpers import (
    PLUMBING_EVENTS,
    case_complexity_df,
    event_case_map,
    flat_event_table,
)
from .feedback_section import _SCENARIO_NAMES
from .styling_helpers import COLOR_SCHEME, section_header, subsection_header

# Categorical color palette
_WITH_COLOR = "#3E7CB1"
_WITHOUT_COLOR = "#A7AEB5"
_SCENARIO_COLOR_MAP = {
    _SCENARIO_NAMES[0]: "#B3541E",  # Large latte & croissant
    _SCENARIO_NAMES[1]: "#3E7CB1",  # 2 espressos (hurry)
    _SCENARIO_NAMES[2]: "#8E3A6E",  # Complaint & resolution
    _SCENARIO_NAMES[3]: "#4E9A43",  # Ask for recommendation
    "Unknown scenario": "#8A8A8A",
}

_FLAG_LABELS = {
    "has_brew_failure": "Brew failure",
    "has_cs_intervention": "CS intervention",
    "has_refund": "Refund",
}
# Below this min cases threshold an avg comparison is mostly noise.
_MIN_CASES = 5
# Rank correlation needs more points than an average to be meaningful.
_MIN_CASES_FOR_CORR = 10
# ≤1 includes tool-free cases (max_repeats 0); upper bucket is open-ended so
# high-repeat retry loops are never silently dropped from the chart.
# 2-3 are collapsed together since they have very similar score in test samples, and makes the chart cleaner
_REPEAT_BUCKETS = [(0, 1, "≤1"), (2, 3, "2–3"), (4, 4, "4"), (5, 999, "5+")]


class ComplexitySection:
    def __init__(self, ocel: ObjectCentricEventlog):
        self._ocel = ocel
        self._df = case_complexity_df(ocel)
        self._fb = self._df.filter(pl.col("feedback_score").is_not_null())

    def panel(self) -> pn.viewable.Viewable:
        if self._fb.is_empty():
            return pn.pane.HTML(
                '<div style="font-size:11px;color:#999;padding:4px 0;">'
                "No feedback scores available — complexity correlation needs "
                "cases with customer feedback.</div>",
                sizing_mode="stretch_width",
            )
        return pn.Column(
            section_header("Feedback × Process Complexity"),
            pn.Row(
                self._flag_impact_chart(),
                self._repeats_impact_chart(),
                sizing_mode="stretch_width",
            ),
            self._activity_impact_table(),
            self._trace_length_scatter(),
            sizing_mode="stretch_width",
        )

    # Chart Avg Feedback by Process Marker 
    def _flag_impact_chart(self) -> pn.viewable.Viewable:
        rows = []
        for flag, label in _FLAG_LABELS.items():
            for present, group in ((True, "Cases with marker"), (False, "Cases without")):
                sub = self._fb.filter(pl.col(flag) == present)
                if sub.is_empty():
                    continue
                avg = float(sub["feedback_score"].mean())
                rows.append({
                    "marker": label, "group": group,
                    "avg": avg, "n": sub.height,
                    "text": f"{avg:.2f} (n={sub.height})",
                })
        if not rows:
            return pn.pane.HTML("")

        fig = px.bar(
            rows,
            x="marker", y="avg", color="group",
            barmode="group", text="text",
            color_discrete_map={
                "Cases with marker": _WITH_COLOR,
                "Cases without": _WITHOUT_COLOR,
            },
            labels={"marker": "", "avg": "Avg feedback score", "group": ""},
        )
        fig.update_traces(textposition="outside", width=0.3, cliponaxis=False)
        fig.update_layout(
            margin=dict(l=30, r=10, t=15, b=25),
            height=220,
            font=dict(size=10),
            plot_bgcolor=COLOR_SCHEME["off-white"],
            yaxis=dict(range=[0, 1.12], tickvals=[0, 0.25, 0.5, 0.75, 1.0]),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        )
        return pn.Column(
            subsection_header(
                f"Avg Feedback by Process Marker (n={self._fb.height} cases)"
            ),
            pn.pane.HTML(
                '<div style="font-size:10px;color:#999;margin-bottom:2px;">'
                "Avg score in cases with vs without the marker — "
                "a wide gap flags the marker as a sign of unhappy customers.</div>",
                sizing_mode="stretch_width",
            ),
            pn.pane.Plotly(fig, height=220, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    # Chart Avg Feedback by Activity Repetition
    def _repeats_impact_chart(self) -> pn.viewable.Viewable:
        """Avg feedback bucketed by max repeats of a single activity per case"""
        rows = []
        for lo, hi, label in _REPEAT_BUCKETS:
            sub = self._fb.filter(pl.col("max_activity_repeats").is_between(lo, hi))
            if sub.is_empty():
                continue
            avg = float(sub["feedback_score"].mean())
            rows.append({
                "bucket": label, "avg": avg, "n": sub.height,
                "text": f"{avg:.2f} (n={sub.height})",
            })
        if len(rows) < 2:
            return pn.pane.HTML("")

        fig = px.bar(
            rows,
            x="bucket", y="avg", text="text",
            color_discrete_sequence=['#B3541E'],
            labels={
                "bucket": "Max repeats of one activity in a case",
                "avg": "Avg feedback score",
            },
        )
        fig.update_traces(textposition="outside", width=0.45, cliponaxis=False)
        fig.update_layout(
            showlegend=False,
            margin=dict(l=30, r=10, t=15, b=35),
            height=220,
            font=dict(size=10),
            plot_bgcolor=COLOR_SCHEME["off-white"],
            yaxis=dict(range=[0, 1.12], tickvals=[0, 0.25, 0.5, 0.75, 1.0]),
            xaxis=dict(type="category"),
        )
        return pn.Column(
            subsection_header("Avg Feedback by Activity Repetition"),
            pn.pane.HTML(
                '<div style="font-size:10px;color:#999;margin-bottom:2px;">'
                "How often the most-repeated single activity occurred within a "
                "case — repeats &ge;3 usually mean agent retry loops.</div>",
                sizing_mode="stretch_width",
            ),
            pn.pane.Plotly(fig, height=220, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    # data for table Activity Impact on Feedback
    def _activity_impact_stats(self) -> pl.DataFrame:
        """Per activity: avg feedback in cases containing it vs cases without."""
        flat = flat_event_table(self._ocel)
        ecm = event_case_map(self._ocel).select("ocel_event_id", "case_id").unique()
        fb = self._fb.select("case_id", "feedback_score")

        act_counts = (
            flat.join(ecm, left_on="ocel_id", right_on="ocel_event_id", how="inner")
            .filter(~pl.col("ocel_type").str.contains("_handover_"))
            .filter(~pl.col("ocel_type").is_in(PLUMBING_EVENTS))
            .group_by(pl.col("ocel_type").alias("activity"), "case_id")
            .agg(pl.len().alias("cnt"))
            .join(fb, on="case_id", how="inner")
        )
        if act_counts.is_empty():
            return act_counts

        n_total = fb.height
        sum_total = float(fb["feedback_score"].sum())
        return (
            act_counts.group_by("activity")
            .agg(
                pl.len().alias("n_with"),
                pl.col("feedback_score").sum().alias("sum_with"),
                # Does feedback drop as the activity repeats more often?
                # Rank correlation over cases containing the activity; 
                # only meaningful with enough cases (_MIN_CASES_FOR_CORR) 
                # and some count variatio across different cases.
                pl.when(
                    (pl.len() >= _MIN_CASES_FOR_CORR)
                    & (pl.col("cnt").n_unique() > 1)
                )
                .then(pl.corr(pl.col("cnt").rank(), pl.col("feedback_score").rank()))
                .otherwise(None)
                .alias("repeat_corr"),
            )
            .with_columns(n_without=n_total - pl.col("n_with"))
            .filter((pl.col("n_with") >= _MIN_CASES) & (pl.col("n_without") >= _MIN_CASES))
            .with_columns(
                avg_with=pl.col("sum_with") / pl.col("n_with"),
                avg_without=(sum_total - pl.col("sum_with")) / pl.col("n_without"),
            )
            .with_columns(delta=pl.col("avg_with") - pl.col("avg_without"))
            .sort("delta")
        )

    def _activity_impact_table(self) -> pn.viewable.Viewable:
        stats = self._activity_impact_stats()
        if stats.is_empty():
            return pn.pane.HTML(
                '<div style="font-size:11px;color:#999;padding:4px 0;">'
                f"No activity occurs in at least {_MIN_CASES} cases on both "
                "sides — not enough data for activity impact.</div>",
                sizing_mode="stretch_width",
            )

        rows_html = ""
        for row in stats.to_dicts():
            delta = row["delta"]
            color = "#8D0209" if delta < -0.05 else "#2E7D32" if delta > 0.05 else "#666"
            corr = row["repeat_corr"]
            if corr is None:
                corr_html = '<span style="color:#bbb;">—</span>'
            else:
                corr_color = (
                    "#8D0209" if corr < -0.2 else "#2E7D32" if corr > 0.2 else "#666"
                )
                corr_html = f'<span style="color:{corr_color};">{corr:+.2f}</span>'
            rows_html += (
                f"<tr>"
                f'<td style="padding:3px 10px 3px 0;">{row["activity"]}</td>'
                f'<td style="padding:3px 10px;text-align:right;color:#666;">{row["n_with"]}</td>'
                f'<td style="padding:3px 10px;text-align:right;">{row["avg_with"]:.2f}</td>'
                f'<td style="padding:3px 10px;text-align:right;">{row["avg_without"]:.2f}</td>'
                f'<td style="padding:3px 10px;text-align:right;'
                f'font-weight:600;color:{color};">{delta:+.2f}</td>'
                f'<td style="padding:3px 0 3px 10px;text-align:right;">{corr_html}</td>'
                f"</tr>"
            )
        table_html = (
            '<table style="font-size:11px;border-collapse:collapse;color:#333;">'
            "<thead><tr style='color:#999;text-align:left;'>"
            '<th style="padding:3px 10px 3px 0;font-weight:600;">Activity</th>'
            '<th style="padding:3px 10px;text-align:right;font-weight:600;">Cases</th>'
            '<th style="padding:3px 10px;text-align:right;font-weight:600;">Avg with</th>'
            '<th style="padding:3px 10px;text-align:right;font-weight:600;">Avg without</th>'
            '<th style="padding:3px 10px;text-align:right;font-weight:600;">&Delta;</th>'
            '<th style="padding:3px 0 3px 10px;text-align:right;font-weight:600;">Repeat corr</th>'
            f"</tr></thead><tbody>{rows_html}</tbody></table>"
        )
        return pn.Column(
            subsection_header("Activity Impact on Feedback"),
            pn.pane.HTML(
                '<div style="font-size:10px;color:#999;margin-bottom:4px;">'
                "Avg feedback in cases where the activity occurs vs cases where it "
                f"does not (activities present in &ge;{_MIN_CASES} cases on both sides). "
                "Negative &Delta; marks activities associated with unhappy customers — "
                "association, not causation. Repeat corr: rank correlation between "
                "how often the activity repeats within a case and the feedback score "
                f"(shown for &ge;{_MIN_CASES_FOR_CORR} cases with varying counts); "
                "negative means repeats hurt.</div>"
                f'<div style="overflow-x:auto;">{table_html}</div>',
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )

    # Chart Feedback vs Trace Length
    def _trace_length_scatter(self) -> pn.viewable.Viewable:
        df = self._fb.with_columns(
            scenario=pl.col("scenario_index")
            .cast(pl.Int64, strict=False)
            .map_elements(
                lambda i: _SCENARIO_NAMES.get(i, f"Scenario {i}"),
                return_dtype=pl.Utf8,
            )
            .fill_null("Unknown scenario")
        ).to_pandas()

        # Fall back to gray for scenario labels outside the fixed map
        color_map = {
            s: _SCENARIO_COLOR_MAP.get(s, "#8A8A8A")
            for s in df["scenario"].unique()
        }
        fig = px.scatter(
            df,
            x="trace_length", y="feedback_score",
            color="scenario",
            color_discrete_map=color_map,
            hover_data={
                "case_id": True,
                "max_activity_repeats": True,
                "scenario": False,
            },
            labels={
                "trace_length": "Events per case",
                "feedback_score": "Feedback score",
                "scenario": "",
            },
        )
        # opacity makes exact-overlap stacks read darker than single points
        fig.update_traces(
            marker=dict(size=9, opacity=0.55, line=dict(width=1.5, color="white"))
        )
        fig.update_layout(
            margin=dict(l=30, r=10, t=15, b=30),
            height=260,
            font=dict(size=10),
            plot_bgcolor=COLOR_SCHEME["off-white"],
            yaxis=dict(range=[-0.05, 1.05]),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        )
        return pn.Column(
            subsection_header(
                f"Feedback vs Trace Length (n={self._fb.height} cases)"
            ),
            pn.pane.HTML(
                '<div style="font-size:10px;color:#999;margin-bottom:2px;">'
                "Trace length largely reflects the scenario — compare points "
                "within one color, not across colors.</div>",
                sizing_mode="stretch_width",
            ),
            pn.pane.Plotly(fig, height=260, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )
