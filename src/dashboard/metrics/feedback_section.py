import panel as pn
import polars as pl

from src.agents import CUSTOMER_SCENARIO_LABELS
from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

from .eventlog_helpers import FEEDBACK_HIGH, FEEDBACK_LOW
from .styling_helpers import kpi_row, section_header, subsection_header, subtitled_kpi_card

_SCENARIO_NAMES = {i: label for i, label in enumerate(CUSTOMER_SCENARIO_LABELS)}


class FeedbackSection:
    def __init__(self, ocel: ObjectCentricEventlog):
        self._ocel = ocel
        self._data = self._load_data()

    def panel(self) -> pn.viewable.Viewable:
        if self._data is None or self._data.is_empty():
            return pn.pane.HTML(
                '<div style="font-size:11px;color:#999;padding:4px 0;">'
                "No customer feedback available for this log.</div>",
                sizing_mode="stretch_width",
            )

        return pn.Column(
            section_header("Feedback Metrics"),
            self._build_kpi_row(),
            self._build_scenario_breakdown(),
            sizing_mode="stretch_width",
        )

    def _load_data(self) -> pl.DataFrame | None:
        # Feedback is stored in event_tables under event_user_feedback
        if hasattr(self._ocel, "event_tables"):
            if "event_user_feedback" in self._ocel.event_tables:
                df = self._ocel.event_tables["event_user_feedback"]
                if df is not None and not df.is_empty():
                    if "feedback_score" in df.columns:
                        df = df.with_columns(pl.col("feedback_score").cast(pl.Float64))
                    return df

        return None

    def _build_kpi_row(self) -> pn.pane.HTML:
        df = self._data
        n = df.height
        avg_score = float(df["feedback_score"].mean())
        excellent = int(df.filter(pl.col("feedback_score") >= FEEDBACK_HIGH).height)
        normal = int(
            df.filter(
                (pl.col("feedback_score") >= FEEDBACK_LOW)
                & (pl.col("feedback_score") < FEEDBACK_HIGH)
            ).height
        )
        not_satisfied = int(df.filter(pl.col("feedback_score") < FEEDBACK_LOW).height)

        if avg_score >= FEEDBACK_HIGH:
            avg_emoji = "😊"
        elif avg_score >= FEEDBACK_LOW:
            avg_emoji = "😐"
        else:
            avg_emoji = "😞"

        cards = [
            (
                "Conversations",
                "Conversations in this window that produced a customer feedback score.",
                str(n),
            ),
            (
                "Avg Score",
                "Mean feedback score across all conversations (0 = worst, 1 = best).",
                f"{avg_score:.2f} {avg_emoji}",
            ),
            (
                "Excellent",
                f"Conversations that scored ≥ {FEEDBACK_HIGH} on the feedback scale.",
                str(excellent),
            ),
            (
                "Normal",
                f"Conversations that scored between {FEEDBACK_LOW} and {FEEDBACK_HIGH}.",
                str(normal),
            ),
            (
                "Not satisfied",
                f"Conversations that scored below {FEEDBACK_LOW} — clearly unhappy customers.",
                str(not_satisfied),
            ),
        ]
        cards_html = "".join(
            subtitled_kpi_card(title, subtitle, value)
            for title, subtitle, value in cards
        )
        return kpi_row(cards_html, columns=5, top_padding=12)

    def _build_scenario_breakdown(self) -> pn.viewable.Viewable:
        df = self._data
        if "scenario_index" not in df.columns:
            return pn.pane.HTML("")

        df_with_scenario = df.filter(pl.col("scenario_index").is_not_null())
        if df_with_scenario.is_empty():
            return pn.pane.HTML("")

        agg = (
            df_with_scenario.group_by("scenario_index")
            .agg(
                pl.col("feedback_score").mean().alias("avg_score"),
                pl.col("feedback_score").count().alias("count"),
            )
            .sort("scenario_index")
        )

        rows_html = ""
        for row in agg.to_dicts():
            idx = int(float(row["scenario_index"]))
            name = _SCENARIO_NAMES.get(idx, f"Scenario {idx}")
            score = row["avg_score"]
            count = row["count"]
            bar_width = int(score * 100)
            color = (
                "#4CAF50"
                if score >= FEEDBACK_HIGH
                else "#FF9800"
                if score >= FEEDBACK_LOW
                else "#F44336"
            )
            rows_html += (
                f'<div style="margin:3px 0;">'
                f'<div style="font-size:10px;color:#555;margin-bottom:2px;">'
                f'S{idx}: {name} <span style="color:#999;">(n={count})</span></div>'
                f'<div style="display:flex;align-items:center;gap:6px;">'
                f'<div style="flex:1;background:#eee;border-radius:3px;height:10px;">'
                f'<div style="width:{bar_width}%;background:{color};height:10px;border-radius:3px;"></div>'
                f"</div>"
                f'<div style="font-size:11px;font-weight:600;color:#333;min-width:32px;">{score:.2f}</div>'
                f"</div>"
                f"</div>"
            )

        return pn.Column(
            subsection_header("Score by Scenario"),
            pn.pane.HTML(
                f'<div style="padding:2px 0;">{rows_html}</div>',
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )
