import json
from pathlib import Path

import panel as pn
import polars as pl

from .styling_helpers import section_header, small_kpi_card, subsection_header

FEEDBACK_STORE_PATH = Path("./feedback_store.json")

_SCENARIO_NAMES = {
    0: "Large latte & croissant",
    1: "2 espressos (hurry)",
    2: "Complaint & resolution",
    3: "Ask for recommendation",
}


class FeedbackSection:
    def __init__(self, log_path: Path):
        self._log_path = log_path
        self._data = self._load_data()

    def panel(self) -> pn.viewable.Viewable:
        if self._data is None or self._data.is_empty():
            return pn.pane.HTML(
                '<div style="font-size:11px;color:#999;padding:4px 0;">'
                "No customer feedback available for this log.</div>",
                sizing_mode="stretch_width",
            )

        return pn.Column(
            section_header("Customer Feedback"),
            self._build_kpi_row(),
            self._build_scenario_breakdown(),
            sizing_mode="stretch_width",
        )

    def _load_data(self) -> pl.DataFrame | None:
        try:
            case_ids = set(pl.read_csv(str(self._log_path))["case_id"].to_list())
        except Exception:
            return None

        if not FEEDBACK_STORE_PATH.exists():
            return None

        with open(FEEDBACK_STORE_PATH) as f:
            store = json.load(f)

        rows = [v for k, v in store.items() if k in case_ids]
        if not rows:
            return None

        return pl.DataFrame(rows, infer_schema_length=len(rows))

    def _build_kpi_row(self) -> pn.pane.HTML:
        df = self._data
        n = df.height
        avg_score = float(df["feedback_score"].mean())
        excellent = int(df.filter(pl.col("feedback_score") >= 0.75).height)
        normal = int(df.filter((pl.col("feedback_score") >= 0.25) & (pl.col("feedback_score") < 0.75)).height)
        not_satisfied = int(df.filter(pl.col("feedback_score") < 0.25).height)

        if avg_score >= 0.75:
            avg_emoji = "😊"
        elif avg_score >= 0.25:
            avg_emoji = "😐"
        else:
            avg_emoji = "😞"

        cards = [
            ("Conversations", str(n)),
            ("Avg Score", f"{avg_score:.2f} {avg_emoji}"),
            ("Excellent", str(excellent)),
            ("Normal", str(normal)),
            ("Not satisfied", str(not_satisfied)),
        ]
        cards_html = "".join(small_kpi_card(label, value) for label, value in cards)
        return pn.pane.HTML(
            f'<div style="padding:2px 0;display:flex;flex-wrap:wrap;">{cards_html}</div>',
            sizing_mode="stretch_width",
        )

    def _build_scenario_breakdown(self) -> pn.viewable.Viewable:
        df = self._data
        if "scenario_index" not in df.columns:
            return pn.pane.HTML("")

        df_with_scenario = df.filter(pl.col("scenario_index").is_not_null())
        if df_with_scenario.is_empty():
            return pn.pane.HTML("")

        agg = (
            df_with_scenario
            .group_by("scenario_index")
            .agg(
                pl.col("feedback_score").mean().alias("avg_score"),
                pl.col("feedback_score").count().alias("count"),
            )
            .sort("scenario_index")
        )

        rows_html = ""
        for row in agg.to_dicts():
            idx = int(row["scenario_index"])
            name = _SCENARIO_NAMES.get(idx, f"Scenario {idx}")
            score = row["avg_score"]
            count = row["count"]
            bar_width = int(score * 100)
            color = "#4CAF50" if score >= 0.75 else "#FF9800" if score >= 0.25 else "#F44336"
            rows_html += (
                f'<div style="margin:3px 0;">'
                f'<div style="font-size:10px;color:#555;margin-bottom:2px;">'
                f'S{idx}: {name} <span style="color:#999;">(n={count})</span></div>'
                f'<div style="display:flex;align-items:center;gap:6px;">'
                f'<div style="flex:1;background:#eee;border-radius:3px;height:10px;">'
                f'<div style="width:{bar_width}%;background:{color};height:10px;border-radius:3px;"></div>'
                f'</div>'
                f'<div style="font-size:11px;font-weight:600;color:#333;min-width:32px;">{score:.2f}</div>'
                f'</div>'
                f'</div>'
            )

        return pn.Column(
            subsection_header("Score by Scenario"),
            pn.pane.HTML(
                f'<div style="padding:2px 0;">{rows_html}</div>',
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )
