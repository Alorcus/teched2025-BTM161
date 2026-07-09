import panel as pn
import plotly.express as px
import polars as pl

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

from .styling_helpers import COLOR_SCHEME, section_header, subsection_header, subtitled_kpi_card


_DENY_COLOR = COLOR_SCHEME["dark_red"]
_FLAG_COLOR = COLOR_SCHEME["orange"]
_GUARDRAIL_TYPE_COLORS = {
    "hard": COLOR_SCHEME["dark_red"],
    "soft": COLOR_SCHEME["yellow"],
    "guardrail": COLOR_SCHEME["orange"],
    "unknown": COLOR_SCHEME["beige"],
}


def _gateway_events(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Concat gateway_deny + gateway_flag with a literal `decision` column.

    Both tables are absent (rather than empty) when the trace was loaded
    without a guardrail log, so we default missing keys to an empty frame
    with the correct schema so the concat is always safe.
    """
    deny = ocel.event_tables.get("event_gateway_deny")
    flag = ocel.event_tables.get("event_gateway_flag")
    frames = []
    if deny is not None and not deny.is_empty():
        frames.append(deny.with_columns(pl.lit("deny").alias("decision")))
    if flag is not None and not flag.is_empty():
        frames.append(flag.with_columns(pl.lit("flag").alias("decision")))
    if not frames:
        return pl.DataFrame(schema={"ocel_id": pl.Utf8, "decision": pl.Utf8})
    return pl.concat(frames, how="diagonal_relaxed")


def _long_triggers(gw: pl.DataFrame) -> pl.DataFrame:
    """Explode `denied_by` / `flagged_by` into one row per (event, guardrail, decision).

    Handles null cells, `|`-separated strings (see the join site in
    src/trace_processing/guardrail_log_loader.py), and stray whitespace. Rows
    where the source column was empty become empty guardrail-id strings and
    are filtered out.
    """
    def _one_side(col: str, decision: str) -> pl.DataFrame:
        if col not in gw.columns:
            return pl.DataFrame(
                schema={"event_id": pl.Utf8, "guardrail_id": pl.Utf8, "decision": pl.Utf8},
            )
        return (
            gw.select(
                pl.col("ocel_id").alias("event_id"),
                pl.col(col).fill_null("").str.split("|").alias("guardrail_id"),
            )
            .explode("guardrail_id")
            .with_columns(
                pl.col("guardrail_id").str.strip_chars(),
                pl.lit(decision).alias("decision"),
            )
            .filter(pl.col("guardrail_id") != "")
        )

    return pl.concat([_one_side("denied_by", "deny"), _one_side("flagged_by", "flag")])


class GuardrailSection:
    def __init__(self, ocel: ObjectCentricEventlog):
        self._ocel = ocel
        self._pane = self._build()

    def panel(self) -> pn.viewable.Viewable:
        return self._pane

    def _build(self) -> pn.Column:
        gw = _gateway_events(self._ocel)
        if gw.is_empty():
            return pn.Column(
                section_header("Guardrails"),
                pn.pane.HTML(
                    '<div style="font-size:11px;color:#999;padding:4px 0;">'
                    "No guardrail activity recorded for this log.</div>",
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            )

        triggers = _long_triggers(gw)

        column = section_header("Guardrails")
        column.append(self._kpi_row(gw, triggers))
        column.append(subsection_header("Trigger Frequency per Guardrail"))
        column.append(self._trigger_frequency_chart(triggers))
        return column

    def _kpi_row(self, gw: pl.DataFrame, triggers: pl.DataFrame) -> pn.pane.HTML:
        denies = int(gw.filter(pl.col("decision") == "deny").height)
        flags = int(gw.filter(pl.col("decision") == "flag").height)
        unique_guardrails = int(triggers["guardrail_id"].n_unique()) if triggers.height else 0
        unique_tools = (
            int(gw.filter(pl.col("tool_name").is_not_null() & (pl.col("tool_name") != ""))
                  ["tool_name"].n_unique())
            if "tool_name" in gw.columns else 0
        )

        # `object_tool_call` gets one row per `gateway_decision` record (allow,
        # flag, or deny — see guardrail_log_loader._project), so its height is
        # the count of every tool call the gateway consulted on. `total` is
        # every decision; `allows` is what's left after subtracting the emitted
        # deny/flag events.
        tool_call_tbl = self._ocel.object_tables.get("object_tool_call")
        if tool_call_tbl is None or tool_call_tbl.is_empty():
            total = denies + flags
            allows = 0
            deny_rate_str = "—"
        else:
            total = int(tool_call_tbl.height)
            allows = max(0, total - denies - flags)
            deny_rate_str = f"{denies / total:.1%}"

        cards = [
            (
                "Total evaluations",
                f"Every tool call the gateway consulted on ({allows} allowed, "
                f"{flags} flagged, {denies} denied).",
                str(total),
            ),
            (
                "Denies",
                "Tool calls the gateway hard-blocked before they could run.",
                str(denies),
            ),
            (
                "Flags",
                "Allowed tool calls that at least one guardrail voted to flag.",
                str(flags),
            ),
            (
                "Guardrails triggered",
                "Distinct guardrails that produced a deny or flag verdict.",
                str(unique_guardrails),
            ),
            (
                "Tools intercepted",
                "Distinct tools that were denied or flagged at least once.",
                str(unique_tools),
            ),
            (
                "Deny rate",
                "Share of gateway-evaluated tool calls that ended in a deny.",
                deny_rate_str,
            ),
        ]
        cards_html = "".join(
            subtitled_kpi_card(label, subtitle, value)
            for label, subtitle, value in cards
        )
        return pn.pane.HTML(
            f'<div style="padding:12px 0 2px;display:grid;grid-template-columns:repeat(6, 1fr);'
            f'gap:8px;width:100%;">{cards_html}</div>',
            sizing_mode="stretch_width",
        )

    def _trigger_frequency_chart(self, triggers: pl.DataFrame) -> pn.viewable.Viewable:
        if triggers.is_empty():
            return pn.pane.Alert("No guardrails were triggered.", alert_type="info")

        agg = (
            triggers
            .group_by(["guardrail_id", "decision"])
            .agg(pl.len().alias("count"))
        )
        n_guardrails = int(agg["guardrail_id"].n_unique())
        height = max(180, 22 * n_guardrails + 60)

        fig = px.bar(
            agg.to_pandas(),
            x="count", y="guardrail_id", color="decision", orientation="h",
            barmode="stack",
            color_discrete_map={"deny": _DENY_COLOR, "flag": _FLAG_COLOR},
            category_orders={"decision": ["deny", "flag"]},
            labels={"guardrail_id": "Guardrail", "count": "Triggers", "decision": ""},
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=180, r=10, t=25, b=25),
            height=height,
            font=dict(size=10),
            plot_bgcolor=COLOR_SCHEME["off-white"],
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        )
        return pn.pane.Plotly(fig, height=height, sizing_mode="stretch_width")