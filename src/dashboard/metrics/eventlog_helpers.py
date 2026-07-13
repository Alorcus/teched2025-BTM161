import polars as pl

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog


_AGENT_OR_USER = ["order_agent", "inventory_agent", "barista_agent",
                  "customer_service_agent", "user"]
_PER_ORDER_SCHEMA = {
    "order_id": pl.Utf8,
    "full_duration_s": pl.Float64,
    "pipeline_duration_s": pl.Float64,
    "confirm_to_tray_s": pl.Float64,
}


def flat_event_table(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Union all per-type event tables into one flat DataFrame."""
    target_schema: dict[str, pl.PolarsDataType] = {
        "ocel_id": pl.Utf8,
        "ocel_time": pl.Datetime,
        "duration": pl.Float64,
        "input_tokens": pl.Float64,
        "response_tokens": pl.Float64,
        "model": pl.Utf8,
    }
    frames = []
    for tbl_name, df in ocel.event_tables.items():
        event_type = tbl_name.removeprefix("event_")
        for col, dtype in target_schema.items():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        df = df.with_columns(
            [pl.col(c).cast(t, strict=False) for c, t in target_schema.items()]
        ).with_columns(pl.lit(event_type).alias("ocel_type"))
        frames.append(df.select(["ocel_id", "ocel_type", "ocel_time",
                                 "duration", "input_tokens", "response_tokens", "model"]))
    if not frames:
        return pl.DataFrame(schema={**target_schema, "ocel_type": pl.Utf8})
    return pl.concat(frames)


def agent_event_counts(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Count events handled by each agent."""
    agent_objects = ocel.objects.filter(pl.col("ocel_type").str.contains("agent"))
    return (
        ocel.event_object
        .join(agent_objects, left_on="ocel_object_id", right_on="ocel_id", how="inner")
        .group_by("ocel_type")
        .agg(pl.len().alias("event_count"))
        .sort("event_count", descending=True)
        .rename({"ocel_type": "agent"})
    )


def handover_matrix(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Build a (source, target, count) matrix from agent-to-agent handover events."""
    handover_events = (
        ocel.events
        .filter(pl.col("ocel_type").str.contains("_handover_"))
        .unique(subset=["ocel_id"])
    )
    if handover_events.is_empty():
        return pl.DataFrame(schema={"source": str, "target": str, "count": pl.UInt32})

    def _split(s: str):
        parts = s.split("_handover_")
        return (
            parts[0].replace("_", " ").title(),
            parts[1].replace("_", " ").title(),
        ) if len(parts) == 2 else (s, "")

    pairs = [_split(t) for t in handover_events["ocel_type"].to_list()]
    return (
        pl.DataFrame({"source": [p[0] for p in pairs], "target": [p[1] for p in pairs]})
        .group_by(["source", "target"])
        .agg(pl.len().alias("count"))
    )


def per_order_durations(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Compute per-order time windows in seconds.

    Returns one row per order with columns:
      order_id, full_duration_s, pipeline_duration_s, confirm_to_tray_s
    Any column may be null when the relevant boundary event is missing
    for that order.
    """
    flat = flat_event_table(ocel)
    case_objs = ocel.objects.filter(pl.col("ocel_type").is_in(_AGENT_OR_USER))
    if case_objs.is_empty() or flat.is_empty():
        return pl.DataFrame(schema=_PER_ORDER_SCHEMA)

    suffix_re = "_(?:" + "|".join(_AGENT_OR_USER) + ")$"
    eo = (
        ocel.event_object
        .join(case_objs, left_on="ocel_object_id", right_on="ocel_id", how="inner")
        .select(
            pl.col("ocel_event_id"),
            pl.col("ocel_object_id").str.replace(suffix_re, "").alias("order_id"),
        )
        .unique()
    )
    events = flat.join(eo, left_on="ocel_id", right_on="ocel_event_id", how="inner")
    if events.is_empty():
        return pl.DataFrame(schema=_PER_ORDER_SCHEMA)

    t = pl.col("ocel_time")
    boundaries = events.group_by("order_id").agg(
        t.filter(pl.col("ocel_type") == "user_prompt").min().alias("first_user_prompt_t"),
        t.filter(pl.col("ocel_type") == "process_order").min().alias("process_order_t"),
        t.max().alias("last_event_t"),
        t.filter(pl.col("ocel_type") == "place_on_tray").max().alias("last_tray_t"),
    )

    def _seconds(end: str, start: str) -> pl.Expr:
        return (pl.col(end) - pl.col(start)).dt.total_milliseconds() / 1000.0

    return boundaries.with_columns(
        _seconds("last_event_t", "first_user_prompt_t").alias("full_duration_s"),
        _seconds("last_event_t", "process_order_t").alias("pipeline_duration_s"),
        _seconds("last_tray_t", "process_order_t").alias("confirm_to_tray_s"),
    ).select(*_PER_ORDER_SCHEMA.keys())
