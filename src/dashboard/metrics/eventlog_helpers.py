import polars as pl

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog


_AGENT_OR_USER = ["order_agent", "inventory_agent", "barista_agent",
                  "customer_service_agent", "user"]
_AGENT_TYPES = ["order_agent", "inventory_agent", "barista_agent",
                 "customer_service_agent"]
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


def _case_object_ids(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Map every event to the case it belongs to, via its agent/user object link.
 
    Agent object ids are ``<case_id>_<agent_type>`` (e.g. ``c123_order_agent``);
    the user object id is bare ``<case_id>``. Stripping the agent-type suffix
    recovers the shared case_id either way. This is the one place that
    "what case is this event part of" logic lives — every case-grouped
    metric below joins through it instead of re-deriving it.
    """
    case_objs = ocel.objects.filter(pl.col("ocel_type").is_in(_AGENT_OR_USER))
    if case_objs.is_empty():
        return pl.DataFrame(schema={"ocel_event_id": pl.Utf8, "case_id": pl.Utf8})
    suffix_re = "_(?:" + "|".join(_AGENT_OR_USER) + ")$"
    return (
        ocel.event_object
        .join(case_objs, left_on="ocel_object_id", right_on="ocel_id", how="inner")
        .select(
            pl.col("ocel_event_id"),
            pl.col("ocel_object_id").str.replace(suffix_re, "").alias("case_id"),
        )
        .unique()
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


def agents_per_case(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Distinct agent-object count per case.
 
    Returns one row per case_id with `agent_count` = how many distinct
    agent objects (order/barista/inventory/customer_service) touched it.
    A case resolved end-to-end by a single agent has agent_count == 1;
    higher counts mean the case bounced between agents.
    """
    agent_objs = ocel.objects.filter(pl.col("ocel_type").is_in(_AGENT_TYPES))
    if agent_objs.is_empty():
        return pl.DataFrame(schema={"case_id": pl.Utf8, "agent_count": pl.UInt32})
    suffix_re = "_(?:" + "|".join(_AGENT_TYPES) + ")$"
    return (
        agent_objs
        .with_columns(pl.col("ocel_id").str.replace(suffix_re, "").alias("case_id"))
        .group_by("case_id")
        .agg(pl.col("ocel_type").n_unique().alias("agent_count"))
    )
 
 
def handover_counts_per_case(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Number of agent-to-agent handovers per case, descending.
 
    Each handover is a single OCEL event that both the departing and the
    receiving agent participate in, so joining through `_case_object_ids`
    and de-duplicating on (event, case) counts each handover exactly once.
    """
    schema = {"case_id": pl.Utf8, "handover_count": pl.UInt32}
    handover_events = (
        ocel.events
        .filter(pl.col("ocel_type").str.contains("_handover_"))
        .unique(subset=["ocel_id"])
    )
    case_ids = _case_object_ids(ocel)
    if handover_events.is_empty() or case_ids.is_empty():
        return pl.DataFrame(schema=schema)
    return (
        handover_events
        .join(case_ids, left_on="ocel_id", right_on="ocel_event_id", how="inner")
        .group_by("case_id")
        .agg(pl.len().alias("handover_count"))
        .sort("handover_count", descending=True)
    )
 
 
def activity_divergence(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Average number of times each activity repeats within a single case.
 
    Only activities that repeat on average (avg_per_case > 1) are returned —
    an activity that fires exactly once per case isn't "diverging", so it's
    filtered out rather than cluttering the ranking. Handover pseudo-events
    are excluded since they're really agent-transition markers, not
    repeatable work items.
    """
    schema = {"ocel_type": pl.Utf8, "avg_per_case": pl.Float64, "max_per_case": pl.UInt32}
    flat = flat_event_table(ocel)
    non_handover = flat.filter(~pl.col("ocel_type").str.contains("_handover_"))
    case_ids = _case_object_ids(ocel)
    if non_handover.is_empty() or case_ids.is_empty():
        return pl.DataFrame(schema=schema)
 
    per_case_counts = (
        non_handover
        .join(case_ids, left_on="ocel_id", right_on="ocel_event_id", how="inner")
        .group_by(["ocel_type", "case_id"])
        .agg(pl.len().alias("n"))
    )
    return (
        per_case_counts.group_by("ocel_type")
        .agg(
            pl.mean("n").alias("avg_per_case"),
            pl.max("n").alias("max_per_case"),
        )
        .filter(pl.col("avg_per_case") > 1.0)
        .sort("avg_per_case", descending=True)
    )
 
 
def tool_call_fanout_per_case(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Per-case tool_call counts, plus how many were flagged/denied by the gateway.
 
    tool_call objects don't encode a case_id in their id (they're arbitrary
    gateway-assigned uuids), so the case is recovered via the "executes"
    event that links a tool_call to the agent who ran it. gateway_flag /
    gateway_deny events are then matched to the same tool_call objects to
    get per-case friction counts.
    """
    schema = {"case_id": pl.Utf8, "tool_call_count": pl.UInt32,
              "flagged_count": pl.UInt32, "denied_count": pl.UInt32}
    case_ids = _case_object_ids(ocel)
    if case_ids.is_empty():
        return pl.DataFrame(schema=schema)
 
    tool_call_cases = (
        ocel.event_object
        .filter(pl.col("ocel_qualifier") == "executes")
        .join(case_ids, on="ocel_event_id", how="inner")
        .select(pl.col("ocel_object_id").alias("tool_call_id"), "case_id")
        .unique()
    )
    if tool_call_cases.is_empty():
        return pl.DataFrame(schema=schema)
 
    tool_call_counts = tool_call_cases.group_by("case_id").agg(
        pl.col("tool_call_id").n_unique().alias("tool_call_count")
    )
 
    def _gateway_case_counts(event_type: str, col_name: str) -> pl.DataFrame:
        gw_event_ids = ocel.events.filter(pl.col("ocel_type") == event_type)["ocel_id"]
        if gw_event_ids.is_empty():
            return pl.DataFrame(schema={"case_id": pl.Utf8, col_name: pl.UInt32})
        return (
            ocel.event_object
            .filter(pl.col("ocel_event_id").is_in(gw_event_ids))
            .join(tool_call_cases, left_on="ocel_object_id", right_on="tool_call_id", how="inner")
            .unique(subset=["ocel_event_id", "case_id"])
            .group_by("case_id")
            .agg(pl.len().alias(col_name))
        )
 
    flagged = _gateway_case_counts("gateway_flag", "flagged_count")
    denied = _gateway_case_counts("gateway_deny", "denied_count")
 
    return (
        tool_call_counts
        .join(flagged, on="case_id", how="left")
        .join(denied, on="case_id", how="left")
        .with_columns(
            pl.col("flagged_count").fill_null(0),
            pl.col("denied_count").fill_null(0),
        )
        .sort("tool_call_count", descending=True)
    )
 