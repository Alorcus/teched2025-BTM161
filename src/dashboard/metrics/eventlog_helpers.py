import polars as pl

from src.trace_processing.eventlog_conversion import ObjectCentricEventlog


_AGENT_OR_USER = ["order_agent", "inventory_agent", "barista_agent",
                  "customer_service_agent", "user"]
_AGENT_TYPES = ["order_agent", "inventory_agent", "barista_agent",
                 "customer_service_agent"]
_SUFFIX_RE = "_(?:" + "|".join(_AGENT_OR_USER) + ")$"
# Events that have no meaning for activity-level metrics excluded
PLUMBING_EVENTS = ["call_llm", "agent_response", "user_prompt", "user_feedback"]
# Unified customer-feedback classification, used by KPI cards and the
# case-centric DFG split:
#   score <  FEEDBACK_LOW   → not satisfied / low
#   FEEDBACK_LOW <= score < FEEDBACK_HIGH → normal / medium
#   score >= FEEDBACK_HIGH  → excellent / high
FEEDBACK_LOW = 0.5
FEEDBACK_HIGH = 0.8
_CASE_FEEDBACK_SCHEMA = {
    "case_id": pl.Utf8,
    "feedback_score": pl.Float64,
    "scenario_index": pl.Float64,
}
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


def event_case_map(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Map every event to the case it belongs to, via its agent/user object link.

    Agent object ids are ``<case_id>_<agent_type>`` (e.g. ``c123_order_agent``);
    the user object id is bare ``<case_id>``. Stripping the agent-type suffix
    recovers the shared case_id either way. This is the one place that
    "what case is this event part of" logic lives — every case-grouped
    metric below joins through it instead of re-deriving it.

    Returns columns: ocel_event_id, case_id, agent_type.
    Note: handover events appear twice (once per involved agent) — select
    ocel_event_id/case_id and .unique() when agent_type is not needed.
    """
    case_objs = ocel.objects.filter(pl.col("ocel_type").is_in(_AGENT_OR_USER))
    if case_objs.is_empty():
        return pl.DataFrame(schema={
            "ocel_event_id": pl.Utf8, "case_id": pl.Utf8, "agent_type": pl.Utf8,
        })
    return (
        ocel.event_object
        .join(case_objs, left_on="ocel_object_id", right_on="ocel_id", how="inner")
        .select(
            pl.col("ocel_event_id"),
            pl.col("ocel_object_id").str.replace(_SUFFIX_RE, "").alias("case_id"),
            pl.col("ocel_type").alias("agent_type"),
        )
        .unique()
    )


def case_feedback_scores(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Feedback per case: case_id | feedback_score | scenario_index."""
    tbl = ocel.event_tables.get("event_user_feedback")
    if tbl is None or "feedback_score" not in tbl.columns:
        return pl.DataFrame(schema=_CASE_FEEDBACK_SCHEMA)
    if "scenario_index" not in tbl.columns:
        tbl = tbl.with_columns(pl.lit(None, dtype=pl.Float64).alias("scenario_index"))
    return (
        tbl
        .select("ocel_id", pl.col("feedback_score").cast(pl.Float64),
                pl.col("scenario_index").cast(pl.Float64, strict=False))
        .drop_nulls(subset=["feedback_score"])
        .join(
            event_case_map(ocel).select("ocel_event_id", "case_id").unique(),
            left_on="ocel_id", right_on="ocel_event_id", how="inner",
        )
        .group_by("case_id")
        .agg(
            pl.col("feedback_score").last(),
            pl.col("scenario_index").last(),
        )
    )


def case_complexity_df(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Per-case process metrics joined with feedback.

    One row per case with columns:
      case_id, trace_length, unique_activities, n_handovers,
      max_activity_repeats, has_cs_intervention, has_brew_failure,
      has_refund, feedback_score, scenario_index
    max_activity_repeats is the highest occurrence count of any single
    tool activity within the case (plumbing/handover events excluded);
    values >= 3 usually indicate agent retry loops.
    feedback_score/scenario_index are null when no feedback exists for a case.
    """
    flat = flat_event_table(ocel)
    ecm_full = event_case_map(ocel)
    ecm = ecm_full.select("ocel_event_id", "case_id").unique()
    if flat.is_empty() or ecm.is_empty():
        return pl.DataFrame(schema={
            **_CASE_FEEDBACK_SCHEMA, "trace_length": pl.UInt32,
            "unique_activities": pl.UInt32, "n_handovers": pl.UInt32,
            "max_activity_repeats": pl.UInt32,
            "has_cs_intervention": pl.Boolean, "has_brew_failure": pl.Boolean,
            "has_refund": pl.Boolean,
        })

    events = flat.join(ecm, left_on="ocel_id", right_on="ocel_event_id", how="inner")
    per_case = events.group_by("case_id").agg(
        pl.len().alias("trace_length"),
        pl.col("ocel_type").n_unique().alias("unique_activities"),
        pl.col("ocel_type").str.contains("_handover_").sum().alias("n_handovers"),
        (pl.col("ocel_type") == "brew_failed").any().alias("has_brew_failure"),
        pl.col("ocel_type").is_in(["offer_refund", "offer_partial_refund"])
            .any().alias("has_refund"),
    )
    max_repeats = (
        events
        .filter(~pl.col("ocel_type").str.contains("_handover_"))
        .filter(~pl.col("ocel_type").is_in(PLUMBING_EVENTS))
        .group_by("case_id", "ocel_type")
        .agg(pl.len().alias("cnt"))
        .group_by("case_id")
        .agg(pl.col("cnt").max().alias("max_activity_repeats"))
    )
    per_case = per_case.join(max_repeats, on="case_id", how="left").with_columns(
        pl.col("max_activity_repeats").fill_null(0)
    )
    # CS involvement is detected via the agent object.
    cs_cases = (
        ecm_full.filter(pl.col("agent_type") == "customer_service_agent")
        .select("case_id")
        .unique()
        .with_columns(has_cs_intervention=pl.lit(True))
    )
    per_case = per_case.join(cs_cases, on="case_id", how="left").with_columns(
        pl.col("has_cs_intervention").fill_null(False)
    )
    return per_case.join(case_feedback_scores(ocel), on="case_id", how="left")


def per_order_durations(ocel: ObjectCentricEventlog) -> pl.DataFrame:
    """Compute per-order time windows in seconds.

    Returns one row per order with columns:
      order_id, full_duration_s, pipeline_duration_s, confirm_to_tray_s
    Any column may be null when the relevant boundary event is missing
    for that order.
    """
    flat = flat_event_table(ocel)
    eo = (
        event_case_map(ocel)
        .select("ocel_event_id", pl.col("case_id").alias("order_id"))
        .unique()
    )
    if eo.is_empty() or flat.is_empty():
        return pl.DataFrame(schema=_PER_ORDER_SCHEMA)
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
    receiving agent participate in, so joining through `event_case_map`
    and de-duplicating on (event, case) counts each handover exactly once.
    """
    schema = {"case_id": pl.Utf8, "handover_count": pl.UInt32}
    handover_events = (
        ocel.events
        .filter(pl.col("ocel_type").str.contains("_handover_"))
        .unique(subset=["ocel_id"])
    )
    case_ids = event_case_map(ocel).select("ocel_event_id", "case_id").unique()
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
    case_ids = event_case_map(ocel).select("ocel_event_id", "case_id").unique()
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
    case_ids = event_case_map(ocel).select("ocel_event_id", "case_id").unique()
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

