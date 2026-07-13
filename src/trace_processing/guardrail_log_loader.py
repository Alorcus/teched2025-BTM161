"""Read `guardrail_log/events.jsonl` and project it into OCEL-shaped tables.

The control-plane gateway (`src/control_plane/gateway.py`) writes one JSONL
line per tool-call evaluation. Each line records which guardrails fired,
what they decided, and the `setup_name` / `snapshot_id` the run was using.
This module turns that raw record stream into:

- synthetic event rows for `gateway_flag` / `gateway_deny` (gateway_allow is
  intentionally NOT emitted — it'd double the event count without signal),
- new object rows for `guardrail`, `setup`, `snapshot`, and `tool_call`,
- E2O rows linking each gateway event to its agent, setup, snapshot,
  consulted guardrails, and tool_call,
- O2O rows linking each snapshot to its agent,
- two backfill maps so existing events can be tagged with their case's
  setup/snapshot.

The loader is forgiving: missing file, empty file, or malformed lines all
resolve to "no extension", letting the OCEL build proceed unchanged.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger("coffee_shop.trace_processing.guardrail_log_loader")


# Event types this loader emits. Listed here so the OCEL converter can
# register matching attribute schemas without importing internal helpers.
GATEWAY_EVENT_TYPES = ("gateway_flag", "gateway_deny")
GATEWAY_OBJECT_TYPES = ("guardrail", "setup", "snapshot", "tool_call")


@dataclass
class GuardrailOcelExtension:
    """OCEL fragments derived from the guardrail log, ready to merge.

    All fields are empty DataFrames / dicts when the log is missing or
    contained no usable rows — callers can unconditionally pass them through
    a `pl.concat` without guarding for emptiness.
    """

    # Per-event-type tables, keyed by raw event type (without the "event_"
    # prefix). Schema matches what `from_eventlog` builds for native types:
    # one ocel_id column + the attributes from EVENT_ATTRIBUTES, including
    # ocel_time. Empty if no gateway_flag / gateway_deny rows were produced.
    event_tables: dict[str, pl.DataFrame] = field(default_factory=dict)

    # Per-object-type tables, keyed by raw object type. Schema: ocel_id + the
    # attributes from OBJECT_ATTRIBUTES for that type.
    object_tables: dict[str, pl.DataFrame] = field(default_factory=dict)

    # Rows to append to the converter's `events` DataFrame
    # (cols: ocel_id, ocel_type).
    events_rows: pl.DataFrame = field(default_factory=lambda: _empty_df({
        "ocel_id": pl.Utf8, "ocel_type": pl.Utf8,
    }))

    # Rows to append to the converter's `objects` DataFrame
    # (cols: ocel_id, ocel_type).
    objects_rows: pl.DataFrame = field(default_factory=lambda: _empty_df({
        "ocel_id": pl.Utf8, "ocel_type": pl.Utf8,
    }))

    # Rows to append to `event_object`
    # (cols: ocel_event_id, ocel_object_id, ocel_qualifier).
    event_object_rows: pl.DataFrame = field(default_factory=lambda: _empty_df({
        "ocel_event_id": pl.Utf8,
        "ocel_object_id": pl.Utf8,
        "ocel_qualifier": pl.Utf8,
    }))

    # Rows for `object_object`
    # (cols: ocel_source_id, ocel_target_id, ocel_qualifier).
    object_object_rows: pl.DataFrame = field(default_factory=lambda: _empty_df({
        "ocel_source_id": pl.Utf8,
        "ocel_target_id": pl.Utf8,
        "ocel_qualifier": pl.Utf8,
    }))

    # Backfill map: case_id (== thread_id) → setup_name. Used by the converter
    # to attach a `setup` link to every pre-existing event in the case.
    case_setup_map: dict[str, str] = field(default_factory=dict)

    # Backfill map: (case_id, agent_id) → snapshot_id. Used by the converter
    # to attach a `snapshot` link to every pre-existing event whose
    # `(case_id, org:resource)` pair shows up in the guardrail log.
    case_agent_snapshot_map: dict[tuple[str, str], str] = field(default_factory=dict)

    # Mapping tool_call_id → tool_call object_id (== tool_call_id). Exposed so
    # the converter can link MLflow tool events to the same tool_call objects
    # via their `tool_call_id` column.
    tool_call_ids: set[str] = field(default_factory=set)

    @property
    def is_empty(self) -> bool:
        return (
            self.events_rows.is_empty()
            and self.objects_rows.is_empty()
            and self.event_object_rows.is_empty()
            and self.object_object_rows.is_empty()
            and not self.case_setup_map
            and not self.case_agent_snapshot_map
        )


def _empty_df(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def load_guardrail_events(path: str | Path) -> GuardrailOcelExtension:
    """Read a guardrail JSONL log and return polars-ready OCEL fragments.

    Missing/empty/malformed-line cases all resolve to an empty extension that
    the converter can pass through as a no-op.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return GuardrailOcelExtension()

    records: list[dict[str, Any]] = []
    bad_lines = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
    if bad_lines:
        logger.warning("guardrail_log: skipped %d malformed line(s) in %s", bad_lines, p)
    if not records:
        return GuardrailOcelExtension()

    # Only gateway_decision rows are meaningful here.
    decisions = [r for r in records if r.get("event_type") == "gateway_decision"]
    if not decisions:
        return GuardrailOcelExtension()

    return project_decisions(decisions)


def load_guardrail_events_from_eventlog(
    eventlog: pl.DataFrame,
) -> GuardrailOcelExtension:
    """Rebuild the OCEL extension from `gateway_decision` rows embedded in
    the flat event log CSV.

    Companion to `load_guardrail_events` — that one reads
    `guardrail_log/events.jsonl` on disk; this one reads the same records
    after they've been folded into `_all_traces.csv`. The two paths converge
    on `project_decisions` so the resulting `GuardrailOcelExtension` is
    identical shape either way.

    Why this exists: `_all_traces.csv` is meant to be shared with users who
    don't have the source JSONL on their machine. Every signal the dashboard
    reads must therefore round-trip through the CSV.

    Timezone contract: the caller MUST pass a frame where `time:timestamp`
    is naive-UTC — either the raw ISO string `_load_gateway_rows` writes,
    or a Datetime that has NOT been through timezone conversion. Any naive
    datetime is tagged as UTC before `.timestamp()` is called; if the caller
    silently handed over a naive-LOCAL Datetime (e.g. after
    `dt.replace_time_zone("UTC").dt.convert_time_zone(local_tz).dt.replace_time_zone(None)`),
    each gateway event's epoch would be shifted by the local UTC offset — a
    silent 2-hour skew in Europe/Berlin summer, invisible on a UTC host.

    Callers that can't guarantee the naive-UTC shape (in practice: the
    dashboard, whose `_load_combined_eventlog` shifts to naive-LOCAL for
    display) MUST route through
    `src.trace_processing.eventlog_conversion._resolve_guardrail_extension`.
    That helper knows to look for a sibling `time:timestamp_utc_naive`
    column — which the dashboard preserves alongside the converted one —
    or to fall back to the on-disk JSONL, whichever avoids the shift.

    Row contract (produced by TraceProcessor.extract_new_traces):
    - `concept:name == "gateway_decision"`
    - `time:timestamp` — ISO-8601 naive-UTC string (with millisecond precision)
    - `case_id` — thread_id
    - `org:resource` — agent_id
    - `gateway_setup_name`, `gateway_snapshot_id`, `gateway_tool_name`,
      `gateway_tool_call_id`, `gateway_final_decision` — scalars
    - `gateway_tool_args_json`, `gateway_verdicts_json` — JSON-encoded
      strings (verdicts is a list, tool_args is a dict)
    """
    if eventlog.is_empty() or "concept:name" not in eventlog.columns:
        return GuardrailOcelExtension()
    rows = eventlog.filter(pl.col("concept:name") == "gateway_decision")
    if rows.is_empty():
        return GuardrailOcelExtension()

    decisions: list[dict[str, Any]] = []
    for r in rows.iter_rows(named=True):
        raw_verdicts = r.get("gateway_verdicts_json") or "[]"
        raw_args = r.get("gateway_tool_args_json") or "{}"
        try:
            verdicts = json.loads(raw_verdicts)
        except (TypeError, ValueError):
            verdicts = []
        try:
            tool_args = json.loads(raw_args)
        except (TypeError, ValueError):
            tool_args = {}

        # `ts` is what project_decisions expects; the CSV carries the human-
        # readable string in time:timestamp. Convert back to an epoch float
        # so the shared projection path treats it identically to JSONL input.
        ts_raw = r.get("time:timestamp")
        ts_epoch: float | None
        if ts_raw is None:
            ts_epoch = None
        else:
            try:
                # Accept both string ("2026-07-12T10:00:00.000") and datetime
                # (when the loader has already parsed the column) shapes. The
                # CSV column is naive-UTC (see _load_gateway_rows); calling
                # `.timestamp()` on a naive datetime otherwise assumes LOCAL
                # and would invert the fix at the producer side. Tag as UTC
                # before converting back to an epoch float.
                if isinstance(ts_raw, str):
                    parsed = datetime.strptime(ts_raw, "%Y-%m-%dT%H:%M:%S.%f")
                else:
                    parsed = ts_raw
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                ts_epoch = parsed.timestamp()
            except (TypeError, ValueError):
                ts_epoch = None

        decisions.append({
            "event_type": "gateway_decision",
            "ts": ts_epoch,
            "thread_id": r.get("case_id"),
            "agent_id": r.get("org:resource"),
            "setup_name": r.get("gateway_setup_name"),
            "snapshot_id": r.get("gateway_snapshot_id"),
            "tool_name": r.get("gateway_tool_name") or "",
            "tool_call_id": r.get("gateway_tool_call_id") or "",
            "tool_args": tool_args,
            "final_decision": r.get("gateway_final_decision") or "allow",
            "verdicts": verdicts,
        })

    if not decisions:
        return GuardrailOcelExtension()
    return project_decisions(decisions)


def project_decisions(decisions: list[dict[str, Any]]) -> GuardrailOcelExtension:
    """Turn gateway_decision records into an OCEL extension.

    Public because both `load_guardrail_events` (JSONL path) and
    `load_guardrail_events_from_eventlog` (CSV path) share this projection —
    keeping them convergent is what guarantees the shared `_all_traces.csv`
    reproduces the same OCEL that a live JSONL would.
    """
    return _project(decisions)


def _project(decisions: list[dict[str, Any]]) -> GuardrailOcelExtension:
    # --- backfill maps -------------------------------------------------
    # `case_setup_map` assumes one setup per thread (true in practice; the
    # JSONL would have to be from multiple experiment runs sharing a
    # thread_id, which doesn't normally happen). If violated, the most recent
    # entry wins and we warn once per offending case.
    case_setup_map: dict[str, str] = {}
    case_setup_conflicts: set[str] = set()
    case_agent_snapshot_map: dict[tuple[str, str], str] = {}

    # Object accumulators (deduped at the end).
    guardrail_obj: dict[str, dict[str, Any]] = {}  # name@version → row
    setup_obj: dict[str, dict[str, Any]] = {}
    snapshot_obj: dict[str, dict[str, Any]] = {}
    tool_call_obj: dict[str, dict[str, Any]] = {}

    # Event accumulators (one row per emitted gateway event).
    flag_rows: list[dict[str, Any]] = []
    deny_rows: list[dict[str, Any]] = []

    # E2O accumulator.
    event_object_rows: list[dict[str, Any]] = []
    # O2O accumulator (snapshot → agent), deduped by (source, target).
    o2o_seen: set[tuple[str, str]] = set()
    object_object_rows: list[dict[str, Any]] = []

    for rec in decisions:
        thread_id = rec.get("thread_id")
        agent_id = rec.get("agent_id")
        setup_name = rec.get("setup_name")
        snapshot_id = rec.get("snapshot_id")
        tool_name = rec.get("tool_name", "")
        tool_call_id = rec.get("tool_call_id", "")
        tool_args = rec.get("tool_args", {})
        final_decision = rec.get("final_decision", "allow")
        verdicts = rec.get("verdicts", []) or []
        ts = rec.get("ts")

        # Skip records missing the identifiers we need to link anything.
        if not (thread_id and agent_id and setup_name and snapshot_id and tool_call_id):
            continue

        ocel_time = _ts_to_datetime(ts)
        if ocel_time is None:
            logger.warning("guardrail_log: skipping record with unparseable ts=%r", ts)
            continue

        # --- backfill maps -------------------------------------------
        prior = case_setup_map.get(thread_id)
        if prior and prior != setup_name and thread_id not in case_setup_conflicts:
            logger.warning(
                "guardrail_log: thread_id %s seen under multiple setups (%s, %s); "
                "keeping the most-recently-seen one for backfill",
                thread_id, prior, setup_name,
            )
            case_setup_conflicts.add(thread_id)
        case_setup_map[thread_id] = setup_name
        case_agent_snapshot_map[(thread_id, agent_id)] = snapshot_id

        # --- objects --------------------------------------------------
        setup_obj.setdefault(setup_name, {"ocel_id": setup_name})

        if snapshot_id not in snapshot_obj:
            snap_agent, snap_version, snap_hash = _parse_snapshot_id(snapshot_id)
            snapshot_obj[snapshot_id] = {
                "ocel_id": snapshot_id,
                "agent_id": snap_agent,
                "version_label": snap_version,
                "hash": snap_hash,
            }

        # tool_call object — id = tool_call_id; tool_name attr; one per call.
        tool_call_obj.setdefault(
            tool_call_id, {"ocel_id": tool_call_id, "tool_name": tool_name},
        )

        # --- O2O: snapshot → agent ------------------------------------
        # Emitted unconditionally for every (snapshot, agent_obj) pair seen,
        # not gated on the event being emitted — the relationship is a fact
        # about the run regardless of whether a deny/flag occurred.
        agent_obj_id = f"{thread_id}_{agent_id}"
        o2o_key = (snapshot_id, agent_obj_id)
        if o2o_key not in o2o_seen:
            o2o_seen.add(o2o_key)
            object_object_rows.append({
                "ocel_source_id": snapshot_id,
                "ocel_target_id": agent_obj_id,
                "ocel_qualifier": "version_of",
            })

        # Sort verdicts by effect for tidy attribute output.
        denied_by: list[str] = []
        flagged_by: list[str] = []
        consulted: list[str] = []
        reasons: list[str] = []
        for v in verdicts:
            name = v.get("guardrail_name", "")
            version = v.get("guardrail_version", "unversioned")
            gtype = v.get("guardrail_type", "")
            effect = v.get("effect", "allow")
            reason = v.get("reason_for_llm") or v.get("reason_internal") or ""

            obj_id = f"{name}@{version}"
            guardrail_obj.setdefault(
                obj_id,
                {"ocel_id": obj_id, "guardrail_type": gtype, "version": version},
            )
            consulted.append(name)
            if effect == "deny":
                denied_by.append(name)
            elif effect == "flag":
                flagged_by.append(name)
            if reason:
                reasons.append(reason)

        # --- emit event? ---------------------------------------------
        emit_deny = final_decision == "deny"
        emit_flag = (not emit_deny) and bool(flagged_by)
        if not (emit_deny or emit_flag):
            # Allowed call with no flags — still produce the tool_call object
            # (already done above), but no synthetic event. Backfill stays.
            continue

        event_type = "gateway_deny" if emit_deny else "gateway_flag"
        event_id = str(uuid.uuid4())
        row = {
            "event_id": event_id,
            "ocel_time": ocel_time,
            "tool_name": tool_name,
            "tool_args": json.dumps(tool_args, sort_keys=True),
            "tool_call_id": tool_call_id,
            "final_decision": final_decision,
            "setup_name": setup_name,
            "snapshot_id": snapshot_id,
            "agent_id": agent_id,
            "denied_by": "|".join(denied_by),
            "flagged_by": "|".join(flagged_by),
            "consulted": "|".join(consulted),
            "n_verdicts": len(verdicts),
            "reason_for_llm": " | ".join(reasons),
        }
        if emit_deny:
            deny_rows.append(row)
        else:
            flag_rows.append(row)

        # --- E2O for this gateway event ------------------------------
        agent_obj_id_for_event = f"{thread_id}_{agent_id}"
        event_object_rows.extend([
            {"ocel_event_id": event_id, "ocel_object_id": agent_obj_id_for_event, "ocel_qualifier": "evaluated_for"},
            {"ocel_event_id": event_id, "ocel_object_id": setup_name, "ocel_qualifier": "under_setup"},
            {"ocel_event_id": event_id, "ocel_object_id": snapshot_id, "ocel_qualifier": "using_snapshot"},
            {"ocel_event_id": event_id, "ocel_object_id": tool_call_id, "ocel_qualifier": "decided_on"},
        ])
        for v in verdicts:
            gname = v.get("guardrail_name", "")
            gver = v.get("guardrail_version", "unversioned")
            effect = v.get("effect", "allow")
            qualifier = (
                "denied_by" if effect == "deny"
                else "flagged_by" if effect == "flag"
                else "consulted"
            )
            event_object_rows.append({
                "ocel_event_id": event_id,
                "ocel_object_id": f"{gname}@{gver}",
                "ocel_qualifier": qualifier,
            })

    # --- finalize --------------------------------------------------------
    ext = GuardrailOcelExtension(
        case_setup_map=case_setup_map,
        case_agent_snapshot_map=case_agent_snapshot_map,
        tool_call_ids=set(tool_call_obj),
    )

    # Per-type event tables. Columns: ocel_id + attrs from EVENT_ATTRIBUTES.
    # Even when one of the two is empty we emit an empty table with the right
    # schema so the converter can register the event type.
    event_columns = [
        "ocel_id", "ocel_time", "tool_name", "tool_args", "tool_call_id",
        "final_decision", "setup_name", "snapshot_id", "agent_id",
        "denied_by", "flagged_by", "consulted", "n_verdicts", "reason_for_llm",
    ]
    event_schema = {
        "ocel_id": pl.Utf8,
        "ocel_time": pl.Datetime("us"),
        "tool_name": pl.Utf8,
        "tool_args": pl.Utf8,
        "tool_call_id": pl.Utf8,
        "final_decision": pl.Utf8,
        "setup_name": pl.Utf8,
        "snapshot_id": pl.Utf8,
        "agent_id": pl.Utf8,
        "denied_by": pl.Utf8,
        "flagged_by": pl.Utf8,
        "consulted": pl.Utf8,
        "n_verdicts": pl.Int64,
        "reason_for_llm": pl.Utf8,
    }

    def _event_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
        if not rows:
            return pl.DataFrame(schema=event_schema)
        # Project to canonical column order + rename event_id → ocel_id.
        df = pl.DataFrame(rows, schema_overrides={"n_verdicts": pl.Int64})
        df = df.rename({"event_id": "ocel_id"})
        return df.select(event_columns).cast(event_schema)

    ext.event_tables = {
        "gateway_flag": _event_df(flag_rows),
        "gateway_deny": _event_df(deny_rows),
    }

    # Per-type object tables.
    ext.object_tables = {
        "guardrail": _object_df(
            guardrail_obj.values(),
            schema={
                "ocel_id": pl.Utf8,
                "guardrail_type": pl.Utf8,
                "version": pl.Utf8,
            },
        ),
        "setup": _object_df(
            setup_obj.values(),
            schema={"ocel_id": pl.Utf8},
        ),
        "snapshot": _object_df(
            snapshot_obj.values(),
            schema={
                "ocel_id": pl.Utf8,
                "agent_id": pl.Utf8,
                "version_label": pl.Utf8,
                "hash": pl.Utf8,
            },
        ),
        "tool_call": _object_df(
            tool_call_obj.values(),
            schema={"ocel_id": pl.Utf8, "tool_name": pl.Utf8},
        ),
    }

    # Plain (ocel_id, ocel_type) projections for the top-level `events` and
    # `objects` DataFrames.
    base_event_rows = (
        [{"ocel_id": r["event_id"], "ocel_type": "gateway_flag"} for r in flag_rows]
        + [{"ocel_id": r["event_id"], "ocel_type": "gateway_deny"} for r in deny_rows]
    )
    ext.events_rows = (
        pl.DataFrame(base_event_rows, schema={"ocel_id": pl.Utf8, "ocel_type": pl.Utf8})
        if base_event_rows
        else _empty_df({"ocel_id": pl.Utf8, "ocel_type": pl.Utf8})
    )

    base_object_rows = (
        [{"ocel_id": gid, "ocel_type": "guardrail"} for gid in guardrail_obj]
        + [{"ocel_id": sid, "ocel_type": "setup"} for sid in setup_obj]
        + [{"ocel_id": sid, "ocel_type": "snapshot"} for sid in snapshot_obj]
        + [{"ocel_id": tid, "ocel_type": "tool_call"} for tid in tool_call_obj]
    )
    ext.objects_rows = (
        pl.DataFrame(base_object_rows, schema={"ocel_id": pl.Utf8, "ocel_type": pl.Utf8})
        if base_object_rows
        else _empty_df({"ocel_id": pl.Utf8, "ocel_type": pl.Utf8})
    )

    if event_object_rows:
        ext.event_object_rows = pl.DataFrame(
            event_object_rows,
            schema={
                "ocel_event_id": pl.Utf8,
                "ocel_object_id": pl.Utf8,
                "ocel_qualifier": pl.Utf8,
            },
        )
    if object_object_rows:
        ext.object_object_rows = pl.DataFrame(
            object_object_rows,
            schema={
                "ocel_source_id": pl.Utf8,
                "ocel_target_id": pl.Utf8,
                "ocel_qualifier": pl.Utf8,
            },
        )

    return ext


def _object_df(rows, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    rows = list(rows)
    if not rows:
        return pl.DataFrame(schema=schema)
    # Some object types only have ocel_id — ensure missing optional cols
    # default to None so the dtype cast succeeds.
    cols = list(schema)
    filled = [{c: r.get(c) for c in cols} for r in rows]
    return pl.DataFrame(filled, schema=schema)


def _parse_snapshot_id(snapshot_id: str) -> tuple[str, str, str]:
    """Split a snapshot_id like `order_agent@v1+d547650bf20d` into parts.

    Falls back to ``("", "", "")`` when the format is unexpected; the OCEL
    is still well-formed in that case, just with empty version metadata.
    """
    agent, _, rest = snapshot_id.partition("@")
    version, _, snap_hash = rest.partition("+")
    return agent, version, snap_hash


def _ts_to_datetime(ts: Any) -> datetime | None:
    """Convert the JSONL `ts` (epoch float seconds) to a naive-UTC datetime.

    Matches the dtype produced by `_preprocess_eventlog`'s
    `pl.col("time_finished").str.to_datetime()` — which yields naive-UTC
    because LogGenerator writes CSV timestamps as naive-UTC (from OTel's
    `start_time_unix_nano`, always UTC). Using `datetime.fromtimestamp(ts)`
    with no `tz=` would return naive-LOCAL and shift gateway events by the
    local UTC offset relative to the rest of the OCEL — invisible on a
    UTC host, hours off on any host in a non-UTC zone. Returns `None` for
    unparseable input so the caller can drop the offending record.
    """
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None
