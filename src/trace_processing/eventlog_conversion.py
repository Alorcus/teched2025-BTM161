from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import json
import logging
import os

import polars as pl

from .guardrail_log_loader import (
    GATEWAY_EVENT_TYPES,
    GATEWAY_OBJECT_TYPES,
    GuardrailOcelExtension,
    load_guardrail_events,
    load_guardrail_events_from_eventlog,
)


logger = logging.getLogger("coffee_shop.trace_processing.eventlog_conversion")


def _resolve_guardrail_extension(
    eventlog: pl.DataFrame,
    guardrail_log_path: str | Path | None,
) -> GuardrailOcelExtension:
    """Prefer gateway_decision rows embedded in the eventlog; fall back to
    the on-disk JSONL for callers that don't hand over a shareable CSV.

    Never merges the two — embedded rows were written from the same JSONL,
    so combining would just duplicate everything.

    Timezone contract: `load_guardrail_events_from_eventlog` interprets any
    naive `time:timestamp` value as UTC. `_load_gateway_rows` writes the
    column as a naive-UTC string, so a freshly-loaded CSV is fine. The
    dashboard, however, parses the column into a Datetime and shifts it to
    naive-LOCAL before this runs — tagging that value as UTC would
    double-shift each gateway event.

    Resolution order (fall through on each miss):
    1. `time:timestamp_utc_naive` — sibling column preserved by the
       dashboard, holding the pre-conversion naive-UTC Datetime.
    2. `time:timestamp` as raw string (plain CSV case) — pass directly.
    3. On-disk JSONL, if `guardrail_log_path` points at an existing file.
    4. Last-resort: run the loader against potentially-shifted embedded
       rows, with a warning.
    """
    if "concept:name" in eventlog.columns:
        has_embedded = (
            eventlog.filter(pl.col("concept:name") == "gateway_decision")
            .height
            > 0
        )
        if has_embedded:
            if "time:timestamp_utc_naive" in eventlog.columns:
                rewired = eventlog.with_columns(
                    pl.col("time:timestamp_utc_naive").alias("time:timestamp"),
                )
                return load_guardrail_events_from_eventlog(rewired)
            ts_dtype = eventlog.schema.get("time:timestamp")
            timestamp_is_string = ts_dtype in (pl.Utf8, pl.String)
            if timestamp_is_string:
                return load_guardrail_events_from_eventlog(eventlog)
            if (
                guardrail_log_path is not None
                and Path(guardrail_log_path).exists()
            ):
                return load_guardrail_events(guardrail_log_path)
            logger.warning(
                "Guardrail rows are embedded in the eventlog but "
                "time:timestamp is not a string column and no "
                "time:timestamp_utc_naive sibling column is available — "
                "gateway event timestamps may be shifted by the local UTC "
                "offset. Provide guardrail_log_path for the authoritative "
                "source, or preserve time:timestamp_utc_naive alongside the "
                "converted column."
            )
            return load_guardrail_events_from_eventlog(eventlog)
    if guardrail_log_path is not None:
        return load_guardrail_events(guardrail_log_path)
    return GuardrailOcelExtension()
EVENT_ATTRIBUTES = {
    "agent_response": ["ocel_time", "duration", "input_tokens", "response_tokens"],
    "call_llm": ["ocel_time", "model", "duration", "input_tokens", "response_tokens"],
    "user_prompt": ["ocel_time"],
    # tools
    "start_preparation": ["ocel_time", "duration"],
    "end_preparation": ["ocel_time", "duration"],
    "estimate_prep_time": ["ocel_time", "duration"],
    "process_order": ["ocel_time", "duration"],
    "modify_order": ["ocel_time", "duration"],
    "check_inventory": ["ocel_time", "duration"],
    "update_stock": ["ocel_time", "duration"],
    "get_order": ["ocel_time", "duration"],
    "transfer_to_agent": ["ocel_time", "duration"],
    "offer_refund": ["ocel_time", "duration"],
    "offer_partial_refund": ["ocel_time", "duration"],
    "get_alternatives": ["ocel_time", "duration"],
    "calculate_total": ["ocel_time", "duration"],
    "prepare_order": ["ocel_time", "duration"],
    "remake_order_item": ["ocel_time", "duration"],
    "place_on_tray": ["ocel_time", "duration"],
    "check_tray": ["ocel_time", "duration"],
    "clean_machine": ["ocel_time", "duration"],
    "user_feedback": [
        "ocel_time",
        "feedback_score",
        "feedback_reason",
        "feedback_valid",
        "scenario_index",
    ],
    # coffee machine
    "job_created": ["ocel_time"],
    "brew_completed": ["ocel_time", "duration"],
    "brew_failed": ["ocel_time", "duration"],
    # handovers
    "order_agent_handover_inventory_agent": [
        "ocel_time",
        "model",
        "duration",
        "input_tokens",
        "response_tokens",
    ],
    "order_agent_handover_barista_agent": [
        "ocel_time",
        "model",
        "duration",
        "input_tokens",
        "response_tokens",
    ],
    "order_agent_handover_customer_service_agent": [
        "ocel_time",
        "model",
        "duration",
        "input_tokens",
        "response_tokens",
    ],
    "barista_agent_handover_order_agent": [
        "ocel_time",
        "model",
        "duration",
        "input_tokens",
        "response_tokens",
    ],
    "barista_agent_handover_inventory_agent": [
        "ocel_time",
        "model",
        "duration",
        "input_tokens",
        "response_tokens",
    ],
    "barista_agent_handover_customer_service_agent": [
        "ocel_time",
        "model",
        "duration",
        "input_tokens",
        "response_tokens",
    ],
    "inventory_agent_handover_order_agent": [
        "ocel_time",
        "model",
        "duration",
        "input_tokens",
        "response_tokens",
    ],
    "inventory_agent_handover_barista_agent": [
        "ocel_time",
        "model",
        "duration",
        "input_tokens",
        "response_tokens",
    ],
    "inventory_agent_handover_customer_service_agent": [
        "ocel_time",
        "model",
        "duration",
        "input_tokens",
        "response_tokens",
    ],
    "customer_service_agent_handover_order_agent": [
        "ocel_time",
        "model",
        "duration",
        "input_tokens",
        "response_tokens",
    ],
    "customer_service_agent_handover_barista_agent": [
        "ocel_time",
        "model",
        "duration",
        "input_tokens",
        "response_tokens",
    ],
    "customer_service_agent_handover_inventory_agent": [
        "ocel_time",
        "model",
        "duration",
        "input_tokens",
        "response_tokens",
    ],
    # gateway
    "gateway_flag": [
        "ocel_time",
        "tool_name",
        "tool_args",
        "tool_call_id",
        "final_decision",
        "setup_name",
        "snapshot_id",
        "agent_id",
        "denied_by",
        "flagged_by",
        "consulted",
        "n_verdicts",
        "reason_for_llm",
    ],
    "gateway_deny": [
        "ocel_time",
        "tool_name",
        "tool_args",
        "tool_call_id",
        "final_decision",
        "setup_name",
        "snapshot_id",
        "agent_id",
        "denied_by",
        "flagged_by",
        "consulted",
        "n_verdicts",
        "reason_for_llm",
    ],
}

OBJECT_ATTRIBUTES = {
    "agent": [],
    "user": [],
    "prompt": ["message"],
    "response": ["message"],
    "order_agent": [],
    "barista_agent": [],
    "inventory_agent": [],
    "customer_service_agent": [],
    "root_agent": [],
    "feedback": [
        "feedback_score",
        "feedback_reason",
        "feedback_valid",
        "scenario_index",
    ],
    "coffee_machine": [],
    "guardrail": ["guardrail_type", "version"],
    "setup": [],
    "snapshot": ["agent_id", "version_label", "hash"],
    "tool_call": ["tool_name"],
}


@dataclass
class ObjectCentricEventlog:
    """
    Minimal OCEL 2.0 container
    """

    events: pl.DataFrame
    objects: pl.DataFrame
    event_object: pl.DataFrame
    object_object: pl.DataFrame
    event_map_type: pl.DataFrame
    object_map_type: pl.DataFrame
    event_tables: dict[str, pl.DataFrame]
    object_tables: dict[str, pl.DataFrame]

    @classmethod
    def from_eventlog(
        cls,
        eventlog: str | pl.DataFrame,
        guardrail_log_path: str | Path | None = None,
    ) -> "ObjectCentricEventlog":
        """
        Create an ObjectCentricEventlog according to the OCEL 2.0 standard from a flat event log.
        The input is either a path to the eventlog or the eventlog a as a polars DataFrame

        Input:
            el : pl.DataFrame holding the raw event log as loaded directly from the CSV.
        """
        if isinstance(eventlog, str):
            eventlog = pl.read_csv(eventlog)

        # Resolve the guardrail extension BEFORE filtering out gateway_decision
        # rows, then strip those rows from the eventlog that feeds the OCEL
        # converter — they've been re-emitted as gateway_flag/gateway_deny by
        # the extension, and leaving the raw rows in would produce a duplicate
        # native `event_gateway_decision` table with no OCEL edges.
        ext = _resolve_guardrail_extension(eventlog, guardrail_log_path)
        if "concept:name" in eventlog.columns:
            eventlog = eventlog.filter(pl.col("concept:name") != "gateway_decision")

        el_enriched = _preprocess_eventlog(eventlog)

        objects = (
            el_enriched.select(
                pl.concat_list(
                    [
                        pl.struct(
                            pl.col("object_id_agent").alias("ocel_id"),
                            pl.col("object_type_agent").alias("ocel_type"),
                        ),
                        pl.struct(
                            pl.col("object_id_message").alias("ocel_id"),
                            pl.col("object_type_message").alias("ocel_type"),
                        ),
                    ]
                )
            )
            .explode("ocel_id")
            .select(pl.col("ocel_id").struct.unnest())
            .drop_nulls()
            .unique()
        )

        if not ext.objects_rows.is_empty():
            objects = pl.concat([objects, ext.objects_rows]).unique()

        events = el_enriched.select(
            ocel_id=pl.col("event_id"), ocel_type=pl.col("event_type")
        )

        if not ext.events_rows.is_empty():
            events = pl.concat([events, ext.events_rows]).unique()

        event_object = (
            el_enriched.select(
                ocel_event_id=pl.col("event_id"),
                ocel_object_id=pl.concat_list(
                    [
                        pl.col("object_id_agent"),
                        pl.col("object_id_message"),
                        pl.col("related_prompt"),
                    ]
                ),
                ocel_qualifier=pl.concat_list(
                    [
                        pl.col("object_type_agent"),
                        pl.col("object_type_message"),
                        pl.lit("prompt"),
                    ]
                ),
            )
            .explode("ocel_object_id", "ocel_qualifier")
            .drop_nulls()
        )

        if (
            "tool_call_id" in el_enriched.columns
            and ext.tool_call_ids
        ):
            tool_call_links = (
                el_enriched
                .filter(pl.col("tool_call_id").is_not_null() & pl.col("tool").is_not_null())
                .filter(pl.col("tool_call_id").is_in(list(ext.tool_call_ids)))
                .select(
                    ocel_event_id=pl.col("event_id"),
                    ocel_object_id=pl.col("tool_call_id"),
                    ocel_qualifier=pl.lit("executes"),
                )
            )
            if not tool_call_links.is_empty():
                event_object = pl.concat([event_object, tool_call_links])

        if ext.case_setup_map:
            setup_map_df = pl.DataFrame(
                {
                    "case_id": list(ext.case_setup_map.keys()),
                    "_setup": list(ext.case_setup_map.values()),
                },
                schema={"case_id": pl.Utf8, "_setup": pl.Utf8},
            )
            setup_links = (
                el_enriched
                .select("event_id", "case_id")
                .unique()
                .join(setup_map_df, on="case_id", how="inner")
                .select(
                    ocel_event_id=pl.col("event_id"),
                    ocel_object_id=pl.col("_setup"),
                    ocel_qualifier=pl.lit("under_setup"),
                )
            )
            if not setup_links.is_empty():
                event_object = pl.concat([event_object, setup_links])

        if ext.case_agent_snapshot_map:
            snap_map_df = pl.DataFrame(
                {
                    "case_id": [k[0] for k in ext.case_agent_snapshot_map],
                    "org:resource": [k[1] for k in ext.case_agent_snapshot_map],
                    "_snapshot": list(ext.case_agent_snapshot_map.values()),
                },
                schema={
                    "case_id": pl.Utf8,
                    "org:resource": pl.Utf8,
                    "_snapshot": pl.Utf8,
                },
            )
            snapshot_links = (
                el_enriched
                .select("event_id", "case_id", "org:resource")
                .unique()
                .join(snap_map_df, on=["case_id", "org:resource"], how="inner")
                .select(
                    ocel_event_id=pl.col("event_id"),
                    ocel_object_id=pl.col("_snapshot"),
                    ocel_qualifier=pl.lit("using_snapshot"),
                )
            )
            if not snapshot_links.is_empty():
                event_object = pl.concat([event_object, snapshot_links])

        if not ext.event_object_rows.is_empty():
            event_object = pl.concat([event_object, ext.event_object_rows])

        event_object = event_object.unique()

        object_object = pl.DataFrame(
            schema={
                "ocel_source_id": str,
                "ocel_target_id": str,
                "ocel_qualifier": str,
            }
        )

        if not ext.object_object_rows.is_empty():
            object_object = pl.concat([object_object, ext.object_object_rows]).unique()

        event_map_type = (
            events.select("ocel_type")
            .unique()
            .with_columns(ocel_type_map=pl.col("ocel_type"))
        )

        object_map_type = (
            objects.select("ocel_type")
            .unique()
            .with_columns(ocel_type_map=pl.col("ocel_type"))
        )

        event_tables = {}
        for evt_type in event_map_type["ocel_type"].to_list():
            if evt_type in GATEWAY_EVENT_TYPES:
                continue
            attrs = EVENT_ATTRIBUTES.get(evt_type, [])
            evt_type_tbl = (
                events.filter(pl.col("ocel_type") == evt_type)
                .join(
                    el_enriched.select(["event_id", *attrs]),
                    left_on="ocel_id",
                    right_on="event_id",
                    how="left",
                )
                .drop("ocel_type")
                .unique()
            )
            event_tables[f"event_{evt_type}"] = evt_type_tbl

        for evt_type, df in ext.event_tables.items():
            event_tables[f"event_{evt_type}"] = df

        object_tables = {}
        for obj_type in object_map_type["ocel_type"].to_list():
            if obj_type in GATEWAY_OBJECT_TYPES:
                continue
            attrs = OBJECT_ATTRIBUTES.get(obj_type, [])
            column_id = (
                "object_id_message"
                if obj_type in ("prompt", "response", "feedback")
                else "object_id_agent"
            )
            obj_type_tbl = (
                objects.filter(pl.col("ocel_type") == obj_type)
                .join(
                    el_enriched.select([column_id, *attrs]),
                    left_on="ocel_id",
                    right_on=column_id,
                    how="left",
                )
                .drop("ocel_type")
                .unique()
            )
            object_tables[f"object_{obj_type}"] = obj_type_tbl

        for obj_type, df in ext.object_tables.items():
            object_tables[f"object_{obj_type}"] = df

        return cls(
            events=events,
            objects=objects,
            event_object=event_object,
            object_object=object_object,
            event_map_type=event_map_type,
            object_map_type=object_map_type,
            event_tables=event_tables,
            object_tables=object_tables,
        )

    def export_to_json(self, export_name: str | None = None) -> None:
        """
        Export the respective ocel to a json file
        """

        def map_dtype(dtype: pl.DataType) -> str:
            if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                return "integer"
            if dtype in (pl.Float32, pl.Float64):
                return "float"
            if dtype == pl.Boolean:
                return "boolean"
            if dtype == pl.Datetime:
                return "time"
            return "string"

        NOW = datetime.utcnow().isoformat() + "Z"

        event_types = []
        for name, df in self.event_tables.items():
            attrs = [
                {"name": col, "type": map_dtype(dtype)}
                for col, dtype in zip(df.columns, df.dtypes)
                if col not in ("ocel_id", "ocel_time")
            ]
            event_types.append({"name": name, "attributes": attrs})

        object_types = []
        for name, df in self.object_tables.items():
            attrs = [
                {"name": col, "type": map_dtype(dtype)}
                for col, dtype in zip(df.columns, df.dtypes)
                if col != "ocel_id"
            ]
            object_types.append({"name": name, "attributes": attrs})

        event_rels = self.event_object.group_by("ocel_event_id").agg(
            pl.struct(["ocel_object_id", "ocel_qualifier"]).alias("rels")
        )
        event_rels_dict = {r["ocel_event_id"]: r["rels"] for r in event_rels.to_dicts()}

        object_rels = self.object_object.group_by("ocel_source_id").agg(
            pl.struct(["ocel_target_id", "ocel_qualifier"]).alias("rels")
        )
        object_rels_dict = {
            r["ocel_source_id"]: r["rels"] for r in object_rels.to_dicts()
        }

        events = []
        for event_type, df in self.event_tables.items():
            for row in df.to_dicts():
                eid = row["ocel_id"]

                events.append(
                    {
                        "id": eid,
                        "type": event_type,
                        "time": row["ocel_time"].isoformat(),
                        "attributes": [
                            {"name": k, "value": str(v)}
                            for k, v in row.items()
                            if k not in ("ocel_id", "ocel_time") and v is not None
                        ],
                        "relationships": [
                            {
                                "objectId": rel["ocel_object_id"],
                                "qualifier": rel["ocel_qualifier"],
                            }
                            for rel in event_rels_dict.get(eid, [])
                        ],
                    }
                )

        objects = []
        for obj_type, df in self.object_tables.items():
            for row in df.to_dicts():
                oid = row["ocel_id"]

                objects.append(
                    {
                        "id": oid,
                        "type": obj_type,
                        "attributes": [
                            {
                                "name": k,
                                "value": str(v),
                                "time": NOW,  # required by schema
                            }
                            for k, v in row.items()
                            if k != "ocel_id" and v is not None
                        ],
                        "relationships": [
                            {
                                "objectId": rel["ocel_target_id"],
                                "qualifier": rel["ocel_qualifier"],
                            }
                            for rel in object_rels_dict.get(oid, [])
                        ],
                    }
                )

        ocel_json = {
            "eventTypes": event_types,
            "objectTypes": object_types,
            "events": events,
            "objects": objects,
        }

        os.makedirs("./generated_ocel/", exist_ok=True)
        if not export_name:
            export_name = f"ocel_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        with open(f"./generated_ocel/{export_name}.json", "w") as f:
            json.dump(ocel_json, f, indent=2)


_HANDOVER_AGENTS = [
    "order_agent",
    "barista_agent",
    "inventory_agent",
    "customer_service_agent",
]


def _preprocess_eventlog(eventlog: pl.DataFrame) -> pl.DataFrame:
    """
    Helper function used to preprocess a given eventlog from the coffee shop.

    """
    el_enriched = (
        eventlog.with_row_index()
        .with_columns(
            object_type_message=(
                pl.when(pl.col("concept:instance") == "prompt")
                .then(pl.col("concept:instance"))
                .when(
                    (pl.col("concept:name") == "call_llm")
                    & (pl.col("message").is_not_null())
                )
                .then(pl.lit("response"))
                .when(pl.col("concept:name") == "user_feedback")
                .then(pl.lit("feedback"))
                .otherwise(pl.lit(None))
            ),
            object_id_agent=pl.when(
                pl.col("org:resource").str.to_lowercase().str.contains("agent")
            )
            .then(pl.col("case_id") + pl.lit("_") + pl.col("org:resource"))
            .otherwise(pl.col("case_id")),
            object_type_agent=pl.col("org:resource"),
            event_id=pl.col("identity:id"),
            event_type=(
                pl.when(
                    (pl.col("concept:name") == "execute_tool")
                    & pl.col("tool").is_not_null()
                )
                .then(pl.col("tool"))
                .when(
                    (pl.col("concept:name") == "call_llm")
                    & (pl.col("message").is_not_null())
                )
                .then(pl.lit("agent_response"))
                .otherwise(pl.col("concept:name"))
            ),
            ocel_time=pl.col("time_finished").str.to_datetime(),
            index=pl.col("index").cast(pl.Float64),
        )
        .with_columns(
            object_id_message=(
                pl.when(pl.col("object_type_message") == "prompt")
                .then(pl.lit("prompt_") + pl.col("identity:id"))
                .when(pl.col("object_type_message") == "response")
                .then(pl.lit("response_") + pl.col("identity:id"))
                .when(pl.col("object_type_message") == "feedback")
                .then(pl.lit("feedback_") + pl.col("identity:id"))
                .otherwise(pl.lit(None))
            ),
            next_agent=pl.col("object_type_agent").shift(-1),
            next_agent_id=pl.col("object_id_agent").shift(-1),
        )
        .with_columns(
            handover_flag=(
                (pl.col("event_type") == "transfer_to_agent")
                & (pl.col("object_type_agent") != pl.col("next_agent"))
                & pl.col("next_agent").is_in(_HANDOVER_AGENTS)
                & pl.col("object_type_agent").is_in(_HANDOVER_AGENTS)
            ),
            previous_event_type=pl.col("event_type").shift(1),
            previous_object_id_message=pl.col("object_id_message").shift(1),
        )
        .with_columns(
            related_prompt=pl.when(
                (pl.col("event_type") == "agent_response")
                & (pl.col("previous_event_type") == "user_prompt")
            )
            .then(pl.col("previous_object_id_message"))
            .otherwise(pl.lit(None))
        )
    )

    cols_to_keep = [
        "index",
        "case_id",
        "ocel_time",
        "event_id",
        "event_type",
        "object_type_agent",
        "object_id_agent",
        "duration",
        "model",
        "input_tokens",
        "response_tokens",
        "feedback_score",
        "feedback_reason",
        "feedback_valid",
        "scenario_index",
    ]

    handover_rows = el_enriched.filter(pl.col("handover_flag")).with_columns(
        index=pl.col("index") + 0.5,
        ocel_time=pl.col("ocel_time") + pl.duration(nanoseconds=1),
        event_type=pl.col("object_type_agent") + "_handover_" + pl.col("next_agent"),
    )

    null_columns = [col for col in handover_rows.columns if col not in cols_to_keep]
    handover_one_direction = handover_rows.with_columns(
        object_type_agent=pl.col("object_type_agent"),
        object_id_agent=pl.col("object_id_agent"),
        *[
            pl.lit(None).alias(col)
            for col in null_columns
            if col not in {"object_type_agent", "object_id_agent"}
        ],
    )
    handover_second_direction = handover_rows.with_columns(
        object_type_agent=pl.col("next_agent"),
        object_id_agent=pl.col("next_agent_id"),
        *[
            pl.lit(None).alias(col)
            for col in null_columns
            if col not in {"object_type_agent", "object_id_agent"}
        ],
    )

    return pl.concat(
        [
            el_enriched.filter(pl.col("handover_flag") == False),
            handover_one_direction,
            handover_second_direction,
        ]
    ).sort("index")
