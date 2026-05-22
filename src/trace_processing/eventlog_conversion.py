from dataclasses import dataclass
from datetime import datetime

import json
import os

import polars as pl


EVENT_ATTRIBUTES = {
    "agent_response": ["ocel_time", "duration", "input_tokens", "response_tokens"],
    "call_llm": ["ocel_time", "model", "duration", "input_tokens", "response_tokens"],
    "user_prompt": ["ocel_time"],

    # tools
    "prepare_order": ["ocel_time", "duration"],
    "estimate_prep_time": ["ocel_time", "duration"],
    "process_order": ["ocel_time", "duration"],
    "check_inventory": ["ocel_time", "duration"],
    "update_stock": ["ocel_time", "duration"],
    "get_order": ["ocel_time", "duration"],
    "remake_order_item": ["ocel_time", "duration"],
    "transfer_to_customer_service": ["ocel_time", "duration"],
    "offer_refund": ["ocel_time", "duration"],
    "offer_partial_refund": ["ocel_time", "duration"],
    "get_alternatives": ["ocel_time", "duration"],
    "calculate_total": ["ocel_time", "duration"],

    "transfer_to_order_agent": ["ocel_time", "duration"],
    "transfer_to_barista": ["ocel_time", "duration"],
    "transfer_to_inventory": ["ocel_time", "duration"],

    # handovers
    "order_agent_handover_inventory_agent": ["ocel_time", "model", "duration", "input_tokens", "response_tokens"],
    "order_agent_handover_barista_agent": ["ocel_time", "model", "duration", "input_tokens", "response_tokens"],
    "order_agent_handover_customer_service_agent": ["ocel_time", "model", "duration", "input_tokens", "response_tokens"],
    "barista_agent_handover_order_agent": ["ocel_time", "model", "duration", "input_tokens", "response_tokens"],
    "barista_agent_handover_inventory_agent": ["ocel_time", "model", "duration", "input_tokens", "response_tokens"],
    "barista_agent_handover_customer_service_agent": ["ocel_time", "model", "duration", "input_tokens", "response_tokens"],
    "inventory_agent_handover_order_agent": ["ocel_time", "model", "duration", "input_tokens", "response_tokens"],
    "inventory_agent_handover_barista_agent": ["ocel_time", "model", "duration", "input_tokens", "response_tokens"],
    "inventory_agent_handover_customer_service_agent": ["ocel_time", "model", "duration", "input_tokens", "response_tokens"],
    "customer_service_agent_handover_order_agent": ["ocel_time", "model", "duration", "input_tokens", "response_tokens"],
    "customer_service_agent_handover_barista_agent": ["ocel_time", "model", "duration", "input_tokens", "response_tokens"],
    "customer_service_agent_handover_inventory_agent": ["ocel_time", "model", "duration", "input_tokens", "response_tokens"],
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
    def from_eventlog(cls, eventlog: str | pl.DataFrame) -> "ObjectCentricEventlog":
        """
        Create an ObjectCentricEventlog according to the OCEL 2.0 standard from a flat event log.
        The input is either a path to the eventlog or the eventlog a as a polars DataFrame

        Input:
            el : pl.DataFrame holding the raw event log as loaded directly from the CSV.

        """
        if isinstance(eventlog, str):
            eventlog = pl.read_csv(eventlog)

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

        events = el_enriched.select(
            ocel_id=pl.col("event_id"), ocel_type=pl.col("event_type")
        )

        event_object = (
            el_enriched.select(
                ocel_event_id=pl.col("event_id"),
                ocel_object_id=pl.concat_list(
                    [pl.col("object_id_agent"), pl.col("object_id_message"), pl.col("related_prompt")]
                ),
                ocel_qualifier=pl.concat_list(
                    [pl.col("object_type_agent"), pl.col("object_type_message"), pl.lit("prompt")]
                ),
            )
            .explode("ocel_object_id", "ocel_qualifier")
            .drop_nulls()
        )

        object_object = pl.DataFrame(
            schema={
                "ocel_source_id": str,
                "ocel_target_id": str,
                "ocel_qualifier": str,
            }
        )

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
            attrs = EVENT_ATTRIBUTES[evt_type]
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

        object_tables = {}
        for obj_type in object_map_type["ocel_type"].to_list():
            attrs = OBJECT_ATTRIBUTES[obj_type]
            column_id = (
                "object_id_message"
                if (obj_type == "prompt" or obj_type == "response")
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

        # ---- eventTypes ----
        event_types = []
        for name, df in self.event_tables.items():
            attrs = [
                {"name": col, "type": map_dtype(dtype)}
                for col, dtype in zip(df.columns, df.dtypes)
                if col not in ("ocel_id", "ocel_time")
            ]
            event_types.append({"name": name, "attributes": attrs})

        # ---- objectTypes ----
        object_types = []
        for name, df in self.object_tables.items():
            attrs = [
                {"name": col, "type": map_dtype(dtype)}
                for col, dtype in zip(df.columns, df.dtypes)
                if col != "ocel_id"
            ]
            object_types.append({"name": name, "attributes": attrs})

        # ---- event relationships (grouped) ----
        event_rels = (
            self.event_object
            .group_by("ocel_event_id")
            .agg(pl.struct(["ocel_object_id", "ocel_qualifier"]).alias("rels"))
        )
        event_rels_dict = {
            r["ocel_event_id"]: r["rels"] for r in event_rels.to_dicts()
        }

        # ---- object relationships (grouped) ----
        object_rels = (
            self.object_object
            .group_by("ocel_source_id")
            .agg(pl.struct(["ocel_target_id", "ocel_qualifier"]).alias("rels"))
        )
        object_rels_dict = {
            r["ocel_source_id"]: r["rels"] for r in object_rels.to_dicts()
        }

        # ---- events ----
        events = []
        for event_type, df in self.event_tables.items():
            for row in df.to_dicts():
                eid = row["ocel_id"]

                events.append({
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
                            "qualifier": rel["ocel_qualifier"]
                        }
                        for rel in event_rels_dict.get(eid, [])
                    ]
                })

        # ---- objects ----
        objects = []
        for obj_type, df in self.object_tables.items():
            for row in df.to_dicts():
                oid = row["ocel_id"]

                objects.append({
                    "id": oid,
                    "type": obj_type,
                    "attributes": [
                        {
                            "name": k,
                            "value": str(v),
                            "time": NOW  # required by schema
                        }
                        for k, v in row.items()
                        if k != "ocel_id" and v is not None
                    ],
                    "relationships": [
                        {
                            "objectId": rel["ocel_target_id"],
                            "qualifier": rel["ocel_qualifier"]
                        }
                        for rel in object_rels_dict.get(oid, [])
                    ]
                })

        # ---- final JSON ----
        ocel_json = {
            "eventTypes": event_types,
            "objectTypes": object_types,
            "events": events,
            "objects": objects,
        }

        # ---- write file ----
        os.makedirs("./generated_ocel/", exist_ok=True)
        if not export_name:
            export_name = f"ocel_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        with open(f"./generated_ocel/{export_name}.json", "w") as f:
            json.dump(ocel_json, f, indent=2)


def _preprocess_eventlog(eventlog: pl.DataFrame) -> pl.DataFrame:
    """
        Helper function used to preprocess a given eventlog from the coffee shop.

    """
    el_enriched = (
        eventlog.with_row_index()
        .with_columns(
            object_type_message=(
                pl.when(pl.col("concept:instance") == "prompt").then(pl.col("concept:instance"))
                .when((pl.col("concept:name") == "call_llm") & (pl.col("message").is_not_null())).then(pl.lit("response"))
                .otherwise(pl.lit(None))),
            object_id_agent=pl.when(pl.col("org:resource").str.contains("agent")).then(pl.col("case_id") + pl.lit("_") + pl.col("org:resource")).otherwise(pl.col("case_id")),
            object_type_agent=pl.col("org:resource"),
            event_id=pl.col("identity:id"),
            event_type=(
                pl.when((pl.col("concept:name") == "execute_tool") & pl.col("tool").is_not_null()).then(pl.col("tool"))
                .when((pl.col("concept:name") == "call_llm") & (pl.col("message").is_not_null())).then(pl.lit("agent_response"))
                .otherwise(pl.col("concept:name"))),
            ocel_time=pl.col("time_finished").str.to_datetime(),
            index=pl.col("index").cast(pl.Float64),
        )
        .with_columns(
            object_id_message=(
                pl.when(pl.col("object_type_message") == "prompt").then(pl.lit("prompt_") + pl.col("identity:id"))
                .when(pl.col("object_type_message") == "response").then(pl.lit("response_") + pl.col("identity:id"))
                .otherwise(pl.lit(None))),
            next_event_type=pl.col("event_type").shift(-1),
            next_agent=pl.col("object_type_agent").shift(-1),
            next_agent_id=pl.col("object_id_agent").shift(-1)
        )
        .with_columns(
            handover_flag=
            (
                (pl.col("event_type") == "call_llm") &
                (pl.col("next_event_type") == "call_llm") &
                (pl.col("object_type_agent") != pl.col("next_agent"))
            ),
            previous_event_type=pl.col("event_type").shift(1),
            previous_object_id_message=pl.col("object_id_message").shift(1),
        )
        .with_columns(
            related_prompt=
                pl.when((pl.col("event_type") == "agent_response") & (pl.col("previous_event_type") == "user_prompt"))
                .then(pl.col("previous_object_id_message"))
                .otherwise(pl.lit(None))
        )
    )

    cols_to_keep = ["index", "case_id", "ocel_time", "event_id", "event_type", "object_type_agent", "object_id_agent", "duration", "model", "input_tokens", "response_tokens"]

    handover_rows = (
        el_enriched.filter(pl.col("handover_flag"))
        .with_columns(
            index=pl.col("index") + 0.5,
            ocel_time=pl.col("ocel_time") + pl.duration(nanoseconds=1),
            event_type=pl.col("object_type_agent") + "_handover_" + pl.col("next_agent"),
        )
    )

    # handover for each agent
    handover_one_direction = (
        handover_rows.with_columns(
            object_type_agent=pl.col("object_type_agent"),
            object_id_agent=pl.col("object_id_agent"),
        )
        .with_columns(pl.all().exclude(cols_to_keep).map_elements(lambda _: None))
    )
    handover_second_direction = (
        handover_rows.with_columns(
            object_type_agent=pl.col("next_agent"),
            object_id_agent=pl.col("next_agent_id"),
        )
        .with_columns(pl.all().exclude(cols_to_keep).map_elements(lambda _: None))
    )

    return (
        pl.concat([
            el_enriched.filter(pl.col("handover_flag") == False),
            handover_one_direction,
            handover_second_direction
        ])
        .sort("index")
    )