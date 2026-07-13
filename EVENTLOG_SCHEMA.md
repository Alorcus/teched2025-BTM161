# Event Log Schema (`_all_traces.csv`)

`_all_traces.csv` is the shared observability interchange for the Metrics Dashboard. The dashboard rebuilds it on page entry from the local MLflow store, but the file is designed to be handed to colleagues who don't have that store on disk. It is a single flat CSV that merges four producer streams into one canonical event-log shape: agent-side spans extracted from MLflow (`LogGenerator`), coffee-machine rows from `services/coffee_machine/logs/coffee_machine.csv`, gateway/guardrail decisions from `guardrail_log/events.jsonl`, and per-case customer feedback from `feedback_store.json`. This document is the single source of truth for its columns.

## ⚠ Sensitivity warning

**Before sharing this file, review the sensitivity table below.**

`_all_traces.csv` embeds verbatim user-turn text (in `message`, on `user_prompt` rows) and free-text LLM reasoning (in `gateway_verdicts_json` — the `reason_internal` and `reason_for_llm` fields — and in `feedback_reason` and `gateway_tool_args_json`). In the TechEd educational context the "customer" is an LLM-simulated persona, so those cells are synthetic and safe to share freely. **In any real deployment, the same columns would carry PII: real customer utterances and unredacted internal model reasoning.** Treat the file as sensitive by default and consult the table below before forwarding it, uploading it to a shared drive, or attaching it to a ticket.

## Column reference

Columns are produced by one of three sites:

- **`LogGenerator`** — `src/trace_processing/log_generator.py`, extracts spans from an MLflow trace.
- **`TraceProcessor`** — `src/trace_processing/trace_processor.py`, merges coffee-machine rows (`_load_coffee_machine_rows`), gateway rows (`_load_gateway_rows`), and per-case feedback rows on top of the `LogGenerator` output, then broadcasts `case_setup` / `case_scenario_index` from MLflow trace tags.
- **`trace_cache`** — `src/dashboard/metrics/trace_cache.py`, owns the compiled shape and its schema version.

Sensitivity classes:

- **public** — structural identifiers, counts, timings, non-content metadata. Safe to share.
- **internal** — engineering-only tokens (agent names, tool IDs, model names). Not directly sensitive but leaks system internals; share within the team.
- **sensitive** — verbatim user text or free-text LLM reasoning. See the warning above.

| Column name              | Producer                             | Dtype             | Sensitivity | Description                                                                                                              |
| ------------------------ | ------------------------------------ | ----------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------ |
| `case_id`                | `LogGenerator`, all merged streams   | string            | public      | LangGraph `thread_id` for the conversation. Primary join key across every stream.                                        |
| `identity:id`            | all producers                        | string (uuid4)    | public      | Unique row identifier. Freshly generated per row; not stable across rebuilds.                                            |
| `time:timestamp`         | all producers                        | string (ISO-8601) | public      | Event start time as naive-UTC ISO-8601 with millisecond precision (see [Timezone contract](#timezone-contract)).         |
| `time_finished`          | all producers                        | string (ISO-8601) | public      | Event end time in the same format. Equal to `time:timestamp` for instantaneous events (gateway decisions, user prompts). |
| `duration`               | `LogGenerator`, coffee machine       | int64 (ns)        | public      | Span duration in nanoseconds. Null for instantaneous events.                                                             |
| `concept:name`           | all producers                        | string            | public      | Event type. Known values: `user_prompt`, `call_llm`, `execute_tool`, `user_feedback`, `gateway_decision`, `job_created`, `process_order`, `brew_completed`, `brew_failed`, `clean_machine`. |
| `concept:instance`       | all producers                        | string            | internal    | Human-readable label combining the agent/actor and the event (e.g. `order_agent calls llm`). May echo tool or drink names. |
| `org:resource`           | all producers                        | string            | internal    | Actor that emitted the row: an agent name (`order_agent`, `barista_agent`, …), `user`, `coffee_machine`, or a gateway `agent_id`. |
| `model`                  | `LogGenerator` (`call_llm` rows)     | string            | internal    | LLM model name reported by the provider (e.g. `claude-3-5-sonnet-20241022`).                                             |
| `input_tokens`           | `LogGenerator` (`call_llm` rows)     | int (nullable)    | public      | Prompt tokens billed for the call.                                                                                       |
| `response_tokens`        | `LogGenerator` (`call_llm` rows)     | int (nullable)    | public      | Completion tokens billed for the call.                                                                                   |
| `message`                | `LogGenerator`, feedback rows        | string (nullable) | **sensitive** | On `user_prompt` rows: the customer's verbatim opening/turn text. On `call_llm` rows: the assistant's free-text response. On `user_feedback` rows: the numeric score as a string. |
| `tool`                   | `LogGenerator` (`execute_tool` rows) | string            | internal    | Tool name invoked (e.g. `process_order`, `transfer_to_customer_service_agent`).                                          |
| `tool_call_id`           | `LogGenerator` (`execute_tool` rows) | string            | internal    | LangChain tool-call identifier. Join key against `gateway_tool_call_id` for guardrail correlation.                       |
| `job_id`                 | coffee machine                       | string            | internal    | Coffee-machine job identifier linking `job_created` → `brew_completed`/`brew_failed`.                                    |
| `drink`                  | coffee machine                       | string            | internal    | Drink type for the machine event (e.g. `latte`, `espresso`).                                                             |
| `gateway_setup_name`     | gateway                              | string            | internal    | Setup that produced the guardrail decision (`baseline`, `all_handovers`, `unconstrained`).                               |
| `gateway_snapshot_id`    | gateway                              | string            | internal    | Snapshot identifier of the guardrail configuration at decision time.                                                     |
| `gateway_tool_name`      | gateway                              | string            | internal    | Tool name the gateway evaluated.                                                                                         |
| `gateway_tool_call_id`   | gateway                              | string            | internal    | Tool-call identifier that ties the decision back to the corresponding `execute_tool` row via `tool_call_id`.             |
| `gateway_final_decision` | gateway                              | string            | internal    | Final verdict: `allow`, `deny`, or a setup-specific label.                                                               |
| `gateway_tool_args_json` | gateway                              | JSON string       | **sensitive** | Arguments passed to the guarded tool, serialized as JSON. May include handoff `context` or `message` fields carrying verbatim customer/agent text. |
| `gateway_verdicts_json`  | gateway                              | JSON string       | **sensitive** | Per-predicate verdicts, serialized as a JSON array. Each entry may contain free-text `reason_internal` and `reason_for_llm` fields written by the LLM judge. |
| `feedback_score`         | feedback rows                        | float             | public      | Customer feedback score in `[0.0, 1.0]` (`1.0` excellent, `0.5` acceptable, `0.0` poor).                                 |
| `feedback_reason`        | feedback rows                        | string            | **sensitive** | Free-text explanation from the LLM-simulated customer. Same treatment as `message`.                                     |
| `feedback_valid`         | feedback rows                        | bool              | public      | Whether the feedback LLM response parsed cleanly (`false` means the score was defaulted to `0.5`).                       |
| `scenario_index`         | feedback rows                        | int               | public      | Scenario the feedback belongs to (`0`–`3` for the presets, `-1` for custom prompts or the Jupyter path).                 |
| `case_setup`             | `TraceProcessor` (broadcast)         | string (nullable) | public      | MLflow `setup` tag lifted onto every row of the case (`baseline`, `all_handovers`, `unconstrained`, or null when untagged). |
| `case_scenario_index`    | `TraceProcessor` (broadcast)         | int               | public      | MLflow `scenario_index` tag lifted onto every row of the case (`0`–`3`, or `-1` when unspecified).                       |

## Timezone contract

All timestamp columns (`time:timestamp`, `time_finished`) are **naive-UTC ISO-8601 strings** with millisecond precision (`%Y-%m-%dT%H:%M:%S.%f` truncated to `.fff`). "Naive-UTC" means the value carries no timezone suffix but represents UTC clock time.

- The Metrics Dashboard's `_load_combined_eventlog` treats every value as UTC and converts to the viewer's local timezone for display.
- Downstream tools that parse the string as *local* time (e.g. `datetime.fromisoformat(...)` on a machine in a non-UTC timezone, then localizing) will drift by the local offset — gateway/coffee-machine rows would appear shifted into the future or past.
- When post-processing, parse with an explicit `tz=UTC` (e.g. `pd.to_datetime(..., utc=True)`) before converting.

Schema version `6` was introduced specifically to fix gateway rows that were previously written as naive-*local*; if you encounter rows with visibly wrong hour offsets, verify the sidecar reports version 6 or later.

## Schema versioning

The compiled schema is versioned by `_SCHEMA_VERSION` in `src/dashboard/metrics/trace_cache.py` and recorded in the sidecar file `_all_traces.meta`, which lives next to the CSV in `generated_event_log/`. On dashboard entry, the loader compares the sidecar version to the current `_SCHEMA_VERSION`:

- **Match** → append-only sync: only new `case_id`s from MLflow are extracted and added.
- **Mismatch or missing sidecar** → the existing CSV is quarantined (renamed to `_all_traces.v{cached_schema}.csv` or `_all_traces.unknown-schema.csv`) and a fresh sync runs at the current version. Quarantined rows are preserved but not loaded — merging them back is a manual step.

Version history is documented inline in `trace_cache.py`.

## Adding a new column

Update this document (`EVENTLOG_SCHEMA.md`) in the same commit as the producer change, and bump `_SCHEMA_VERSION` in `src/dashboard/metrics/trace_cache.py`. Both are load-bearing: the doc is the recipient-facing contract, and the version bump forces already-built caches to rebuild instead of silently mixing old and new row shapes.
