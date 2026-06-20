# Unified Event Log Schema

> **Status:** proposal — not yet implemented end-to-end. Drafted on the
> `lotte/refine-ocel-and-metrics` branch to align on a target format before
> the coffee machine and guardrail gateway start contributing rows directly.

## Goal

Today, behaviour from different parts of the coffee shop is logged in
**three separate places** with **three different schemas**. We want a
single merged event log per session that:

- keeps the **agent event log's column shape** as the canonical schema,
- lets the **coffee machine** and (soon) the **guardrail gateway** insert
  their events into that same file,
- distinguishes who emitted each row via `org:resource`,
- stays sorted by `time:timestamp` so the file reads top-to-bottom as
  one chronological story per case.

This is _not_ an OCEL log. The downstream conversion in
`src/trace_processing/eventlog_conversion.py` already lifts the flat log
into OCEL 2.0 — that step stays.

---

## Where logs live today

| #   | Source                                      | Writer                                                         | Output                                            | Schema                                                                                                                                                                                                                      |
| --- | ------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | LangGraph swarm (agents + LLM + tool calls) | `src/trace_processing/log_generator.py` → `trace_processor.py` | `./generated_event_log/<timestamp>.eventlog.csv`  | `case_id, identity:id, time:timestamp, concept:instance, concept:name, org:resource, message, time_finished, duration, model, input_tokens, response_tokens, tool` (+ `feedback_*` on the trailing `customer_feedback` row) |
| 2   | Coffee machine FastAPI service              | `services/coffee_machine/logger.py`, `state.py`                | `services/coffee_machine/logs/coffee_machine.csv` | `case_id, concept:name, ocel_time, duration, org:resource, job_id, drink`                                                                                                                                                   |
| 3   | Guardrail gateway                           | _not merged yet_                                               | TBD                                               | TBD — must conform to the schema below                                                                                                                                                                                      |

Stream 1 is the source of truth for the schema. Streams 2 and 3 must
adapt their column names so the rows can be concatenated and re-sorted.

> **Stream 2 schema is implicit, not enforced.**
> `services/coffee_machine/logger.py` generates the CSV header from the
> `**attrs.keys()` of the **first** row written to a fresh file. Adding a
> kwarg in `state.py`'s `emit_event(...)` silently changes the header for new
> files but corrupts existing ones (header has N columns, rows have N+1).
> Tracked in [Open thoughts](#open-thoughts).

---

## Canonical columns

The merged log uses the agent-log column names. Columns are grouped by
who's expected to fill them.

### Always required (every row)

| Column             | Type          | Meaning                                                                                                            |
| ------------------ | ------------- | ------------------------------------------------------------------------------------------------------------------ |
| `case_id`          | string (UUID) | Process instance. Equals LangGraph `thread_id` and the coffee machine `correlation_id`. **One value per session.** |
| `identity:id`      | string (UUID) | Unique event id. Generated per row by whoever emits it.                                                            |
| `time:timestamp`   | ISO-8601 ms   | Event start time, format `YYYY-MM-DDTHH:MM:SS.fff`.                                                                |
| `time_finished`    | ISO-8601 ms   | Event end time, same format. For instantaneous events: equal to `time:timestamp`.                                  |
| `concept:name`     | string        | Activity type. Closed vocabulary — see [Activity vocabulary](#activity-vocabulary).                                |
| `concept:instance` | string        | Mostly a label, but the value `"prompt"` is **load-bearing** — see note below.                          |
| `org:resource`     | string        | Who emitted the event. Closed vocabulary — see [Resources](#resources).                                            |

> **`concept:instance` is partly fixed.** For `user_prompt` rows, the value
> MUST equal the literal string `"prompt"` —
> `_preprocess_eventlog` in `eventlog_conversion.py:411` keys on it to mint
> the `prompt` object. Other rows use free-form labels like
> `"order_agent calls llm"` or `"barista_agent uses tool start_preparation"`,
> which are decoration only.

### Conditionally required (depending on `concept:name`)

| Column            | Type     | Required for                               | Meaning                                                                       |
| ----------------- | -------- | ------------------------------------------ | ----------------------------------------------------------------------------- |
| `duration`        | int (ns) | All events with measurable duration        | Event duration in nanoseconds. Matches OTel native units; read directly off the span in `log_generator.py:101`. |
| `message`         | string   | `user_prompt`, `call_llm` (assistant text) | The natural-language content.                                                 |
| `model`           | string   | `call_llm`                                 | Model id (`ministral-3:14b`, `claude-…`).                                     |
| `input_tokens`    | int      | `call_llm`                                 | Prompt-side token count.                                                      |
| `response_tokens` | int      | `call_llm`                                 | Completion-side token count.                                                  |
| `tool`            | string   | `execute_tool`                             | Tool name (e.g. `process_order`, `start_preparation`).                        |

> **Unit mismatch today.** The agent log already emits `duration` in
> nanoseconds (OTel). The coffee machine writes seconds (Python `time.time()`
> deltas). The merge step multiplies stream 2's `duration` by 1e9 — see merge
> step 4.

### Optional / source-specific (nullable for rows that don't apply)

| Column            | Type   | Filled by               | Meaning                                                                         |
| ----------------- | ------ | ----------------------- | ------------------------------------------------------------------------------- |
| `feedback_score`  | float  | `customer_feedback` row | 0.0–1.0 customer rating.                                                        |
| `feedback_reason` | string | `customer_feedback` row | Short free-text explanation.                                                    |
| `feedback_valid`  | bool   | `customer_feedback` row | Whether the LLM response parsed cleanly.                                        |
| `job_id`          | string | coffee machine          | Internal brew job id. Multiple `job_id`s per `case_id` are possible (re-brews). |
| `drink`           | string | coffee machine          | Drink type for the brew job.                                                    |

Optional columns that don't apply to a row are written as the empty CSV
token. Polars reads them back as `null`, and the
OCEL converter checks via `is_not_null()`. Emitters MUST NOT write a literal
`""` deliberately — that would produce a non-null empty string and slip past
those checks.

---

## Activity vocabulary

`concept:name` values, grouped by emitter. **Closed set** — adding a new
activity means updating this doc _and_ `EVENT_ATTRIBUTES` in
`src/trace_processing/eventlog_conversion.py`.

### From the agent log (stream 1)

| `concept:name`      | When                                                         | `org:resource` |
| ------------------- | ------------------------------------------------------------ | -------------- |
| `user_prompt`       | Customer agent submits a turn to the swarm                   | `user`         |
| `call_llm`          | An agent invokes its LLM                                     | `<agent_name>` |
| `execute_tool`      | An agent runs a tool (the tool name lands in the `tool` col) | `<agent_name>` |
| `customer_feedback` | Trailing per-case feedback row appended by `TraceProcessor`  | `user`         |

### From the coffee machine (stream 2)

| `concept:name`   | When                                                                                                                                   | `org:resource`   |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| `process_order`  | Brew job transitions to `brewing`                                                                                                      | `coffee_machine` |
| `brew_completed` | Brew job finishes successfully                                                                                                         | `coffee_machine` |
| `brew_failed`    | Brew job fails                                                                                                                         | `coffee_machine` |
| `clean_machine`  | Machine cleaned (today emitted by the agent's tool call — could double-emit from the machine itself once it tracks cleaning lifecycle) | `coffee_machine` |

> **Note on overlap with stream 1.** The barista agent calls
> `start_preparation` / `end_preparation` (logged as `execute_tool` rows
> with `org:resource=barista_agent`; both are emitted per brew — see
> `src/agents/barista_agent.py:448`). The machine then emits its own
> `process_order` / `brew_completed` / `brew_failed` rows with
> `org:resource=coffee_machine`. **Both are kept** — they describe the
> same job from two perspectives (agent intent vs. physical reality).

---

## Resources

`org:resource` is also a closed set:

- `user` — the (simulated) customer
- `order_agent`, `inventory_agent`, `barista_agent`, `customer_service_agent` — the coffee shop agents
- `coffee_machine` — the FastAPI service

The OCEL conversion uses this column to derive object types (see
`object_type_agent` in `eventlog_conversion.py`). New resource names
must be added to that mapping too.

---

## Merge procedure

The merged file is produced by a post-processing pass after a session
ends. Today this is `TraceProcessor.process_all_traces()`. The pass:

1. **Read** the MLflow traces under `./mlruns/` and convert each to a
   per-case DataFrame via `LogGenerator.generate_event_log_df()`.
   _(existing behaviour)_
2. **Concat** all per-case DataFrames into one combined log.
   _(existing behaviour)_
3. **Append** the customer-feedback row per case (one per `case_id`) at
   the end of that case. _(existing behaviour)_
4. **Read** `services/coffee_machine/logs/coffee_machine.csv`. Map its
   columns into the canonical schema:
   - `concept:name` → as-is
   - `ocel_time` (epoch float) → `time:timestamp` (ISO-8601 ms),
     `time_finished` = `time:timestamp + duration`
   - `duration` → as-is, but converted to nanoseconds for parity with
     stream 1
   - `org:resource` → `coffee_machine` (already correct)
   - `job_id`, `drink` → carry over into the optional columns
   - generate a fresh `identity:id` per row
   - synthesise `concept:instance` as e.g. `f"coffee machine {concept:name} ({drink})"`
   - leave `message`, `model`, `input_tokens`, `response_tokens`, `tool`,
     `feedback_*` empty
5. **Read** the guardrail log (once it exists). Same mapping idea.
6. **Concat** everything.
7. **Filter** to rows whose `case_id` exists in the agent log (drops
   stale machine/guardrail entries from earlier sessions whose mlruns
   directory was deleted).
8. **Sort** by `time:timestamp`, write to `./generated_event_log/<ts>.eventlog.csv`.

The existing `customer_feedback` injection is the template — the
coffee-machine merge follows the same shape with a different source CSV.

---

## Open thoughts

These don't block writing the merge code, but worth flagging:

1. **Coffee-machine CSV lifecycle.** Decided: append-only, same as
   `mlflow.db`. Every export re-merges every row whose `case_id` is still
   in MLflow (the filter in step 7 drops the rest). The raw CSV grows
   across sessions; rotation is out of scope for now.
2. **Stream 2 header fragility** — see the warning under [Where logs live
   today](#where-logs-live-today). The fix is to make
   `services/coffee_machine/logger.py` enforce a fixed header (and reject
   unknown kwargs), but that's a separate change.
3. **Identical-timestamp ordering.** `_preprocess_eventlog` in
   `eventlog_conversion.py` uses `pl.col(...).shift(-1)` to detect
   handovers. If two rows of the merged log share `time:timestamp` (e.g.
   barista `execute_tool` and machine `process_order` rounded to the same
   millisecond), the order after `sort_values` is not guaranteed and could
   produce phantom or missed handovers. Today `time.time()` gives μs
   resolution so collisions are rare — but not impossible. Could be fixed
   later by adding a per-case monotonic `seq` column at merge time.
4. **Fate of `concept:instance`.** With `"prompt"` being the only
   load-bearing value, the rest of the column is decoration. Could be
   dropped entirely if the OCEL converter is updated to key on
   `concept:name == "user_prompt"` instead. Separate cleanup.
