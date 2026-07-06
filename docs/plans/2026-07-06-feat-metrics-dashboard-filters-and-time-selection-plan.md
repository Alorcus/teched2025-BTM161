---
title: Metrics Dashboard — filters & improved time selection
type: feat
status: completed
date: 2026-07-06
origin: docs/brainstorms/2026-07-06-metrics-dashboard-filters-and-time-selection-brainstorm.md
---

# Metrics Dashboard — filters & improved time selection

## Overview

Redesign the Metrics Dashboard's left sidebar so users can slice the metric sections by three dimensions instead of one: **time** (preset buttons + editable `DatetimePicker` inputs), **scenario** (`CUSTOMER_SCENARIOS` 0–3 plus `-1` for custom/unspecified), and **configuration** (setup name — `baseline`, `all_handovers`, `unconstrained`, …). Filters stage locally and only re-render the metric sections when an **Apply filters** button is clicked; the trace count label and filtered-span hint stay live during staging.

To make this possible, MLflow traces must carry the setup name and the played scenario as trace tags. A shared helper attaches those tags at the three sites that call `app.stream(...)` (simulate/headless, dashboard-interactive, Jupyter). The extractor reads the tags into two new case-level columns in `_all_traces.csv`. A migration step (`poetry run reset-traces -y`) is required in the same PR because old traces have no tags — this is scoped to a demo/teaching repo where a clean slate is acceptable (see brainstorm: docs/brainstorms/2026-07-06-metrics-dashboard-filters-and-time-selection-brainstorm.md, "Old-trace migration" decision).

## Problem Statement

Two problems on the same page:

1. **Time selection is unusable at wide date ranges.** The current `DatetimeRangeSlider` spans the min/max of every timestamp in the loaded event log. With a 7-day dataset compressed into a ~280 px sidebar the slider offers roughly 0.6 pixels per minute; selecting a 10-minute window is a 6-pixel target. The workaround pattern (drag, miss, drag again) is friction the user reports repeatedly.
2. **There is no way to filter by scenario or configuration.** The metric sections aggregate across every conversation in the store — a baseline run and an all_handovers run are indistinguishable, and a scenario-0 conversation is mixed with scenario-3. Because setup is chosen at app startup and scenario is a per-conversation input, the data was never persisted per-trace.

## Proposed Solution

### Sidebar redesign

Three collapsible `pn.Card` sections stacked in the sidebar (`agent_panel.py:151` is the existing precedent):

- **Time** — five preset buttons (`Last 10 min`, `Last hour`, `Last 24h`, `Today`, `All`) + two editable `DatetimePicker` inputs (`allow_input=True`, `enable_time=True`, `enable_seconds=True`). The existing `DatetimeRangeSlider` is removed.
- **Scenario** — `pn.widgets.CheckBoxGroup(inline=False)` with human-readable labels; underlying values are the raw indices from `CUSTOMER_SCENARIOS` (`0..3`) plus `-1` ("Custom / Unspecified").
- **Configuration** — `pn.widgets.CheckBoxGroup(inline=False)` populated from the distinct `setup` values found in the loaded case-metadata table. `(unknown)` is a bucket for cases whose trace has no `setup` tag.

Each card shows a small count badge in its title when filters are active. Below the three cards: a live **trace count label** (contained cases + partial-excluded count, identical structure to today's `_format_count_label`), a live **span hint** ("Selected cases span YYYY-MM-DD HH:MM → HH:MM"), and a primary **Apply filters** button. Apply is disabled when staged state equals applied state.

### Filter semantics

- **AND across groups**: a case passes iff its `first_t/last_t` are fully contained in the time window AND (scenario filter is empty OR its `case_scenario_index` ∈ scenario filter) AND (setup filter is empty OR its `case_setup` ∈ setup filter).
- **Empty checkbox group = no filter** (all pass). Non-empty = whitelist.
- **Time**: keeps existing "fully-contained cases only" semantics — partial cases are still excluded and reported separately in the count label (see brainstorm: "Resolved Questions / Time filter granularity").

### MLflow trace tagging (write path)

New module-level helper in `src/coffee_shop.py`:

```python
# src/coffee_shop.py — near the bottom of the module
def _tag_current_trace(setup_name: str, scenario_index: int) -> None:
    """Attach setup + scenario as trace tags to the most recently completed
    autolog trace. Safe to call when mlflow is disabled — no-ops if
    get_last_active_trace_id() returns None."""
    trace_id = mlflow.get_last_active_trace_id()
    if trace_id is None:
        return
    mlflow.set_trace_tag(trace_id, "setup", setup_name)
    mlflow.set_trace_tag(trace_id, "scenario_index", str(scenario_index))
```

**API verified against installed `mlflow==3.14.0`**: `mlflow.set_trace_tag(trace_id, key, value)` works on trace IDs *after* the autolog span closes — this is the correct API. `mlflow.update_current_trace(tags=...)` would have looked right but silently no-ops after `app.stream(...)` returns because there is no active span (`fluent.py:1506-1513`; docstring explicitly requires an active trace). Values are stringified server-side; passing `str(scenario_index)` avoids a warning.

The helper is called at three sites immediately after `app.stream(...)` completes:

| Site | File:line | `setup_name` source | `scenario_index` source |
|---|---|---|---|
| Simulate / headless / `run_conversation` | `src/conversation.py:45-53` (after the `for sm in extract_messages(stream)` loop) | New `setup_name` param on `ConversationEngine.__init__` (threaded from `CoffeeShop.open_shop` → `self.config.setup_name`) | Read `self.customer_agent.scenario_index` — see note below |
| Dashboard interactive | `src/dashboard/interaction/conversation_runner.py` around line 301 (needs `import mlflow` — not currently imported) | `self.shop.config.setup_name` | `self._current_scenario_index`, set at start of `_run_conversation` (line 162) from the `scenario_index` param (already 0..3 or `-1`) |
| Jupyter | `src/notebook_ui.py:216` (right next to the existing `mlflow.get_last_active_trace_id()` call in `continue_conversation_interactive`) | `self.shop.config.setup_name` | Constant `-1` (no scenario in the notebook path) |

**Scenario source-of-truth simplification (unblocks flow):** `CustomerAgent.reset(None)` already picks a random scenario internally and stores the resolved integer on `self.scenario_index` (`customer_agent.py:45-50`). For the simulate path, the trace-tag helper reads `shop.customer_agent.scenario_index` *after* `reset()` runs — this covers `--scenario random` (which today passes `None` down the chain) without changing `pick_scenario_index` semantics. For the interactive path, `scenario_select.value` is already the raw int (`-1` when custom prompt in use — the existing watcher at `interaction_page.py:126-155` flips it there), so `runner._current_scenario_index = scenario_index` at `_run_conversation` line 162 is the same value.

### Extractor (read path)

`src/trace_processing/trace_processor.py` currently reads MLflow traces and emits per-event rows into `_all_traces.csv`. Extend it to also read `trace.info.tags.get("setup")` and `trace.info.tags.get("scenario_index")` per trace, and stamp them onto every event row of that case as two new columns: **`case_setup`** and **`case_scenario_index`**. Column names are namespaced to avoid collision with the existing per-event `scenario_index` (currently on `user_feedback` events only, per `eventlog_conversion.py:38`).

Null-tag policy:
- Missing `case_setup` → written as null → filter shows as `(unknown)` bucket.
- Missing `case_scenario_index` → written as `-1` (unify with "custom / unspecified" — one less bucket for the user).

`src/dashboard/metrics/trace_cache.py:42` bumps `SCHEMA_VERSION = 3` → `4` so the cache auto-rebuilds on the next dashboard load. The extractor is idempotent by design.

### Case-metadata table (OCEL clean)

Per the grilled brainstorm decision, OCEL structures (`EVENT_ATTRIBUTES`, `_preprocess_eventlog.cols_to_keep`) are **not** touched (see brainstorm: "OCEL impact — None"). Instead, `_load_combined_eventlog` in `src/dashboard/metrics/metrics_page.py` builds a lightweight polars frame alongside the event log:

```python
# after loading combined event CSV in _load_combined_eventlog
case_metadata = (
    combined.select(["case_id", "case_setup", "case_scenario_index"])
    .drop_nulls("case_id")
    .unique(subset=["case_id"])
)
```

Filter helpers (`_contained_case_ids`, new `_apply_filters`) inner-join `case_metadata` on `case_id` to select the pool, then filter the full event log by `case_id.is_in(...)`.

## System-Wide Impact

### Interaction graph

- `poetry run simulate --setup S --scenario X --traces N` → `simulate.py` loop → `CoffeeShop.run_conversation(scenario_index=idx)` → `CustomerAgent.reset(idx)` (stamps `.scenario_index`) → `ConversationEngine.run_automated` → `send_message` → `app.stream(...)` (autolog creates trace) → `_tag_current_trace(setup, agent.scenario_index)` → `mlflow.set_trace_tag(...)` → later, `TraceProcessor.process_all_traces` reads tags and writes CSV rows with `case_setup`/`case_scenario_index`.
- Dashboard interactive Run → `on_run` reads `scenario_select.value` (already `-1` when custom prompt is set) → `runner.start(scenario_index, custom_prompt)` → `_run_conversation` stamps `self._current_scenario_index` → `_stream_with_events` runs stream → helper tags trace.
- Metrics page load → `ensure_trace_cache` detects `SCHEMA_VERSION` bump → rebuilds `_all_traces.csv` (writes new columns) → `_load_combined_eventlog` builds case-metadata → sidebar cards populate options from distinct values → user stages filters → Apply → `_render_metrics(filter_state)` re-renders.

### Error propagation

- `mlflow.get_last_active_trace_id()` returning `None` (mlflow disabled or trace failed): helper early-returns, no exception. Downstream extractor sees no tag → null → `(unknown)` bucket / `-1`.
- `mlflow.set_trace_tag` errors (network / bad ID): should propagate as-is; we don't want silent tag drops. Tag values are auto-stringified (with a warning for non-string) — we cast explicitly to keep logs clean.
- Panel `DatetimePicker` param validation raises `ValueError` if `value` is outside `[start, end]`. Bounds are computed as `[data_min, max(data_max, datetime.now())]` so preset windows anchored at "now" always fit even when data is stale.
- Extractor: if a trace object lacks `.info.tags`, `.get("setup")` returns `None`. No exception path.

### State lifecycle risks

- **Cache rebuild vs concurrent write**: `ensure_trace_cache` reads `mlflow.db` and writes `_all_traces.csv`. If a simulate run is writing traces while the dashboard is loading, the CSV represents a snapshot. Not a new risk — same as today.
- **Trace tag ordering**: `set_trace_tag` is called *after* `app.stream(...)` returns. If the process is killed between stream-return and tag-set, the trace exists without tags. Extractor writes `null` for that trace's `case_setup` and `-1` for `case_scenario_index`. User can filter them out via `(unknown)`. Acceptable.
- **Filter cardinality changes**: options come from data. If user has scenario 1 checked and a new simulate run adds scenario 2, on next page load the option list grows; the applied "scenario=1" filter still yields the same result. No stale-state bug.

### API surface parity

- Three trace-writer sites (simulate/headless, dashboard-interactive, Jupyter) all route through the shared `_tag_current_trace` helper. No fourth site currently produces traces; verified via `grep -rE "app\.stream|self\.app\.stream" src/`.
- `simulate.py`, `interaction_page.py`, `notebook_ui.py` — no CLI or UI additions needed; scenario_index is already threaded to `run_conversation` / `runner.start`. Only the tag call is new.

### Integration test scenarios

Extend `tests/test_simulation_e2e.py` (which already runs `--setup baseline --scenario 0 --traces 1`):

1. After the subprocess completes, use `MlflowClient.search_traces(...)` to fetch the just-created trace and assert `trace.info.tags["setup"] == "baseline"` and `trace.info.tags["scenario_index"] == "0"`.
2. Run `TraceProcessor().process_all_traces()` and assert `_all_traces.csv` has `case_setup` and `case_scenario_index` columns with the expected values.
3. Run once more with `--scenario random`; assert the resulting trace tag is a string in `{"0","1","2","3"}` (not `"None"`, not `"-1"`).

New unit tests (`unittest`, following the `TestCase`/`test_*` pattern used across `tests/`):

- `tests/test_metrics_filters.py::TestCaseMetadataBuild` — feed a synthetic 3-case polars frame with mixed `case_setup`/`case_scenario_index`, assert the resulting metadata frame has one row per case.
- `tests/test_metrics_filters.py::TestApplyFilters` — 4-6 case pool with varying attributes; assert AND semantics, empty-checkbox = pass-all, `(unknown)` bucket behavior.
- `tests/test_metrics_filters.py::TestPresetWindows` — freeze "now" via a stub; assert `Last 10 min` returns `(now - 10min, now)`, `Today` returns `(midnight, now)`, `All` returns `(data_min, data_max)`.
- `tests/test_coffee_shop_tagging.py::TestTagCurrentTrace` — mock `mlflow.get_last_active_trace_id` + `mlflow.set_trace_tag`; assert the helper skips silently when trace_id is None and calls `set_trace_tag` twice with stringified args otherwise.

## Acceptance Criteria

### Functional requirements

- [x] Sidebar shows three collapsible `pn.Card` sections: Time, Scenario, Configuration.
- [x] Time card contains five preset buttons and two editable `DatetimePicker` inputs; the `DatetimeRangeSlider` is removed. Preset click writes both picker values simultaneously.
- [x] Scenario `CheckBoxGroup` shows human-readable labels for indices `0..3` and a `-1` "Custom / Unspecified" option. Underlying values are the raw ints.
- [x] Configuration `CheckBoxGroup` shows the distinct `case_setup` values from the loaded data, plus `(unknown)` if any case has null setup.
- [x] Trace count label updates live as any filter widget's value changes (before Apply).
- [x] Span hint ("Selected cases span …") updates live alongside the count label.
- [x] Apply button is disabled when staged state equals applied state; enabled otherwise.
- [x] Clicking Apply re-renders the full metric-sections pane with the applied filter; nothing else triggers a re-render.
- [x] All three trace-writing paths (simulate, dashboard interactive, Jupyter) attach `setup` and `scenario_index` tags to their traces.
- [x] Extractor emits `case_setup` and `case_scenario_index` columns in `_all_traces.csv`; existing columns are preserved unchanged.
- [x] `SCHEMA_VERSION` in `trace_cache.py` is bumped so the cache rebuilds on next dashboard load.
- [x] Empty filter combinations render the generic empty-state message: "No traces match the current filters."

### Non-functional requirements

- [x] MLflow tag calls do not raise when `mlflow_enabled=False`.
- [x] Apply → re-render latency is not worse than the current slider-commit re-render (same underlying `_render_metrics`).
- [x] No changes to OCEL (`EVENT_ATTRIBUTES`, `_preprocess_eventlog.cols_to_keep`).
- [ ] `ruff-check` and `ruff-format` pass (pre-commit hook).

### Quality gates

- [ ] `tests/test_simulation_e2e.py` passes with the new tag assertions (requires `dangerouslyDisableSandbox: true` when the Anthropic LLM proxy is used at `localhost:6655` — see AGENTS.md line 112).
- [x] New unit tests for filter helpers and the tag helper pass.
- [ ] Manual smoke check: run `poetry run reset-traces -y`, then `poetry run simulate --setup baseline --scenario 0 --traces 3`, then `poetry run simulate --setup all_handovers --scenario random --traces 3`, start dashboard, verify: (a) both setups appear in Configuration card; (b) scenarios 0 and one of 0..3 appear in Scenario card; (c) filter combinations render expected subsets; (d) Apply button state toggles correctly.

## Implementation Phases

The four phases below are ordered to be individually shippable/testable. Everything ships in **one PR** (per grilled brainstorm decision) so the `reset-traces` migration and the reader/writer changes stay atomic.

### Phase 1 — Trace tagging + extractor + schema bump

**Goal:** Every new trace carries `setup` and `scenario_index` tags; extractor lifts them to CSV columns; cache auto-rebuilds.

Files:

- `src/coffee_shop.py` — add module-level `_tag_current_trace(setup_name, scenario_index)` helper (as shown in Proposed Solution).
- `src/conversation.py`
  - `ConversationEngine.__init__(app, mlflow_enabled, setup_name)` — new `setup_name: str` param. Update the two call sites in `coffee_shop.py` (search for `ConversationEngine(`) to pass `self.config.setup_name`.
  - `send_message` — after the `for sm in extract_messages(self.app.stream(...))` loop completes and before the existing `mlflow.get_last_active_trace_id()` call (line 50), call `_tag_current_trace(self.setup_name, self.customer_agent.scenario_index)`. Note: `send_message` currently doesn't know about the customer agent. `run_automated` does (`customer_agent` param at line 57). Simpler: put the tag call inside `run_automated` right after each per-turn `send_message` — but that re-tags on every turn (harmless; last write wins). Cleanest: hoist the tag call into `run_automated`, which owns the customer_agent reference and runs the per-conversation lifecycle. Confirm at implementation time by reading conversation.py:26-100.
- `src/dashboard/interaction/conversation_runner.py`
  - Add `import mlflow` at top.
  - `_run_conversation(self, scenario_index, custom_prompt=None)` at line 162 — set `self._current_scenario_index = scenario_index` before the stream loop.
  - `_stream_with_events` at line 301 — after the outer `for chunk in self.shop.app.stream(...)` loop completes, call `_tag_current_trace(self.shop.config.setup_name, self._current_scenario_index)`. Import the helper from `src.coffee_shop`.
- `src/notebook_ui.py:216` — in `continue_conversation_interactive`, right before or after the existing `trace_id = mlflow.get_last_active_trace_id()` line, call `_tag_current_trace(self.shop.config.setup_name, -1)`.
- `src/trace_processing/trace_processor.py`
  - In the trace-iteration loop (currently produces feedback rows around lines 197-225), lift `setup_tag = trace.info.tags.get("setup")` and `scenario_tag_raw = trace.info.tags.get("scenario_index")`. Convert `scenario_tag_raw` to int via `int(scenario_tag_raw) if scenario_tag_raw not in (None, "None") else -1`.
  - Add both to every event row of that trace's case. Two new columns: `case_setup` (nullable str) and `case_scenario_index` (int, default `-1` when tag missing).
- `src/dashboard/metrics/trace_cache.py:42` — `SCHEMA_VERSION = 3` → `4`.

Success: `poetry run reset-traces -y && poetry run simulate --setup baseline --scenario 0 --traces 1 && python3 -c "from src.trace_processing import TraceProcessor; TraceProcessor().process_all_traces()"`; then `head -1 generated_event_log/_all_traces.csv` shows `case_setup` and `case_scenario_index` in the header, and every row for that trace has `baseline` / `0`.

### Phase 2 — Sidebar UI: three cards, staged state, Apply button

**Goal:** Sidebar reshaped to match the new filter model; existing slider fully removed; nothing wired to the metric pane yet (Apply is a no-op stub).

Files:

- `src/dashboard/metrics/metrics_page.py`
  - Remove lines 44–85 (slider construction, `slider_label`, both watchers).
  - `_load_combined_eventlog` — after building `combined`, also build `case_metadata` polars frame with columns `case_id`, `case_setup`, `case_scenario_index`, `first_t`, `last_t` (last two via `_case_bounds`). Return both.
  - Build widgets:
    - Five `pn.widgets.Button(name="Last 10 min", button_type="light", ...)` (etc.), stacked in a `pn.Row` or short `pn.Column`.
    - `start_picker = pn.widgets.DatetimePicker(name="From", value=full_start, start=data_bound_min, end=data_bound_max, allow_input=True, enable_seconds=True)` (and analog `end_picker`). Bounds: `data_bound_min = full_start`, `data_bound_max = max(full_end, datetime.now())` — accommodates preset windows anchored at "now" when data is stale.
    - `scenario_group = pn.widgets.CheckBoxGroup(name="", inline=False, options={"0: Latte & croissant …": 0, "1: 2 espressos …": 1, "2: Cold cappuccino complaint …": 2, "3: New recommendation …": 3, "Custom / Unspecified": -1}, value=[])`. Labels reused from `interaction_page.py:99-106` where the same option dict is built.
    - `setup_group = pn.widgets.CheckBoxGroup(name="", inline=False, options=<distinct case_setup values incl. "(unknown)" if any nulls>, value=[])`.
    - `apply_button = pn.widgets.Button(name="Apply filters", button_type="primary", disabled=True)`.
    - Three `pn.Card(<contents>, title="Time" | "Scenario" | "Configuration", collapsed=False, sizing_mode="stretch_width")`.
  - Assemble: `sidebar = pn.Column(<title>, <trace count summary>, time_card, scenario_card, setup_card, trace_count_label, span_hint_label, apply_button, width=320, styles={"padding": "10px 12px 10px 16px"})`.
  - **Staged state** = the widgets themselves. **Applied state** = a small tuple/dict held in a closure (`applied = {"start": full_start, "end": full_end, "scenarios": [], "setups": []}`).
  - Preset button callbacks (`preset_last_10min.on_click(_apply_preset_10min)` etc.) write both picker values.
  - `_restage(event=None)`:
    - Read staged values off widgets.
    - Compute (contained, partial) counts against `case_metadata` filtered by staged criteria; update `trace_count_label`.
    - Compute filtered-subset span; update `span_hint_label`.
    - `apply_button.disabled = (staged == applied)`.
  - Watch every filter widget: `start_picker.param.watch(_restage, "value")`, ..., `scenario_group.param.watch(_restage, "value")`, `setup_group.param.watch(_restage, "value")`.
  - `apply_button.on_click(_on_apply)` — swaps `applied = staged.copy()`, calls `metrics_pane[:] = [_render_metrics(combined, case_metadata, applied)]`, then `_restage()` to reset button state.

Success: `poetry run dashboard --setup baseline`, sidebar renders three cards + Apply. Staging widgets updates the live count and span hint; Apply button toggles enabled/disabled. Clicking Apply still renders the same metric sections as today (filter logic goes in Phase 3).

### Phase 3 — Filter logic + empty state

**Goal:** Apply actually filters.

Files:

- `src/dashboard/metrics/metrics_page.py`
  - New helper `_apply_filters(case_metadata, start, end, scenarios, setups) -> pl.Series` — returns filtered `case_id` series. Semantics:
    - Time: `first_t >= start AND last_t <= end` (existing "fully contained").
    - Scenario: `len(scenarios) == 0 OR case_scenario_index.is_in(scenarios)`. Empty setups is treated as "no filter"; `case_scenario_index` is always non-null (falls back to `-1`).
    - Setup: `len(setups) == 0 OR case_setup.is_in(setups)`. If `(unknown)` is in the selected setups, that maps to null (`case_setup.is_null()`).
  - Refactor `_render_metrics(eventlog, start, end)` → `_render_metrics(eventlog, case_metadata, applied)` where `applied` is the dict from Phase 2. Internally: `keep_ids = _apply_filters(...)`; `filtered = eventlog.filter(pl.col("case_id").is_in(keep_ids))`.
  - Empty case: if `keep_ids` is empty, return `pn.pane.Alert("No traces match the current filters.", alert_type="warning")`. Replaces the current "No events in the selected timeframe." message at line 220-222.
  - `_case_counts(case_metadata, start, end, scenarios, setups) -> tuple[int, int]` — extended to take the same filter args so the live count reflects the full AND.

Success: manually check flow K in the ultrathink section — pick "Last 10 min" on stale data, count = 0. Check scenario 0 + setup baseline; verify metric sections filter down.

### Phase 4 — Tests + migration + polish

**Goal:** Ship-ready.

Files:

- `tests/test_metrics_filters.py` — new unit tests (see "Integration test scenarios" above).
- `tests/test_coffee_shop_tagging.py` — new unit test for `_tag_current_trace`.
- `tests/test_simulation_e2e.py` — extend to assert trace tags exist after the subprocess run and that `_all_traces.csv` contains the new columns.
- Manual migration in the PR description:
  ```
  poetry run reset-traces -y   # stop the dashboard first — reset-traces refuses if port 5006 is bound
  poetry install               # already needed if pyproject / lockfile moved
  # re-run any demo traces you want available
  poetry run simulate --setup baseline --scenario 0 --traces 3
  ```
- Small polish:
  - Card title badges (active-filter counts) — nice-to-have; if it adds >30 min, defer.
  - Span hint styling — reuse `styling_helpers.py`'s HTML color palette.

Success: `pytest -q` green; manual smoke script above works end-to-end.

## Alternative Approaches Considered

1. **Push `case_setup`/`case_scenario_index` into OCEL `EVENT_ATTRIBUTES` on every event type.** Rejected — requires editing 33 event-type whitelist entries plus `cols_to_keep`, and pollutes an event-level data model with case-level attributes that only the view layer needs (see brainstorm: "OCEL impact / Decision B").
2. **`mlflow.update_current_trace(tags=...)` inside a wrapping `mlflow.start_span(...)` context.** Rejected — adds an extra root span to every trace, changing its shape for the process-mining consumer. `set_trace_tag(trace_id, ...)` is the drop-in.
3. **Backfill `case_scenario_index` from the `user_feedback` event (original brainstorm proposal).** Rejected — the value doesn't exist for conversations that never reach feedback (crashes, timeouts) and can't be known until *after* the whole trace has run. Pre-declaring at conversation start is simpler and more honest (see grilled brainstorm Q5).
4. **Keep the `DatetimeRangeSlider` alongside presets.** Rejected — at wide date ranges the slider is low-precision; adding presets while keeping a bad primary control just adds visual noise. The precision problem the user reported wouldn't be fixed by adding controls next to it (see grilled brainstorm Q7).
5. **Live re-render on every filter change instead of Apply button.** Rejected — even at 3 checkboxes rapidly toggled, users hit two-to-four unnecessary re-renders per intent. Apply is one click for one predictable render (see grilled brainstorm Q12).
6. **Adaptive time bounds that snap to the filtered subset.** Rejected — creates circular feedback (change scenario → time bounds shrink → previously-set time range gets clipped) and confuses users. Global bounds + a static span hint gives the information without the surprise (see grilled brainstorm Q11).

## Dependencies & Risks

**Dependencies (all satisfied):**
- `mlflow>=3.4.0,<4.0.0` (installed: 3.14.0). `mlflow.set_trace_tag` present since 2.20; safe.
- `panel==1.9.3`. `DatetimePicker` (with `allow_input`), `pn.Card`, `pn.widgets.CheckBoxGroup(inline=False)` all present.
- `polars` (any version currently in use). `.join`, `.is_in`, `.group_by().agg()` — nothing exotic.

**Risks:**
1. **`ConversationEngine`'s tag site**: `send_message` is called per-turn, `run_automated` is per-conversation. Placing the tag call in the wrong one either re-tags every turn (harmless — `set_trace_tag` overwrites) or misses solo `send_message` calls from the dashboard's per-turn user input. **Mitigation**: put the tag call in the *only* place both paths funnel through, which is right after `app.stream(...)` in `send_message`. Confirm the run_automated → send_message topology at implementation time.
2. **Concurrent simulate + dashboard**: rebuilding `_all_traces.csv` mid-run could show partial data. Not new; same as today's cache.
3. **`reset-traces` port check**: refuses when port 5006 is bound (`reset.py:106-121`). Users need to stop the dashboard first. Call this out in the PR description.
4. **AGENTS.md claims "no test suite"** but tests exist. Update AGENTS.md line 99 as part of this PR? Out of scope — leave as follow-up unless the review flags it.
5. **`--scenario random` currently sends `None` down the chain and the customer agent picks randomly.** Reading `customer_agent.scenario_index` *after* reset returns the actual played index. If the future change threads scenario differently (e.g., passing directly to the LangGraph node), the read location may need to move. Not urgent.

## Success Metrics

- Manual: opening the dashboard with mixed setup/scenario data, a user can filter to any (setup, scenario, time) intersection in ≤3 clicks and 1 Apply.
- Automatic: the extended `test_simulation_e2e.py` passes on first run against a freshly reset store.
- Zero regression in the existing metric sections when filters are all-off (identical output to today).

## Documentation Plan

- `AGENTS.md` — mention the trace-tag convention in a new short "MLflow trace tags" section (setup + scenario). Small.
- Inline docstrings for `_tag_current_trace`, `_apply_filters`, and the sidebar restage helper. Follow existing docstring style in `metrics_page.py`.
- PR description covers the `reset-traces` migration step.

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-07-06-metrics-dashboard-filters-and-time-selection-brainstorm.md](../brainstorms/2026-07-06-metrics-dashboard-filters-and-time-selection-brainstorm.md). Key decisions carried forward:
  - Drop the `DatetimeRangeSlider`; use preset buttons + editable `DatetimePicker` inputs.
  - Persist `setup` and `scenario_index` as MLflow trace tags (not derived from events).
  - Case-metadata table lives *outside* OCEL — clean separation of concerns.
  - Staged filters + Apply button; live count and span hint during staging.
  - Empty-state message is generic; no smart hint.
  - Ship in one PR with a `poetry run reset-traces -y` migration step.

### Internal References

- Slider pattern to remove: `src/dashboard/metrics/metrics_page.py:44-85`.
- `pn.Card` precedent: `src/dashboard/interaction/agent_panel.py:151-161`.
- Existing `get_last_active_trace_id` sites: `src/conversation.py:50`, `src/notebook_ui.py:216`.
- Scenario option dict already built for interactive dashboard: `src/dashboard/interaction/interaction_page.py:99-106`.
- Custom-prompt / scenario watcher that keeps `scenario_select.value == -1` when a custom prompt is set: `src/dashboard/interaction/interaction_page.py:126-155`.
- Setup name resolution: `src/setups.py:36-56`.
- Simulate scenario handling: `src/simulate.py:21-46, 59-64, 168-190`.
- Config field carrying setup: `src/config.py:25`.
- Reset script: `src/reset.py:24-121`.
- Extractor entry: `src/trace_processing/trace_processor.py:82-225`.
- Cache schema version: `src/dashboard/metrics/trace_cache.py:42`.
- OCEL structures we do NOT touch: `src/trace_processing/eventlog_conversion.py:10-129, 487-503`.

### External References

- `mlflow.set_trace_tag` — `mlflow==3.14.0`, `.venv/lib/python3.13/site-packages/mlflow/tracing/fluent.py:1576-1599`. Docstring explicitly allows tagging traces that have already ended.
- `mlflow.update_current_trace` — same file, lines 1395-1573. Does NOT work post-stream; keep as documentation for why we chose `set_trace_tag`.
- `pn.widgets.DatetimePicker` — Panel 1.9.3, `.venv/lib/python3.13/site-packages/panel/widgets/input.py:645-832`. `allow_input=True` required for typed editing.
- `pn.Card` — Panel 1.9.3, `.venv/lib/python3.13/site-packages/panel/layout/card.py:19-126`.
- `pn.widgets.CheckBoxGroup` — Panel 1.9.3, `.venv/lib/python3.13/site-packages/panel/widgets/select.py:1153-1249`.

### Related Work

- No related PRs or open issues in this repo — no external project tracker configured (verified via `AGENTS.md`, `CLAUDE.md`, and `.github/`).
- Adjacent recent change: `f493f70 add time frame window to metrics dashboard` (git log) — this plan extends that commit's slider work.
