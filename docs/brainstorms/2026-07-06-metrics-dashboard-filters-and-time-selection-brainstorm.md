---
title: Metrics Dashboard — filters & improved time selection
date: 2026-07-06
status: brainstorm (grilled)
---

# Metrics Dashboard — filters & improved time selection

## What We're Building

Two related improvements to the Metrics Dashboard sidebar (`src/dashboard/metrics/metrics_page.py`):

1. **Better time-window control.** Replace the current `DatetimeRangeSlider` + static label with **preset buttons + two editable `DatetimePicker` inputs**. The slider is dropped — it's low-precision on wide date ranges, which was the original pain point.

2. **Case-level filters for Scenario and Configuration.** Two new collapsible filter cards in the left sidebar:
   - **Scenario** — multi-select over the four `CUSTOMER_SCENARIOS` plus a `-1` "Custom / Unspecified" bucket.
   - **Configuration** — multi-select over the active setup names (`baseline` / `all_handovers` / `unconstrained`) plus an "(unknown)" bucket for untagged cases.

3. **Staged filters + Apply button.** Nothing re-renders until "Apply filters" is clicked. Trace count label stays live while staging so the user knows in advance whether their staged filter selects zero cases.

## Why This Approach

- **Presets + datetime inputs cover all three interaction modes** without the slider: common windows for the 80% case, precise input when the timestamp is known. The slider becomes ornamental at wide ranges (~0.6 px/min over a week), so it goes.
- **Persist `setup` and `scenario_index` per trace via MLflow tags**, then extract into the event-log CSV. This is the only way to filter after the fact once mixed configurations/scenarios accumulate.
- **Scenario is a pre-declared input** (from the dashboard toggle or `--scenario` CLI arg) — *not* backfilled from the terminal `user_feedback` event. This mirrors the setup mechanism and works even for conversations that never reach feedback.
- **Case-metadata table lives outside OCEL.** OCEL's `EVENT_ATTRIBUTES` and `_preprocess_eventlog.cols_to_keep` are whitelists — polluting them with view-time filter fields (setup, per-case scenario) is the wrong direction. The filter joins a lightweight `case_id → (scenario_index, setup)` table built in `_load_combined_eventlog`.
- **AND across filter groups** matches the mental model ("baseline runs of scenario 1 in the last hour").
- **Collapsible sidebar cards** keep the sidebar tidy as more filters land later.
- **Clean-slate reset (`poetry run reset-traces`)** avoids the "null column on all old traces" problem. Shipped in one PR with the tagging + extractor bump.

## Key Decisions

| Decision | Choice |
|---|---|
| Time-selection UX | **Preset buttons + two editable `DatetimePicker` inputs**. Slider dropped. |
| Presets shipped | Last 10 min, Last hour, Last 24h, Today, All |
| Filter apply model | **Staged filters + "Apply filters" button** — single re-render trigger |
| Trace count label | **Updates live** while filters stage (before Apply) |
| Apply button state | Disabled when staged state equals applied state |
| Filter combine | AND across groups (time ∧ scenario ∧ configuration) |
| Filter granularity | Multi-select within a group (checkbox list) |
| Sidebar layout | Three collapsible cards: Time, Scenario, Configuration; badge shows active-filter count |
| Empty state | Generic: "No traces match the current filters." No smart hint, no reset button in scope. |
| Time bounds | Global (over all loaded cases). Hint shows filtered subset's actual span. |
| Configuration card with 1 value | Always visible |
| Case-metadata storage | Lightweight polars frame `case_id → (scenario_index, setup)` built in `_load_combined_eventlog`, joined at filter time. **OCEL untouched.** |
| Setup tag flow | Threaded via `CoffeeShopConfig.setup_name`; `ConversationEngine.__init__` grows a `setup_name` param. Three trace sites call a shared `_tag_trace_with(setup_name, scenario_index)` helper in `coffee_shop.py` right after `app.stream(...)`. |
| Scenario tag flow | Per-conversation state on the runner/engine (not param-threaded). Dashboard reads toggle at conversation start; simulate reads `--scenario` (default: random `randint(0, len(CUSTOMER_SCENARIOS)-1)`). |
| Scenario tag value | 0-3 for preset scenarios; **`-1`** when `custom_prompt` is used, and for the Jupyter path (`NotebookUI`). |
| Configuration null bucket | "(unknown)" — traces from crashed conversations or any that slip past tagging are visible and filterable. |
| Migration | `poetry run reset-traces` in the same PR as the tagging + extractor bump. Clean slate. |

### Trace-tagging helper

```python
# coffee_shop.py
def _tag_trace_with(setup_name: str, scenario_index: int) -> None:
    trace_id = mlflow.get_last_active_trace_id()
    if trace_id is None:
        return
    mlflow.update_current_trace(tags={"setup": setup_name, "scenario_index": str(scenario_index)})
```

Called by:
- `ConversationEngine.send_message` (`conversation.py:40` — needs `setup_name` from new constructor arg + `scenario_index` from per-conversation state)
- `ConversationRunner._stream_with_events` (`conversation_runner.py:301` — has `self.shop.config.setup_name`; scenario from `self._current_scenario_index`)
- `NotebookUI._stream_to_output` (`notebook_ui.py:208` — has `self.shop.config.setup_name`; scenario always `-1`)

### Must-verify before implementing

- **`mlflow.update_current_trace(tags=...)` behaviour under `mlflow.langchain.autolog()`**. The autolog wrapper creates the trace implicitly; we need to confirm tags applied after `app.stream(...)` returns actually attach to that trace.

## Resolved Questions

- **`setup` source at extraction time.** MLflow trace tag `"setup"`, written by the shared helper at three `app.stream` sites. Extractor reads the tag into a new `setup` column at cache-build time.
- **Time filter granularity.** Fully-contained cases only. Unchanged from today.
- **Configuration card when only one setup exists.** Always show the card.
- **Old-trace migration.** `poetry run reset-traces` — clean slate, ships in the same PR.
- **Scenario source.** Pre-declared per conversation (dashboard toggle / `--scenario` CLI). Not derived from feedback.
- **Custom prompt / no-scenario tag value.** `-1`.
- **Slider vs. inputs vs. presets.** Slider dropped; presets + datetime inputs only.
- **Filter apply behaviour.** Staged with an Apply button; count label stays live.
- **Time bounds.** Global; hint shows filtered subset span.
- **OCEL impact.** None. Case-metadata table lives alongside OCEL.
- **Configuration null bucket.** "(unknown)" bucket.
- **Empty state hint.** Generic message; no smart suggestion, no reset button in scope.

## Open Questions

_None — ready for planning._

## Non-Goals

- No new metric sections.
- No changes to the MLflow storage schema itself — only trace tags.
- No cross-filtering from charts (click a scenario bar → apply filter). Follow-up.
- No "smart" empty-state hints (which filter to relax, how many extra traces it would surface). Follow-up if the generic message proves inadequate.
- No adaptive time bounds. Global only for this change.

## Files Likely Touched

- `src/dashboard/metrics/metrics_page.py` — sidebar refactor (three cards + Apply button), remove slider, add datetime inputs + preset buttons, wire staged filter state, build case-metadata frame in `_load_combined_eventlog`, add filter-subset span hint, live count-label wiring.
- `src/dashboard/metrics/trace_cache.py` — bump `SCHEMA_VERSION`; write new `setup` column from MLflow trace tag.
- `src/trace_processing/trace_processor.py` — read `setup` tag from MLflow trace, emit to CSV row.
- `src/coffee_shop.py` — new shared helper `_tag_trace_with(setup_name, scenario_index)`.
- `src/conversation.py` — `ConversationEngine.__init__` grows `setup_name` param + per-conversation `scenario_index` state; `send_message` calls helper after `app.stream`.
- `src/dashboard/interaction/conversation_runner.py` — track `self._current_scenario_index` at conversation start (from dashboard toggle / `custom_prompt` sentinel `-1`); call helper after `stream`.
- `src/notebook_ui.py` — call helper after `stream` with `scenario_index=-1`.
- `src/simulate.py` — add `--scenario` CLI arg (append, default random per conversation); pass into per-conversation flow.
- `src/dashboard/interaction/interaction_page.py` — no logic change; scenario toggle value is already available (`interaction_page.py:222`).

## Next Step

Run `/ce:plan` to design the concrete implementation slices and verify the MLflow-autolog tag interaction before coding.
