# Updating the Process Model

This guide lists every file that must be touched when a new BPMN process
diagram is dropped into the repo (e.g. `docs/order-process-vN.bpmn`) so the
Process Supervisor reflects the new flow.

The supervisor reads three things at runtime:

1. A **markdown narrative** of the BPMN (loaded as the LLM prompt's
   "Process description" block).
2. A **YAML activity catalog** that maps `(agent, trigger, tool)` triples
   to BPMN activity IDs/names — used by both the LLM (catalog block) and
   the deterministic fallback matcher.
3. The **supervisor source** itself, but only when the new BPMN introduces
   structural patterns that need a code change (see "When the supervisor
   itself must change" below).

## Files to update

| # | File | What to change |
|---|------|----------------|
| 1 | `docs/<new-name>.bpmn` | The new BPMN XML. Source of truth for the diagram. Keep the old file alongside if useful for diffs. |
| 2 | `docs/order-process-flow.md` | Rewrite to mirror the new BPMN: lanes, activities, gateways, control-flow ASCII diagram, sequence-flow table, control-flow semantics, cross-lane handoffs. The supervisor reads this file *verbatim* as the LLM "Process description" — accuracy here drives LLM-mode quality. |
| 3 | `config/process_model.yaml` | The activity catalog. Bump `name:` (e.g. `coffee_shop_order_v4`). For each BPMN activity, add an entry with a stable `id` (`AnnA01..AnnNN`), a slug `name`, the `display_name` from the BPMN label, the `agent` (lane), the `trigger` (`message` / `tool_call`), the `tool` name (for tool-call triggers), and `terminal: true` for activities that immediately precede an End event. Leave `description_source: docs/order-process-flow.md` as-is unless you renamed the markdown. |
| 4 | `src/control_plane/process_supervisor.py` | **Only if needed.** Extend `_deterministic_pick` whenever two activities share the same `(agent, trigger, tool)` key — i.e. the deterministic matcher cannot tell them apart from the message alone. Today: `A01` vs `A07` (`order_agent`, `message`) and `A08` vs `A10` (`customer_service_agent`, `message`). Pattern: branch on whether a prior activity-id has been seen via `_has_seen_activity_id("A04:", ...)`. |

That's it. The supervisor picks up the new model on the next process
restart — restart the dashboard (`poetry run dashboard`) for the changes
to take effect; modules are not hot-reloaded.

## Step-by-step recipe

1. **Diff the BPMN.** List added/removed lanes, activities, gateways, end events. For each new activity note: lane (= agent), whether it's tool-driven or message-driven, the matching tool name in `src/agents/`.
2. **Rewrite `docs/order-process-flow.md`** end-to-end. The supervisor injects the entire markdown into every LLM call, so prefer concise tables over prose. Existing structure (Lanes → Activities → Events & Gateways → Control Flow ASCII → Sequence Flows → Semantics → Cross-Lane Handoffs) is the intended template.
3. **Edit `config/process_model.yaml`:**
    - Bump `name:` so `process_log/` consumers can tell logs apart by version.
    - For each BPMN activity, add a YAML block. Pick IDs that group logically (`A01..A07` order branch, `A08..A10` complaint branch) — IDs are matched in the deterministic fallback by exact string, but humans read them in the log.
    - For `tool_call` activities, the `tool:` field MUST equal the `name` of the `@tool`-decorated function in `src/agents/<agent>.py`. Mismatches silently produce `Violation:no_activity_for_…`.
    - Mark `terminal: true` on every activity whose outgoing edge is a BPMN End event.
4. **Check for new ambiguities.** Group your YAML entries by `(agent, trigger, tool)`. Any group with more than one entry needs a disambiguator branch in `_deterministic_pick`. The pattern:
    ```python
    if agent == "<lane>" and trigger == "message" and tool is None:
        target_id = "<terminal-id>" if self._has_seen_activity_id("<earlier-id>:") else "<entry-id>"
        return self._activities_by_id.get(target_id)
    ```
5. **Smoke-test.** Run a synthetic trace per new branch through `ProcessSupervisor.observe(...)` in deterministic mode (no LLM needed) and inspect the log file. Each happy-path message should emit `Execution:` / `Termination:` lines; nothing should land as `Violation:no_activity_for_…` unless intended.
6. **Restart** the dashboard so the changes load.

## When the supervisor itself must change

The matcher knows about three trigger kinds: `message`, `tool_call`,
`handoff`. New BPMN constructs may need code support:

- **New gateway types** (event-based, complex) — the supervisor doesn't model gateways; they're just transitions between activities. Usually no code change needed.
- **Message events that aren't agent-initiated** — the current model treats inbound `HumanMessage` as `NonAction:user`. If a BPMN message-flow needs to map to an activity, extend `_classify_message`.
- **A new "trigger" kind** (e.g. timer, signal) — extend `_classify_message` and the `Activity.trigger` field.
- **Hierarchical processes / sub-processes** — would need a parent/child stack rather than the current flat in-memory log; a bigger change.

For most BPMN edits (adding/removing activities, adding XOR/AND branches,
new lanes for new agent kinds) the changes are confined to files 1–3.

## Common mistakes

- **Forgetting `terminal: true`** — the supervisor will emit `Execution:` instead of `Termination:` and never close the case.
- **Tool-name typos** — the `tool:` field in YAML must match the registered tool name letter-for-letter; the deterministic matcher silently falls through to `Violation:no_activity_for_…`.
- **Mixed up lanes** — assigning an activity to the wrong `agent:` makes it a violation when the actual lane invokes it.
- **Reusing an ID across versions without bumping `name:`** — log readers conflate runs from different model versions.
- **Editing the BPMN but not the markdown** — the deterministic matcher will still work, but the LLM-mode supervisor sees stale narrative and tends to over-flag.
- **Not restarting the dashboard** — the in-memory `ProcessSupervisor` is constructed once at `CoffeeShop.open_shop()`; YAML/markdown changes only take effect on restart.
