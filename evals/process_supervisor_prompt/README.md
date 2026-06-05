# Process Supervisor Prompt Eval

Goal: find a single, dense, LLM-friendly description of the BPMN process that beats the
current setup (markdown narrative `docs/order-process-flow.md` + YAML catalog
`config/process_model.yaml` jammed into the prompt together).

## Files

- `ground_truth.jsonl` — one JSON object per line, derived from
  `process_log/process_meta.log` (11 messages from one full conversation). Each entry has:
  - `i` — line number in the source log
  - `agent`, `kind`, `trigger`, `tool`, `target`, `content_excerpt` — message attributes
    that get passed to `_llm_decide`
  - `expected` — the **correct** supervisor decision, NOT what the existing log says.
    The previous supervisor was Opus-era and had the BPMN-ID parse bug, so most of its
    `Violation:llm_unparseable_output` lines are wrong. We re-labelled from the BPMN +
    YAML catalog directly.
  - `rationale` — short explanation of why this is the correct answer

## Labelling rules

1. **Tool-call messages** with a tool that maps uniquely to one `(agent, trigger=tool_call, tool)`
   entry in `config/process_model.yaml` → `Execution:<id>:<slug>`.
2. **Handoff messages** (`transfer_to_agent`) → `Termination:<id>:<slug>:via_handoff_to_<target>`,
   where `<id>` is the most recent open Execution by the source agent. (See
   `process_supervisor._terminate_for_handoff`.)
3. **Plain-text messages** from `order_agent` → `A01` if no AND-split branch fired yet,
   else `A07` (terminal). From `customer_service_agent` → `A08` then `A10`.
   Other agents have no message-trigger activity → Violation.
4. **Tool-call with a tool not in the catalog**, or a tool bound to a different agent's
   lane → `Violation:no_activity_for_<agent>_<trigger>_<tool>`.

## Scoring (when the harness runs)

- **Exact-match accuracy** on `(decision_kind, activity_id)` is the headline number.
- **Reason text** for Violations is *not* exact-matched — only the `Violation` decision
  kind is required. Different prompts will produce different (still valid) reason
  phrasings.
- **`llm_unparseable_output` rate** is tracked separately — that's a format-following
  failure, not a wrong-answer failure.
- **Mean per-call latency** is tracked too.

## Status

**WAITING FOR HUMAN APPROVAL** of `ground_truth.jsonl` before running the eval against
the candidate prompts.
