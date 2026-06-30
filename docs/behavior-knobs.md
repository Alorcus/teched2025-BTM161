# Behavior Knobs Overview

A *knob* is anywhere a developer or operator can change runtime behavior of the
multi-agent coffee shop. This document is a map of every knob in the repo,
grouped by family, with the loading path, the consumer, and a copy-pasteable
pattern for adding a new rule under each family.

- **Direct knob** — its job is to steer behavior. Editing it is the intended
  way to change what the system does (e.g. an agent prompt, a guardrail entry).
- **Indirect knob** — it steers behavior as a side effect of doing something
  else (e.g. a tool implementation, an enum value, a log schema, a CLI flag).

> All file paths below are relative to repo root. Line numbers may shift as the
> code evolves — search by symbol name when in doubt.

---

## 1. At-a-glance matrix

| Knob | Kind | Directness | Behavior types | Where |
|------|------|------------|----------------|-------|
| Agent YAML (`base_prompt`, `tools`, `guardrails`, `guidelines`, `allowed_handovers`) | YAML | direct | content, routing, tool-use, compliance | `config/setups/<setup>/agents/*.yaml` |
| Guideline YAML (appended to every agent that references it) | YAML | direct | content | `config/setups/<setup>/guidelines/*.yaml` |
| Guardrail YAML (hard predicate or soft judge) | YAML | direct | compliance, tool-use, observability | `config/setups/<setup>/guardrails/*.yaml` |
| Predicate registry (the Python that backs hard guardrails) | Python | direct | compliance | `src/control_plane/predicates.py` |
| Process model (BPMN activity catalog) | YAML | direct | compliance, observability | `config/process_model.yaml` |
| Process supervisor prompt | YAML | direct | compliance, content | `config/setups/<setup>/agents/process_supervisor_agent.yaml` |
| Retrospective prompt + role overrides | YAML + Python | direct | evaluation | `config/setups/<setup>/agents/retrospective_agent.yaml`, `src/control_plane/retrospective.py` (`_AGENT_ROLES`, `_SYNTHESIS_PROMPT`) |
| Customer scenarios | Python list | direct | simulation | `src/agents/customer_agent.py` (`CUSTOMER_SCENARIOS`) |
| Customer feedback rubric (0.0–1.0 anchors) | Python prompt | direct | evaluation | `src/agents/customer_agent.py` (`get_feedback`) |
| `CoffeeShopConfig` dataclass (paths, toggles, retry caps, recursion limit) | Python | direct | control-flow, persistence, observability | `src/config.py` |
| CLI flags on `simulate` (`--traces`, `--scenario`, `--setup`, `--process-supervisor`, `--retrospective`, `--reset-inventory`, …) | CLI | direct | simulation, control-flow | `src/simulate.py` |
| LLM provider env vars (`LLM_PROVIDER`, `OLLAMA_MODEL`, `ANTHROPIC_MODEL`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`) | env | direct | content (which model speaks) | `src/llm.py`, `.env` |
| Setup directory (`config/setups/<name>/`) — selects the active bundle | dir + env | direct | meta (selects all the above) | `src/setups.py` (`COFFEE_SHOP_SETUP`) |
| Tool registry (which Python callables are mountable by name) | Python | direct | tool-use | `src/control_plane/tool_registry.py` |
| Tool implementations (the body of each `@tool` function) | Python | indirect | tool-use, content | `src/agents/*_agent.py`, `src/agents/order_store.py`, `src/agents/tray_tools.py` |
| Menu catalog + extras + size pricing | Python | indirect | content, tool-use | `src/agents/shared_components.py` (`MENU`, `ALLOWED_EXTRAS`, `process_order` body) |
| Order state machine (allowed transitions) | Python | indirect | control-flow, compliance | `src/agents/order_state_machine.py` (`ALLOWED_TRANSITIONS`) |
| Tray module (in-memory tray store) | Python | indirect | tool-use, content | `src/agents/tray.py` |
| Context isolation hook (per-agent message slicing + briefing) | Python | indirect | content, control-flow | `src/agents/context_isolation.py` |
| Handoff deferrer (parallel tool-call mitigation for Ollama) | Python | indirect | control-flow | `src/llm.py` (`_HandoffDeferrer`) |
| Coffee-machine simulator (`FAILURE_RATE`, `SEED`, contamination logic) | Python service | indirect | stochasticity, content | `services/coffee_machine/state.py` |
| Subgraph topology (LLM → Gateway → Tools loop, batch-deny policy) | Python | indirect | control-flow, compliance | `src/control_plane/subgraph.py` |
| Swarm wiring + `default_active_agent` | Python | indirect | routing | `src/graph.py` |
| Snapshot ID (hash over agent+guardrails+guidelines for log filtering) | Python | indirect | observability | `src/control_plane/snapshot.py` |
| JSONL log sink + path conventions | Python + config | indirect | observability, persistence | `src/control_plane/log_sink.py`, `CoffeeShopConfig` paths |
| Trace processing (MLflow → XES/OCEL CSV, injects `customer_feedback` event) | Python | indirect | observability | `src/trace_processing/` |
| MLflow experiment name + tracking URI | dataclass | indirect | observability, persistence | `CoffeeShopConfig.mlflow_experiment`, `mlflow_tracking_uri` |
| Conversation engine (max turns, transcript builder, feedback persistence) | Python | indirect | control-flow, evaluation, persistence | `src/conversation.py`, `CustomerAgent.max_turns` |
| Retrospective parsing (enum, grounding check, lenient JSON repair) | Python | indirect | evaluation | `src/control_plane/retrospective.py` |
| Dashboard runner + handover-pause toggle | Python + dataclass | indirect | control-flow (dashboard only) | `src/dashboard/interaction/conversation_runner.py`, `CoffeeShopConfig.handover_pause_default` |

---

## 2. Cross-cutting patterns

Several patterns recur and are worth naming up front — once you recognize one,
you can predict where to add a new rule.

### 2.1 YAML → loader → runtime pipeline

YAML files are loaded once at `CoffeeShop.open_shop()`. There is no hot
reload. Editing YAML requires a process restart (`simulate` re-run, dashboard
restart, or Jupyter kernel restart).

Loaders:

- `src/control_plane/agent_repo.py` — `AgentRepo(config_dir)` reads
  `<config_dir>/agents/*.yaml` into immutable `AgentDefinition` dataclasses.
- `src/control_plane/catalog.py` — `Catalog(config_dir)` reads
  `<config_dir>/guardrails/*.yaml` and `<config_dir>/guidelines/*.yaml`.
- `src/control_plane/process_supervisor.py` — `load_process_model(path)` reads
  `config/process_model.yaml` (note: NOT under the setup directory).

The `config_dir` itself is the **setup** — a single directory under
`config/setups/<name>/` containing `agents/`, `guardrails/`, and
`guidelines/` subdirectories. Picked by `--setup <name>` CLI flag or the
`COFFEE_SHOP_SETUP` env var (env supersedes the flag); see `src/setups.py`.
Default: `baseline`.

Important: there is **no merging or layering**. A setup is a complete,
independent configuration. Two top-level directories (`config/agents/` and
`config/guidelines/`) exist for legacy/reference but are *not loaded by the
runtime* — only `config/setups/<name>/` is read at `open_shop()`. `config/
process_model.yaml`, however, IS loaded (it lives outside the setup tree).

### 2.2 Predicate registry

Hard guardrails are declarative in YAML but back onto Python predicates by
name. The registry is `PREDICATE_REGISTRY` in
`src/control_plane/predicates.py`. A predicate is either a function taking
`GuardrailContext → Verdict`, or a *factory* that returns one when called with
`predicate_args` from the YAML.

YAML example (`config/setups/baseline/guardrails/coffee_shop.yaml`):

```yaml
- id: discount_within_30pct
  type: hard
  effect: flag
  tools: [calculate_total]
  predicate: discount_within_limit
  predicate_args: { max_pct: 30 }
```

The loader resolves `predicate: discount_within_limit` to
`discount_within_limit_predicate(max_pct=30)`. Snapshot IDs include
`predicate_args` so the same predicate parameterized differently hashes
distinctly (`src/control_plane/snapshot.py`).

### 2.3 `{placeholder}` substitution surface

Prompts loaded from YAML are run through Python `.format()` with a fixed set
of placeholders. The placeholders themselves are knobs — editing the YAML to
add a new placeholder requires changing the caller too.

- **Process supervisor** (`config/setups/<setup>/agents/process_supervisor_agent.yaml`,
  consumer: `ProcessSupervisor._llm_decide`):
  `{activity_catalog}`, `{prior_log_tail}`, `{message_brief}`.
- **Retrospective** (`config/setups/<setup>/agents/retrospective_agent.yaml`,
  consumer: `Retrospective._ask_agent`):
  `{agent_name}`, `{agent_role}`, `{peer_agents}`, `{agent_transcript}`.
- **Synthesis pass** is hard-coded as `_SYNTHESIS_PROMPT` in
  `src/control_plane/retrospective.py` (not loaded from YAML).
- **Operator agent prompts** are *not* templated — they are concatenated with
  a `## Guidelines` appendix in `src/control_plane/factory.py:29-34`.

### 2.4 Setup overlay vs. legacy top-level

`config/agents/` and `config/guidelines/` exist at the top level but are
*not* read by the runtime. The runtime reads only
`config/setups/<setup>/...`. Treat the top-level copies as historical; when
adding a new rule, add it under `config/setups/baseline/` (or whichever setup
is active).

---

## 3. Knob families

### 3.1 Agent prompts and per-agent wiring

**File**: `config/setups/<setup>/agents/<agent>.yaml`
**Loader**: `src/control_plane/agent_repo.py` (`AgentDefinition`)
**Consumer**: `src/control_plane/factory.py` (`build`) → `subgraph.create_agent_subgraph`

Fields per YAML:

| Field | What it controls |
|-------|------------------|
| `id` | Stable agent identifier used everywhere (state, logs, swarm router) |
| `version` | Participates in `snapshot_id` (`src/control_plane/snapshot.py`); bump when you change `base_prompt` or tool set |
| `base_prompt` | The agent's system prompt — its job description, decision flow, tone |
| `tools` | Names from `TOOL_REGISTRY` (`src/control_plane/tool_registry.py`); resolved to `@tool` callables |
| `guardrails` | Hard/soft guardrail IDs from the active setup's `guardrails/` |
| `guidelines` | Guideline IDs from `guidelines/` — appended verbatim under `## Guidelines` |
| `allowed_handovers` | Targets the `transfer_to_agent` guardrail will allow; also drives swarm `destinations` (`src/graph.py:42`) |
| `model` (optional) | Currently parsed (`AgentDefinition.model_ref`) but not yet consumed — one LLM is shared across all agents |

Composition rule (`src/control_plane/factory.py:29-34`): the final system
prompt is `base_prompt` + `\n\n## Guidelines\n\n- <prompt1>\n\n- <prompt2>…`.
Guidelines are an additive, shared layer — same guideline reused across agents.

**Pattern: add a new rule via agent prompt**

1. Edit `config/setups/baseline/agents/<agent>.yaml`.
2. Modify `base_prompt` (or add a one-line rule to the existing flow).
3. Bump `version: v2` so `snapshot_id` changes and log readers can tell runs
   apart.
4. Restart the process. No code changes.

**Pattern: add a new guideline (shared across agents)**

1. Add an entry under `guidelines:` in
   `config/setups/baseline/guidelines/coffee_shop.yaml` with `id`, `version`,
   `prompt`.
2. Reference the `id` in each agent's `guidelines:` list.

**Gotchas**

- Operator-agent prompts are not `.format()`-ed — using `{placeholder}` in
  `base_prompt` will not substitute anything; it will appear literally.
- The customer agent (`src/agents/customer_agent.py`) is NOT loaded from YAML
  — its prompt is hard-coded. To change customer behavior, edit
  `CUSTOMER_SCENARIOS` or `_system_prompt`.
- The process supervisor and retrospective agents *are* loaded from the same
  agent YAML directory but consumed by different runtime paths (see 3.3, 3.4).

### 3.2 Guardrails (hard predicates and soft judges)

**File**: `config/setups/<setup>/guardrails/coffee_shop.yaml`
**Loader**: `src/control_plane/catalog.py` (`_build_guardrail`)
**Consumer**: `src/control_plane/gateway.py` (`Gateway.evaluate_call`),
invoked per tool call from `src/control_plane/subgraph.py`.

Two kinds:

- **Hard** — backed by a Python `predicate` in `PREDICATE_REGISTRY`. Returns
  a `Verdict` deterministically. Effects: `deny` (block + return reason to
  the LLM), `flag` (log only), `allow`.
- **Soft** — backed by a `judge_prompt` (LLM-as-judge). **Currently stubbed:
  `SoftGuardrail.eval` in `src/control_plane/guardrails.py:58-65` always
  returns ALLOW with `reason_internal="soft guardrail evaluation skipped
  (stub)"`.** Soft guardrails are declarable but inert until the stub is
  replaced.

Batch policy (`src/control_plane/subgraph.py`): all-or-nothing per LLM turn.
If ANY tool call in a single AIMessage is denied, synthetic `ToolMessage`s
are emitted for every `tool_call_id` and the LLM is asked to retry. This is
necessary for the Anthropic `tool_use ↔ tool_result` invariant.

**Pattern: add a new hard guardrail**

1. Implement the predicate in `src/control_plane/predicates.py`. Signature:
   `def my_predicate(context: GuardrailContext) -> Verdict` or, for
   parameterized rules, `def my_predicate_factory(arg: int) -> Callable[[GuardrailContext], Verdict]`.
2. Register it: add an entry in `PREDICATE_REGISTRY` keyed by the YAML name.
3. Declare it in `config/setups/baseline/guardrails/coffee_shop.yaml`:
   ```yaml
   - id: my_new_rule
     type: hard
     version: v1
     tools: [some_tool_name]   # empty list = applies to all
     effect: deny              # or flag, allow
     predicate: my_predicate
     predicate_args: { ... }   # optional; participates in snapshot hash
   ```
4. Reference the guardrail `id` in each agent's `guardrails:` list in
   `config/setups/baseline/agents/<agent>.yaml`.

**Pattern: declare a new soft guardrail (will be inert until stub replaced)**

```yaml
- id: my_soft_rule
  type: soft
  tools: [transfer_to_agent]
  effect: allow
  judge_prompt: "Is this handover appropriate given conversation state?"
  state_dependencies: [conversation]
```

To actually run it, replace `SoftGuardrail.eval` in
`src/control_plane/guardrails.py` with an LLM call against `judge_prompt`
and the listed `state_dependencies`.

**Gotchas**

- `tools: []` means "applies to every tool", not "applies to nothing".
- A `flag` verdict still ALLOWS the call — it only marks the log entry.
- Per-call verdicts are logged individually via
  `Gateway._log_decision`, but the batch result is what routes the message.

### 3.3 Process supervisor (BPMN compliance observer)

**Files**:
- BPMN diagram (reference only): `docs/order-process-w-compliant.bpmn`
- Activity catalog: `config/process_model.yaml` (loaded at runtime)
- Supervisor prompt: `config/setups/<setup>/agents/process_supervisor_agent.yaml`
- Update procedure: `docs/updating-process-model.md`

**Loader**: `src/control_plane/process_supervisor.py` (`load_process_model`)
**Consumer**: `ProcessSupervisor.observe` (called per LangGraph message in
the dashboard/simulation runner)

The activity catalog defines `A01..A10` and `A05b`, `A09b` — one entry per
BPMN activity with `id`, `name` (slug), `display_name`, `agent` (lane),
`trigger` (`message` / `tool_call`), `tool` (for tool-call triggers), and
`terminal: true` for End-event predecessors.

Per message, the supervisor emits one log line to
`./process_log/process_meta.log`:

```
Execution:<ActivityID>:<ActivityName> | <serialized message>
Termination:<ActivityID>:<ActivityName>:terminal | <serialized message>
Termination:<ActivityID>:<ActivityName>:via_handoff_to_<target> | <serialized message>
Violation:<reason> | <serialized message>
```

Handoffs are special-cased in code: they always emit `Termination:` for the
source agent's most recent open activity (`_terminate_for_handoff`,
`_last_open_activity_for`). Tool-result messages are dropped. The supervisor
prompt is asked to classify everything else; LLM output is validated against
a regex (`_validate_llm_line`) and unparseable output is logged as
`Violation:llm_unparseable_output`.

**Active mode** (`CoffeeShopConfig.process_supervisor_active`, default
False): when violations occur, the supervisor can re-issue a critique
(`ProcessSupervisor.critique`) and the runner retries up to
`process_supervisor_max_retries` (default 3). This is wired in the dashboard
runner; the headless `simulate` path observes passively.

**Pattern: change the to-be process**

The full recipe lives in `docs/updating-process-model.md`. Summary:

1. Update the BPMN diagram (`docs/<new>.bpmn`).
2. Edit `config/process_model.yaml`:
   - Bump `name:` (so log readers can distinguish versions).
   - For each new activity: stable `id`, snake_case `name`, `display_name`,
     `agent` (= lane), `trigger`, `tool` (for tool-call triggers),
     `terminal: true` if it precedes End.
   - Tool name MUST match the `@tool`-decorated function name exactly,
     otherwise the deterministic matcher silently produces
     `Violation:no_activity_for_…`.
3. If two activities share `(agent, trigger, tool)`, add a disambiguator in
   `_deterministic_pick` in `src/control_plane/process_supervisor.py`.
4. Restart (no hot reload; the supervisor is constructed once in
   `CoffeeShop.open_shop`).

**Pattern: change the supervisor's decision style**

Edit `base_prompt` in
`config/setups/<setup>/agents/process_supervisor_agent.yaml`. The three
placeholders (`{activity_catalog}`, `{prior_log_tail}`, `{message_brief}`)
are injected by `_llm_decide`. Output must remain parseable by
`_validate_llm_line` regexes (`Execution:`, `Termination:`, `Violation:`).

**Gotchas**

- `config/process_model.yaml` is loaded from a hard-coded path
  (`CoffeeShopConfig.process_model_path`), NOT from the setup directory. The
  process model is shared across setups.
- `recent_tail=20` in `ProcessSupervisor.__init__` controls how many prior
  log lines the LLM sees; constructor arg, not currently exposed via config.
- Supervisor is off by default in the dashboard but on by default in
  `simulate` (see CLI flag defaults).

### 3.4 Retrospective (per-agent goal-friction review + synthesis)

**Files**:
- Per-agent prompt: `config/setups/<setup>/agents/retrospective_agent.yaml`
- Synthesis prompt: hard-coded `_SYNTHESIS_PROMPT` in
  `src/control_plane/retrospective.py`
- Role overrides (informational): `_AGENT_ROLES` dict, same file

**Consumer**: `Retrospective.run`, invoked at the end of every conversation
when `CoffeeShopConfig.retrospective_enabled=True`.

Output: one JSON file per conversation under `./retrospective_log/<UTC>.json`,
containing `entries` (per-agent answers) and `synthesis` (team-level view).

Per-agent answer schema (enforced by `_parse_retrospective` +
`_check_quote_grounding`):

```json
{
  "inferred_goal": "...",
  "goal_evidence_quote": "...",     // must appear verbatim in transcript
  "obstacle_moment_quote": "...",   // must appear verbatim in transcript
  "obstacle_source": "own_action | peer_agent | customer | process | tools_or_info | none",
  "obstacle_target": "<peer name when peer_agent, else null>",
  "what_was_in_the_way": "...",
  "next_time_change": "..."
}
```

Lenient JSON repair (`_parse_json_object_lenient`,
`_repair_truncated_json`) attempts to recover from truncated LLM responses
(the synthesis pass has its own `max_tokens=4096` budget).

**Pattern: change the retrospective question**

Edit `base_prompt` in
`config/setups/<setup>/agents/retrospective_agent.yaml`. Placeholders
available: `{agent_name}`, `{agent_role}`, `{peer_agents}`,
`{agent_transcript}`. If you change the JSON shape, also update
`_REQUIRED_KEYS`, `_OBSTACLE_SOURCES`, and `_parse_retrospective` in
`src/control_plane/retrospective.py`.

**Pattern: change the assigned role descriptions**

Edit `_AGENT_ROLES` in `src/control_plane/retrospective.py`. This is what
gets injected as `{agent_role}` and is the baseline against which
"role drift" is measured by the synthesis pass.

**Pattern: change the team-level synthesis**

Edit `_SYNTHESIS_PROMPT` in `src/control_plane/retrospective.py`. The
placeholders are `{transcript}` and `{retrospectives}` (JSON-encoded list of
per-agent entries).

**Gotchas**

- Quote-grounding rejects entries whose quoted lines don't appear in the
  transcript (after smart-quote normalization). LLMs that paraphrase will be
  recorded as `valid: false`.
- The customer agent does NOT participate in the retrospective. Its
  perspective is captured separately via `CustomerAgent.get_feedback` (3.7).
- The process supervisor receives a different transcript view — the tail of
  its own critique log, not the conversation
  (`conversation.py:_build_retrospective_views`).

### 3.5 Tools, state, and physical capabilities

**Files**: `src/agents/*_agent.py`, `src/agents/order_store.py`,
`src/agents/tray.py`, `src/agents/tray_tools.py`,
`src/agents/shared_components.py`
**Registry**: `src/control_plane/tool_registry.py` (`TOOL_REGISTRY`)
**Resolver**: `resolve_tools(names)` called from
`src/control_plane/factory.py:25`

Tools are LangChain `@tool`-decorated callables. Each tool's `args_schema`
(Pydantic BaseModel) is the LLM's view of how to call it. Adding a new tool
is a code change; binding it to an agent is a YAML change.

Related indirect knobs:

- **Menu and pricing**: `MENU` dict and `process_order` body in
  `src/agents/shared_components.py` and `src/agents/order_agent.py`. Size and
  extras pricing live in `process_order`.
- **Allowed extras**: `ALLOWED_EXTRAS` set in `shared_components.py`. Items
  outside this set are rejected by `process_order`.
- **Order state machine**: `ALLOWED_TRANSITIONS` in
  `src/agents/order_state_machine.py`. Any tool that calls
  `state_machine.transition` is gated by this table.
- **Tray**: in-memory dict `_trays` in `src/agents/tray.py`. The dashboard
  panels (`src/dashboard/interaction/tray_panel.py`) read this.

**Pattern: add a new tool**

1. Implement it as a `@tool(args_schema=...)` callable in a file under
   `src/agents/`. Return `json.dumps({...})` so the LLM gets structured
   output.
2. Import it in `src/control_plane/tool_registry.py` and add it to the
   `TOOL_REGISTRY` literal.
3. Reference its `name` (the `@tool` name, defaults to the function name) in
   each agent's `tools:` list in
   `config/setups/baseline/agents/<agent>.yaml`.
4. If the tool should be subject to a guardrail, add the guardrail to its
   `tools:` selector.
5. If the tool corresponds to a BPMN activity, add an entry to
   `config/process_model.yaml` with `trigger: tool_call` and the matching
   `tool: <name>`.

**Pattern: add an item to the menu**

Add a `MenuItem(...)` to `MENU` in `shared_components.py`. Adjust
`process_order` only if the new item needs special pricing or category
handling.

**Pattern: change order lifecycle**

Edit `ALLOWED_TRANSITIONS` in `src/agents/order_state_machine.py`. Add the
new status to `OrderStatus` in `shared_components.py` if it doesn't exist.

**Gotchas**

- `transfer_to_agent` (`shared_components.py`) is a `Command`-returning tool
  — it both updates state and routes execution to the target agent. The
  `allowed_handover_targets` hard guardrail is what enforces the
  `allowed_handovers` list in YAML.
- The handoff defererer (`src/llm.py:_HandoffDeferrer`) is an Ollama-only
  workaround for parallel-tool-call invariants: if a model emits a handoff
  alongside other tool calls in the same response, the handoff is stripped
  and re-injected when the model next replies with no tool calls. Anthropic
  goes through `parallel_tool_calls=False` instead.

### 3.6 Coffee machine (stochastic external service)

**Files**: `services/coffee_machine/*.py`
**Default URL**: `http://127.0.0.1:8001` (set in
`CoffeeShopConfig.coffee_machine_url` and a constant in
`src/agents/barista_agent.py`)
**Lifecycle**: auto-started as a subprocess by `start_coffee_machine` in
`src/agents/barista_agent.py` if the port is free.

This is the main source of stochasticity in the system. Knobs:

| Knob | Where | Effect |
|------|-------|--------|
| `FAILURE_RATE = 0.2` | `services/coffee_machine/state.py:16` | Probability that a brew job fails (the "intentional 20% barista error rate" from AGENTS.md) |
| `SEED` env var (`COFFEE_MACHINE_SEED`, default 100) | `services/coffee_machine/state.py:15` | RNG seed; makes failures reproducible |
| Contamination logic | `state.py:84-85` and `compute_status` | A failed brew marks the machine dirty; the next successful brew is `contaminated=True` until `clean_machine` is called |
| `duration = rng.uniform(1, 3)` | `state.py:71` | Brew duration in seconds |
| `outcome_queue` length (4) | `state.py:33` | How many pre-rolled outcomes are buffered; affects `/queue` visibility |

**Pattern: tune failure rate or seed for an experiment**

- Set `COFFEE_MACHINE_SEED=<n>` before starting the dashboard / `simulate`.
- POST to `/reseed` with `{"seed": <n>}` at runtime.
- Edit `FAILURE_RATE` directly for non-default rates (no env var for this
  one).

### 3.7 Customer simulation and feedback

**Files**: `src/agents/customer_agent.py`, `src/conversation.py`

The customer is NOT part of the swarm — it drives the conversation
externally via `ConversationEngine.run_automated`. Knobs:

| Knob | Where | Effect |
|------|-------|--------|
| `CUSTOMER_SCENARIOS` list | `customer_agent.py:9-14` | The four scenarios the simulator can run; `--scenario 0..3`, `--scenario all` (round-robin), `--scenario random` |
| `CustomerAgent.max_turns = 15` | `customer_agent.py:38` | Hard cap on conversation length |
| Customer system prompt | `_system_prompt` / `build_default_prompt` | The customer's persona, brevity rule, and `DONE` end-token |
| Feedback rubric (0.0/0.5/1.0 anchors) | `get_feedback` | What "good/acceptable/bad service" means to the simulated customer |
| Mid-conversation experience injection | `inject_experience` | Used by `ConversationEngine._consume_tray` to flag contaminated coffee |

The DONE token is the exit condition — `respond_to` returns `None` when the
customer's reply is the single word "DONE" (or contains it within 10 chars).

**Pattern: add a new scenario**

Append a string to `CUSTOMER_SCENARIOS`. The `--scenario all` flag will pick
it up automatically. The number range in `simulate.py` (`0-3`) is computed
from the list length at runtime.

**Pattern: change what "good service" means**

Edit the rubric inside `CustomerAgent.get_feedback`. The score lands in
`feedback_store.json` keyed by `thread_id` and is also injected as the
single `customer_feedback` event at the end of each case by the trace
processor.

### 3.8 Simulation control (CLI)

**File**: `src/simulate.py` (entry point: `poetry run simulate`)

Flags (all visible via `simulate --help`):

| Flag | Default | What it changes |
|------|---------|-----------------|
| `--traces N` | 1 | Number of conversations to run |
| `--scenario {0..3, all, random}` | `random` | Which `CUSTOMER_SCENARIOS` index |
| `--export-logs` | off | Run `TraceProcessor.process_all_traces` after the run |
| `--reset-inventory / --no-reset-inventory` | reset | Reset DB inventory before each trace |
| `--quiet` | off | Suppress per-message output |
| `--full-messages` | off | Don't truncate message content to 200 chars |
| `--log-level {debug,info,warning,error}` | info | `coffee_shop` logger level |
| `--setup <name>` | env or `baseline` | Which `config/setups/<name>/` to load |
| `--list-setups` | — | Print available setups and exit |
| `--process-supervisor / --no-process-supervisor` | on | Enable the supervisor |
| `--retrospective / --no-retrospective` | off | Run the per-agent retro at end of each conversation |

The `COFFEE_SHOP_SETUP` env var supersedes `--setup`.

The dashboard entry point is `poetry run dashboard`
(`src/dashboard/app.py`); its knobs are dashboard-only and live in the
`CoffeeShopConfig` (e.g. `handover_pause_default`).

### 3.9 LLM provider

**File**: `src/llm.py` (`create_chat_llm`), `.env` / `.env.example`

| Env var | Default | Purpose |
|---------|---------|---------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `anthropic` |
| `OLLAMA_MODEL` | `ministral-3:14b` | Local Ollama model |
| `ANTHROPIC_MODEL` | `anthropic--claude-4.6-opus` | Model name when going through Hyperspace proxy |
| `ANTHROPIC_BASE_URL` | `http://localhost:6655/anthropic/` | Hyperspace AI Local LLM Proxy |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |

`bind_tools_sequential` (same file) applies provider-specific mitigations
for parallel tool calls (Anthropic: `parallel_tool_calls=False`; Ollama:
post-processing via `_HandoffDeferrer`).

### 3.10 Observability and persistence

| Sink / path | Where it's configured | What it captures |
|-------------|----------------------|------------------|
| `./guardrail_log/events.jsonl` | `CoffeeShopConfig.guardrail_log_path` → `JsonlLogSink` | Per-tool-call gateway decisions + tool execution markers, tagged with `setup_name` and `snapshot_id` |
| `./process_log/process_meta.log` | `CoffeeShopConfig.process_log_path` | One line per LangGraph message classified by the supervisor |
| `./retrospective_log/<UTC>.json` | `CoffeeShopConfig.retrospective_log_dir` (or `RETROSPECTIVE_LOG_DIR` env in `simulate`) | One JSON per conversation: per-agent entries + synthesis |
| `./feedback_store.json` | `conversation.py:FEEDBACK_STORE_PATH` | `thread_id → {feedback_score, feedback_reason, valid, …}` |
| MLflow `./mlruns/`, `mlflow.db` | `CoffeeShopConfig.mlflow_experiment` / `mlflow_tracking_uri` / `mlflow_enabled` | LangChain autolog traces |
| `./generated_event_log/`, `./generated_ocel/`, `./generated_visualizations/` | `src/trace_processing/` defaults | XES/OCEL exports for process mining, with `customer_feedback` injected as a terminal event per case |
| Coffee machine OCEL CSV | `services/coffee_machine/logger.py` | Brew/clean events keyed by `correlation_id` (= LangGraph `thread_id` = MLflow `case_id`) |

The `snapshot_id` (`src/control_plane/snapshot.py`) is a stable hash over
`(agent_id, agent_version, sorted guardrails+predicate_args, sorted
guidelines)`. Every gateway log line carries the snapshot, so logs from two
different setups (or two versions of the same setup) can be filtered apart
even when they share `./guardrail_log/events.jsonl`.

**Pattern: emit a new event type from the gateway**

1. Define the event in a new `Gateway.log_<event>` method, calling
   `self.log_sink.append({"event_type": "<name>", ...})`.
2. Call it from wherever in `src/control_plane/subgraph.py` is appropriate.
3. Downstream consumers (notably `src/trace_processing/eventlog_conversion.py`)
   may need to learn about the new `event_type`.

### 3.11 Wiring: setups, swarm, subgraph

These are deeper structural knobs — touch them only when adding a new
configuration channel, not for routine rule changes.

**Setups** (`src/setups.py`, `config/setups/`): the meta-knob that selects
all the YAML above. To add a new setup:

1. `cp -r config/setups/baseline config/setups/<new-name>` and edit.
2. Run with `--setup <new-name>` or `COFFEE_SHOP_SETUP=<new-name>`.
3. Both `agents/` and `guidelines/` subdirectories are required; missing
   `guardrails/` raises at load time.

**Swarm wiring** (`src/graph.py`): `AGENT_IDS` is the hard-coded tuple of
agents that the swarm router instantiates. `default_active_agent="order_agent"`
sets where new conversations start. Adding a new operator agent requires a
new YAML in the setup AND adding the ID to `AGENT_IDS`.

**Subgraph topology** (`src/control_plane/subgraph.py`):
`LLM → conditional → Gateway → conditional → Tools → LLM` loop. The
batch-deny policy and the context-isolation hook are wired here. Changing
this affects every agent uniformly.

### 3.12 `CoffeeShopConfig` dataclass

A central collection of runtime toggles and paths
(`src/config.py:CoffeeShopConfig`). Highlights not already covered:

| Field | Default | Effect |
|-------|---------|--------|
| `mlflow_enabled` | True | Turn off all LangChain autolog and trace IDs |
| `process_supervisor_enabled` | False | Default off; `simulate` flips it on via CLI |
| `process_supervisor_active` | False | When True, supervisor critiques violations and the runner retries (dashboard only) |
| `process_supervisor_max_retries` | 3 | Cap on active-mode retries |
| `retrospective_enabled` | True | Run the retro at end of each conversation |
| `handover_pause_default` | False | Initial state of the dashboard's "pause at next handover" toggle |
| `recursion_limit` | 100 | LangGraph recursion limit (not currently wired through to the swarm `.compile()` — present as a config-level intent) |

---

## 4. Adding a new behavior — decision tree

> "I want to change…"

- **…what an agent says or how it makes decisions** →
  agent YAML `base_prompt` (§3.1). Bump `version`.
- **…which agents an agent can hand off to** →
  agent YAML `allowed_handovers` (§3.1). The hard guardrail enforces this
  automatically.
- **…a rule that applies to many agents** →
  guideline YAML (§3.1), reference its `id` in each agent.
- **…what counts as a forbidden tool call** →
  guardrail YAML + predicate (§3.2).
- **…what counts as a non-compliant process step** →
  `config/process_model.yaml` + (if ambiguous) `_deterministic_pick` (§3.3).
- **…the question we ask agents after each run** →
  retrospective YAML (§3.4); update `_REQUIRED_KEYS` if shape changes.
- **…what an agent can physically do** →
  new tool (§3.5), register it, bind it in agent YAML, optionally add a
  process-model activity.
- **…how often the coffee machine fails** →
  `FAILURE_RATE` / `COFFEE_MACHINE_SEED` (§3.6).
- **…how the simulated customer behaves** →
  `CUSTOMER_SCENARIOS`, `max_turns`, or feedback rubric (§3.7).
- **…how a simulation is invoked** →
  `simulate` CLI flags or `CoffeeShopConfig` defaults (§3.8, §3.12).
- **…which LLM speaks** →
  `.env` (§3.9).
- **…where logs go or what's in them** →
  `CoffeeShopConfig` paths + `JsonlLogSink` shape (§3.10).
- **…a structural property of the swarm itself** →
  `src/graph.py` and `src/control_plane/subgraph.py` (§3.11).
- **…the whole configuration as a unit, for an experiment** →
  create a new setup under `config/setups/<name>/` (§3.11).

---

## 5. Quick reference: where each behavior actually lives

| Symptom you want to change | First file to open |
|---------------------------|--------------------|
| Agent tone or decision flow | `config/setups/baseline/agents/<agent>.yaml` |
| Handovers blocked unexpectedly | `config/setups/baseline/guardrails/coffee_shop.yaml` + `src/control_plane/predicates.py` |
| Activity logged as `Violation:no_activity_for_…` | `config/process_model.yaml` (check `agent`/`trigger`/`tool` triple) |
| LLM produces bad retrospective JSON | `config/setups/baseline/agents/retrospective_agent.yaml` + `src/control_plane/retrospective.py` |
| Coffee always (or never) fails | `services/coffee_machine/state.py` (`FAILURE_RATE`, `SEED`) |
| Customer ends conversation too early | `CustomerAgent.max_turns`, DONE-token rule in `respond_to` |
| Run logs are mixed across experiments | check `setup_name` in `guardrail_log/events.jsonl`; bump agent/guideline versions to differentiate `snapshot_id` |
| Process model changed but supervisor ignores it | did you restart? `ProcessSupervisor` is constructed once at `open_shop` |
| Feedback always 0.5 | `valid: false` in `feedback_store.json` → LLM response didn't parse; check `get_feedback` |
| Anthropic tool-call errors | `bind_tools_sequential` in `src/llm.py`; check `parallel_tool_calls=False` is reaching the model |
