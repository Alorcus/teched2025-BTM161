# Agentic Coffee Shop

A multi-agent coffee shop system for exploring the behavior of LLM-based agents. Specialized agents collaborate to take orders, manage stock, prepare drinks, and resolve customer issues. Their interactions are traced via MLflow and can be exported as event logs for process mining and analysis.

## Overview

Five agents work together in a LangGraph Swarm:

- **Order Agent** — takes and prices orders
- **Inventory Agent** — checks stock and suggests alternatives
- **Barista Agent** — prepares drinks (with a simulated 20% failure rate to create variants)
- **Customer Service Agent** — handles complaints and refunds
- **Customer Agent** — drives the conversation from outside the swarm, simulating a customer

The repository contains a CLI for headless trace generation and a Panel-based observatory dashboard for live exploration and metrics.

## Requirements

- [Python](https://www.python.org/downloads/) >= 3.13
- (Recommended) [Poetry](https://python-poetry.org/) for dependency and virtualenv management.
  - Alternative: pip with the provided `requirements.txt`.
- An API key for an [LLM provider supported by LangChain](https://python.langchain.com/docs/integrations/chat/#featured-providers), or a local Ollama runtime.

## Installation

1. Install dependencies: `poetry install`
2. Activate the venv: run `poetry env activate` and use the printed command (or prefix commands with `poetry run`).
3. Install the LangChain integration for your LLM provider, for example:
   ```
   pip install "langchain[openai]<1.0.0"
   pip install "langchain[anthropic]<1.0.0"
   ```
4. Configure your LLM provider via a `.env` file (see `.env.example`). Set `LLM_PROVIDER=ollama` (default) or `LLM_PROVIDER=anthropic`.

## Pre-commit Hook

Runs `ruff check` and `ruff format` on staged files. CI enforces the same on every PR.

```bash
brew install pre-commit          # macOS
pip install pre-commit           # Linux (or: poetry install)
pre-commit install
```

## Setups

A **setup** is a self-contained configuration of agents, guardrails, and guidelines under `config/setups/<name>/` (subdirs: `agents/`, `guardrails/`, `guidelines/`). Both `simulate` and `dashboard` require a setup to be selected. Guardrail predicate *logic* lives in Python ([src/control_plane/predicates.py](src/control_plane/predicates.py)) and is referenced by name from the guardrail YAML — varying `predicate_args` (e.g. `max_pct: 10`) is a YAML-only change.

**Available setups:**

The catalogue is designed as a spectrum from *no rules* to *hostile rules*, so you can compare how the same swarm behaves under different control-plane pressure. The first three are the reference points; the remaining five dial specific knobs (range caps, lifecycle gates, effect mode) around them.

Reference points:

- `unconstrained` — every business agent can transfer to every other agent, with no guardrails, no guidelines, and no supervisor preamble — maximum agent freedom for observing emergent behavior.
- `baseline` — the standard coffee shop: each agent can only hand off to the next role in the workflow, `deny` lifecycle gates enforce the order state machine (`pending → inventory_confirmed → in_preparation → completed/preparation_error → refunded`), and every agent prompt declares that a runtime process supervisor is watching.
- `all_handovers` — every business agent can transfer to every other agent, and an `order_id_in_handoff` flag guardrail (plus matching `handoff_order_id` guideline) requires handoffs to carry an `ORDXXXX` once an order exists.

Governance dials on top of `baseline`:

- `sensible_ranges` — adds `deny` range caps that let normal orders through but block outliers: order size 1–6 units, total ≤ $20, discount ≤ 30 %, partial refund ≤ 50 %. Shows the "happy path still works, only outliers get stopped" regime.
- `sensible_ranges_flag` — identical caps to `sensible_ranges`, but every range guardrail is `flag` (observe-only). Agents behave as if unconstrained on magnitudes while the guardrail log records every trip — useful for measuring how often a proposed cap *would* bite before you enforce it.
- `overconstrained` — the same range guardrails cranked so far that normal business cannot happen: max one unit per order, total ≤ $3, zero discounts, zero partial refunds. Demonstrates the failure mode of over-tight governance — most orders never get created.
- `lifecycle_flag` — same lifecycle preconditions as `baseline`, but the order-state gates run in `flag` mode instead of `deny`. Agents behave as if the state machine were unenforced; every illegal transition is labeled in the log so you can compare emergent order flow against the enforced one.
- `anti_flow` — the lifecycle gates are deliberately **inverted** (each tool is only "allowed" from a status it can never legitimately be in), so every fulfillment step is denied from its real predecessor. Combined with a severed barista handover, orders get trapped mid-flow. Useful as a worst-case for showing how mis-configured guardrails cause deadlocks rather than safety.

**Selecting a setup:**

```bash
poetry run simulate --setup baseline --traces 1
poetry run dashboard --setup baseline
poetry run simulate --list-setups          # show available setups
```

Repeat `--setup` to run multiple setups sequentially in one `simulate` invocation
(e.g. `--setup baseline --setup all_handovers`); each setup runs `--traces N` conversations.

**Adding a new setup** — copy `baseline` and edit the YAMLs:

```bash
cp -r config/setups/baseline config/setups/my_setup
# edit config/setups/my_setup/{agents,guardrails,guidelines}/*.yaml
poetry run simulate --setup my_setup --traces 1
```

## Headless Simulation

You can generate traces in bulk without the dashboard UI using the `simulate` CLI command. This runs the Customer Agent against the coffee shop swarm and captures MLflow traces for each conversation.

### Usage

All examples below require `--setup <name>`; see [Setups](#setups).

```bash
# Run a single trace with a random scenario
poetry run simulate --setup baseline

# Run 10 traces cycling through all 7 scenarios
poetry run simulate --setup baseline --traces 10 --scenario all

# Run 5 traces with a specific scenario (index 0-6)
poetry run simulate --setup baseline --traces 5 --scenario 2

# Run with minimal output (no message content)
poetry run simulate --setup baseline --traces 10 --quiet

# Run with debug logging enabled
poetry run simulate --setup baseline --traces 5 --log-level debug

# Export event logs after simulation
poetry run simulate --setup baseline --traces 10 --scenario all --export-logs
```

### Arguments

| Argument        | Default   | Description                                                                       |
| --------------- | --------- | --------------------------------------------------------------------------------- |
| `--setup NAME`  | required  | Setup under `config/setups/` to load; repeat the flag to run multiple setups     |
| `--list-setups` | off       | List available setups and exit                                                    |
| `--traces N`    | `1`       | Number of conversation traces to run                                              |
| `--scenario`    | `random`  | Scenario index (`0`–`6`), `all` (round-robin), or `random`                        |
| `--export-logs` | off       | Generate event log CSV after simulation                                           |
| `--quiet`       | off       | Minimal output: only trace numbers, scenarios, and summary                        |
| `--log-level`   | `warning` | Set the logging level for agent diagnostics (`debug`, `info`, `warning`, `error`) |

### Available Scenarios

Defined in `src/agents/customer_agent.py` (`CUSTOMER_SCENARIO_DEFS`) — single source of truth for both the label and the LLM prompt.

| Index | Description                                                       |
| ----- | ----------------------------------------------------------------- |
| 0     | Order a plain espresso — nothing more, nothing less               |
| 1     | Order a large latte and a croissant (friendly)                    |
| 2     | Order 2 espressos (in a hurry)                                    |
| 3     | Complain about a cold cappuccino and seek resolution              |
| 4     | Ask for a recommendation and order based on the suggestion        |
| 5     | Order a tea and stubbornly refuse anything else                   |
| 6     | Rich customer buys everything until the store is empty            |

### Batch Script

For mixing setups and scenarios in a single run (e.g. 10 traces of scenario 0 under `baseline`, then 10 of scenario 2 under `unconstrained`), use `scripts/run_batches.py`. Three ways to drive it — no-flag runs use the module-level defaults, or pass explicit batches via CLI or JSON:

```bash
# 1. Run the built-in default batch set (edit the module-level BATCHES to change it)
poetry run python -m scripts.run_batches

# 2. Pass batches on the command line as `setup:scenario:count` triples
poetry run python -m scripts.run_batches --batches baseline:0:10 unconstrained:2:10

# 3. Load batches (and toggles) from a JSON config file
poetry run python -m scripts.run_batches --config batches.json
```

Boolean toggles: `--reset-inventory` / `--no-reset-inventory`, `--process-supervisor` / `--no-process-supervisor`, `--export-logs` / `--no-export-logs`. The JSON config schema is `{"batches": [["baseline", 0, 50], ...], "reset_inventory": true, "process_supervisor": false, "export_logs": false}`.

Make sure the Poetry virtual environment is active (`poetry env activate`) or prefix with `poetry run` as shown; the script imports from `src/` and needs the project's dependencies. Batches sharing a setup reuse the same `CoffeeShop` instance, so keep same-setup entries consecutive in the list.

## Agent Observatory Dashboard

A two-page observability dashboard built with [Panel](https://panel.holoviz.org/):

- **Interaction Observatory** (`/`) — a real-time view of all agents in a grid layout. Each panel displays the system prompt, available tools, current status, handoff context, context-isolated message history, and tool call log, updating live as a conversation streams through the system.
- **Metrics Dashboard** (`/metrics`) — analytics over previously-generated event logs (KPIs, per-agent workload, per-order timings, OCEL-based visualizations).

### Launch

```bash
# Start the dashboard (opens browser at http://localhost:5006)
poetry run dashboard --setup baseline
```

### Features

#### Interaction Observatory (/)

- **2x2 grid layout** showing all 4 agents at once (scales to 3x3 for up to 9)
- **Live status badges**: idle / thinking / executing tool / handed off
- **Handoff context display**: see what each agent received from the previous agent
- **Tool call log**: arguments and results for every tool invocation
- **Context-isolated messages**: the same filtered view each agent's LLM actually sees
- **Sidebar controls**: scenario selector, log-level filter, customizable customer prompt, run button, and global conversation log
- **Customer mode toggle**: switch between the simulated AI customer and a manual customer mode where you can type messages yourself and submit feedback at the end of the conversation

#### Metrics Dashboard (/metrics)

- **Automatic trace cache**: on every page entry, the dashboard reconciles its data source with the MLflow store. If new conversations have been recorded since the last build (whether from the Interaction Observatory or `poetry run simulate`), it re-runs the trace processor and consolidates every trace into a single `generated_event_log/_all_traces.csv`. Staleness is decided by comparing MLflow's trace count to the count recorded in `_all_traces.meta` at the last build — no manual export step is required. The directory is owned by the cache: any per-run CSVs left over from earlier exports are removed during the build.
- **Timeframe filter**: a dual-handle range slider over the cached event log, defaulting to the full span of available traces. Drag either handle to narrow the window; the sidebar label shows the number of fully-contained traces and, separately, the count of partial traces excluded (any whose conversation started before the start or ended after the end). On release, every section recomputes against the filtered traces.
- **Overview**: KPI cards summarizing the events in the selected window
- **System Metrics**: per-agent workload and activity breakdown
- **Time Metrics**: per-order durations and timing distributions
- **Visualization**: OCEL-based diagrams (object-type mapping, OC-DFG, OC-PN) generated via the `Visualizer`

### Workflow

Generate traces either through the Interaction Observatory or the headless simulator, then analyze them in the Metrics Dashboard:

1. **(Optional) Generate traces in bulk** via the CLI: `poetry run simulate --setup baseline --traces 10 --scenario all` — runs N conversations and stores their MLflow traces. The `--export-logs` flag is no longer needed for the dashboard to see them.
2. **Open the dashboard** with `poetry run dashboard --setup baseline`.
3. **Explore conversations live** in the Interaction Observatory (run a scenario, watch agents collaborate). Every conversation you run here is automatically picked up.
4. **Switch to the Metrics Dashboard** tab. On entry it reconciles its cache with MLflow — new conversations are processed on the spot — then use the timeframe slider to scope the analysis to a window of interest.

### Resetting Trace State

To wipe MLflow tracking state, generated event logs / OCELs / visualizations, the coffee-shop SQLite, and the auxiliary log directories in one step:

```bash
poetry run reset-traces             # interactive — prompts y/N
poetry run reset-traces --yes       # skip the prompt (CI / scripted resets)
poetry run reset-traces --dry-run   # preview without deleting
```

The command refuses to run while the dashboard is listening on port 5006 — stop it first so deleted SQLite/WAL files don't get rewritten through unlinked inodes.

### How It Works

The dashboard runs the same `CoffeeShop` multi-agent graph used by the CLI. A background thread drives the conversation (using the simulated Customer Agent), while the Panel UI polls for events every 100ms. Stream events from LangGraph are parsed into typed dashboard events (agent messages, tool calls, handoffs, etc.) and dispatched to the corresponding agent panel.

The Metrics Dashboard loads the consolidated `_all_traces.csv` cache into an `ObjectCentricEventlog` and renders sections from it. The cache is rebuilt from MLflow on page entry whenever the trace count has changed; aside from that single write, the page is read-only.

### Sharing the event log CSV

`generated_event_log/_all_traces.csv` is designed to be handed to colleagues who don't have your MLflow store — but it embeds free-text content: verbatim customer utterances (in `message` on `user_prompt` rows), LLM assistant responses, and LLM reasoning from the guardrail gateway (in `gateway_tool_args_json`, `gateway_verdicts_json`, and `feedback_reason`). In this TechEd repository the "customer" is an LLM-simulated persona, so those cells are synthetic. **In any real deployment, the same columns would carry PII and unredacted model reasoning — review the file before sharing.** The complete column list, per-column sensitivity classification, and timezone/versioning contract live in [`EVENTLOG_SCHEMA.md`](EVENTLOG_SCHEMA.md); consult it before forwarding the CSV or attaching it to a ticket.

## Observing the Database

Orders and inventory are persisted in a local SQLite database (`coffee_shop.db`). To inspect the database while the agents are running, install the [SQLite Viewer](https://marketplace.visualstudio.com/items?itemName=qwtel.sqlite-viewer) extension in VS Code. Open `coffee_shop.db` from the file explorer and the extension will display the tables in a browsable grid view. Refresh to see the latest orders and stock levels as they are updated.

## Agent Architecture

### Order Agent

**Role**: Takes and processes customer orders
**Tools**:

- `process_order()` — Parse customer orders
- `calculate_total()` — Calculate pricing with discount capabilities
- `transfer_to_agent` — Handoff tool

**Responsibilities**:

- Welcome customers and take orders
- Validate menu items and quantities
- Calculate totals and apply discounts
- Transfer to inventory for availability checks

### Inventory Agent

**Role**: Manages stock levels and availability
**Tools**:

- `check_inventory()` — Verify item availability for orders
- `update_stock()` — Decrease inventory after confirmed orders
- `get_alternatives()` — Find substitute items for out-of-stock products
- `transfer_to_agent` — Handoff tool

**Responsibilities**:

- Check item availability against current stock
- Update inventory after order confirmation
- Suggest alternatives for unavailable items
- Transfer to barista when items are available
- Escalate to customer service for stock issues

### Barista Agent

**Role**: Handles order preparation and quality
**Tools**:

- `start_preparation()` — Start coffee preparation
- `remake_order_item()` — Handle preparation errors and remakes
- `estimate_prep_time()` — Provide accurate timing estimates
- `transfer_to_agent` — Handoff tool

**Responsibilities**:

- Prepare drinks and food items
- Handle preparation errors (20% failure rate simulation)
- Provide preparation time estimates
- Quality control and remake capabilities

### Customer Agent

**Role**: Simulates a customer interacting with the coffee shop, and can also be driven manually from the dashboard
**Scenarios**:

- Ordering a plain espresso — nothing more, nothing less
- Ordering a latte and croissant
- Quickly ordering two espressos
- Complaining about a cold drink and seeking resolution
- Asking for a recommendation and ordering based on the suggestion
- Ordering a tea and stubbornly refusing anything else
- Buying everything in the store until it is empty

**Manual mode**:

- Use the dashboard's Customer mode switch to switch from the simulated AI customer to a manual experience.
- Type customer messages directly in the sidebar, send them to the swarm, and submit feedback once the conversation is finished.

**Behavior**:

- Picks a scenario randomly (or by index via `reset()`)
- Drives the conversation by sending an opening message and responding to agent replies
- Ends the conversation after at most 8 turns, or when the goal is achieved (signals `DONE`)

### Customer Service Agent

**Role**: Manages customer satisfaction and issue resolution
**Tools**:

- `offer_refund()` — Process refunds when necessary
- `offer_partial_refund()` — Process a partial refund when necessary
- `transfer_to_agent` — Handoff tool

**Responsibilities**:

- Handle customer complaints with empathy
- Offer appropriate compensation (remakes, refunds, discounts)
- Suggest alternatives with customer service touch
- Coordinate with other agents for resolution

## Visualization

Visuals can be generated with the `Visualizer` class. Currently, an Object-Type Mapping, OC-DFGs, and OC-PNs are supported.

**How to:**

1. Create a `VisualizationConfig` with `ocel_path`, `out_dir`, and `export_format`.
2. Create a new `Visualizer` instance and call `run`.
3. The visuals will be saved at the specified output path. A dictionary of output paths is returned.

Alternatively, edit the example parameters in `visualizer.py` and run the script directly.

**Adding new visualizations:** create a private export method (e.g. using `pm4py` to discover the object from the OCEL event log, then `gviz` to render and save), and register it in the public `run` method of the visualizer.

## Running the Tests

Use the Python interpreter from the Poetry virtual environment (`poetry env activate` first, or prefix with `poetry run python`).

```bash
python -m unittest discover -s tests -v
```

Individual test modules can be run directly:

```bash
python -m unittest tests/test_tools_order.py -v
```
