# Agentic Coffee Shop

A multi-agent coffee shop system for exploring the behavior of LLM-based agents. Specialized agents collaborate to take orders, manage stock, prepare drinks, and resolve customer issues. Their interactions are traced via MLflow and can be exported as event logs for process mining and analysis.

## Overview

Five agents work together in a LangGraph Swarm:

- **Order Agent** — takes and prices orders
- **Inventory Agent** — checks stock and suggests alternatives
- **Barista Agent** — prepares drinks (with a simulated 20% failure rate to create variants)
- **Customer Service Agent** — handles complaints and refunds
- **Customer Agent** — drives the conversation from outside the swarm, simulating a customer

The repository contains three Jupyter notebooks for stepping through the system, a CLI for headless trace generation, and a Panel-based observatory dashboard for live exploration and metrics.

## Requirements

- [Python](https://www.python.org/downloads/) >= 3.13
- (Recommended) [Poetry](https://python-poetry.org/) for dependency and virtualenv management.
  - Alternative: pip with the provided `requirements.txt`.
- (Recommended) [poetry-jupyter-plugin](https://pypi.org/project/poetry-jupyter-plugin/) to register the Poetry venv as a Jupyter kernel:

  ```
  $ poetry self add poetry-jupyter-plugin
  ```

- An API key for an [LLM provider supported by LangChain](https://python.langchain.com/docs/integrations/chat/#featured-providers), or a local Ollama runtime.

## Installation

1. Install dependencies: `poetry install`
2. Install the Jupyter kernel: `poetry jupyter install`
3. Activate the venv: run `poetry env activate` and use the printed command (or prefix commands with `poetry run`).
4. Install the LangChain integration for your LLM provider, for example:
   ```
   pip install "langchain[openai]<1.0.0"
   pip install "langchain[anthropic]<1.0.0"
   ```
5. Configure your LLM provider via a `.env` file (see `.env.example`). Set `LLM_PROVIDER=ollama` (default) or `LLM_PROVIDER=anthropic`.
6. Start Jupyter: `jupyter notebook`

## Pre-commit Hook

Runs `ruff check` and `ruff format` on staged files. CI enforces the same on every PR.

```bash
brew install pre-commit          # macOS
pip install pre-commit           # Linux (or: poetry install)
pre-commit install
```

## Notebooks

Three self-contained exercises:

1. [`1_Standard_agentic_coffee_shop`](1_Standard_agentic_coffee_shop.ipynb) — get familiar with the setup and generate a first trace.
2. [`2_Exceptions_agentic_coffee_shop`](2_Exceptions_agentic_coffee_shop.ipynb) — explore agent behavior under errors and edge cases, producing process variants.
3. [`3_Extending_agentic_coffee_shop`](3_Extending_agentic_coffee_shop.ipynb) — experiment with agent definitions (instructions, tools) and observe how changes affect the multi-agent system.

## Setups

A **setup** is a self-contained configuration of agents, guardrails, and guidelines under `config/setups/<name>/` (subdirs: `agents/`, `guardrails/`, `guidelines/`). Both `simulate` and `dashboard` require a setup to be selected. Guardrail predicate *logic* lives in Python ([src/control_plane/predicates.py](src/control_plane/predicates.py)) and is referenced by name from the guardrail YAML — varying `predicate_args` (e.g. `max_pct: 10`) is a YAML-only change.

**Available setups:**

- `baseline` — the standard coffee shop: each agent can only hand off to the next role in the workflow, and every agent prompt declares that a runtime process supervisor is watching.
- `all_handovers` — every business agent can transfer to every other agent, and an `order_id_in_handoff` flag guardrail (plus matching `handoff_order_id` guideline) requires handoffs to carry an `ORDXXXX` once an order exists.
- `unconstrained` — every business agent can transfer to every other agent, with no guardrails, no guidelines, and no supervisor preamble — maximum agent freedom for observing emergent behavior.

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

You can generate traces in bulk without the jupyter UI using the `simulate` CLI command. This runs the Customer Agent against the coffee shop swarm and captures MLflow traces for each conversation.

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

## Agent Observatory Dashboard

A two-page observability dashboard built with [Panel](https://panel.holoviz.org/):

- **Interaction Observatory** (`/`) — a real-time view of all agents in a grid layout. Each panel displays the system prompt, available tools, current status, handoff context, context-isolated message history, and tool call log, updating live as a conversation streams through the system.
- **Metrics Observatory** (`/metrics`) — analytics over previously-generated event logs (KPIs, per-agent workload, per-order timings, OCEL-based visualizations).

Switch between pages via the tabs in the header.

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

#### Metrics Observatory (/metrics)

- **Event log selector**: choose any CSV in `generated_event_log/` (defaults to most recent)
- **Overview**: KPI cards summarizing the selected log
- **System Metrics**: per-agent workload and activity breakdown
- **Time Metrics**: per-order durations and timing distributions
- **Visualization**: OCEL-based diagrams (object-type mapping, OC-DFG, OC-PN) generated via the `Visualizer`

### Workflow

The Interaction Observatory does **not** save event logs. Generate logs separately via the headless simulator, then explore them in the Metrics Observatory:

1. **Generate logs** via the CLI: `poetry run simulate --traces 10 --scenario all --export-logs` — this produces CSVs in `generated_event_log/` from MLflow traces (with full token counts and durations).
2. **Open the dashboard** with `poetry run dashboard`.
3. **Explore conversations live** in the Interaction Observatory (run a scenario, watch agents collaborate).
4. **Use the Customer mode toggle** to switch to manual mode and type customer messages yourself for ad-hoc conversations.
5. **Switch to the Metrics Observatory** tab and pick any generated log to analyze.

### How It Works

The dashboard runs the same `CoffeeShop` multi-agent graph used by the notebooks and CLI. A background thread drives the conversation (using the simulated Customer Agent), while the Panel UI polls for events every 100ms. Stream events from LangGraph are parsed into typed dashboard events (agent messages, tool calls, handoffs, etc.) and dispatched to the corresponding agent panel.

The Metrics Observatory loads CSV event logs into an `ObjectCentricEventlog` and renders sections from those logs — it is read-only and does not write to disk.

## Trace Table Dashboard

The Trace Table is the third page of the multi-page Agent Observatory dashboard, focused on the global message trace: one row per emitted message, with columns per agent plus a Process Supervisor column. It shares the same `CoffeeShop` graph and event bus as the Interaction Observatory but presents the conversation as a single, globally ordered table next to the live tray, stock, and coffee machine status.

### Launch

```bash
# The Trace Table is served by the regular dashboard command
poetry run dashboard
# Then open: http://localhost:5006/trace
```

Use the header tabs to switch between the Interaction, Metrics, and Trace pages.

### Features

- **Global trace table**: every agent message, tool call, tool result, and handoff as one row, in emission order
- **Top status strip**: tray, stock, and coffee machine widgets shared with the Agent Observatory
- **Sidebar controls**: scenario picker, log level, editable customer prompt, run button
- **Conversation log**: chat-style log below the sidebar with smart auto-scroll (sticks to bottom only when already at the bottom)

---

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
