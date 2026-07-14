# AGENTS.md

## Project Overview

A multi-agent coffee shop system that demonstrates process mining of LLM-based agents. Five specialized agents (Order, Inventory, Barista, Customer Service, Customer) collaborate via LangGraph Swarm, with interactions logged via MLflow for trace analysis.

## Tech Stack

- **Python 3.13+** with Poetry (primary) or pip
- **LangGraph + LangGraph Swarm** — multi-agent orchestration
- **LangChain < 1.0.0** — LLM provider abstraction
- **Ollama** (default LLM runtime, model: `ministral-3:14b`) or **Anthropic** via Hyperspace AI proxy
- **MLflow** — experiment tracking and OpenTelemetry tracing
- **Panel** — interactive dashboard UI
- **Pandas** — event log processing

## Project Structure

```
├── src/
│   ├── coffee_shop.py                      # Main CoffeeShop facade
│   ├── simulate.py                         # Headless CLI simulation script
│   ├── agents/
│   │   ├── shared_components.py            # Data models, menu, handoff tools
│   │   ├── order_agent.py                  # Order taking & pricing
│   │   ├── inventory_agent.py              # Stock management
│   │   ├── barista_agent.py                # Order prep (20% simulated failure rate)
│   │   ├── customer_service_agent.py       # Issue resolution & refunds
│   │   └── customer_agent.py               # Simulated customer with scenarios
│   ├── dashboard/                          # Panel-based observability dashboard
│   └── trace_processing/
│       ├── trace_processor.py              # MLflow trace discovery & batch processing
│       └── log_generator.py                # OpenTelemetry trace → CSV event log
```

## Setup & Running

```bash
# Poetry (recommended)
poetry install

# Pip fallback
pip install -r requirements.txt
pip install "langchain[ollama]<1.0.0"

# Headless simulation (generate traces without UI)
poetry run simulate --setup baseline --traces 10 --scenario all --export-logs

# Observability dashboard
poetry run dashboard --setup baseline
```

LLM provider is configured via a `.env` file (see `.env.example`). Set `LLM_PROVIDER=ollama` (default) or `LLM_PROVIDER=anthropic` for the Hyperspace AI proxy. The factory lives in `src/llm.py`.

## Key Architecture Notes

- Agents are non-hierarchical (swarm pattern), coordinated via `create_handoff_tool()`
- Agents run sequentially (not in parallel) for Ollama compatibility
- The Customer Agent drives conversations externally — it is not part of the swarm graph
- Order status lifecycle: `pending → inventory_confirmed → completed/preparation_error → refunded`
- MLflow traces are stored under `./mlruns/` and converted to XES-compatible CSV event logs in `./generated_event_log/`

### Event log schema

The consolidated `_all_traces.csv` (and its schema version, timezone contract, and per-column sensitivity classification) is documented in [`EVENTLOG_SCHEMA.md`](EVENTLOG_SCHEMA.md). Treat that file as the single source of truth for the CSV's columns — update it in the same commit as any producer change in `LogGenerator`, `TraceProcessor`, or `trace_cache`.

## Customer Feedback

After each conversation, `CustomerAgent.get_feedback()` invokes the LLM to rate service quality from the customer perspective. The result is a structured record with:

- `feedback_score` — float `0.0–1.0` (anchors: `1.0` excellent, `0.5` acceptable, `0.0` poor)
- `feedback_reason` — short free-text explanation
- `valid` — whether the LLM response parsed cleanly (falls back to `0.5` if not)

Feedback is persisted to `./feedback_store.json` keyed by `thread_id` (the conversation UUID, identical to `case_id` in the event log). During log export, `TraceProcessor` injects exactly one `customer_feedback` event at the end of each case. Both the headless `simulate` path and the `dashboard` runner capture feedback.

To export event logs with feedback after a dashboard session:

```bash
python3 -c "from src.trace_processing import TraceProcessor; TraceProcessor().process_all_traces()"
```

## MLflow Trace Tags

Every LangGraph trace (one MLflow trace per `app.stream(...)` call) is tagged
with two attributes the Metrics Dashboard filters against:

- `setup` — the active setup name (`baseline`, `all_handovers`, `unconstrained`)
- `scenario_index` — the customer scenario played (`0`–`3` for a preset, `-1`
  for a custom prompt where no scenario applies)

Tags are written by `_tag_trace(trace_id, setup, scenario)` in
`src/conversation.py`, called at every site that produces a coffee-shop trace
(`ConversationEngine.send_message`, `ConversationRunner._stream_with_events`). The trace processor lifts the
tags into `case_setup` and `case_scenario_index` columns on every event row
of the CSV cache; the extractor's `SCHEMA_VERSION` is bumped whenever the
column shape changes so already-built caches are rebuilt on demand.

## Branch Naming

New branches follow: `<two-initials>/<feature-description-with-dashes>`
Example: `al/add-login-page`

## Merging

Squash branches before merging into `main` so each merged change is a single commit — keeps history linear and makes reverts straightforward.

## Code Conventions

- Agent tools follow the pattern: `@tool(args_schema=Schema)` with docstrings (required by LangChain)
- Tool functions return `json.dumps(result)` for structured output
- Snake_case for functions/variables, PascalCase for classes
- Pydantic models and dataclasses for data structures
- `unittest`-based test suite in `tests/`; new tests follow `class TestX(unittest.TestCase)` with `test_*` methods (see `tests/test_tools_order.py` for the canonical shape)
- Follow the guiding principle: **good code documents itself.** Prefer clear names and structure over comments; only add a comment when it captures a non-obvious *why* (external constraint, subtle invariant, workaround) that the code can't express on its own.

## Important Constraints

- This is educational/demo material, not production code
- In-memory checkpointing only (no persistence across sessions)
- The 20% barista error rate is intentional — it creates process variants for mining analysis

## Running Tests

### E2E test (`tests/test_simulation_e2e.py`)

The E2E simulation test invokes the full multi-agent flow with real LLM calls. When `LLM_PROVIDER=anthropic` and the LLM is served via a **local proxy** (e.g., Hyperspace AI at `localhost:6655`), the test subprocess must be able to reach localhost. In Claude Code, this means the Bash tool call **must** use `dangerouslyDisableSandbox: true` — the default sandbox blocks localhost network access, causing connection failures or SOCKS proxy errors (`socksio` not installed).

This does NOT apply when using Ollama or a remote LLM endpoint — only when the Anthropic base_url points to localhost.

