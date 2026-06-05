# AGENTS.md

## Project Overview

A multi-agent coffee shop system that demonstrates process mining of LLM-based agents. Five specialized agents (Order, Inventory, Barista, Customer Service, Customer) collaborate via LangGraph Swarm, with interactions logged via MLflow for trace analysis.

## Tech Stack

- **Python 3.13+** with Poetry (primary) or pip
- **LangGraph + LangGraph Swarm** — multi-agent orchestration
- **LangChain < 1.0.0** — LLM provider abstraction
- **Ollama** (default LLM runtime, model: `ministral-3:14b`) or **Anthropic** via Hyperspace AI proxy
- **MLflow** — experiment tracking and OpenTelemetry tracing
- **Jupyter Notebook + ipywidgets** — interactive UI
- **Pandas** — event log processing

## Project Structure

```
├── 1_Standard_agentic_coffee_shop.ipynb    # Exercise 1: basic order flow
├── 2_Exceptions_agentic_coffee_shop.ipynb  # Exercise 2: error handling
├── 3_Extending_agentic_coffee_shop.ipynb   # Exercise 3: agent customization
├── src/
│   ├── coffee_shop.py                      # Main CoffeeShop class & Jupyter UI
│   ├── simulate.py                         # Headless CLI simulation script
│   ├── styles.py                           # CSS for chat interface
│   ├── agents/
│   │   ├── shared_components.py            # Data models, menu, handoff tools
│   │   ├── order_agent.py                  # Order taking & pricing
│   │   ├── inventory_agent.py              # Stock management
│   │   ├── barista_agent.py                # Order prep (20% simulated failure rate)
│   │   ├── customer_service_agent.py       # Issue resolution & refunds
│   │   └── customer_agent.py               # Simulated customer with scenarios
│   └── trace_processing/
│       ├── trace_processor.py              # MLflow trace discovery & batch processing
│       └── log_generator.py                # OpenTelemetry trace → CSV event log
```

## Setup & Running

```bash
# Poetry (recommended)
poetry install
poetry jupyter install

# Pip fallback
pip install -r requirements.txt
pip install "langchain[ollama]<1.0.0"

# Run notebooks
jupyter notebook
# Then open notebooks 1–3 in order

# Headless simulation (generate traces without UI)
poetry run simulate --traces 10 --scenario all --export-logs
```

LLM provider is configured via a `.env` file (see `.env.example`). Set `LLM_PROVIDER=ollama` (default) or `LLM_PROVIDER=anthropic` for the Hyperspace AI proxy. The factory lives in `src/llm.py`.

## Key Architecture Notes

- Agents are non-hierarchical (swarm pattern), coordinated via `create_handoff_tool()`
- Agents run sequentially (not in parallel) for Ollama compatibility
- The Customer Agent drives conversations externally — it is not part of the swarm graph
- Order status lifecycle: `pending → inventory_confirmed → completed/preparation_error → refunded`
- MLflow traces are stored under `./mlruns/` and converted to XES-compatible CSV event logs in `./generated_event_log/`

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
- No automated test suite — validation is via interactive notebook exercises

## Important Constraints

- This is educational/demo material, not production code
- No automated tests exist
- In-memory checkpointing only (no persistence across sessions)
- The 20% barista error rate is intentional — it creates process variants for mining analysis

## Running Tests

### E2E test (`tests/test_simulation_e2e.py`)

The E2E simulation test invokes the full multi-agent flow with real LLM calls. When `LLM_PROVIDER=anthropic` and the LLM is served via a **local proxy** (e.g., Hyperspace AI at `localhost:6655`), the test subprocess must be able to reach localhost. In Claude Code, this means the Bash tool call **must** use `dangerouslyDisableSandbox: true` — the default sandbox blocks localhost network access, causing connection failures or SOCKS proxy errors (`socksio` not installed).

This does NOT apply when using Ollama or a remote LLM endpoint — only when the Anthropic base_url points to localhost.

