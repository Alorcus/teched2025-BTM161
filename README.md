# BTM161 - AI agent mining and governance with SAP Signavio solutions

## Description
This repository hosts the (self-contained but simplified) material for the SAP TechEd 2025 session "BTM161 - AI agent mining and governance with SAP Signavio solutions"

## Overview
This SAP Sample introduces attendees to analyzing the behavior of LLM-based multi-agent systems with [SAP Signavio Process Intelligence](https://www.signavio.com/products/process-intelligence/). Within three Jupyter notebooks of an agentic coffee shop, specialized agents work together to provide a complete coffee shop experience. Each agent has specific tools and responsibilities, and they intelligently hand off to each other based on the situation to serve the "best" coffee for its customers. The interactions with the coffee shop and the actions taken by the agents are logged and, afterwards, can be analyzed by generating event logs suitable for process mining with SAP Signavio Process Intelligence.

Please be aware that this repository includes a simplified version which generates CSV-files that can be easily consumed by your own SAP Signavio Process Intelligence. This will allow you to try this self-contained version on your own afterwards as well adapt it to your needs but it will not include all features that have been presented at TechEd.

For the TechEd hands-on session, we will host a dedicated Jupyter notebook instance, offer access to SAP Signavio Process Intelligence, and several advanced agent mining features in a dedicated service. 

## Requirements

This section covers the prerequisites and the installation process required for running the coffee shop MAS.

### Prerequisites
- [Python](https://www.python.org/downloads/) >= 3.13
- (Recommended) [poetry](https://python-poetry.org/) is used for managing packages and the virtual environment and needs to be [installed](https://python-poetry.org/docs/#installing-with-pipx).
    - Alternative: Use pip to install the dependencies based on the provided `requirements.txt`.
- (Recommended) Use the [poetry-jupyter-plugin](https://pypi.org/project/poetry-jupyter-plugin/) to install the virtual environment created by Poetry as a Jupyter kernel:
    ```
    $ poetry self add poetry-jupyter-plugin
    ```
    - Alternative: Set up the Jupyter kernel manually.
- You need an API key for an [LLM provider supported by LangChain](https://python.langchain.com/docs/integrations/chat/#featured-providers).

### Installation
1. Install the project by running `poetry install` in this directory.
1. Install a Juypter kernel via poetry by running `poetry jupyter install`.
1. Activate the virtual environment created by poetry. To obtain the appropriate activation command for this run `poetry env activate`. (Alternative: Prefix all subsequent commands with `poetry run`, which will execute the command in the virtual environment.)
1. Install the appropriate langchain integration package fitting your supported LLM provider:
    1. Identify the name of the package using the [documentation](https://github.com/langchain-ai/langgraph/blob/a10a66cbd151c92f89d6476fb70e5e405ce50b98/docs/docs/snippets/chat_model_tabs.md)), e.g., `langchain[openai]` or `langchain[anthropic]`.
    2. Install the package _in the version fitting this repository_ by adjusting and running `pip install "langchain[PROVIDER]<1.0.0"`, e.g., `pip install "langchain[openai]<1.0.0"` or `pip install "langchain[anthropic]<1.0.0"`.
3. Open `src/coffee_shop.py` and, starting line 20, configure access to the LLM, like credentials or API keys, according to the [documentation](https://github.com/langchain-ai/langgraph/blob/a10a66cbd151c92f89d6476fb70e5e405ce50b98/docs/docs/snippets/chat_model_tabs.md).
1. Start the Jupyter server by running `jupyter notebook`.
1. Now, you should be able to open and run the first notebook: `1_Standard_agentic_coffee_shop.ipynb`.

## Exercises
The session contains three exercises in the form of Juypter notebooks, which are self-contained and thus include the respective instructions for completing the exercise:
1. Exercise: [`1_Standard_agentic_coffee_shop`](1_Standard_agentic_coffee_shop.ipynb). This exercise is for getting to know the overall setup and generate a first trace by interacting with the coffee shop, uploading a mapped CSV, and analyzing it with SAP Signavio Process Intelligence.
2. Exercise [`2_Exceptions_agentic_coffee_shop`](2_Exceptions_agentic_coffee_shop.ipynb). This exercise is about exploring the behavior of the agents in case of errors and when experiencing edge cases. This will lead to several process variants by agents and shows how to analyze their differences over time.
3. Exercise [`3_Extending_agentic_coffee_shop`](3_Extending_agentic_coffee_shop.ipynb). This exercise is for experimenting with the agents' definitions in order to change their behavior, for example, by changing their instructions and the available tools. With the help of SAP Signavio Process Intelligence, you will find out how this tool can support you in monitoring a multi-agent system during development. 

## Headless Simulation

You can generate traces in bulk without the Jupyter UI using the `simulate` CLI command. This runs the Customer Agent against the coffee shop swarm and captures MLflow traces for each conversation.

### Usage

```bash
# Run a single trace with a random scenario
poetry run simulate

# Run 10 traces cycling through all 4 scenarios
poetry run simulate --traces 10 --scenario all

# Run 5 traces with a specific scenario (index 0-3)
poetry run simulate --traces 5 --scenario 2

# Run with minimal output (no message content)
poetry run simulate --traces 10 --quiet

# Run with debug logging enabled
poetry run simulate --traces 5 --log-level debug

# Export event logs after simulation
poetry run simulate --traces 10 --scenario all --export-logs
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--traces N` | `1` | Number of conversation traces to run |
| `--scenario` | `random` | Scenario index (`0`–`3`), `all` (round-robin), or `random` |
| `--export-logs` | off | Generate event log CSV after simulation |
| `--quiet` | off | Minimal output: only trace numbers, scenarios, and summary |
| `--log-level` | `warning` | Set the logging level for agent diagnostics (`debug`, `info`, `warning`, `error`) |

### Available Scenarios

| Index | Description |
|---|---|
| 0 | Order a large latte and a croissant (friendly) |
| 1 | Order 2 espressos (in a hurry) |
| 2 | Complain about a cold cappuccino and seek resolution |
| 3 | Ask for a recommendation and order based on suggestion |

## Agent Observatory Dashboard

A real-time observability dashboard built with [Panel](https://panel.holoviz.org/) that shows all agents simultaneously in a grid layout. Each agent panel displays its system prompt, available tools, current status, handoff context, message history (context-isolated), and tool call log — all updating live as a conversation streams through the system.

### Launch

```bash
# Start the dashboard (opens browser at http://localhost:5006)
poetry run dashboard

# Or via Panel CLI
panel serve src/dashboard/app.py --show --port 5006
```

### Features

- **2x2 grid layout** showing all 4 agents at once (scales to 3x3 for up to 9)
- **Live status badges**: idle / thinking / executing tool / handed off
- **Handoff context display**: see what each agent received from the previous agent
- **Tool call log**: arguments and results for every tool invocation
- **Context-isolated messages**: the same filtered view each agent's LLM actually sees
- **Sidebar controls**: scenario selector, run button, and global conversation log

### How It Works

The dashboard runs the same `CoffeeShop` multi-agent graph used by the notebooks and CLI. A background thread drives the conversation (using the simulated Customer Agent), while the Panel UI polls for events every 100ms. Stream events from LangGraph are parsed into typed dashboard events (agent messages, tool calls, handoffs, etc.) and dispatched to the corresponding agent panel.

---

## Observing the Database

Orders and inventory are persisted in a local SQLite database (`coffee_shop.db`). To inspect the database while the agents are running, install the [SQLite Viewer](https://marketplace.visualstudio.com/items?itemName=qwtel.sqlite-viewer) extension in VS Code. Once installed, simply open `coffee_shop.db` from the file explorer and the extension will display the tables in a browsable grid view. You can refresh the view at any time to see the latest orders and stock levels as they are updated by the agents during a simulation.

## Agent Architecture

### 🛒 Order Agent
**Role**: Takes and processes customer orders
**Tools**:
- `process_order()` - Parse customer orders
- `calculate_total()` - Calculate pricing with discount capabilities
- Handoff tools to inventory and customer service agents

**Responsibilities**:
- Welcome customers and take orders
- Validate menu items and quantities
- Calculate totals and apply discounts
- Transfer to inventory for availability checks

### 📦 Inventory Agent
**Role**: Manages stock levels and availability
**Tools**:
- `check_inventory()` - Verify item availability for orders
- `update_stock()` - Decrease inventory after confirmed orders
- `get_alternatives()` - Find substitute items for out-of-stock products
- Handoff tools to barista and customer service agents

**Responsibilities**:
- Check item availability against current stock
- Update inventory after order confirmation
- Suggest alternatives for unavailable items
- Transfer to barista when items are available
- Escalate to customer service for stock issues

### ☕ Barista Agent
**Role**: Handles order preparation and quality
**Tools**:
- `prepare_order()` - Simulate order preparation with realistic error handling
- `remake_order_item()` - Handle preparation errors and remakes
- `estimate_prep_time()` - Provide accurate timing estimates
- Handoff tool to customer service for issues

**Responsibilities**:
- Prepare drinks and food items
- Handle preparation errors (20% failure rate simulation)
- Provide preparation time estimates
- Quality control and remake capabilities

### 🙋 Customer Agent
**Role**: Simulates a customer interacting with the coffee shop
**Scenarios**:
- Ordering a latte and croissant
- Quickly ordering two espressos
- Complaining about a cold drink and seeking resolution
- Asking for a recommendation and ordering based on the suggestion

**Behavior**:
- Picks a scenario randomly (or by index via `reset()`)
- Drives the conversation by sending an opening message and responding to agent replies
- Ends the conversation after at most 8 turns, or when the goal is achieved (signals `DONE`)

---

### 🤝 Customer Service Agent
**Role**: Manages customer satisfaction and issue resolution
**Tools**:
- `offer_refund()` - Process refunds when necessary
- `offer_partial_refund()` - Process a partial refund when necessary
- Handoff tools to all other agents

**Responsibilities**:
- Handle customer complaints with empathy
- Offer appropriate compensation (remakes, refunds, discounts)
- Suggest alternatives with customer service touch
- Coordinate with other agents for resolution

## Running the Tests

```bash
python -m unittest discover -s tests -v
```

Individual test modules can be run directly, e.g.:

```bash
python -m unittest tests/test_tools_order.py -v
```

Use the Python interpreter from the Poetry virtual environment (`poetry env activate` first, or prefix with `poetry run python`).

---
## Contributing
Please read the [CONTRIBUTING.md](./CONTRIBUTING.md) to understand the contribution guidelines.

## Code of Conduct
Please read the [SAP Open Source Code of Conduct](https://github.com/SAP-samples/.github/blob/main/CODE_OF_CONDUCT.md).

## How to obtain support

Support for the content in this repository is available during the actual time of the online session for which this content has been designed. Otherwise, you may request support via the [Issues](../../issues) tab.

## License
Copyright (c) 2025 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.
