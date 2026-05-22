import logging
from collections import defaultdict

import mlflow

from src.llm import create_chat_llm
from src.config import CoffeeShopConfig
from src.agents.order_store import create_order_store_engine, set_engine
from src.agents import (
    init_db, reset_inventory, set_item_stock, get_all_inventory,
    CustomerAgent, CUSTOMER_SCENARIOS,
)
from src.graph import build_coffee_shop_graph
from src.conversation import ConversationEngine
from src.notebook_ui import NotebookUI, AGENT_CONFIG

_coffee_shop_logger = logging.getLogger("coffee_shop")
_coffee_shop_logger.setLevel(logging.INFO)
if not _coffee_shop_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s — %(message)s"))
    _coffee_shop_logger.addHandler(_handler)


class CoffeeShop:
    """Facade combining graph construction, conversation engine, and UI."""

    def __init__(self, config: CoffeeShopConfig | None = None):
        self.config = config or CoffeeShopConfig()
        self.agent_definitions = defaultdict(str)
        self.traces_of_latest_conversations = []
        self.verbose_mode = True
        self.customer_agent_enabled = False
        self.customer_agent = None
        self._last_agent_message = None
        self.agent_config = AGENT_CONFIG
        self._engine = None
        self._ui = None

    def set_agent_definition(self, agent, definition):
        """Set or update the definition for a specific agent before starting the shop"""
        self.agent_definitions[agent] = definition

    def open_shop(self):
        """Start the coffee shop application after potentially updating agent definitions"""
        engine = create_order_store_engine(self.config.db_url)
        set_engine(engine)
        self._engine = engine

        init_db()
        reset_inventory()

        llm = self.config.llm or create_chat_llm()

        self.customer_agent = CustomerAgent(llm)

        if self.config.mlflow_enabled:
            mlflow.langchain.autolog()
            if not mlflow.get_experiment_by_name(self.config.mlflow_experiment):
                mlflow.create_experiment(self.config.mlflow_experiment)
            mlflow.set_experiment(self.config.mlflow_experiment)

        self.app = build_coffee_shop_graph(llm, self.agent_definitions)

        self._conversation_engine = ConversationEngine(
            self.app, mlflow_enabled=self.config.mlflow_enabled
        )

    def _get_config(self, thread_id):
        return {"configurable": {"thread_id": thread_id}}

    def send_message(self, thread_id, message):
        """Send a message through the swarm and return the last customer-facing agent response."""
        result = self._conversation_engine.send_message(thread_id, message)
        self.traces_of_latest_conversations = self._conversation_engine.traces_of_latest_conversations
        self._last_agent_message = result
        return result

    def run_conversation(self, scenario_index=None, on_message=None):
        """Run a full automated conversation using the CustomerAgent."""
        reset_inventory()
        trace_ids = self._conversation_engine.run_automated(
            self.customer_agent, scenario_index=scenario_index, on_message=on_message
        )
        self.traces_of_latest_conversations = self._conversation_engine.traces_of_latest_conversations
        return trace_ids

    def create_interactive_interface(self, success_only=False):
        """Create an enhanced interactive widget interface for the coffee shop"""
        self._ui = NotebookUI(
            self.app, self.customer_agent, mlflow_enabled=self.config.mlflow_enabled
        )
        self._ui.traces_of_latest_conversations = self.traces_of_latest_conversations
        return self._ui.create_interactive_interface(success_only=success_only)

    def display_current_inventory(self):
        if self._ui:
            self._ui.display_current_inventory()
