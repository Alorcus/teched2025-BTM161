import logging

import mlflow

from src.llm import create_chat_llm
from src.config import CoffeeShopConfig
from src.setups import setup_dir
from src.agents.order_store import create_order_store_engine, set_engine
from src.agents import (
    init_db, reset_inventory, CustomerAgent,
)
from src.control_plane import AgentRepo, Catalog, JsonlLogSink
from src.graph import build_coffee_shop_graph
from src.conversation import ConversationEngine

AGENT_CONFIG = {
    'order_agent': {
        'icon': '\U0001f4dd',
        'name': 'Order Agent',
        'color': '#2196F3',
        'bg_color': '#E3F2FD',
    },
    'inventory_agent': {
        'icon': '\U0001f4e6',
        'name': 'Inventory Agent',
        'color': '#FF9800',
        'bg_color': '#FFF3E0',
    },
    'barista_agent': {
        'icon': '☕',
        'name': 'Barista Agent',
        'color': '#8BC34A',
        'bg_color': '#F1F8E9',
    },
    'customer_service_agent': {
        'icon': '\U0001f4ac',
        'name': 'Customer Service',
        'color': '#E91E63',
        'bg_color': '#FCE4EC',
    },
    'user': {
        'icon': '\U0001f464',
        'name': 'You',
        'color': '#424242',
        'bg_color': '#F5F5F5',
    },
}

class _PaddedNameFormatter(logging.Formatter):
    """Left-pads %(name)s to the widest logger name seen so far, so child loggers stay aligned."""
    _max_width = 0

    def format(self, record):
        type(self)._max_width = max(self._max_width, len(record.name))
        original = record.name
        record.name = record.name.ljust(self._max_width)
        try:
            return super().format(record)
        finally:
            record.name = original

_coffee_shop_logger = logging.getLogger("coffee_shop")
_coffee_shop_logger.setLevel(logging.INFO)
if not _coffee_shop_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(_PaddedNameFormatter("[%(levelname)-8s] %(name)s — %(message)s"))
    _coffee_shop_logger.addHandler(_handler)


class CoffeeShop:
    """Facade combining graph construction and the conversation engine."""

    def __init__(self, config: CoffeeShopConfig | None = None):
        self.config = config or CoffeeShopConfig()
        self.traces_of_latest_conversations = []
        self.customer_agent = None
        self.agent_config = AGENT_CONFIG
        self._engine = None
        self.agent_repo: AgentRepo | None = None
        self.catalog: Catalog | None = None
        self.log_sink: JsonlLogSink | None = None
        self.gateways: dict = {}

    def open_shop(self, reset_inventory_first=True):
        engine = create_order_store_engine(self.config.db_url)
        set_engine(engine)
        self._engine = engine

        init_db()
        if reset_inventory_first:
            _coffee_shop_logger.info("Resetting inventory to initial stock levels")
            reset_inventory()

        llm = self.config.llm or create_chat_llm()

        self.customer_agent = CustomerAgent(llm)

        if self.config.mlflow_enabled:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.langchain.autolog()
            if not mlflow.get_experiment_by_name(self.config.mlflow_experiment):
                mlflow.create_experiment(self.config.mlflow_experiment)
            mlflow.set_experiment(self.config.mlflow_experiment)

        if not self.config.setup_name:
            raise ValueError(
                "CoffeeShopConfig.setup_name is required — pick a setup from config/setups/"
            )
        config_dir = setup_dir(self.config.setup_name)
        self.agent_repo = AgentRepo(config_dir)
        self.catalog = Catalog(config_dir)
        self.log_sink = JsonlLogSink(self.config.guardrail_log_path, setup_name=self.config.setup_name)
        _coffee_shop_logger.info(
            f"control plane: setup={self.config.setup_name} | agents={self.agent_repo.ids()} | log={self.config.guardrail_log_path}"
        )

        self.app, self.gateways = build_coffee_shop_graph(
            llm, self.agent_repo, self.catalog, self.log_sink
        )

        self._conversation_engine = ConversationEngine(
            self.app,
            mlflow_enabled=self.config.mlflow_enabled,
            setup_name=self.config.setup_name,
        )

    def _get_config(self, thread_id):
        return {"configurable": {"thread_id": thread_id}}

    def send_message(self, thread_id, message):
        """Send a message through the swarm and return the last customer-facing agent response."""
        result = self._conversation_engine.send_message(thread_id, message)
        self.traces_of_latest_conversations = self._conversation_engine.traces_of_latest_conversations
        return result

    def run_conversation(self, scenario_index=None, on_message=None, reset_inventory_first=True):
        """Run a full automated conversation using the CustomerAgent."""
        if reset_inventory_first:
            reset_inventory()
        trace_ids = self._conversation_engine.run_automated(
            self.customer_agent, scenario_index=scenario_index, on_message=on_message
        )
        self.traces_of_latest_conversations = self._conversation_engine.traces_of_latest_conversations
        return trace_ids

    def capture_feedback(self, thread_id: str, order_id: str | None = None) -> dict:
        """Capture customer feedback for a completed conversation and persist it."""
        feedback = self.customer_agent.get_feedback()
        self._conversation_engine.feedback_log[thread_id] = {
            "thread_id": thread_id,
            "order_id": order_id,
            "scenario_index": self.customer_agent.scenario_index,
            **feedback,
        }
        self._conversation_engine._save_feedback_store()
        return feedback

    def get_last_feedback(self) -> dict | None:
        """Return the most recently recorded customer feedback entry."""
        log = self._conversation_engine.feedback_log
        return next(reversed(log.values()), None) if log else None
