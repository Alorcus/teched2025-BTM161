import logging
from pathlib import Path

import mlflow

from src.llm import create_chat_llm
from src.config import CoffeeShopConfig
from src.setups import setup_dir
from src.agents.order_store import create_order_store_engine, set_engine
from src.agents import (
    init_db, reset_inventory, set_item_stock, get_all_inventory,
    CustomerAgent, CUSTOMER_SCENARIOS,
)
from src.control_plane import AgentRepo, Catalog, JsonlLogSink, ProcessSupervisor
from src.graph import build_coffee_shop_graph
from src.conversation import ConversationEngine
from src.notebook_ui import NotebookUI, AGENT_CONFIG

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
    """Facade combining graph construction, conversation engine, and UI."""

    def __init__(self, config: CoffeeShopConfig | None = None):
        self.config = config or CoffeeShopConfig()
        self.traces_of_latest_conversations = []
        self.verbose_mode = True
        self.customer_agent_enabled = False
        self.customer_agent = None
        self._last_agent_message = None
        self.agent_config = AGENT_CONFIG
        self._engine = None
        self._ui = None
        self.agent_repo: AgentRepo | None = None
        self.catalog: Catalog | None = None
        self.log_sink: JsonlLogSink | None = None
        self.process_supervisor: ProcessSupervisor | None = None

    def open_shop(self, reset_inventory_first=True):
        """Start the coffee shop application after potentially updating agent definitions"""
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
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
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

        self.app = build_coffee_shop_graph(llm, self.agent_repo, self.catalog, self.log_sink)

        if self.config.process_supervisor_enabled:
            self.process_supervisor = ProcessSupervisor(
                process_model_path=self.config.process_model_path,
                log_path=self.config.process_log_path,
                llm=llm,
                prompt_template=self.agent_repo.get("process_supervisor").base_prompt,
            )
            _coffee_shop_logger.info(
                "process supervisor: model=%s | log=%s",
                self.config.process_model_path,
                self.config.process_log_path,
            )
        else:
            _coffee_shop_logger.info(
                "process supervisor: DISABLED — no observation, no critique, no process_meta.log entries"
            )

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

    def run_conversation(self, scenario_index=None, on_message=None, reset_inventory_first=True):
        """Run a full automated conversation using the CustomerAgent."""
        if reset_inventory_first:
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

    def capture_feedback(self, thread_id: str, order_id: str | None = None) -> dict:
        """Capture customer feedback for a completed conversation and persist it."""
        feedback = self.customer_agent.get_feedback()
        self._conversation_engine.feedback_log[thread_id] = {
            "thread_id": thread_id,
            "order_id": order_id,
            **feedback,
        }
        self._conversation_engine._save_feedback_store()
        return feedback

    def get_last_feedback(self) -> dict | None:
        """Return the most recently recorded customer feedback entry."""
        log = self._conversation_engine.feedback_log
        return next(reversed(log.values()), None) if log else None

    def display_current_inventory(self):
        if self._ui:
            self._ui.display_current_inventory()
