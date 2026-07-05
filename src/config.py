from dataclasses import dataclass
from typing import Any


@dataclass
class CoffeeShopConfig:
    """Configuration for the coffee shop system.

    `setup_name` selects which `config/setups/<name>/` directory to load.
    There is no default — callers must pick one explicitly.

    `handover_pause_default` seeds the dashboard's "pause at next handover"
    toggle initial state on page load. When `True`, the dashboard starts in
    Pause mode and the next inter-agent handover will halt until the user
    clicks the toggle back to Go. When `False` (default), handovers proceed
    normally. Only affects the dashboard runner; the headless `simulate` path
    has no consumer for this signal.
    """
    llm: Any = None
    db_url: str | None = None
    mlflow_experiment: str = "lg-coffee-mas"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_enabled: bool = True
    coffee_machine_url: str = "http://127.0.0.1:8001"
    setup_name: str | None = None
    guardrail_log_path: str = "./guardrail_log/events.jsonl"
    process_model_path: str = "./config/process_model.yaml"
    process_log_path: str = "./process_log/process_meta.log"
    process_supervisor_enabled: bool = False
    process_supervisor_active: bool = False
    process_supervisor_max_retries: int = 3
    handover_pause_default: bool = False
    recursion_limit: int = 100
