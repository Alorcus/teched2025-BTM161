from dataclasses import dataclass
from typing import Any


@dataclass
class CoffeeShopConfig:
    """Configuration for the coffee shop system.

    `setup_name` selects which `config/setups/<name>/` directory to load.
    There is no default — callers must pick one explicitly.
    """

    llm: Any = None
    db_url: str | None = None
    mlflow_experiment: str = "lg-coffee-mas"
    mlflow_enabled: bool = True
    coffee_machine_url: str = "http://127.0.0.1:8001"
    setup_name: str | None = "baseline"
    guardrail_log_path: str = "./guardrail_log/events.jsonl"
    process_model_path: str = "./config/process_model.yaml"
    process_log_path: str = "./process_log/process_meta.log"
    process_supervisor_enabled: bool = True
    process_supervisor_active: bool = False
    process_supervisor_max_retries: int = 3
