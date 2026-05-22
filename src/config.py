from dataclasses import dataclass, field
from typing import Any


@dataclass
class CoffeeShopConfig:
    """Configuration for the coffee shop system.

    All fields are optional — None means "use default from environment".
    """
    llm: Any = None
    db_url: str | None = None
    mlflow_experiment: str = "lg-coffee-mas"
    mlflow_enabled: bool = True
    coffee_machine_url: str = "http://127.0.0.1:8001"
