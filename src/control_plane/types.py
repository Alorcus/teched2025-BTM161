from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class Effect(str, Enum):
    DENY = "deny"
    ALLOW = "allow"
    FLAG = "flag"


@dataclass
class Verdict:
    allowed: Effect
    guardrail_name: str
    guardrail_type: str
    reason_internal: str = ""
    reason_for_llm: str = ""


@dataclass
class GuardrailContext:
    agent_id: str
    tool_name: str
    tool_args: dict[str, Any]
    state: dict[str, Any] = field(default_factory=dict)
    allowed_handovers: list[str] = field(default_factory=list)
