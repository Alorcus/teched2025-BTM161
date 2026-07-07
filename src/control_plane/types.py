from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Effect(str, Enum):
    DENY = "deny"
    ALLOW = "allow"
    FLAG = "flag"


@dataclass
class Verdict:
    effect: Effect
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
    # Set only for "on_output" stage evaluation: the agent's final reply text
    # to be checked (e.g. by NeMo output rails). Empty for pre-call evaluation.
    output_text: str = ""
