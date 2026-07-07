from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from .types import Effect, GuardrailContext, Verdict


@dataclass
class Guardrail(ABC):
    """Base guardrail. Subclassed by evaluation mechanism."""

    name: str
    version: str = "unversioned"
    tools: list[str] = field(default_factory=list)
    effect: Effect = Effect.DENY
    description: str = ""
    # When the gateway evaluates this guardrail: "pre_call" (before a tool call,
    # the default and only stage the existing guardrails use) or "on_output"
    # (on the agent's final user-facing reply, via Gateway.evaluate_output).
    stage: str = "pre_call"

    @abstractmethod
    def eval(self, context: GuardrailContext) -> Verdict: ...

    @property
    def type(self) -> str:
        return "guardrail"

    def applies_to(self, tool_name: str) -> bool:
        return not self.tools or tool_name in self.tools


@dataclass
class HardGuardrail(Guardrail):
    """Deterministic rule, deterministic evaluation."""

    predicate: Callable[[GuardrailContext], Verdict] | None = None
    predicate_args: dict | None = None

    def eval(self, context: GuardrailContext) -> Verdict:
        if self.predicate is None:
            raise ValueError(f"HardGuardrail {self.name!r} has no predicate")
        verdict = self.predicate(context)
        if not verdict.guardrail_name:
            verdict.guardrail_name = self.name
        if not verdict.guardrail_type:
            verdict.guardrail_type = self.type
        return verdict

    @property
    def type(self) -> str:
        return "hard"


@dataclass
class SoftGuardrail(Guardrail):
    """LLM-as-judge evaluation. Stubbed for MVP — always allows, logs skipped."""

    judge_prompt: str = ""
    state_dependencies: list[str] = field(default_factory=list)

    def eval(self, context: GuardrailContext) -> Verdict:
        return Verdict(
            effect=Effect.ALLOW,
            guardrail_name=self.name,
            guardrail_type=self.type,
            reason_internal="soft guardrail evaluation skipped (stub)",
            reason_for_llm="",
        )

    @property
    def type(self) -> str:
        return "soft"
