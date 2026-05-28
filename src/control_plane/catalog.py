"""Concrete guardrail and guideline objects, keyed by id.

Hard-guardrail predicates and ids live in Python (typed, testable).
Guideline prose lives in YAML and is loaded at construction.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

from .guardrails import Guardrail, HardGuardrail, SoftGuardrail
from .predicates import (
    allowed_handover_targets_predicate,
    discount_within_limit_predicate,
)
from .types import Effect


@dataclass(frozen=True)
class Guideline:
    id: str
    prompt: str
    version: str = "v1"


_GUARDRAILS: dict[str, Guardrail] = {
    "allowed_handover_targets": HardGuardrail(
        name="allowed_handover_targets",
        version="v1",
        tools=["transfer_to_agent"],
        effect=Effect.DENY,
        description="Handover target must be in the calling agent's allowed_handovers list.",
        predicate=allowed_handover_targets_predicate,
    ),
    "discount_within_30pct": HardGuardrail(
        name="discount_within_30pct",
        version="v1",
        tools=["calculate_total"],
        effect=Effect.FLAG,
        description="Flag (do not block) calculate_total invocations with discount_percent > 30.",
        predicate=discount_within_limit_predicate(30),
    ),
    "handover_appropriateness_soft_stub": SoftGuardrail(
        name="handover_appropriateness_soft_stub",
        version="v1",
        tools=["transfer_to_agent"],
        effect=Effect.ALLOW,
        description="Stub soft guardrail asking 'is this handover appropriate?' (always allow for MVP).",
        judge_prompt="Is the proposed handover appropriate given conversation state and guidelines?",
        state_dependencies=["conversation"],
    ),
}


class Catalog:
    """Resolves guardrail and guideline ids to concrete objects."""

    def __init__(self, config_dir: Path):
        self._guardrails: dict[str, Guardrail] = dict(_GUARDRAILS)
        self._guidelines: dict[str, Guideline] = {}
        guidelines_dir = Path(config_dir) / "guidelines"
        if guidelines_dir.exists():
            for yaml_path in sorted(guidelines_dir.glob("*.yaml")):
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                for entry in data.get("guidelines", []):
                    gl = Guideline(
                        id=entry["id"],
                        prompt=entry["prompt"],
                        version=entry.get("version", "v1"),
                    )
                    self._guidelines[gl.id] = gl

    def guardrails(self, ids: list[str]) -> list[Guardrail]:
        missing = [i for i in ids if i not in self._guardrails]
        if missing:
            raise KeyError(f"Unknown guardrail ids: {missing}")
        return [self._guardrails[i] for i in ids]

    def guidelines(self, ids: list[str]) -> list[Guideline]:
        missing = [i for i in ids if i not in self._guidelines]
        if missing:
            raise KeyError(f"Unknown guideline ids: {missing}")
        return [self._guidelines[i] for i in ids]
