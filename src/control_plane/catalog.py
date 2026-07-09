"""Concrete guardrail and guideline objects, keyed by id.

Guardrails and guidelines are both loaded from YAML at construction time.
Predicate logic itself lives in `predicates.py` and is referenced by name
through `PREDICATE_REGISTRY` so YAML stays declarative.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

from .guardrails import Guardrail, HardGuardrail, SoftGuardrail
from .predicates import PREDICATE_REGISTRY
from .types import Effect
from .temporal_constraints import TemporalConstraint
from .temporal_guardrail import TemporalConstraintGuardrail, create_temporal_guardrail


@dataclass(frozen=True)
class Guideline:
    id: str
    prompt: str
    version: str = "v1"


def _build_guardrail(entry: dict) -> Guardrail:
    guardrail_id = entry["id"]
    guardrail_type = entry.get("type")
    if guardrail_type not in ("hard", "soft"):
        raise ValueError(f"Guardrail {guardrail_id!r}: type must be 'hard' or 'soft', got {guardrail_type!r}")

    try:
        effect = Effect(entry.get("effect", "flag"))
    except ValueError as exc:
        raise ValueError(f"Guardrail {guardrail_id!r}: invalid effect {entry.get('effect')!r}") from exc

    common = {
        "name": guardrail_id,
        "version": entry.get("version", "v1"),
        "tools": list(entry.get("tools", [])),
        "effect": effect,
        "description": entry.get("description", ""),
    }

    if guardrail_type == "hard":
        predicate_name = entry.get("predicate")
        if predicate_name not in PREDICATE_REGISTRY:
            known = sorted(PREDICATE_REGISTRY)
            raise ValueError(
                f"Guardrail {guardrail_id!r}: unknown predicate {predicate_name!r}. Known: {known}"
            )
        predicate_args = entry.get("predicate_args") or None
        callable_ = PREDICATE_REGISTRY[predicate_name]
        predicate = callable_(**predicate_args) if predicate_args else callable_
        return HardGuardrail(predicate=predicate, predicate_args=predicate_args, **common)

    return SoftGuardrail(
        judge_prompt=entry.get("judge_prompt", ""),
        state_dependencies=list(entry.get("state_dependencies", [])),
        **common,
    )


class Catalog:
    """Resolves guardrail and guideline ids to concrete objects."""

    def __init__(self, config_dir: Path):
        self._guardrails: dict[str, Guardrail] = {}
        self._guidelines: dict[str, Guideline] = {}
        self._temporal_guardrail: TemporalConstraintGuardrail | None = None

        guardrails_dir = Path(config_dir) / "guardrails"
        if not guardrails_dir.exists():
            raise FileNotFoundError(f"Guardrails directory not found: {guardrails_dir}")
        else:
            for yaml_path in sorted(guardrails_dir.glob("*.yaml")):
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                for entry in data.get("guardrails", []):
                    guardrail = _build_guardrail(entry)
                    self._guardrails[guardrail.name] = guardrail

        guidelines_dir = Path(config_dir) / "guidelines"
        if not guidelines_dir.exists():
            raise FileNotFoundError(f"Guidelines directory not found: {guidelines_dir}")
        else:
            for yaml_path in sorted(guidelines_dir.glob("*.yaml")):
                data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                for entry in data.get("guidelines", []):
                    guideline = Guideline(
                        id=entry["id"],
                        prompt=entry["prompt"],
                        version=entry.get("version", "unversioned"),
                    )
                    self._guidelines[guideline.id] = guideline

        constraints_path = Path(config_dir) / "constraints" / "temporal_order.yaml"
        print(f"🔍 Looking for temporal constraints at: {constraints_path}")
        print(f"📁 File exists: {constraints_path.exists()}")

        if constraints_path.exists():
            try:
                self._temporal_guardrail = create_temporal_guardrail(constraints_path)
                print(f"✅ Temporal guardrail created with {len(self._temporal_guardrail.constraints)} constraints")
                print(f"   Tools monitored: {self._temporal_guardrail.tools}")
            except Exception as e:
                print(f"❌ Error loading temporal guardrail: {e}")
                self._temporal_guardrail = None
        else:
            print(f"⚠️ No temporal constraints file found")
            self._temporal_guardrail = None

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
    
    def get_temporal_guardrail(self) -> TemporalConstraintGuardrail | None:
        """Get the temporal constraint guardrail if configured."""
        return self._temporal_guardrail