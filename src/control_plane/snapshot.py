import hashlib
import json
from typing import Iterable

from .guardrails import Guardrail, HardGuardrail


def snapshot_id(
    agent_id: str,
    agent_version: str,
    guardrails: Iterable[Guardrail],
    guidelines: Iterable[tuple[str, str]],
) -> str:
    """Stable id for an (agent, guardrails, guidelines) build.

    For hard guardrails with `predicate_args`, the args participate in the
    hash so the same predicate parameterized differently (e.g. max_pct=10
    vs max_pct=30) gets a distinct snapshot id.

    guidelines is an iterable of (id, version) pairs.
    """
    parts: list[str] = [f"agent:{agent_id}@{agent_version}"]
    for g in sorted(guardrails, key=lambda g: g.name):
        part = f"gr:{g.name}@{g.version}"
        if isinstance(g, HardGuardrail) and g.predicate_args:
            part += "+" + json.dumps(g.predicate_args, sort_keys=True)
        parts.append(part)
    for gl_id, gl_version in sorted(guidelines):
        parts.append(f"gl:{gl_id}@{gl_version}")
    joined = "|".join(parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    return f"{agent_id}@{agent_version}+{digest}"
