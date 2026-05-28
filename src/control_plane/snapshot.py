import hashlib
from typing import Iterable

from .guardrails import Guardrail


def snapshot_id(
    agent_id: str,
    agent_version: str,
    guardrails: Iterable[Guardrail],
    guidelines: Iterable[tuple[str, str]],
) -> str:
    """Stable id for an (agent, guardrails, guidelines) build.

    guidelines is an iterable of (id, version) pairs.
    """
    parts: list[str] = [f"agent:{agent_id}@{agent_version}"]
    for g in sorted(guardrails, key=lambda g: g.name):
        parts.append(f"gr:{g.name}@{g.version}")
    for gl_id, gl_version in sorted(guidelines):
        parts.append(f"gl:{gl_id}@{gl_version}")
    joined = "|".join(parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    return f"{agent_id}@{agent_version}+{digest}"
