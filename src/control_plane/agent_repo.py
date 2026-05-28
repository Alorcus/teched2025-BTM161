from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AgentDefinition:
    id: str
    base_prompt: str
    tools: tuple[str, ...]
    guardrail_ids: tuple[str, ...]
    guideline_ids: tuple[str, ...]
    allowed_handovers: tuple[str, ...]
    version: str = "v1"
    model_ref: str | None = None


def load_agent_definition(path: Path) -> AgentDefinition:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AgentDefinition(
        id=data["id"],
        base_prompt=data["base_prompt"],
        tools=tuple(data.get("tools", [])),
        guardrail_ids=tuple(data.get("guardrails", [])),
        guideline_ids=tuple(data.get("guidelines", [])),
        allowed_handovers=tuple(data.get("allowed_handovers", [])),
        version=data.get("version", "v1"),
        model_ref=data.get("model"),
    )


class AgentRepo:
    """Reads AgentDefinitions from `<config_dir>/agents/*.yaml` at construction."""

    def __init__(self, config_dir: Path):
        self._defs: dict[str, AgentDefinition] = {}
        agents_dir = Path(config_dir) / "agents"
        for yaml_path in sorted(agents_dir.glob("*.yaml")):
            d = load_agent_definition(yaml_path)
            self._defs[d.id] = d

    def get(self, agent_id: str) -> AgentDefinition:
        if agent_id not in self._defs:
            raise KeyError(f"Unknown agent_id {agent_id!r}; known: {list(self._defs)}")
        return self._defs[agent_id]

    def ids(self) -> list[str]:
        return list(self._defs)

    def all(self) -> dict[str, AgentDefinition]:
        return dict(self._defs)
