"""Retrospective collector.

After a conversation ends, asks each operator agent the four After-Action
Review questions in isolation (one LLM call per agent, all four questions
in one shot). Per-agent results are appended to a single JSON file per
conversation: ``<log_dir>/<thread_id>.json``.

The retrospective never raises into the caller — a failure for any one
agent is recorded as an entry with ``valid=false`` and the run continues.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import normalize_content

logger = logging.getLogger("coffee_shop.control_plane.retrospective")


_AGENT_ROLES: dict[str, str] = {
    "order_agent": "Take and price customer orders, then hand off to the inventory agent.",
    "inventory_agent": "Check stock for the order and either confirm or escalate, then hand off.",
    "barista_agent": "Prepare drinks and place items on the tray. Some preparations fail.",
    "customer_service_agent": "Resolve complaints, issue refunds, and recover from problems.",
    "process_supervisor": "Observe every message, classify it against the process model, flag violations.",
}


_REQUIRED_KEYS = ("q1_supposed", "q2_actual", "q3_why_diff", "q4_next_time")


class Retrospective:
    """Run a per-agent After-Action Review after a conversation ends."""

    def __init__(self, llm, prompt_template: str, log_dir: Path | str):
        self.llm = llm
        self.prompt_template = prompt_template
        self.log_dir = Path(log_dir)

    def run(
        self,
        thread_id: str,
        agents: Iterable[str],
        transcripts: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Ask every agent the AAR questions, write one JSON file per conversation.

        ``transcripts`` maps agent_name → that agent's view of the conversation
        (already filtered by the caller — operator agents see their own swarm
        slice; the process supervisor sees its own critique log tail).

        Agents with empty transcripts are skipped (silent agents add no signal).
        """
        results: list[dict[str, Any]] = []
        for agent_name in agents:
            transcript = transcripts.get(agent_name, "").strip()
            if not transcript:
                logger.debug("retrospective: skipping %s (no transcript)", agent_name)
                continue
            try:
                entry = self._ask_agent(agent_name, transcript)
            except Exception as exc:
                logger.exception("retrospective: %s failed", agent_name)
                entry = {
                    "agent_name": agent_name,
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(entry)

        self._write(thread_id, results)
        return results

    def _ask_agent(self, agent_name: str, transcript: str) -> dict[str, Any]:
        prompt = self.prompt_template.format(
            agent_name=agent_name,
            agent_role=_AGENT_ROLES.get(agent_name, "Participating agent."),
            agent_transcript=transcript,
        )
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Return your retrospective JSON now."),
        ]
        logger.debug("retrospective: asking %s with prompt:\n%s", agent_name, prompt)
        response = self.llm.invoke(messages)
        raw = normalize_content(response.content).strip()
        parsed = _parse_retrospective(raw)

        if parsed is None:
            return {
                "agent_name": agent_name,
                "valid": False,
                "raw_response": raw,
            }

        return {
            "agent_name": agent_name,
            "valid": True,
            **{k: parsed[k] for k in _REQUIRED_KEYS},
            "raw_response": raw,
        }

    def _write(self, thread_id: str, entries: list[dict[str, Any]]) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{thread_id}.json"
        payload = {"thread_id": thread_id, "entries": entries}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("retrospective: wrote %d entries to %s", len(entries), path)


def _parse_retrospective(raw: str) -> dict[str, Any] | None:
    """Extract the four-key AAR JSON object from an LLM response, or None."""
    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    if not all(k in parsed and isinstance(parsed[k], dict) for k in _REQUIRED_KEYS):
        return None
    return parsed
