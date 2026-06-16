"""Retrospective collector.

After a conversation ends, runs a three-stage retrospective:

1. Per-agent peer-aware After-Action Review — each operator agent sees the
   FULL conversation transcript and answers the four AAR questions about its
   own actions, plus a peer_review section critiquing the other agents it
   interacted with.
2. Synthesis pass — one final LLM call ingests all per-agent retrospectives
   plus the conversation transcript and produces a team-level summary of
   systemic issues, not bound to any one agent's perspective.
3. Persist — one JSON file per conversation at
   ``<log_dir>/<UTC-timestamp>.json``. The thread_id is preserved inside the
   payload for cross-referencing with traces.

The retrospective never raises into the caller — a failure for any one agent
is recorded as an entry with ``valid=false`` and the run continues.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
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


_SYNTHESIS_PROMPT = """You are reviewing a multi-agent coffee shop conversation that just ended. You have the full transcript and each operator agent's self-review plus their peer reviews of each other.

Your job: produce a TEAM-LEVEL synthesis. Look across all retrospectives for systemic issues (not individual mistakes). Where do the agents agree? Where do they contradict each other? What broke down between agents that no single agent can see alone?

Conversation transcript:
{transcript}

Per-agent retrospectives (JSON):
{retrospectives}

Return ONLY valid JSON, no prose, no code fences:
{{
  "what_worked":      {{"summary": "...", "evidence": "concrete moment from transcript or retrospectives"}},
  "what_broke":       {{"summary": "...", "evidence": "concrete moment"}},
  "agreements":       {{"summary": "where multiple agents independently identified the same issue", "evidence": "which agents, what they said"}},
  "contradictions":   {{"summary": "where agents disagree about what happened", "evidence": "which agents, what conflicting claims"}},
  "systemic_fix":     {{"summary": "the single highest-leverage change for next time", "evidence": "why this fix, grounded in the data above"}}
}}
"""


class Retrospective:
    """Run a peer-aware AAR + synthesis pass after a conversation ends."""

    def __init__(self, llm, prompt_template: str, log_dir: Path | str):
        self.llm = llm
        self.prompt_template = prompt_template
        self.log_dir = Path(log_dir)

    def run(
        self,
        thread_id: str,
        agents: Iterable[str],
        transcripts: dict[str, str],
    ) -> dict[str, Any]:
        """Run per-agent AARs + synthesis, write one JSON file per conversation.

        ``transcripts`` maps agent_name → its transcript view. Operator agents
        receive the full conversation (peer-aware); the process supervisor
        receives its own critique log tail.

        Returns the full payload (also persisted to disk).
        """
        agent_list = list(agents)
        operator_agents = [a for a in agent_list if a != "process_supervisor"]

        entries: list[dict[str, Any]] = []
        for agent_name in agent_list:
            transcript = transcripts.get(agent_name, "").strip()
            if not transcript:
                logger.debug("retrospective: skipping %s (no transcript)", agent_name)
                continue
            peers = [a for a in operator_agents if a != agent_name]
            try:
                entry = self._ask_agent(agent_name, transcript, peers)
            except Exception as exc:
                logger.exception("retrospective: %s failed", agent_name)
                entry = {
                    "agent_name": agent_name,
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            entries.append(entry)

        # Use the first non-empty operator transcript as the canonical
        # conversation view for the synthesis pass — they're all the same
        # full transcript in peer-aware mode.
        canonical_transcript = next(
            (transcripts[a] for a in operator_agents if transcripts.get(a, "").strip()),
            "",
        )
        synthesis: dict[str, Any] | None = None
        if entries and canonical_transcript:
            try:
                synthesis = self._synthesize(canonical_transcript, entries)
            except Exception as exc:
                logger.exception("retrospective synthesis failed")
                synthesis = {"valid": False, "error": f"{type(exc).__name__}: {exc}"}

        payload = self._write(thread_id, entries, synthesis)
        return payload

    def _ask_agent(
        self, agent_name: str, transcript: str, peer_agents: list[str]
    ) -> dict[str, Any]:
        peer_str = ", ".join(peer_agents) if peer_agents else "(none)"
        prompt = self.prompt_template.format(
            agent_name=agent_name,
            agent_role=_AGENT_ROLES.get(agent_name, "Participating agent."),
            peer_agents=peer_str,
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
            "peer_review": parsed.get("peer_review", []),
            "raw_response": raw,
        }

    def _synthesize(
        self, transcript: str, entries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        # Strip raw_response from the input — it's noisy and the structured
        # fields carry the same content.
        slim_entries = [
            {k: v for k, v in e.items() if k != "raw_response"} for e in entries
        ]
        prompt = _SYNTHESIS_PROMPT.format(
            transcript=transcript,
            retrospectives=json.dumps(slim_entries, indent=2),
        )
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Return the synthesis JSON now."),
        ]
        response = self.llm.invoke(messages)
        raw = normalize_content(response.content).strip()
        parsed = _parse_json_object(raw)
        if parsed is None:
            return {"valid": False, "raw_response": raw}
        return {"valid": True, **parsed, "raw_response": raw}

    def _write(
        self,
        thread_id: str,
        entries: list[dict[str, Any]],
        synthesis: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # UTC, ISO-8601-ish, filesystem-safe.
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.log_dir / f"{timestamp}.json"
        # If a file already exists for this timestamp (rare, but possible if
        # two conversations end in the same second), append a short suffix.
        if path.exists():
            suffix = thread_id[:8]
            path = self.log_dir / f"{timestamp}_{suffix}.json"
        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "thread_id": thread_id,
            "entries": entries,
            "synthesis": synthesis,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(
            "retrospective: wrote %d entries + synthesis=%s to %s",
            len(entries),
            "yes" if synthesis else "no",
            path,
        )
        return payload


def _parse_retrospective(raw: str) -> dict[str, Any] | None:
    """Extract the AAR JSON object from an LLM response, or None.

    Requires the four AAR keys (each a dict). ``peer_review`` is optional —
    parsed as a list when present, defaulted to [] when absent.
    """
    parsed = _parse_json_object(raw)
    if parsed is None:
        return None
    if not all(k in parsed and isinstance(parsed[k], dict) for k in _REQUIRED_KEYS):
        return None
    pr = parsed.get("peer_review", [])
    if not isinstance(pr, list):
        pr = []
    parsed["peer_review"] = pr
    return parsed


def _parse_json_object(raw: str) -> dict[str, Any] | None:
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
    return parsed
