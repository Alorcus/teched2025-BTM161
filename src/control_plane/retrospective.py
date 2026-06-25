"""Retrospective collector.

After a conversation ends, runs a three-stage retrospective:

1. Per-agent goal-friction review — each operator agent sees the FULL
   conversation transcript and answers ONE sharp question: where did its own
   goal become hard or impossible to reach, and what was in the way. The
   answer is structured (inferred_goal + evidence quote + obstacle moment +
   obstacle source enum) so it's both auditable and aggregatable.
2. Synthesis pass — one final LLM call ingests all per-agent answers plus
   the transcript and produces a team-level view: assigned-vs-enacted goal
   drift, the obstacle_source distribution, and the single highest-leverage
   process fix.
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


_REQUIRED_KEYS = (
    "inferred_goal",
    "goal_evidence_quote",
    "obstacle_moment_quote",
    "obstacle_source",
    "what_was_in_the_way",
    "next_time_change",
)

_OBSTACLE_SOURCES = {
    "own_action",
    "peer_agent",
    "customer",
    "process",
    "tools_or_info",
    "none",
}

# Token budgets. Per-agent retros answer one structured question and easily
# fit in the default; the synthesis pass writes five evidence-bearing sections
# and was empirically observed to get truncated mid-JSON around ~1k tokens.
_SYNTHESIS_MAX_TOKENS = 4096


_SYNTHESIS_PROMPT = """You are reviewing a multi-agent coffee shop conversation that just ended. You have the full transcript and each operator agent's answer to one sharp question: where did its own goal become hard or impossible to reach, and what was in the way.

Each per-agent entry contains:
- inferred_goal: the goal the agent attributes to itself based on what it actually did (NOT what its assigned role says).
- goal_evidence_quote: one of its own lines that it cites as evidence of that goal.
- obstacle_moment_quote: the single moment where the goal got hard.
- obstacle_source: one of own_action | peer_agent | customer | process | tools_or_info | none.
- obstacle_target: the peer named (only when obstacle_source is peer_agent).
- what_was_in_the_way: a concrete description of the obstacle.
- next_time_change: the verb-led instruction the agent gave its future self.

Your job: produce a TEAM-LEVEL synthesis grounded in this data. Look for:
- ROLE DRIFT: where does an agent's inferred_goal diverge from the goal it should have had given the process? A drift is itself a process signal.
- OBSTACLE PATTERN: what does the obstacle_source distribution say? Are obstacles concentrated on one source (process? a specific peer?) or spread out?
- CONVERGENCE / DIVERGENCE: do multiple agents independently point at the same moment or the same obstacle? Do any agents contradict each other about what happened?
- HIGHEST-LEVERAGE FIX: the single change to the process or to an agent's role that would have prevented the most-cited obstacle.

Conversation transcript:
{transcript}

Per-agent answers (JSON):
{retrospectives}

Return ONLY valid JSON, no prose, no code fences:
{{
  "role_drift":        {{"summary": "where inferred_goal diverged from assigned role, and for whom", "evidence": "concrete agent + quote"}},
  "obstacle_pattern":  {{"summary": "what the obstacle_source distribution shows", "evidence": "counts or specific agents and their sources"}},
  "convergence":       {{"summary": "where multiple agents independently point at the same moment or obstacle", "evidence": "which agents, what they said"}},
  "contradictions":    {{"summary": "where agents disagree about what happened", "evidence": "which agents, what conflicting claims"}},
  "systemic_fix":      {{"summary": "the single highest-leverage process or role change for next time", "evidence": "why this fix, grounded in the data above"}}
}}
"""


class Retrospective:
    """Run a per-agent goal-friction review + synthesis pass after a conversation ends."""

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
        """Run per-agent retrospectives + synthesis, write one JSON file per conversation.

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

        grounding_error = _check_quote_grounding(parsed, transcript)
        if grounding_error is not None:
            return {
                "agent_name": agent_name,
                "valid": False,
                "error": grounding_error,
                "raw_response": raw,
            }

        return {
            "agent_name": agent_name,
            "valid": True,
            **{k: parsed[k] for k in _REQUIRED_KEYS},
            "obstacle_target": parsed.get("obstacle_target"),
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
        # Give synthesis its own token budget — it produces the longest output
        # of any call in the system. .bind() returns a wrapper, leaving the
        # shared self.llm untouched.
        try:
            llm = self.llm.bind(max_tokens=_SYNTHESIS_MAX_TOKENS)
        except Exception:
            llm = self.llm
        response = llm.invoke(messages)
        raw = normalize_content(response.content).strip()
        parsed, recovered = _parse_json_object_lenient(raw)
        if parsed is None:
            return {"valid": False, "raw_response": raw}
        result: dict[str, Any] = {"valid": True, **parsed, "raw_response": raw}
        if recovered:
            result["recovered_from_truncation"] = True
        return result

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
    """Extract the retrospective JSON object from an LLM response, or None.

    Requires the six string fields, validates the obstacle_source enum, and
    enforces the obstacle_target cross-field rule (set iff source is
    'peer_agent', otherwise coerced to None). Returns None on any structural
    failure — the caller records ``valid=false`` and keeps going.
    """
    parsed = _parse_json_object(raw)
    if parsed is None:
        return None
    for k in _REQUIRED_KEYS:
        v = parsed.get(k)
        if not isinstance(v, str) or not v.strip():
            return None
    if parsed["obstacle_source"] not in _OBSTACLE_SOURCES:
        return None
    target = parsed.get("obstacle_target")
    if parsed["obstacle_source"] == "peer_agent":
        if not isinstance(target, str) or not target.strip():
            return None
    else:
        # LLMs commonly fill this harmlessly when the enum doesn't require it.
        # Coerce rather than reject so we don't lose otherwise-good entries.
        parsed["obstacle_target"] = None
    return parsed


def _check_quote_grounding(
    parsed: dict[str, Any], transcript: str
) -> str | None:
    """Verify the agent's quotes actually come from the transcript.

    Returns an error string when a quote is fabricated, or None when grounding
    is fine. Whitespace is normalized before substring check to tolerate
    stylistic variation; smart quotes are folded to straight quotes.
    """
    norm_transcript = _normalize_quote(transcript)
    goal_quote = parsed["goal_evidence_quote"].strip()
    if goal_quote != "NO_OWN_MESSAGES":
        if _normalize_quote(goal_quote) not in norm_transcript:
            return "goal_evidence_quote_not_in_transcript"
    obstacle_quote = parsed["obstacle_moment_quote"].strip()
    if _normalize_quote(obstacle_quote) not in norm_transcript:
        return "obstacle_moment_quote_not_in_transcript"
    return None


def _normalize_quote(s: str) -> str:
    return (
        s.replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .strip()
    )


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


def _parse_json_object_lenient(
    raw: str,
) -> tuple[dict[str, Any] | None, bool]:
    """Strict parse first; if that fails, try to repair a truncated response.

    Returns ``(parsed, recovered)`` where ``recovered`` is True when the strict
    parse failed and the repair succeeded. The repair closes an unterminated
    string (if we ended mid-value), drops a trailing partial key/comma, and
    appends enough ``}`` to balance the braces. Strings, escapes, and unicode
    are tracked correctly so the repair is safe on arbitrary text.

    This is intentionally conservative: it only fires when strict parsing has
    already failed, and it returns None if the repaired text still doesn't
    parse — partial garbage is preferred over confidently-wrong output.
    """
    strict = _parse_json_object(raw)
    if strict is not None:
        return strict, False

    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    start = text.find("{")
    if start == -1:
        return None, False
    body = text[start:]

    repaired = _repair_truncated_json(body)
    if repaired is None:
        return None, False
    try:
        parsed = json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        return None, False
    if not isinstance(parsed, dict):
        return None, False
    return parsed, True


def _repair_truncated_json(text: str) -> str | None:
    """Best-effort close of a JSON object whose tail was cut off.

    Walks the text tracking string state (with backslash escapes) and brace
    depth. If we end inside a string, append ``"``. Then drop any trailing
    comma / dangling key (``,"foo":``-style) and append the right number of
    ``}`` to balance the structure.
    """
    in_string = False
    escape = False
    depth = 0
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                # More closes than opens — give up, the structure is corrupt.
                return None

    if depth == 0 and not in_string:
        # Already balanced — strict parse would have caught it, so this is
        # likely a trailing-comma situation. Strip trailing commas and retry.
        return _strip_trailing_commas(text)

    repaired = text
    if in_string:
        repaired += '"'

    # Drop trailing whitespace / comma / a dangling "key": at end-of-buffer
    # (e.g. '... "evidence":' with no value yet, or '...,' with no next key).
    repaired = repaired.rstrip()
    # Strip a dangling ``"key":`` with optional trailing whitespace.
    repaired = re.sub(r',\s*"[^"\\]*"\s*:\s*$', "", repaired)
    # Strip a trailing standalone comma.
    repaired = repaired.rstrip().rstrip(",").rstrip()

    repaired = _strip_trailing_commas(repaired)
    repaired += "}" * depth
    return repaired


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before ``}`` or ``]`` so json.loads accepts the
    text. Safe to run on already-valid JSON."""
    return re.sub(r",(\s*[}\]])", r"\1", text)
