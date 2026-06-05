"""Process Supervisor Agent.

Observes every new LangGraph message during a conversation and appends one
metadata line per loggable message to a growing log file. Tool-result
messages are dropped (the activity was already recorded on the AIMessage
that issued the tool_call). The line shape is:

    Execution:<ActivityID>:<ActivityName> | <serialized input message>
    Termination:<ActivityID>:<ActivityName>:<reason> | <serialized input message>

or, when the message violates the process model:

    Violation:<reason> | <serialized input message>

`reason` after a Termination is `terminal` for a natural end (the BPMN End
event, e.g. A07 Handout Order) or `via_handoff_to_<target>` when an agent
delegates to another agent — the BPMN sequence-flow edge between lanes
terminates the source agent's currently-running activity. The decision is
written FIRST so the log reads as "what the supervisor decided | the
evidence it saw".

Decision engine: one small LLM call per message, given the process
description (loaded from docs/order-process-flow.md when available), the
recent log tail, and the new message. Output must match a strict regex;
unparseable output is recorded as a Violation.

Activities mirror docs/order-process-flow.md (BPMN model):
  A01 Identify Customer Request → A02 Create Order → A03 Check Stock →
  AND-split { A04 Place Food on Tray, A05 Brew Coffee, A06 Purchase Order } →
  AND-join → A07 Handout Order (terminal).
"""
from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

logger = logging.getLogger("coffee_shop.control_plane.process_supervisor")

_EXECUTION_RE = re.compile(
    r"^\s*Execution\s*:\s*(?P<id>A\d+[a-z]?)\s*:\s*(?P<name>[^|\r\n]+?)\s*$"
)
_TERMINATION_RE = re.compile(
    r"^\s*Termination\s*:\s*(?P<id>A\d+[a-z]?)\s*:\s*(?P<name>[^|\r\n:]+?)\s*:\s*(?P<reason>[a-zA-Z0-9_\-]+)\s*$"
)
_VIOLATION_RE = re.compile(r"^\s*Violation\s*:\s*(?P<reason>.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Activity:
    id: str
    name: str
    agent: str
    trigger: str  # "message" | "tool_call" | "handoff"
    tool: str | None
    target: str | None
    terminal: bool
    display_name: str | None = None


DEFAULT_PROMPT_TEMPLATE = (
    "You are the process supervisor for a multi-agent coffee shop.\n\n"
    "Process description:\n{process_description}\n\n"
    "Allowed activities:\n{activity_catalog}\n\n"
    "Prior log tail:\n{prior_log_tail}\n\n"
    "New message: {message_brief}\n\n"
    "Reply with exactly ONE line in one of these formats. Use the\n"
    "activity's `slug` (snake_case) as <ActivityName>, not the display name:\n"
    "  Execution:<ActivityID>:<ActivityName>\n"
    "  Termination:<ActivityID>:<ActivityName>:terminal\n"
    "  Violation:<short_reason_without_spaces_or_with_underscores>\n"
    "No prose, no quotes."
)


def load_process_model(
    path: str | os.PathLike,
) -> tuple[str, list[Activity], str]:
    """Read the YAML process model. Returns (description, activities, prompt_template).

    If the YAML sets `description_source: <relative path>`, the description is
    loaded from that file (resolved relative to the YAML); the markdown is the
    source of truth for the BPMN narrative. Otherwise the inline `description:`
    field is used.

    `prompt_template:` is an optional top-level YAML field. When absent, the
    built-in DEFAULT_PROMPT_TEMPLATE is used. Supported placeholders:
    {process_description}, {activity_catalog}, {prior_log_tail}, {message_brief}.
    """
    yaml_path = Path(path)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    description = ""
    src = data.get("description_source")
    if src:
        candidate = (yaml_path.parent / src).resolve()
        if not candidate.exists():
            candidate = Path(src).resolve()
        if candidate.exists():
            description = candidate.read_text(encoding="utf-8").strip()
        else:
            logger.warning("description_source %s not found; using inline description", src)
    if not description:
        description = (data.get("description") or "").strip()

    activities = [
        Activity(
            id=a["id"],
            name=a["name"],
            agent=a["agent"],
            trigger=a["trigger"],
            tool=a.get("tool"),
            target=a.get("target"),
            terminal=bool(a.get("terminal", False)),
            display_name=a.get("display_name"),
        )
        for a in data.get("activities", [])
    ]
    prompt_template = (data.get("prompt_template") or DEFAULT_PROMPT_TEMPLATE).rstrip() + "\n"
    return description, activities, prompt_template


def _serialize_input_message(msg: BaseMessage, agent_name: str) -> str:
    """One-line serialization of the inbound message."""
    msg_type = type(msg).__name__
    if isinstance(msg, AIMessage):
        if getattr(msg, "tool_calls", None):
            calls = ";".join(
                f"{tc.get('name', '?')}({_short_args(tc.get('args'))})"
                for tc in msg.tool_calls
            )
            payload = f"tool_calls=[{calls}]"
        else:
            payload = f"text={_one_line(str(msg.content or ''))}"
    elif isinstance(msg, ToolMessage):
        payload = (
            f"name={getattr(msg, 'name', '?')} "
            f"result={_one_line(str(msg.content or ''))[:200]}"
        )
    elif isinstance(msg, HumanMessage):
        payload = f"text={_one_line(str(msg.content or ''))}"
    else:
        payload = f"raw={_one_line(str(getattr(msg, 'content', msg)))}"
    return f"{msg_type}[{agent_name}] {payload}"


def _one_line(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def _short_args(args: Any) -> str:
    if not args:
        return ""
    try:
        s = str(dict(args))
    except (TypeError, ValueError):
        s = str(args)
    s = _one_line(s)
    return s[:120] + ("..." if len(s) > 120 else "")


def _classify_message(msg: BaseMessage) -> tuple[str, str | None, str | None]:
    """Return (trigger, tool_name, handoff_target).

    trigger ∈ {"message", "tool_call", "handoff", "tool_result", "user"}.
    """
    if isinstance(msg, AIMessage):
        tcs = getattr(msg, "tool_calls", None) or []
        if tcs:
            tc = tcs[0]
            name = tc.get("name") or ""
            if name.startswith("transfer_to"):
                target = (tc.get("args") or {}).get("target_agent")
                return "handoff", name, target
            return "tool_call", name, None
        return "message", None, None
    if isinstance(msg, ToolMessage):
        return "tool_result", getattr(msg, "name", None), None
    if isinstance(msg, HumanMessage):
        return "user", None, None
    return "message", None, None


class ProcessSupervisor:
    """Append-only, thread-safe process-conformance observer."""

    def __init__(
        self,
        process_model_path: str | os.PathLike,
        log_path: str | os.PathLike,
        llm: Any,
        recent_tail: int = 20,
        prompt_template_override: str | None = None,
    ):
        if llm is None:
            raise ValueError("ProcessSupervisor requires an LLM instance")
        self.description, self.activities, file_template = load_process_model(process_model_path)
        self.prompt_template = prompt_template_override if prompt_template_override is not None else file_template
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.recent_tail = recent_tail
        self._lock = threading.Lock()
        self._lines: list[str] = []
        self._activities_by_id = {a.id: a for a in self.activities}

    def observe(self, msg: BaseMessage, agent_name: str) -> str | None:
        """Decide metadata for one message and append it. Returns the written
        line, or None if the message was intentionally not logged (e.g. tool
        results — they're function returns, not agent decisions)."""
        line = self._build_line(msg, agent_name)
        if line is None:
            return None
        self._append(line)
        return line

    def _build_line(self, msg: BaseMessage, agent_name: str) -> str | None:
        serialized = _serialize_input_message(msg, agent_name)
        trigger, tool, target = _classify_message(msg)

        # Tool results are function returns, not agent decisions — drop them
        # from the log entirely. The decision was already recorded on the
        # AIMessage that issued the tool_call.
        if trigger == "tool_result":
            return None

        # User messages are process input, not agent activity, but we keep
        # them as bookkeeping so the file shows the conversation entry point.
        if trigger == "user":
            return f"NonAction:{trigger} | {serialized}"

        # Handoffs (transfer_to_*) are BPMN sequence-flow edges between lanes.
        # The semantics are: the source agent's currently-running activity
        # terminates and control passes to the target agent. Emit a Termination
        # line against that prior activity.
        if trigger == "handoff":
            return f"{self._terminate_for_handoff(agent_name, target)} | {serialized}"

        decision = self._llm_decide(msg, agent_name, trigger, tool, target)
        return f"{decision} | {serialized}"

    def _terminate_for_handoff(self, source_agent: str, target_agent: str | None) -> str:
        """Find the most recent non-terminated activity by source_agent and
        emit a Termination line for it. If none is found, that's a violation."""
        prior = self._last_open_activity_for(source_agent)
        target = target_agent or "?"
        if prior is None:
            return f"Violation:handoff_without_prior_activity_{source_agent}_to_{target}"
        return f"Termination:{prior[0]}:{prior[1]}:via_handoff_to_{target}"

    def _last_open_activity_for(self, agent: str) -> tuple[str, str] | None:
        """Walk the in-memory log backwards. Return the (id, name) of the most
        recent Execution: line for `agent` that has NOT yet been followed by a
        matching Termination: line. None if there is no open activity."""
        terminated_ids: set[str] = set()
        for line in reversed(self._lines):
            head = line.split(" | ", 1)[0]
            if (m := _TERMINATION_RE.match(head)):
                terminated_ids.add(m.group("id"))
                continue
            if (m := _EXECUTION_RE.match(head)):
                act_id = m.group("id")
                if act_id in terminated_ids:
                    continue
                known = self._activities_by_id.get(act_id)
                if known and known.agent == agent:
                    return act_id, m.group("name")
        return None

    def _llm_decide(
        self,
        msg: BaseMessage,
        agent_name: str,
        trigger: str,
        tool: str | None,
        target: str | None,
    ) -> str:
        catalog = "\n".join(
            f"  {a.id} ({a.display_name or a.name}, slug={a.name})"
            f" — agent={a.agent} trigger={a.trigger}"
            f"{' tool=' + a.tool if a.tool else ''}"
            f"{' target=' + a.target if a.target else ''}"
            f"{' [terminal]' if a.terminal else ''}"
            for a in self.activities
        )
        prior_tail = "\n".join(self._lines[-self.recent_tail:]) or "(empty)"
        msg_brief = (
            f"agent={agent_name} trigger={trigger} "
            f"tool={tool or '-'} target={target or '-'} "
            f"content={_one_line(str(getattr(msg, 'content', '')))[:200]}"
        )
        prompt = self.prompt_template.format(
            process_description=self.description,
            activity_catalog=catalog,
            prior_log_tail=prior_tail,
            message_brief=msg_brief,
        )
        result = self.llm.invoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        if isinstance(text, list):
            text = next(
                (c.get("text", "") for c in text if isinstance(c, dict) and c.get("type") == "text"),
                "",
            )
        text = str(text).strip().splitlines()[0] if text else ""
        validated = self._validate_llm_line(text)
        if validated is None:
            return f"Violation:llm_unparseable_output"
        return validated

    def _validate_llm_line(self, text: str) -> str | None:
        if (m := _EXECUTION_RE.match(text)):
            act_id, name = m.group("id"), m.group("name")
            known = self._activities_by_id.get(act_id)
            if known and name in (known.name, known.display_name):
                if known.terminal:
                    return f"Termination:{act_id}:{known.name}:terminal"
                return f"Execution:{act_id}:{known.name}"
            return f"Violation:llm_unknown_activity_{act_id}"
        if (m := _TERMINATION_RE.match(text)):
            act_id, name, reason = m.group("id"), m.group("name"), m.group("reason")
            known = self._activities_by_id.get(act_id)
            if known and name in (known.name, known.display_name):
                if reason == "terminal" and not known.terminal:
                    return f"Execution:{act_id}:{known.name}"
                return f"Termination:{act_id}:{known.name}:{reason}"
            return f"Violation:llm_unknown_activity_{act_id}"
        if (v := _VIOLATION_RE.match(text)):
            reason = re.sub(r"\s+", "_", v.group("reason").strip())
            return f"Violation:{reason}"
        return None

    def _append(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def lines(self) -> list[str]:
        with self._lock:
            return list(self._lines)
