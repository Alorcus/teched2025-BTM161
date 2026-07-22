import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from .types import Effect, GuardrailContext, Verdict

logger = logging.getLogger("coffee_shop.control_plane.guardrails")

MAX_JUDGE_MESSAGE_LENGTH = 8000
MAX_JUDGE_REASON_LENGTH = 400


@dataclass
class Guardrail(ABC):
    """Base guardrail. Subclassed by evaluation mechanism."""

    name: str
    version: str = "unversioned"
    tools: list[str] = field(default_factory=list)
    effect: Effect = Effect.DENY
    description: str = ""

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


JudgeInvoker = Callable[[str, str], str]


@dataclass
class SoftGuardrail(Guardrail):
    """LLM-as-judge evaluation.

    The guardrail sends the assistant's proposed message (or tool-call args) to
    an LLM together with the configured `judge_prompt` and asks for a strict
    JSON verdict of the form `{"decision": "allow"|"deny", "reason": "..."}`.

    The judge is invoked through `judge_invoker` (system_prompt, user_prompt) →
    raw string. `default` is used when the injected invoker is None: it lazily
    builds one from `src.llm.create_chat_llm`. Tests inject a stub to keep the
    evaluation deterministic and offline.

    Template variables inside `judge_prompt` (and `user_template`) are filled
    from `_template_vars(context)` before the call.
    """

    judge_prompt: str = ""
    user_template: str = "{message}"
    state_dependencies: list[str] = field(default_factory=list)
    judge_invoker: JudgeInvoker | None = None

    def eval(self, context: GuardrailContext) -> Verdict:
        message = self._extract_message(context)
        if not message.strip():
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name=self.name,
                guardrail_type=self.type,
                reason_internal="empty message — nothing to evaluate",
                reason_for_llm="",
            )

        if len(message) > MAX_JUDGE_MESSAGE_LENGTH:
            message = message[:MAX_JUDGE_MESSAGE_LENGTH]

        variables = self._template_vars(context, message)
        system_prompt = _safe_format(self.judge_prompt, variables)
        user_prompt = _safe_format(self.user_template, variables)

        invoker = self.judge_invoker or _default_invoker
        try:
            raw = invoker(system_prompt, user_prompt)
        except Exception as exc:
            logger.warning(
                "soft guardrail %r judge invocation failed: %s — allowing by default",
                self.name, exc,
            )
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name=self.name,
                guardrail_type=self.type,
                reason_internal=f"judge error: {exc}",
                reason_for_llm="",
            )

        raw_truncated = raw[:MAX_JUDGE_MESSAGE_LENGTH]
        decision, reason = _parse_judge_response(raw_truncated)
        if decision == "deny":
            bounded_reason = (reason or "")[:MAX_JUDGE_REASON_LENGTH]
            reason_for_llm = (
                bounded_reason
                or "Your message referenced items not available on our menu. "
                "Only recommend items from the menu."
            )
            return Verdict(
                effect=self.effect,
                guardrail_name=self.name,
                guardrail_type=self.type,
                reason_internal=raw_truncated.strip(),
                reason_for_llm=reason_for_llm,
            )
        return Verdict(
            effect=Effect.ALLOW,
            guardrail_name=self.name,
            guardrail_type=self.type,
            reason_internal=raw_truncated.strip(),
            reason_for_llm="",
        )

    @property
    def type(self) -> str:
        return "soft"

    def _extract_message(self, context: GuardrailContext) -> str:
        args = context.tool_args or {}
        message_keys = ("content", "message", "text")
        for key in message_keys:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value
        if any(key in args for key in message_keys):
            return ""
        if isinstance(args, dict) and args:
            return json.dumps(args, ensure_ascii=False, sort_keys=True)
        return ""

    def _template_vars(self, context: GuardrailContext, message: str) -> dict[str, Any]:
        from src.agents.shared_components import ALLOWED_EXTRAS, MENU

        menu_lines = [
            f"- {item.name} (${item.price:.2f}) — {item.category}"
            for item in MENU.values()
        ]
        return {
            "message": message,
            "agent_id": context.agent_id,
            "tool_name": context.tool_name,
            "menu": "\n".join(menu_lines),
            "menu_items": ", ".join(sorted(MENU.keys())),
            "allowed_extras": ", ".join(sorted(ALLOWED_EXTRAS)),
        }


def _safe_format(template: str, variables: dict[str, Any]) -> str:
    """Format `template` with `variables`, leaving unknown placeholders intact.

    A judge prompt that references `{unknown}` should not raise KeyError — it
    is authored in YAML and the writer might make a typo. Missing placeholders
    are preserved verbatim so the omission is visible in the LLM's input.
    """
    if not template:
        return ""

    class _Missing(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    try:
        return template.format_map(_Missing(variables))
    except (IndexError, ValueError):
        return template


_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_response(raw: str) -> tuple[str, str]:
    """Extract (decision, reason) from a judge's raw text response.

    Accepts a JSON object anywhere in the text (LLMs often wrap output in prose
    or code fences). Unknown / ambiguous responses return ('allow', '') so a
    broken judge never silently blocks conversations — matching the fail-open
    policy in `SoftGuardrail.eval`. No keyword sniffing: any recovery from
    non-JSON output would be a footgun (any judge output containing the word
    "deny" — even in a benign context — would otherwise trip a rejection with
    the full raw output as the reason, feeding an unbounded prompt-injection
    payload back into the agent).
    """
    if not raw:
        return "allow", ""

    match = _JSON_OBJECT_PATTERN.search(raw)
    if not match:
        return "allow", ""
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return "allow", ""
    if not isinstance(parsed, dict):
        return "allow", ""
    decision = str(parsed.get("decision", "")).lower().strip()
    reason = str(parsed.get("reason", "")).strip()
    if decision in ("allow", "deny"):
        return decision, reason
    return "allow", ""


_DEFAULT_LLM = None
_DEFAULT_LLM_LOCK = threading.Lock()


def _default_invoker(system_prompt: str, user_prompt: str) -> str:
    """Lazily construct a chat LLM from the project factory and invoke it.

    Cached at module scope so we don't rebuild the client per call. Double-
    checked locking so two threads racing on first use don't both call
    `create_chat_llm()`; the LangChain chat clients themselves are documented
    as safe for concurrent `.invoke()`, so no per-call lock is needed.
    """
    global _DEFAULT_LLM
    if _DEFAULT_LLM is None:
        with _DEFAULT_LLM_LOCK:
            if _DEFAULT_LLM is None:
                from src.llm import create_chat_llm

                _DEFAULT_LLM = create_chat_llm()

    from langchain_core.messages import HumanMessage, SystemMessage
    from src.llm import normalize_content

    response = _DEFAULT_LLM.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return normalize_content(getattr(response, "content", ""))
