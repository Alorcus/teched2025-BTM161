"""NeMo Guardrails integrated as a control-plane guardrail type.

A ``NeMoGuardrail`` runs a NeMo rails configuration's input **or** output rails
through NeMo's validation-only ``LLMRails.check()`` API (no full generation) and
maps the returned ``RailStatus`` onto our ``Effect``:

    PASSED   -> ALLOW
    MODIFIED -> FLAG    (spike: the change is logged; we do not rewrite tool args)
    BLOCKED  -> DENY    (result.rail names the blocking rail)

Because it subclasses ``Guardrail`` and returns a ``Verdict``, an *input*-stage
NeMo guardrail is evaluated pre-tool-call by ``Gateway.evaluate_call`` exactly
like any hard/soft guardrail — no gateway changes needed. An *output*-stage NeMo
guardrail is evaluated on the agent's final reply by ``Gateway.evaluate_output``.

The LLM backing NeMo's LLM-based rails (e.g. ``self check input``) is the same
LangChain model the agents use, injected via ``bind_llm`` / ``bind_llm_to_nemo``.

``nemoguardrails`` is imported lazily (only when a nemo guardrail is bound or
evaluated) so the rest of the control plane does not hard-depend on it.
"""

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

# NeMo must know to use the LangChain LLM framework before it constructs any LLM.
# Set it at import time; also re-asserted in _get_rails via set_default_framework.
os.environ.setdefault("NEMOGUARDRAILS_LLM_FRAMEWORK", "langchain")

from .guardrails import Guardrail
from .types import Effect, GuardrailContext, Verdict

logger = logging.getLogger("coffee_shop.control_plane.nemo")

# One LLMRails per config dir, shared across agents (the LLM is graph-wide).
_RAILS_CACHE: dict[str, object] = {}
_CACHE_LOCK = threading.Lock()

# Lazily-built ChatOllama subclass (see _nemo_safe_llm).
_SAFE_OLLAMA_CLS = None


def _nemo_safe_llm(llm):
    """Make a LangChain LLM safe to hand to NeMo.

    NeMo binds sampling params (``temperature``, ``max_tokens``) onto the model
    at call time. langchain-ollama's ``ChatOllama`` then leaks those straight to
    the ollama client as top-level kwargs, which it rejects
    ("AsyncClient.chat() got an unexpected keyword argument 'temperature'").
    Return a ``ChatOllama`` subclass that folds those kwargs into ``options``
    (where the ollama client expects them). Non-Ollama models (e.g.
    ChatAnthropic) accept the kwargs natively and pass through unchanged.
    """
    global _SAFE_OLLAMA_CLS
    if type(llm).__name__ != "ChatOllama":
        return llm
    if _SAFE_OLLAMA_CLS is None:
        from langchain_ollama import ChatOllama

        class _NeMoSafeChatOllama(ChatOllama):
            def _chat_params(self, messages, stop=None, **kwargs):
                options = dict(kwargs.pop("options", None) or {})
                if "max_tokens" in kwargs:
                    options.setdefault("num_predict", kwargs.pop("max_tokens"))
                for key in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "seed",
                    "repeat_penalty",
                    "num_predict",
                ):
                    if key in kwargs:
                        options.setdefault(key, kwargs.pop(key))
                params = super()._chat_params(messages, stop=stop, **kwargs)
                merged = dict(params.get("options") or {})
                merged.update(options)
                params["options"] = merged
                return params

        _SAFE_OLLAMA_CLS = _NeMoSafeChatOllama
    # Reconstruct with the same config (model, base_url, temperature, ...).
    fields = {k: getattr(llm, k) for k in type(llm).model_fields if hasattr(llm, k)}
    return _SAFE_OLLAMA_CLS(**fields)


def _latest_human_text(state: dict) -> str:
    """Best-effort extraction of the most recent user message text from state."""
    messages = (state or {}).get("messages", []) or []
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            content = getattr(message, "content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ).strip()
            return str(content)
    return ""


def _run_check(rails, messages, rail_types):
    """Call ``rails.check`` even if an event loop is already running.

    LangGraph runs these subgraph nodes synchronously, so the common path is the
    plain sync ``check()``. If we are ever inside a running loop (async graph,
    notebook), run ``check_async`` to completion on a dedicated thread instead of
    exploding with "event loop is already running".
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return rails.check(messages, rail_types=rail_types)

    box: dict = {}

    def _worker():
        box["result"] = asyncio.run(rails.check_async(messages, rail_types=rail_types))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    return box["result"]


@dataclass
class NeMoGuardrail(Guardrail):
    """Guardrail backed by a NeMo rails config, evaluated via ``LLMRails.check``."""

    config_path: Path | None = None
    rail_types: list[str] = field(default_factory=lambda: ["input"])

    # Plain class attribute (no annotation) so dataclass does NOT treat it as a
    # field/constructor arg. Set per-instance by bind_llm before first eval.
    _llm = None

    @property
    def type(self) -> str:
        return "nemo"

    def bind_llm(self, llm) -> None:
        """Inject the LLM NeMo should use and warm the shared rails instance."""
        self._llm = llm
        self._get_rails()  # surface config/import errors at build time

    def _get_rails(self):
        from nemoguardrails import LLMRails, RailsConfig, set_default_framework

        set_default_framework("langchain")
        key = str(self.config_path)
        with _CACHE_LOCK:
            rails = _RAILS_CACHE.get(key)
            if rails is None:
                config = RailsConfig.from_path(str(self.config_path))
                rails = LLMRails(config, llm=_nemo_safe_llm(self._llm))
                _RAILS_CACHE[key] = rails
                logger.info(f"built NeMo LLMRails from {self.config_path}")
        return rails

    def _messages_for(self, context: GuardrailContext):
        """Build the (messages, rail_types) pair for this guardrail's stage."""
        from nemoguardrails.rails.llm.options import RailType

        user_text = _latest_human_text(context.state)

        if "output" in self.rail_types:
            # Output rails validate an already-generated assistant reply. A user
            # turn is supplied for context; the assistant turn is what's checked.
            messages = [
                {"role": "user", "content": user_text or "(no user message)"},
                {"role": "assistant", "content": context.output_text or ""},
            ]
            return messages, [RailType.OUTPUT]

        # Input rails: feed the triggering user request plus a compact render of
        # the tool call the agent is about to make (so both the user's intent and
        # the tool arguments are moderated / topic-checked).
        tool_render = ""
        if context.tool_name:
            tool_render = f"[tool call] {context.tool_name} args={context.tool_args}"
        content = (
            "\n\n".join(part for part in (user_text, tool_render) if part) or "(empty)"
        )
        messages = [
            {
                "role": "context",
                "content": {
                    "agent_id": context.agent_id,
                    "tool_name": context.tool_name,
                },
            },
            {"role": "user", "content": content},
        ]
        return messages, [RailType.INPUT]

    def eval(self, context: GuardrailContext) -> Verdict:
        # Everything that can touch NeMo is inside the try so a rails/config/LLM
        # error fails open (ALLOW) rather than breaking the agent loop.
        logger.debug(f"NeMo guardrail {self.name!r} evaluating context: {context}")
        try:
            rails = self._get_rails()
            messages, rail_types = self._messages_for(context)
            result = _run_check(rails, messages, rail_types)
            status = result.status.value  # "passed" | "modified" | "blocked"
            rail = result.rail
            content = result.content
        except Exception as exc:
            logger.warning(f"NeMo guardrail {self.name!r} check failed: {exc}")
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name=self.name,
                guardrail_type=self.type,
                reason_internal=f"nemo check errored (fail-open): {exc!r}",
            )

        stage = "output" if "output" in self.rail_types else "input"

        if status == "blocked":
            return Verdict(
                effect=Effect.DENY,
                guardrail_name=self.name,
                guardrail_type=self.type,
                reason_internal=f"NeMo {stage} rail {rail!r} blocked the content",
                reason_for_llm=(
                    content or "This request was blocked by a safety guardrail."
                ),
            )
        if status == "modified":
            return Verdict(
                effect=Effect.FLAG,
                guardrail_name=self.name,
                guardrail_type=self.type,
                reason_internal=f"NeMo {stage} rail modified the content (flagged, not blocked): {content!r}",
            )
        return Verdict(
            effect=Effect.ALLOW,
            guardrail_name=self.name,
            guardrail_type=self.type,
            reason_internal=f"NeMo {stage} rails passed",
        )


def bind_llm_to_nemo(guardrails, llm) -> None:
    """Bind the agents' LLM to every NeMoGuardrail in the list (no-op otherwise)."""
    for guardrail in guardrails:
        if isinstance(guardrail, NeMoGuardrail):
            guardrail.bind_llm(llm)
