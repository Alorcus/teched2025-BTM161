import logging
from dataclasses import dataclass
from typing import Any

from .guardrails import Guardrail
from .log_sink import JsonlLogSink, NullLogSink
from .types import Effect, GuardrailContext, Verdict

logger = logging.getLogger("coffee_shop.control_plane.gateway")


@dataclass
class CallDecision:
    """Per-tool-call evaluation result."""

    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any]
    verdicts: list[Verdict]
    final_decision: Effect  # ALLOW or DENY (FLAG collapses to ALLOW for routing)
    deny_reason_for_llm: str = ""


class Gateway:
    """Per-agent gateway. Evaluates proposed tool calls against guardrails,
    logs every decision, and returns CallDecision objects so the subgraph
    can route allowed/denied calls.
    """

    def __init__(
        self,
        agent_id: str,
        guardrails: list[Guardrail],
        allowed_handovers: list[str],
        snapshot_id: str,
        log_sink: JsonlLogSink | NullLogSink,
    ):
        self.agent_id = agent_id
        self.guardrails = guardrails
        self.allowed_handovers = list(allowed_handovers)
        self.snapshot_id = snapshot_id
        self.log_sink = log_sink

    def evaluate_call(
        self,
        tool_call: dict[str, Any],
        state: dict[str, Any],
        thread_id: str | None = None,
    ) -> CallDecision:
        tool_name = tool_call.get("name", "")
        tool_args = dict(tool_call.get("args", {}))
        tool_call_id = tool_call.get("id", "")

        context = GuardrailContext(
            agent_id=self.agent_id,
            tool_name=tool_name,
            tool_args=tool_args,
            state=state,
            allowed_handovers=list(self.allowed_handovers),
        )

        applicable = [
            guardrail
            for guardrail in self.guardrails
            if guardrail.stage == "pre_call" and guardrail.applies_to(tool_name)
        ]
        # Cheap deterministic rules first, then LLM-backed checks (nemo/soft).
        applicable.sort(key=lambda g: 0 if g.type == "hard" else 1)

        verdicts: list[Verdict] = []
        final = Effect.ALLOW
        deny_reason = ""
        for guardrail in applicable:
            verdict = guardrail.eval(context)
            verdicts.append(verdict)
            if verdict.effect == Effect.DENY and final != Effect.DENY:
                final = Effect.DENY
                deny_reason = verdict.reason_for_llm or verdict.reason_internal
                break
            if verdict.effect == Effect.FLAG and final == Effect.ALLOW:
                final = Effect.FLAG  # observability only; doesn't block

        decision = CallDecision(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_args=tool_args,
            verdicts=verdicts,
            final_decision=final,
            deny_reason_for_llm=deny_reason,
        )
        self._log_decision(decision, thread_id)
        return decision

    def evaluate_output(
        self,
        text: str,
        state: dict[str, Any],
        thread_id: str | None = None,
    ) -> CallDecision:
        """Evaluate an agent's final user-facing reply against on_output-stage
        guardrails (e.g. NeMo output rails). Returns a CallDecision whose
        deny_reason_for_llm holds the replacement/refusal text when denied.
        """
        applicable = [g for g in self.guardrails if g.stage == "on_output"]
        if not applicable:
            # Nothing to check — avoid logging noise on every reply.
            return CallDecision(
                tool_call_id="",
                tool_name="<agent_output>",
                tool_args={},
                verdicts=[],
                final_decision=Effect.ALLOW,
            )

        context = GuardrailContext(
            agent_id=self.agent_id,
            tool_name="",
            tool_args={},
            state=state,
            allowed_handovers=list(self.allowed_handovers),
            output_text=text,
        )

        verdicts: list[Verdict] = []
        final = Effect.ALLOW
        deny_reason = ""
        for guardrail in applicable:
            verdict = guardrail.eval(context)
            verdicts.append(verdict)
            if verdict.effect == Effect.DENY and final != Effect.DENY:
                final = Effect.DENY
                deny_reason = verdict.reason_for_llm or verdict.reason_internal
                break
            if verdict.effect == Effect.FLAG and final == Effect.ALLOW:
                final = Effect.FLAG

        decision = CallDecision(
            tool_call_id="",
            tool_name="<agent_output>",
            tool_args={"output_preview": text[:200]},
            verdicts=verdicts,
            final_decision=final,
            deny_reason_for_llm=deny_reason,
        )
        self._log_decision(decision, thread_id, event_type="gateway_output_decision")
        return decision

    def _log_decision(
        self,
        decision: CallDecision,
        thread_id: str | None,
        event_type: str = "gateway_decision",
    ) -> None:
        self.log_sink.append(
            {
                "event_type": event_type,
                "snapshot_id": self.snapshot_id,
                "agent_id": self.agent_id,
                "thread_id": thread_id,
                "tool_name": decision.tool_name,
                "tool_call_id": decision.tool_call_id,
                "tool_args": decision.tool_args,
                "final_decision": decision.final_decision.value,
                "verdicts": [
                    {
                        "guardrail_name": v.guardrail_name,
                        "guardrail_type": v.guardrail_type,
                        "effect": v.effect.value,
                        "reason_internal": v.reason_internal,
                        "reason_for_llm": v.reason_for_llm,
                    }
                    for v in decision.verdicts
                ],
            }
        )

    def log_tool_execution(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        result_preview: str,
        thread_id: str | None = None,
    ) -> None:
        self.log_sink.append(
            {
                "event_type": "tool_execution",
                "snapshot_id": self.snapshot_id,
                "agent_id": self.agent_id,
                "thread_id": thread_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result_preview": result_preview[:500],
            }
        )
