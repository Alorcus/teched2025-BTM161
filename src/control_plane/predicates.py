from .types import Effect, GuardrailContext, Verdict


def allowed_handover_targets_predicate(context: GuardrailContext) -> Verdict:
    """Hard guardrail: target_agent of transfer_to_agent must be in allowed_handovers."""
    target = context.tool_args.get("target_agent", "")
    allowed = context.allowed_handovers
    if target in allowed:
        return Verdict(
            allowed=Effect.ALLOW,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"{target!r} is in allowed_handovers={allowed}",
        )
    return Verdict(
        allowed=Effect.DENY,
        guardrail_name="",
        guardrail_type="",
        reason_internal=f"{target!r} not in allowed_handovers={allowed} for agent {context.agent_id!r}",
        reason_for_llm=(
            f"Handover to {target!r} is not allowed from {context.agent_id!r}. "
            f"You may only transfer to: {', '.join(allowed) or '(none)'}."
        ),
    )


def discount_within_limit_predicate(max_pct: int):
    """Factory: FLAG verdict when calculate_total is called with discount_percent above max_pct."""

    def _eval(context: GuardrailContext) -> Verdict:
        pct = int(context.tool_args.get("discount_percent", 0) or 0)
        if pct <= max_pct:
            return Verdict(
                allowed=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"discount_percent={pct} within limit {max_pct}",
            )
        return Verdict(
            allowed=Effect.FLAG,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"discount_percent={pct} exceeds limit {max_pct} (flagged, not blocked)",
        )

    return _eval
