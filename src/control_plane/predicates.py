from .types import Effect, GuardrailContext, Verdict


def allowed_handover_targets_predicate(context: GuardrailContext) -> Verdict:
    """Hard guardrail: target_agent of transfer_to_agent must be in allowed_handovers."""
    target = context.tool_args.get("target_agent", "")
    allowed = context.allowed_handovers
    if target in allowed:
        return Verdict(
            effect=Effect.ALLOW,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"{target!r} is in allowed_handovers={allowed}",
        )
    return Verdict(
        effect=Effect.DENY,
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
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"discount_percent={pct} within limit {max_pct}",
            )
        return Verdict(
            effect=Effect.FLAG,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"discount_percent={pct} exceeds limit {max_pct} (flagged, not blocked)",
        )

    return _eval


def transfer_includes_order_id_predicate(context: GuardrailContext) -> Verdict:
    """Hard guardrail: FLAG if transfer_to_agent does not include order_id in tool_args."""

    order_id_keys = ["order_id", "order_number", "order_num", "ORD"]

    if any(key in str(context.tool_args) for key in order_id_keys):
        return Verdict(
            effect=Effect.ALLOW,
            guardrail_name="",
            guardrail_type="",
            reason_internal="transfer includes order_id",
        )
    return Verdict(
        effect=Effect.FLAG,
        guardrail_name="",
        guardrail_type="",
        reason_internal="transfer missing order_id (flagged, not blocked)",
        reason_for_llm=(
            "When transferring to another agent, always include the order id so the next agent can look up order details."
            " Include the order id when handing over in this pattern: ORDXXXX"
        ),
    )


PREDICATE_REGISTRY = {
    "allowed_handover_targets": allowed_handover_targets_predicate,
    "discount_within_limit": discount_within_limit_predicate,
    "transfer_includes_order_id": transfer_includes_order_id_predicate,
}
