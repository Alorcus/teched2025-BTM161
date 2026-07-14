import re

from src.agents.order_store import load_order

from .types import Effect, GuardrailContext, Verdict

_ORDER_ID_RE = re.compile(r"\bORD\d{3,}\b")


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
    """Hard guardrail: FLAG if transfer_to_agent does not include an ORD#### order id."""

    summary = str(context.tool_args.get("context_summary", ""))
    if _ORDER_ID_RE.search(summary):
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


def require_order_status_predicate(allowed: list[str], effect: str = "deny"):
    """Factory: block (or flag) a tool call unless the order's *current* status is in
    `allowed`. This is how the order lifecycle is enforced now that there is no state
    machine — each state-gated tool declares the source statuses it may be called from.

    The order is resolved from the tool's `order_id` argument. The transition target the
    tool computes at runtime is always legal from an allowed source, so a source-status
    precondition is equivalent to validating the transition. Unresolvable orders return
    ALLOW so the tool itself can report 'not found'. `effect` selects deny vs flag.
    """
    allowed_set = {str(s) for s in allowed}
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        order_id = str(context.tool_args.get("order_id", ""))
        order = load_order(order_id) if order_id else None
        if order is None:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"order {order_id!r} not resolvable; precondition not evaluated",
            )
        current = order.status.value
        if current in allowed_set:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"order {order_id} status={current} in allowed={sorted(allowed_set)}",
            )
        return Verdict(
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"order {order_id} status={current} not in allowed={sorted(allowed_set)}",
            reason_for_llm=(
                f"Cannot call {context.tool_name!r} while order {order_id} is '{current}'. "
                f"This tool is only valid when the order status is one of: "
                f"{', '.join(sorted(allowed_set))}."
            ),
        )

    return _eval


PREDICATE_REGISTRY = {
    "allowed_handover_targets": allowed_handover_targets_predicate,
    "discount_within_limit": discount_within_limit_predicate,
    "transfer_includes_order_id": transfer_includes_order_id_predicate,
    "require_order_status": require_order_status_predicate,
}
