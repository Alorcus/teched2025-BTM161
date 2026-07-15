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


def discount_within_limit_predicate(max_pct: int, effect: str = "flag"):
    """Factory: verdict when calculate_total is called with discount_percent above max_pct.

    `effect` selects deny vs flag on violation; defaults to flag so existing setups
    (which pass only `max_pct`) keep their observe-only behavior.
    """
    violation_effect = Effect(effect)

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
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"discount_percent={pct} exceeds limit {max_pct}",
            reason_for_llm=(
                f"A discount of {pct}% is not allowed; the maximum permitted discount is {max_pct}%."
            ),
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


def order_size_within_range_predicate(
    max_units: int, min_units: int = 1, effect: str = "deny"
):
    """Factory: constrain order size at process_order time. Size is counted in *units*
    (summed quantities) read straight from the proposed `tool_args["order"]` — the order
    row does not exist yet, so there is no order_id to load. Violation when total units
    fall outside [min_units, max_units]. Missing/malformed `order` → ALLOW (nothing to
    evaluate; the tool itself validates the payload).
    """
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        order = context.tool_args.get("order")
        if not isinstance(order, list):
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal="no order list in tool_args; size not evaluated",
            )
        units = sum(
            int(item.get("quantity", 1) or 1)
            for item in order
            if isinstance(item, dict)
        )
        if min_units <= units <= max_units:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"order size {units} units within [{min_units}, {max_units}]",
            )
        return Verdict(
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"order size {units} units outside [{min_units}, {max_units}]",
            reason_for_llm=(
                f"This order has {units} item(s), which is outside the permitted range of "
                f"{min_units}–{max_units}. Adjust the order to fit within the allowed size."
            ),
        )

    return _eval


def refund_within_limit_predicate(max_pct: int, effect: str = "deny"):
    """Factory: constrain offer_partial_refund's `refund_percent`. Violation when the
    requested percentage exceeds `max_pct`. Mirrors the tool default of 50 when absent.
    """
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        pct = int(context.tool_args.get("refund_percent", 50) or 0)
        if pct <= max_pct:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"refund_percent={pct} within limit {max_pct}",
            )
        return Verdict(
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"refund_percent={pct} exceeds limit {max_pct}",
            reason_for_llm=(
                f"A partial refund of {pct}% is not allowed; the maximum permitted refund is {max_pct}%."
            ),
        )

    return _eval


def order_total_within_limit_predicate(max_total: float, effect: str = "deny"):
    """Factory: constrain the order's total (in dollars). Resolves the order from
    `tool_args["order_id"]` and reads `order.total`; violation when it exceeds `max_total`.
    Unresolvable orders → ALLOW. Enforced at calculate_total, so it reads the pre-discount
    total set by process_order.
    """
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        order_id = str(context.tool_args.get("order_id", ""))
        order = load_order(order_id) if order_id else None
        if order is None:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"order {order_id!r} not resolvable; total not evaluated",
            )
        if order.total <= max_total:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"order {order_id} total ${order.total:.2f} within limit ${max_total:.2f}",
            )
        return Verdict(
            effect=violation_effect,
            guardrail_name="",
            guardrail_type="",
            reason_internal=f"order {order_id} total ${order.total:.2f} exceeds limit ${max_total:.2f}",
            reason_for_llm=(
                f"Order {order_id} totals ${order.total:.2f}, which exceeds the maximum permitted "
                f"order total of ${max_total:.2f}."
            ),
        )

    return _eval


PREDICATE_REGISTRY = {
    "allowed_handover_targets": allowed_handover_targets_predicate,
    "discount_within_limit": discount_within_limit_predicate,
    "transfer_includes_order_id": transfer_includes_order_id_predicate,
    "require_order_status": require_order_status_predicate,
    "order_size_within_range": order_size_within_range_predicate,
    "refund_within_limit": refund_within_limit_predicate,
    "order_total_within_limit": order_total_within_limit_predicate,
}
