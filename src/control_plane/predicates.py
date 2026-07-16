import re

from src.agents.order_store import load_order
from src.agents.shared_components import ALLOWED_EXTRAS, MENU

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
    precondition is equivalent to validating the transition. A missing or unresolvable
    `order_id` yields the same violation effect as an out-of-set status: hallucinated
    IDs must not silently bypass the gate. `effect` selects deny vs flag.
    """
    allowed_set = {str(s) for s in allowed}
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        order_id = str(context.tool_args.get("order_id", ""))
        order = load_order(order_id) if order_id else None
        if order is None:
            return Verdict(
                effect=violation_effect,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"order {order_id!r} not resolvable; treated as violation",
                reason_for_llm=(
                    f"Cannot call {context.tool_name!r} for order {order_id!r}: "
                    f"the order does not exist. Verify the order id before retrying."
                ),
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


_ITEM_TERMINALS: frozenset[str] = frozenset(MENU.keys()) | {
    "coffee", "mocha", "macchiato", "chai", "frappe", "frappé",
    "brew", "blend", "tea", "cortado", "matcha", "affogato",
}
_MULTIWORD_TERMINALS: tuple[tuple[str, ...], ...] = (
    ("flat", "white"),
    ("pumpkin", "spice"),
    ("hot", "chocolate"),
    ("cold", "brew"),
    ("london", "fog"),
    ("house", "blend"),
    ("iced", "tea"),
)
_REJECTION_TRIGGERS: tuple[str, ...] = (
    "don't", "do not", "not on", "not have", "not serve", "not offer",
    "out of", " no ", "cannot", "can't", "isn't", "aren't", "unfortunately",
)
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "some", "any", "one", "two", "three",
    "small", "medium", "large", "normal", "regular",
    "your", "my", "his", "her", "their",
    "please", "just", "only",
})
_CONNECTORS: frozenset[str] = frozenset({
    "and", "or", "but", "to", "for", "in", "on", "at", "of", "from",
    "have", "has", "had", "like", "want", "try", "add", "serve", "serves",
    "get", "got", "make", "makes", "do", "does", "did", "put", "puts",
    "recommend", "recommends", "suggest", "suggests", "offer", "offers",
    "we", "you", "i", "they", "he", "she", "it", "us", "them",
    "would", "could", "should", "will", "can", "may", "might",
    "our", "how", "about", "perhaps", "what", "with",
})
_EXTRAS_TOKENS: frozenset[str] = frozenset(
    tok for extra in ALLOWED_EXTRAS for tok in extra.split()
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]


def _is_rejection_context(sentence: str) -> bool:
    lower = sentence.lower()
    return any(trigger in lower for trigger in _REJECTION_TRIGGERS)


def _tokens(sentence: str) -> list[str]:
    return re.findall(r"[a-zA-Zé']+", sentence.lower())


def _find_candidate_phrases(tokens: list[str]) -> list[tuple[list[str], int]]:
    """Return (phrase_tokens, terminal_index) for each item-shape phrase in tokens.

    A phrase terminates at either a single-token terminal (in `_ITEM_TERMINALS`) or
    a multi-word terminal (e.g. "flat white"). The phrase extends left through
    *content* tokens (anything not in `_CONNECTORS`) — this is how off-menu
    modifiers like "hazelnut" or "honey" end up inside the phrase and cause it to
    fail on-menu validation. Extension stops when a connector (verb, preposition,
    pronoun, conjunction) is hit; connectors are not included in the phrase.
    """
    phrases: list[tuple[list[str], int]] = []
    i = 0
    n = len(tokens)

    def _extend_left(start_left: int) -> list[str]:
        acc: list[str] = []
        left = start_left
        while left >= 0 and tokens[left] not in _CONNECTORS and tokens[left] not in _ITEM_TERMINALS:
            acc.insert(0, tokens[left])
            left -= 1
        return acc

    while i < n:
        matched_multi = False
        for mw in _MULTIWORD_TERMINALS:
            if tuple(tokens[i : i + len(mw)]) == mw:
                terminal_end = i + len(mw) - 1
                phrase_tokens = _extend_left(i - 1) + list(tokens[i : i + len(mw)])
                phrases.append((phrase_tokens, terminal_end))
                i += len(mw)
                matched_multi = True
                break
        if matched_multi:
            continue
        if tokens[i] in _ITEM_TERMINALS:
            phrase_tokens = _extend_left(i - 1) + [tokens[i]]
            phrases.append((phrase_tokens, i))
        i += 1
    return phrases


def _phrase_is_on_menu(phrase: list[str]) -> bool:
    """A phrase is on-menu iff its terminal is a MENU key AND every preceding token
    is an ALLOWED_EXTRAS token or a stop word. Anything else — including family-word
    terminals like 'mocha' or 'flat white' — is off-menu."""
    if not phrase:
        return True
    terminal_tokens: list[str] = []
    for mw in _MULTIWORD_TERMINALS:
        if tuple(phrase[-len(mw):]) == mw:
            terminal_tokens = list(mw)
            break
    if terminal_tokens:
        return False
    terminal = phrase[-1]
    if terminal not in MENU:
        return False
    for tok in phrase[:-1]:
        if tok in _STOP_WORDS:
            continue
        if tok in _EXTRAS_TOKENS:
            continue
        return False
    return True


def off_menu_recommendation_predicate(effect: str = "deny"):
    """Factory: detect off-menu drink/food recommendations in an assistant message.

    Applies to the synthetic `assistant_message` tool call synthesized in the
    subgraph's `response_gateway` node — `tool_args["content"]` holds the raw
    `AIMessage.content`. Violation when the message contains a recommendation-shaped
    sentence naming a drink/food whose composed tokens are not in MENU + ALLOWED_EXTRAS.
    Sentences without a recommendation intent verb, or in a rejection context
    ("we don't serve mocha"), are ignored. Any internal error → ALLOW.
    """
    violation_effect = Effect(effect)

    def _eval(context: GuardrailContext) -> Verdict:
        try:
            content = context.tool_args.get("content", "")
            if not isinstance(content, str) or not content.strip():
                return Verdict(
                    effect=Effect.ALLOW,
                    guardrail_name="",
                    guardrail_type="",
                    reason_internal="empty or non-string content; nothing to scan",
                )

            offenders: list[str] = []
            for sentence in _sentences(content):
                if _is_rejection_context(sentence):
                    continue
                for phrase, _terminal_idx in _find_candidate_phrases(_tokens(sentence)):
                    if not _phrase_is_on_menu(phrase):
                        offenders.append(" ".join(phrase))

            if not offenders:
                return Verdict(
                    effect=Effect.ALLOW,
                    guardrail_name="",
                    guardrail_type="",
                    reason_internal="no off-menu recommendation detected",
                )

            unique = sorted(set(offenders))
            return Verdict(
                effect=violation_effect,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"off-menu recommendations detected: {unique}",
                reason_for_llm=(
                    "Your last message recommended items we do not offer: "
                    f"{', '.join(repr(o) for o in unique)}. "
                    f"The menu is: {', '.join(sorted(MENU.keys()))}. "
                    f"Available extras: {', '.join(sorted(ALLOWED_EXTRAS))}. "
                    "Recommend only from these; do not invent flavors, syrups, or drinks."
                ),
            )
        except Exception as exc:
            return Verdict(
                effect=Effect.ALLOW,
                guardrail_name="",
                guardrail_type="",
                reason_internal=f"predicate error, defaulting to ALLOW: {exc!r}",
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
    "off_menu_recommendation": off_menu_recommendation_predicate,
}
