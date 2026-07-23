"""Isolation tests for the `soft_baseline` setup — LLM-as-judge counterparts to
each baseline hard guardrail.

Each new soft guardrail is loaded through the real YAML → Catalog path (so the
tests double as a wiring check) and its `judge_invoker` is replaced with an
in-memory stub. That keeps the tests deterministic, offline, and fast while
still validating:

- the applicable-tool wiring (`gr.tools`, `gr.applies_to(tool_name)`),
- the effect the YAML declared (`gr.effect`),
- that the stub judge receives the intended template variables (menu,
  allowed_handovers, order_status, tool_args_json),
- that a deny verdict lifts to a `Verdict(effect=DENY)` (or FLAG when
  configured so),
- and that an allow verdict is passed through cleanly.

The soft_baseline setup keeps every baseline hard rule as a `flag`-only
shadow; the soft counterparts are the actual enforcers (or, for
`soft_discount_within_30pct`, a `flag`-only shadow that mirrors baseline's
observability-only intent for that particular rule).
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Callable

from src.agents.order_store import init_db, reset_inventory, save_order
from src.agents.shared_components import Order, OrderItem, OrderStatus
from src.control_plane import Catalog, SoftGuardrail
from src.control_plane.types import Effect, GuardrailContext

SOFT_BASELINE_DIR = Path("config/setups/soft_baseline")


def _catalog() -> Catalog:
    return Catalog(SOFT_BASELINE_DIR)


def _install_stub(catalog: Catalog, guardrail_id: str, invoker: Callable[[str, str], str]) -> SoftGuardrail:
    """Attach a stub judge to the named soft guardrail loaded from YAML."""
    [gr] = catalog.guardrails([guardrail_id])
    assert isinstance(gr, SoftGuardrail), f"{guardrail_id} should be a SoftGuardrail"
    gr.judge_invoker = invoker
    return gr


def _order_with_status(status: OrderStatus, customer: str = "IsoTest") -> str:
    order = Order(
        customer=customer,
        status=status,
        total=4.0,
        items=[OrderItem(name="latte", quantity=1, price=4.0, size=None, extras=[])],
    )
    save_order(order)
    return order.order_id_str


def _allow_response(reason: str = "") -> str:
    return json.dumps({"decision": "allow", "reason": reason})


def _deny_response(reason: str) -> str:
    return json.dumps({"decision": "deny", "reason": reason})


class SoftBaselineCatalogWiringTest(unittest.TestCase):
    """The YAML must produce the exact guardrail shape the runtime expects."""

    def test_all_soft_counterparts_load(self):
        catalog = _catalog()
        expected = [
            "soft_allowed_handover_targets",
            "soft_discount_within_30pct",
            "soft_gate_check_inventory",
            "soft_gate_update_stock",
            "soft_gate_start_preparation",
            "soft_gate_end_preparation",
            "soft_gate_offer_refund",
        ]
        grs = {g.name: g for g in catalog.guardrails(expected)}
        self.assertEqual(sorted(grs), sorted(expected))
        for name, gr in grs.items():
            self.assertIsInstance(gr, SoftGuardrail, f"{name} must be soft")
            self.assertTrue(gr.tools, f"{name} must declare tools")

    def test_hard_shadows_still_flag_only(self):
        """Every baseline hard rule survives in soft_baseline as a flag-only shadow."""
        catalog = _catalog()
        for hard_id in (
            "allowed_handover_targets",
            "discount_within_30pct",
            "gate_check_inventory",
            "gate_update_stock",
            "gate_start_preparation",
            "gate_end_preparation",
            "gate_offer_refund",
        ):
            [gr] = catalog.guardrails([hard_id])
            self.assertEqual(
                gr.effect, Effect.FLAG,
                f"{hard_id} must be flag-only in soft_baseline; got {gr.effect}",
            )
            self.assertEqual(gr.type, "hard")


class SoftAllowedHandoverTargetsTest(unittest.TestCase):
    """LLM-as-judge counterpart to allowed_handover_targets."""

    def _eval(self, invoker, target: str, allowed_handovers: list[str], agent_id: str = "order_agent"):
        catalog = _catalog()
        gr = _install_stub(catalog, "soft_allowed_handover_targets", invoker)
        context = GuardrailContext(
            agent_id=agent_id,
            tool_name="transfer_to_agent",
            tool_args={
                "target_agent": target,
                "context_summary": "handoff test",
                "expectation": "test",
            },
            state={},
            allowed_handovers=allowed_handovers,
        )
        return gr, gr.eval(context)

    def test_wiring(self):
        [gr] = _catalog().guardrails(["soft_allowed_handover_targets"])
        self.assertEqual(gr.effect, Effect.DENY)
        self.assertIn("transfer_to_agent", gr.tools)
        self.assertTrue(gr.applies_to("transfer_to_agent"))
        self.assertFalse(gr.applies_to("process_order"))

    def test_allowed_target_passes_when_judge_allows(self):
        _gr, verdict = self._eval(
            lambda _s, _u: _allow_response(),
            target="inventory_agent",
            allowed_handovers=["inventory_agent", "customer_service_agent"],
        )
        self.assertEqual(verdict.effect, Effect.ALLOW)

    def test_disallowed_target_denied_when_judge_denies(self):
        _gr, verdict = self._eval(
            lambda _s, _u: _deny_response("barista_agent is not in the allowed list"),
            target="barista_agent",
            allowed_handovers=["inventory_agent", "customer_service_agent"],
        )
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("barista_agent", verdict.reason_for_llm)

    def test_prompt_receives_allowed_handovers_and_target(self):
        """Use a target that is NOT in the allowed list so the two variables
        are disambiguated: `barista_agent` must appear only in the user prompt
        (via tool_args_json), and only the allowed names may appear in the
        system prompt. Swapping the two variables in the YAML would then
        break the assertions."""
        captured: dict[str, str] = {}

        def invoker(system: str, user: str) -> str:
            captured["system"] = system
            captured["user"] = user
            return _allow_response()

        self._eval(
            invoker,
            target="barista_agent",
            allowed_handovers=["inventory_agent", "customer_service_agent"],
            agent_id="order_agent",
        )
        self.assertIn("order_agent", captured["system"])
        self.assertIn("inventory_agent", captured["system"])
        self.assertIn("customer_service_agent", captured["system"])
        self.assertNotIn("barista_agent", captured["system"])
        self.assertIn("barista_agent", captured["user"])

    def test_prompt_notes_empty_allowed_list_and_denies(self):
        captured: dict[str, str] = {}

        def invoker(system: str, user: str) -> str:
            captured["system"] = system
            return _deny_response("no handovers allowed")

        _gr, verdict = self._eval(
            invoker, target="anything", allowed_handovers=[]
        )
        self.assertIn("(none)", captured["system"])
        self.assertEqual(
            verdict.effect, Effect.DENY,
            "empty allowed_handovers must not short-circuit ALLOW",
        )


class SoftDiscountWithinLimitTest(unittest.TestCase):
    """LLM-as-judge counterpart to discount_within_30pct — mirrors baseline flag-only."""

    def _eval(self, invoker, discount: int):
        catalog = _catalog()
        gr = _install_stub(catalog, "soft_discount_within_30pct", invoker)
        context = GuardrailContext(
            agent_id="order_agent",
            tool_name="calculate_total",
            tool_args={"order_id": "ORD9999", "discount_percent": discount},
            state={},
            allowed_handovers=[],
        )
        return gr, gr.eval(context)

    def test_wiring_flag_only(self):
        """Baseline discount rule was already flag-only; the soft mirror stays flag."""
        [gr] = _catalog().guardrails(["soft_discount_within_30pct"])
        self.assertEqual(gr.effect, Effect.FLAG)
        self.assertIn("calculate_total", gr.tools)

    def test_flag_on_judge_deny(self):
        """Judge says deny → verdict effect must remain FLAG (does not block)."""
        _gr, verdict = self._eval(
            lambda _s, _u: _deny_response("50% exceeds 30% cap"),
            discount=50,
        )
        self.assertEqual(verdict.effect, Effect.FLAG)
        self.assertIn("50", verdict.reason_for_llm)

    def test_allow_on_judge_allow(self):
        _gr, verdict = self._eval(
            lambda _s, _u: _allow_response(),
            discount=10,
        )
        self.assertEqual(verdict.effect, Effect.ALLOW)

    def test_prompt_carries_discount_percent_via_tool_args(self):
        """Regression: `discount_percent` must reach the judge through the
        tool_args_json substitution. Renaming that template variable would
        otherwise silently break this guardrail."""
        captured: dict[str, str] = {}

        def invoker(_system: str, user: str) -> str:
            captured["user"] = user
            return _allow_response()

        self._eval(invoker, discount=42)
        self.assertIn("42", captured["user"])
        self.assertIn("discount_percent", captured["user"])


class SoftGateCheckInventoryTest(unittest.TestCase):
    def setUp(self):
        init_db()
        reset_inventory()

    def _eval(self, invoker, order_id: str):
        catalog = _catalog()
        gr = _install_stub(catalog, "soft_gate_check_inventory", invoker)
        context = GuardrailContext(
            agent_id="inventory_agent",
            tool_name="check_inventory",
            tool_args={"order_id": order_id},
            state={},
            allowed_handovers=[],
        )
        return gr, gr.eval(context)

    def test_wiring(self):
        [gr] = _catalog().guardrails(["soft_gate_check_inventory"])
        self.assertEqual(gr.effect, Effect.DENY)
        self.assertIn("check_inventory", gr.tools)

    def test_pending_order_allowed_by_judge(self):
        order_id = _order_with_status(OrderStatus.PENDING)
        _gr, verdict = self._eval(lambda _s, _u: _allow_response(), order_id)
        self.assertEqual(verdict.effect, Effect.ALLOW)

    def test_non_pending_order_denied_by_judge(self):
        order_id = _order_with_status(OrderStatus.INVENTORY_CONFIRMED)
        _gr, verdict = self._eval(
            lambda _s, _u: _deny_response("status is 'inventory_confirmed', not 'pending'"),
            order_id,
        )
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("inventory_confirmed", verdict.reason_for_llm)

    def test_prompt_contains_current_order_status(self):
        order_id = _order_with_status(OrderStatus.INVENTORY_CONFIRMED)
        captured: dict[str, str] = {}

        def invoker(system: str, _user: str) -> str:
            captured["system"] = system
            return _deny_response("wrong status")

        self._eval(invoker, order_id)
        self.assertIn("inventory_confirmed", captured["system"])
        self.assertIn(order_id, captured["system"])

    def test_unresolvable_order_id_leaves_status_blank(self):
        captured: dict[str, str] = {}

        def invoker(system: str, _user: str) -> str:
            captured["system"] = system
            return _deny_response("no order")

        self._eval(invoker, "ORD999999")
        self.assertIn('""', captured["system"])
        self.assertIn("ORD999999", captured["system"])


class SoftGateUpdateStockTest(unittest.TestCase):
    def setUp(self):
        init_db()
        reset_inventory()

    def _eval(self, invoker, order_id: str):
        catalog = _catalog()
        gr = _install_stub(catalog, "soft_gate_update_stock", invoker)
        context = GuardrailContext(
            agent_id="inventory_agent",
            tool_name="update_stock",
            tool_args={"order_id": order_id},
            state={},
            allowed_handovers=[],
        )
        return gr, gr.eval(context)

    def test_wiring(self):
        [gr] = _catalog().guardrails(["soft_gate_update_stock"])
        self.assertEqual(gr.effect, Effect.DENY)
        self.assertIn("update_stock", gr.tools)

    def test_confirmed_allowed(self):
        order_id = _order_with_status(OrderStatus.INVENTORY_CONFIRMED)
        _gr, verdict = self._eval(lambda _s, _u: _allow_response(), order_id)
        self.assertEqual(verdict.effect, Effect.ALLOW)

    def test_pending_denied(self):
        order_id = _order_with_status(OrderStatus.PENDING)
        _gr, verdict = self._eval(
            lambda _s, _u: _deny_response("status pending, not inventory_confirmed"),
            order_id,
        )
        self.assertEqual(verdict.effect, Effect.DENY)

    def test_completed_denied(self):
        order_id = _order_with_status(OrderStatus.COMPLETED)
        _gr, verdict = self._eval(
            lambda _s, _u: _deny_response("status completed, not inventory_confirmed"),
            order_id,
        )
        self.assertEqual(verdict.effect, Effect.DENY)

    def test_prompt_carries_order_status_and_id(self):
        order_id = _order_with_status(OrderStatus.COMPLETED)
        captured: dict[str, str] = {}

        def invoker(system: str, _user: str) -> str:
            captured["system"] = system
            return _deny_response("wrong status")

        self._eval(invoker, order_id)
        self.assertIn("completed", captured["system"])
        self.assertIn(order_id, captured["system"])


class SoftGateStartPreparationTest(unittest.TestCase):
    def setUp(self):
        init_db()
        reset_inventory()

    def _eval(self, invoker, order_id: str):
        catalog = _catalog()
        gr = _install_stub(catalog, "soft_gate_start_preparation", invoker)
        context = GuardrailContext(
            agent_id="barista_agent",
            tool_name="start_preparation",
            tool_args={"order_id": order_id},
            state={},
            allowed_handovers=[],
        )
        return gr, gr.eval(context)

    def test_wiring(self):
        [gr] = _catalog().guardrails(["soft_gate_start_preparation"])
        self.assertEqual(gr.effect, Effect.DENY)
        self.assertIn("start_preparation", gr.tools)

    def test_confirmed_allowed(self):
        order_id = _order_with_status(OrderStatus.INVENTORY_CONFIRMED)
        _gr, verdict = self._eval(lambda _s, _u: _allow_response(), order_id)
        self.assertEqual(verdict.effect, Effect.ALLOW)

    def test_in_preparation_allowed_for_retry(self):
        order_id = _order_with_status(OrderStatus.IN_PREPARATION)
        _gr, verdict = self._eval(lambda _s, _u: _allow_response(), order_id)
        self.assertEqual(verdict.effect, Effect.ALLOW)

    def test_preparation_error_allowed_for_retry(self):
        order_id = _order_with_status(OrderStatus.PREPARATION_ERROR)
        _gr, verdict = self._eval(lambda _s, _u: _allow_response(), order_id)
        self.assertEqual(verdict.effect, Effect.ALLOW)

    def test_pending_denied(self):
        order_id = _order_with_status(OrderStatus.PENDING)
        _gr, verdict = self._eval(
            lambda _s, _u: _deny_response("not yet confirmed"),
            order_id,
        )
        self.assertEqual(verdict.effect, Effect.DENY)


class SoftGateEndPreparationTest(unittest.TestCase):
    def setUp(self):
        init_db()
        reset_inventory()

    def _eval(self, invoker, order_id: str):
        catalog = _catalog()
        gr = _install_stub(catalog, "soft_gate_end_preparation", invoker)
        context = GuardrailContext(
            agent_id="barista_agent",
            tool_name="end_preparation",
            tool_args={"order_id": order_id},
            state={},
            allowed_handovers=[],
        )
        return gr, gr.eval(context)

    def test_wiring(self):
        [gr] = _catalog().guardrails(["soft_gate_end_preparation"])
        self.assertEqual(gr.effect, Effect.DENY)
        self.assertIn("end_preparation", gr.tools)

    def test_in_preparation_allowed(self):
        order_id = _order_with_status(OrderStatus.IN_PREPARATION)
        _gr, verdict = self._eval(lambda _s, _u: _allow_response(), order_id)
        self.assertEqual(verdict.effect, Effect.ALLOW)

    def test_completed_denied(self):
        order_id = _order_with_status(OrderStatus.COMPLETED)
        _gr, verdict = self._eval(
            lambda _s, _u: _deny_response("already completed"),
            order_id,
        )
        self.assertEqual(verdict.effect, Effect.DENY)

    def test_pending_denied(self):
        order_id = _order_with_status(OrderStatus.PENDING)
        _gr, verdict = self._eval(
            lambda _s, _u: _deny_response("not in preparation"),
            order_id,
        )
        self.assertEqual(verdict.effect, Effect.DENY)

    def test_prompt_carries_order_status_and_id(self):
        order_id = _order_with_status(OrderStatus.IN_PREPARATION)
        captured: dict[str, str] = {}

        def invoker(system: str, _user: str) -> str:
            captured["system"] = system
            return _allow_response()

        self._eval(invoker, order_id)
        self.assertIn("in_preparation", captured["system"])
        self.assertIn(order_id, captured["system"])


class SoftGateOfferRefundTest(unittest.TestCase):
    def setUp(self):
        init_db()
        reset_inventory()

    def _eval(self, invoker, order_id: str):
        catalog = _catalog()
        gr = _install_stub(catalog, "soft_gate_offer_refund", invoker)
        context = GuardrailContext(
            agent_id="customer_service_agent",
            tool_name="offer_refund",
            tool_args={"order_id": order_id},
            state={},
            allowed_handovers=[],
        )
        return gr, gr.eval(context)

    def test_wiring(self):
        [gr] = _catalog().guardrails(["soft_gate_offer_refund"])
        self.assertEqual(gr.effect, Effect.DENY)
        self.assertIn("offer_refund", gr.tools)

    def test_completed_allowed(self):
        order_id = _order_with_status(OrderStatus.COMPLETED)
        _gr, verdict = self._eval(lambda _s, _u: _allow_response(), order_id)
        self.assertEqual(verdict.effect, Effect.ALLOW)

    def test_pending_denied(self):
        order_id = _order_with_status(OrderStatus.PENDING)
        _gr, verdict = self._eval(
            lambda _s, _u: _deny_response("not completed"),
            order_id,
        )
        self.assertEqual(verdict.effect, Effect.DENY)

    def test_status_visible_to_judge(self):
        order_id = _order_with_status(OrderStatus.PREPARATION_ERROR)
        captured: dict[str, str] = {}

        def invoker(system: str, _user: str) -> str:
            captured["system"] = system
            return _deny_response("wrong status")

        self._eval(invoker, order_id)
        self.assertIn("preparation_error", captured["system"])


class SoftGuardrailPromptContentTest(unittest.TestCase):
    """Cross-cutting: verify the new template variables (`tool_args_json`,
    `allowed_handovers`, `order_id`, `order_status`) reach every soft judge."""

    def setUp(self):
        init_db()
        reset_inventory()

    def test_tool_args_json_reaches_user_template(self):
        catalog = _catalog()
        captured: dict[str, str] = {}

        def invoker(_system: str, user: str) -> str:
            captured["user"] = user
            return _allow_response()

        gr = _install_stub(catalog, "soft_gate_start_preparation", invoker)
        order_id = _order_with_status(OrderStatus.INVENTORY_CONFIRMED)
        gr.eval(GuardrailContext(
            agent_id="barista_agent",
            tool_name="start_preparation",
            tool_args={"order_id": order_id, "hint": "smoke-test"},
            state={},
            allowed_handovers=[],
        ))
        self.assertIn(order_id, captured["user"])
        self.assertIn("smoke-test", captured["user"])

    def test_soft_state_gate_does_not_short_circuit_on_empty_content(self):
        """Parity guard: a tool call whose args happen to include a blank
        `content` key must still be evaluated by the judge (not silently
        allowed via the `assistant_message` empty-message shortcut). The
        parallel hard predicate denies the same call — the soft judge must
        get a chance to enforce the same rule."""
        catalog = _catalog()
        invocations: list[tuple[str, str]] = []

        def invoker(system: str, user: str) -> str:
            invocations.append((system, user))
            return _deny_response("blank order_id, cannot verify status")

        gr = _install_stub(catalog, "soft_gate_check_inventory", invoker)
        verdict = gr.eval(GuardrailContext(
            agent_id="inventory_agent",
            tool_name="check_inventory",
            tool_args={"content": "", "order_id": ""},
            state={},
            allowed_handovers=[],
        ))
        self.assertEqual(
            len(invocations), 1,
            "judge must be invoked even when tool_args has a blank content key",
        )
        self.assertEqual(verdict.effect, Effect.DENY)

    def test_off_menu_recommendation_still_works(self):
        """Regression: the extended _template_vars must not break the pre-existing
        off_menu_recommendation soft guardrail. Captures BOTH prompts and asserts
        that the menu / allowed_extras placeholders are still populated — a bare
        stub that only sniffs the user message would not detect a dropped
        `{menu}` placeholder in the system prompt."""
        catalog = _catalog()
        captured: dict[str, str] = {}

        def invoker(system: str, user: str) -> str:
            captured["system"] = system
            captured["user"] = user
            if "hazelnut" in user.lower():
                return _deny_response("hazelnut latte not on menu")
            return _allow_response()

        gr = _install_stub(catalog, "off_menu_recommendation", invoker)
        deny_verdict = gr.eval(GuardrailContext(
            agent_id="order_agent",
            tool_name="assistant_message",
            tool_args={"content": "How about a hazelnut latte?"},
            state={},
            allowed_handovers=[],
        ))
        self.assertEqual(deny_verdict.effect, Effect.DENY)
        allow_verdict = gr.eval(GuardrailContext(
            agent_id="order_agent",
            tool_name="assistant_message",
            tool_args={"content": "Would you like a latte?"},
            state={},
            allowed_handovers=[],
        ))
        self.assertEqual(allow_verdict.effect, Effect.ALLOW)

        self.assertIn("espresso", captured["system"])
        self.assertIn("latte", captured["system"])
        self.assertIn("oat milk", captured["system"])
        self.assertNotIn("{menu}", captured["system"])
        self.assertNotIn("{allowed_extras}", captured["system"])


if __name__ == "__main__":
    unittest.main()
