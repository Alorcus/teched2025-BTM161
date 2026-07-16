"""Tests for the `off_menu_recommendation` predicate.

Cases derive from the SpecFlow analysis in
`docs/plans/2026-07-16-feat-menu-adherence-guardrail-plan.md` — the twelve
scenarios span the two common failure modes (menu enumeration false-positive,
family-word false-negative like 'mocha' / 'flat white').
"""
import unittest

from src.control_plane.predicates import off_menu_recommendation_predicate
from src.control_plane.types import Effect, GuardrailContext


def _ctx(content: str) -> GuardrailContext:
    return GuardrailContext(
        agent_id="order_agent",
        tool_name="assistant_message",
        tool_args={"content": content},
        state={},
        allowed_handovers=[],
    )


class TestOffMenuRecommendation(unittest.TestCase):
    def setUp(self):
        self.eval_deny = off_menu_recommendation_predicate("deny")

    def _assert_deny(self, text: str):
        verdict = self.eval_deny(_ctx(text))
        self.assertEqual(
            verdict.effect, Effect.DENY,
            f"expected DENY for {text!r}, got {verdict.effect} — {verdict.reason_internal}",
        )

    def _assert_allow(self, text: str):
        verdict = self.eval_deny(_ctx(text))
        self.assertEqual(
            verdict.effect, Effect.ALLOW,
            f"expected ALLOW for {text!r}, got {verdict.effect} — {verdict.reason_internal}",
        )

    def test_01_hazelnut_latte_denied(self):
        self._assert_deny("How about a hazelnut latte?")

    def test_02_oat_milk_latte_allowed(self):
        self._assert_allow("Would you like an oat milk latte coming up.")

    def test_03_menu_enumeration_allowed(self):
        self._assert_allow("We have espresso, latte, cappuccino, americano.")

    def test_04_customer_echo_rejection_allowed(self):
        self._assert_allow("You asked about mocha — we don't serve that.")

    def test_05_pumpkin_spice_denied(self):
        self._assert_deny("Try our seasonal pumpkin spice.")

    def test_06_pricing_statement_allowed(self):
        self._assert_allow("That'll be $4.50 for the iced americano.")

    def test_07_flat_white_denied(self):
        self._assert_deny("Perhaps a flat white?")

    def test_08_decaf_latte_vanilla_syrup_allowed(self):
        self._assert_allow("Would you like one decaf latte with vanilla syrup?")

    def test_09_honey_addon_out_of_scope(self):
        # "Add some honey to your latte" is an off-menu *addon* recommendation,
        # not an off-menu drink/food. The tool-call guardrail on `process_order`
        # already rejects unknown extras when the order is submitted; the
        # conversation-layer scanner focuses on hallucinated drinks/food. Left
        # as an explicit ALLOW so the boundary is visible; upgrading to catch
        # this case is deferred (would require an "add X to Y" heuristic).
        self._assert_allow("We could add some honey to your latte.")

    def test_10_mocha_denied(self):
        self._assert_deny("We could do a mocha for you.")

    def test_11_no_item_phrase_allowed(self):
        self._assert_allow("I'd suggest checking with our barista.")

    def test_12_frappe_denied(self):
        self._assert_deny("Would you like a frappé?")

    def test_empty_content_allowed(self):
        self._assert_allow("")

    def test_missing_content_allowed(self):
        verdict = self.eval_deny(GuardrailContext(
            agent_id="order_agent", tool_name="assistant_message",
            tool_args={}, state={}, allowed_handovers=[],
        ))
        self.assertEqual(verdict.effect, Effect.ALLOW)

    def test_predicate_error_defaults_to_allow(self):
        verdict = self.eval_deny(GuardrailContext(
            agent_id="order_agent", tool_name="assistant_message",
            tool_args={"content": 12345},
            state={}, allowed_handovers=[],
        ))
        self.assertEqual(verdict.effect, Effect.ALLOW)

    def test_flag_effect_variant(self):
        eval_flag = off_menu_recommendation_predicate("flag")
        verdict = eval_flag(_ctx("How about a hazelnut latte?"))
        self.assertEqual(verdict.effect, Effect.FLAG)

    def test_reason_for_llm_lists_menu(self):
        verdict = self.eval_deny(_ctx("Would you like a frappé?"))
        self.assertIn("frappé", verdict.reason_for_llm)
        self.assertIn("espresso", verdict.reason_for_llm)
        self.assertIn("vanilla syrup", verdict.reason_for_llm)

    def test_scenario_4_multi_line_recommendation(self):
        # Reproduces the actual bad turn from the Scenario 4 log — the reason
        # this guardrail exists. Must catch at least the three unambiguous
        # off-menu names.
        actual_turn = (
            "If you want to stay in that comfort zone but level it up:\n"
            "- Hazelnut Latte — still got that rich, creamy goodness.\n"
            "- Caramel Macchiato — vanilla cousin with a caramel kick.\n"
            "\n"
            "If you're feeling a little adventurous:\n"
            "- Honey Cinnamon Latte — warm, slightly spiced vibe.\n"
        )
        verdict = self.eval_deny(_ctx(actual_turn))
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("hazelnut latte", verdict.reason_internal)
        self.assertIn("caramel macchiato", verdict.reason_internal)
        self.assertIn("honey cinnamon latte", verdict.reason_internal)


if __name__ == "__main__":
    unittest.main()
