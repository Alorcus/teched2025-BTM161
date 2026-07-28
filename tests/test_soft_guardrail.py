"""Isolation tests for `SoftGuardrail` — LLM-as-judge evaluation.

The judge LLM is replaced by an in-memory `judge_invoker` stub so these tests
are deterministic, offline, and fast. Every branch of the parse/apply logic is
covered here so the graph-level tests don't need to re-verify them.
"""
import json
import unittest

from src.control_plane.guardrails import SoftGuardrail, _parse_judge_response, _safe_format
from src.control_plane.types import Effect, GuardrailContext


def _ctx(content: str = "", agent_id: str = "order_agent", **extra_args) -> GuardrailContext:
    args = {"content": content, "agent_id": agent_id}
    args.update(extra_args)
    return GuardrailContext(
        agent_id=agent_id,
        tool_name="assistant_message",
        tool_args=args,
        state={},
        allowed_handovers=[],
    )


class TestSoftGuardrailAllow(unittest.TestCase):
    def test_allow_verdict_from_json(self):
        def invoker(_system: str, _user: str) -> str:
            return json.dumps({"decision": "allow", "reason": "on menu"})

        guardrail = SoftGuardrail(
            name="off_menu_recommendation",
            effect=Effect.DENY,
            tools=["assistant_message"],
            judge_prompt="Judge this",
            judge_invoker=invoker,
        )
        verdict = guardrail.eval(_ctx("Would you like a latte?"))
        self.assertEqual(verdict.effect, Effect.ALLOW)
        self.assertEqual(verdict.guardrail_name, "off_menu_recommendation")
        self.assertEqual(verdict.guardrail_type, "soft")
        self.assertEqual(verdict.reason_for_llm, "")

    def test_allow_when_judge_returns_fenced_json(self):
        """LLMs frequently wrap JSON in prose or code fences. Parser must survive it."""
        def invoker(_s: str, _u: str) -> str:
            return "Here is my verdict:\n```json\n" + json.dumps(
                {"decision": "allow", "reason": ""}
            ) + "\n```"

        guardrail = SoftGuardrail(name="g", effect=Effect.DENY, judge_invoker=invoker)
        self.assertEqual(guardrail.eval(_ctx("hi")).effect, Effect.ALLOW)


class TestSoftGuardrailDeny(unittest.TestCase):
    def test_deny_verdict_produces_reason_for_llm(self):
        def invoker(_s: str, _u: str) -> str:
            return json.dumps({
                "decision": "deny",
                "reason": "hazelnut latte is not on the menu",
            })

        guardrail = SoftGuardrail(
            name="off_menu_recommendation",
            effect=Effect.DENY,
            tools=["assistant_message"],
            judge_prompt="Judge",
            judge_invoker=invoker,
        )
        verdict = guardrail.eval(_ctx("How about a hazelnut latte?"))
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertIn("hazelnut latte", verdict.reason_for_llm)
        self.assertIn("hazelnut latte", verdict.reason_internal)

    def test_deny_fallback_reason_when_judge_omits_it(self):
        def invoker(_s: str, _u: str) -> str:
            return json.dumps({"decision": "deny", "reason": ""})

        guardrail = SoftGuardrail(name="g", effect=Effect.DENY, judge_invoker=invoker)
        verdict = guardrail.eval(_ctx("Try our matcha frappe!"))
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertTrue(verdict.reason_for_llm)  # must not be empty


class TestSoftGuardrailFailureModes(unittest.TestCase):
    def test_empty_message_allows_without_calling_judge(self):
        calls: list[tuple[str, str]] = []

        def invoker(s: str, u: str) -> str:
            calls.append((s, u))
            return json.dumps({"decision": "deny", "reason": "should not run"})

        guardrail = SoftGuardrail(name="g", effect=Effect.DENY, judge_invoker=invoker)
        verdict = guardrail.eval(_ctx(""))
        self.assertEqual(verdict.effect, Effect.ALLOW)
        self.assertEqual(calls, [])

    def test_invoker_exception_defaults_to_allow(self):
        def invoker(_s: str, _u: str) -> str:
            raise RuntimeError("boom")

        guardrail = SoftGuardrail(name="g", effect=Effect.DENY, judge_invoker=invoker)
        verdict = guardrail.eval(_ctx("please try a caramel macchiato"))
        self.assertEqual(verdict.effect, Effect.ALLOW)
        self.assertIn("boom", verdict.reason_internal)

    def test_unparseable_response_defaults_to_allow(self):
        def invoker(_s: str, _u: str) -> str:
            return "I couldn't decide — sorry"

        guardrail = SoftGuardrail(name="g", effect=Effect.DENY, judge_invoker=invoker)
        verdict = guardrail.eval(_ctx("hi"))
        self.assertEqual(verdict.effect, Effect.ALLOW)

    def test_prose_containing_deny_does_not_trip_keyword_sniff(self):
        """Historic regression guard: a judge that returns free-form prose
        containing the word "deny" (e.g. "I don't deny lattes are popular")
        must NOT be treated as a deny verdict — that would leak the entire
        raw judge output back to the agent as an injection payload."""
        def invoker(_s: str, _u: str) -> str:
            return "I really don't deny that lattes are popular here!"

        guardrail = SoftGuardrail(name="g", effect=Effect.DENY, judge_invoker=invoker)
        self.assertEqual(guardrail.eval(_ctx("hi")).effect, Effect.ALLOW)


class TestSoftGuardrailEffectSemantics(unittest.TestCase):
    def test_effect_allow_soft_guardrail_stays_allow_on_deny_verdict(self):
        """A guardrail configured as `effect: allow` (observability-only) must
        NOT block even when the judge returns deny."""
        def invoker(_s: str, _u: str) -> str:
            return json.dumps({"decision": "deny", "reason": "off menu"})

        guardrail = SoftGuardrail(
            name="observability_only",
            effect=Effect.ALLOW,
            judge_invoker=invoker,
        )
        verdict = guardrail.eval(_ctx("Try our hazelnut latte!"))
        self.assertEqual(verdict.effect, Effect.ALLOW)

    def test_effect_flag_stays_flag_on_deny_verdict(self):
        def invoker(_s: str, _u: str) -> str:
            return json.dumps({"decision": "deny", "reason": "off menu"})

        guardrail = SoftGuardrail(
            name="flag_only",
            effect=Effect.FLAG,
            judge_invoker=invoker,
        )
        verdict = guardrail.eval(_ctx("Try our hazelnut latte!"))
        self.assertEqual(verdict.effect, Effect.FLAG)


class TestSoftGuardrailBounding(unittest.TestCase):
    def test_oversized_message_is_truncated_before_judge_call(self):
        """A malicious or accidental oversized reply must not be forwarded
        verbatim to the judge — it could cause a context-length error and
        fail-open the guardrail. The message is capped at
        MAX_JUDGE_MESSAGE_LENGTH before templating."""
        from src.control_plane.guardrails import MAX_JUDGE_MESSAGE_LENGTH

        captured = {}

        def invoker(_s: str, user: str) -> str:
            captured["user"] = user
            return json.dumps({"decision": "allow", "reason": ""})

        guardrail = SoftGuardrail(
            name="g",
            effect=Effect.DENY,
            judge_invoker=invoker,
            user_template="{message}",
        )
        oversized = "a" * (MAX_JUDGE_MESSAGE_LENGTH * 2)
        guardrail.eval(_ctx(oversized))
        self.assertLessEqual(len(captured["user"]), MAX_JUDGE_MESSAGE_LENGTH)

    def test_oversized_reason_bounded_in_verdict(self):
        """A judge that returns an oversized `reason` cannot inject a large
        payload into the agent's next-turn correction prompt."""
        from src.control_plane.guardrails import MAX_JUDGE_REASON_LENGTH

        long_reason = "b" * (MAX_JUDGE_REASON_LENGTH * 3)

        def invoker(_s: str, _u: str) -> str:
            return json.dumps({"decision": "deny", "reason": long_reason})

        guardrail = SoftGuardrail(name="g", effect=Effect.DENY, judge_invoker=invoker)
        verdict = guardrail.eval(_ctx("Try our hazelnut latte!"))
        self.assertEqual(verdict.effect, Effect.DENY)
        self.assertLessEqual(len(verdict.reason_for_llm), MAX_JUDGE_REASON_LENGTH)


class TestJudgeInvokerContract(unittest.TestCase):
    def test_menu_and_extras_substituted_into_prompt(self):
        captured: dict = {}

        def invoker(system: str, user: str) -> str:
            captured["system"] = system
            captured["user"] = user
            return json.dumps({"decision": "allow", "reason": ""})

        guardrail = SoftGuardrail(
            name="g",
            effect=Effect.DENY,
            judge_prompt="MENU:\n{menu}\nEXTRAS: {allowed_extras}",
            user_template="AUDIT:\n{message}",
            judge_invoker=invoker,
        )
        guardrail.eval(_ctx("Would you like a latte with oat milk?"))

        self.assertIn("espresso", captured["system"])
        self.assertIn("latte", captured["system"])
        self.assertIn("oat milk", captured["system"])
        self.assertIn("Would you like a latte", captured["user"])

    def test_unknown_placeholder_is_preserved_not_raising(self):
        def invoker(_s: str, _u: str) -> str:
            return json.dumps({"decision": "allow", "reason": ""})

        guardrail = SoftGuardrail(
            name="g",
            effect=Effect.DENY,
            judge_prompt="This has {unknown} placeholder",
            judge_invoker=invoker,
        )
        self.assertEqual(guardrail.eval(_ctx("hi")).effect, Effect.ALLOW)


class TestParseJudgeResponse(unittest.TestCase):
    def test_allow_from_object(self):
        self.assertEqual(
            _parse_judge_response('{"decision": "allow", "reason": ""}'),
            ("allow", ""),
        )

    def test_deny_from_object(self):
        self.assertEqual(
            _parse_judge_response('{"decision": "deny", "reason": "bad"}'),
            ("deny", "bad"),
        )

    def test_json_extracted_from_prose(self):
        self.assertEqual(
            _parse_judge_response('Sure! {"decision": "deny", "reason": "x"} yep'),
            ("deny", "x"),
        )

    def test_missing_fields_default_to_allow(self):
        self.assertEqual(_parse_judge_response('{"foo": "bar"}'), ("allow", ""))

    def test_empty_response_allows(self):
        self.assertEqual(_parse_judge_response(""), ("allow", ""))


class TestSafeFormat(unittest.TestCase):
    def test_known_placeholders_replaced(self):
        self.assertEqual(_safe_format("{a}-{b}", {"a": 1, "b": 2}), "1-2")

    def test_unknown_placeholders_preserved(self):
        self.assertEqual(_safe_format("{a}-{missing}", {"a": 1}), "1-{missing}")

    def test_empty_template_returns_empty(self):
        self.assertEqual(_safe_format("", {"a": 1}), "")


if __name__ == "__main__":
    unittest.main()
