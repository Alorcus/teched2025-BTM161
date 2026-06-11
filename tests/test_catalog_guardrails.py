"""Tests for YAML-driven guardrail loading in `Catalog`."""
import tempfile
import unittest
from pathlib import Path

from src.control_plane.catalog import Catalog
from src.control_plane.guardrails import HardGuardrail, SoftGuardrail
from src.control_plane.types import Effect, GuardrailContext


def _write_setup(root: Path, yaml_text: str) -> Path:
    setup = root / "baseline"
    (setup / "guardrails").mkdir(parents=True)
    (setup / "guidelines").mkdir(parents=True)
    (setup / "guardrails" / "coffee_shop.yaml").write_text(yaml_text, encoding="utf-8")
    return setup


class TestCatalogGuardrails(unittest.TestCase):
    def test_hard_guardrail_with_predicate_args(self):
        yaml_text = """\
guardrails:
  - id: discount_within_10pct
    type: hard
    version: v2
    tools: [calculate_total]
    effect: flag
    description: Flag big discounts.
    predicate: discount_within_limit
    predicate_args:
      max_pct: 10
"""
        with tempfile.TemporaryDirectory() as d:
            setup = _write_setup(Path(d), yaml_text)
            catalog = Catalog(setup)
            [gr] = catalog.guardrails(["discount_within_10pct"])

            self.assertIsInstance(gr, HardGuardrail)
            self.assertEqual(gr.name, "discount_within_10pct")
            self.assertEqual(gr.version, "v2")
            self.assertEqual(gr.effect, Effect.FLAG)
            self.assertEqual(gr.predicate_args, {"max_pct": 10})

            ctx_under = GuardrailContext(
                agent_id="order_agent", tool_name="calculate_total",
                tool_args={"discount_percent": 5}, state={}, allowed_handovers=[],
            )
            self.assertEqual(gr.eval(ctx_under).effect, Effect.ALLOW)

            ctx_over = GuardrailContext(
                agent_id="order_agent", tool_name="calculate_total",
                tool_args={"discount_percent": 25}, state={}, allowed_handovers=[],
            )
            self.assertEqual(gr.eval(ctx_over).effect, Effect.FLAG)

    def test_hard_guardrail_without_args(self):
        yaml_text = """\
guardrails:
  - id: allowed_handover_targets
    type: hard
    tools: [transfer_to_agent]
    effect: deny
    predicate: allowed_handover_targets
"""
        with tempfile.TemporaryDirectory() as d:
            setup = _write_setup(Path(d), yaml_text)
            catalog = Catalog(setup)
            [gr] = catalog.guardrails(["allowed_handover_targets"])

            self.assertIsInstance(gr, HardGuardrail)
            self.assertIsNone(gr.predicate_args)

    def test_soft_guardrail(self):
        yaml_text = """\
guardrails:
  - id: handover_appropriateness_soft_stub
    type: soft
    tools: [transfer_to_agent]
    effect: allow
    judge_prompt: Is the proposed handover appropriate?
    state_dependencies: [conversation]
"""
        with tempfile.TemporaryDirectory() as d:
            setup = _write_setup(Path(d), yaml_text)
            catalog = Catalog(setup)
            [gr] = catalog.guardrails(["handover_appropriateness_soft_stub"])

            self.assertIsInstance(gr, SoftGuardrail)
            self.assertEqual(gr.judge_prompt, "Is the proposed handover appropriate?")
            self.assertEqual(gr.state_dependencies, ["conversation"])

    def test_unknown_predicate_raises(self):
        yaml_text = """\
guardrails:
  - id: bogus
    type: hard
    tools: [foo]
    effect: deny
    predicate: does_not_exist
"""
        with tempfile.TemporaryDirectory() as d:
            setup = _write_setup(Path(d), yaml_text)
            with self.assertRaisesRegex(ValueError, "unknown predicate 'does_not_exist'"):
                Catalog(setup)

    def test_invalid_type_raises(self):
        yaml_text = """\
guardrails:
  - id: bogus
    type: medium
    tools: [foo]
    effect: deny
"""
        with tempfile.TemporaryDirectory() as d:
            setup = _write_setup(Path(d), yaml_text)
            with self.assertRaisesRegex(ValueError, "type must be 'hard' or 'soft'"):
                Catalog(setup)

    def test_unknown_id_raises_at_resolution(self):
        yaml_text = """\
guardrails:
  - id: known
    type: hard
    tools: [foo]
    effect: deny
    predicate: allowed_handover_targets
"""
        with tempfile.TemporaryDirectory() as d:
            setup = _write_setup(Path(d), yaml_text)
            catalog = Catalog(setup)
            with self.assertRaises(KeyError):
                catalog.guardrails(["unknown"])


if __name__ == "__main__":
    unittest.main()
