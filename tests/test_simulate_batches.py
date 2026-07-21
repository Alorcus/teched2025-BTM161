import argparse
import unittest

from src.simulate import (
    Batch,
    parse_batch_triple,
    parse_batches_arg,
    parse_scenario_token,
    resolve_scenario,
)


class TestParseBatchTriple(unittest.TestCase):
    def test_full_triple(self):
        self.assertEqual(
            parse_batch_triple("baseline:2:10"), Batch("baseline", 2, 10)
        )

    def test_omitted_count_defaults_to_1(self):
        self.assertEqual(parse_batch_triple("baseline:2"), Batch("baseline", 2, 1))

    def test_omitted_scenario_and_count_default(self):
        self.assertEqual(parse_batch_triple("baseline"), Batch("baseline", 0, 1))

    def test_empty_setup_falls_back_to_default(self):
        batch = parse_batch_triple("::10")
        self.assertEqual((batch.scenario, batch.count), (0, 10))
        self.assertEqual(batch.setup, "baseline")

    def test_all_empty_parts_default_baseline_0_1(self):
        self.assertEqual(parse_batch_triple("::"), Batch("baseline", 0, 1))

    def test_scenario_random_token(self):
        self.assertEqual(
            parse_batch_triple("baseline:random:5"), Batch("baseline", "random", 5)
        )

    def test_scenario_all_token(self):
        self.assertEqual(
            parse_batch_triple("baseline:all:5"), Batch("baseline", "all", 5)
        )

    def test_non_integer_scenario_raises(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_batch_triple("baseline:not-an-int:1")

    def test_non_integer_count_raises(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_batch_triple("baseline:0:xx")

    def test_zero_count_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_batch_triple("baseline:0:0")

    def test_negative_count_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_batch_triple("baseline:0:-3")

    def test_out_of_range_scenario_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_batch_triple("baseline:99:1")

    def test_too_many_colons_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_batch_triple("baseline:0:1:extra")


class TestParseBatchesArg(unittest.TestCase):
    def test_multiple_values(self):
        self.assertEqual(
            parse_batches_arg(["baseline:0:5", "unconstrained:2:3"]),
            [Batch("baseline", 0, 5), Batch("unconstrained", 2, 3)],
        )

    def test_comma_separated_within_one_value(self):
        self.assertEqual(
            parse_batches_arg(["baseline:0:1,unconstrained:2:3"]),
            [Batch("baseline", 0, 1), Batch("unconstrained", 2, 3)],
        )

    def test_empty_chunks_ignored(self):
        self.assertEqual(
            parse_batches_arg(["baseline:0:1,,unconstrained:2:3"]),
            [Batch("baseline", 0, 1), Batch("unconstrained", 2, 3)],
        )

    def test_all_empty_raises(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_batches_arg([","])


class TestParseScenarioToken(unittest.TestCase):
    def test_random(self):
        self.assertEqual(parse_scenario_token("random", "x"), "random")

    def test_all(self):
        self.assertEqual(parse_scenario_token("all", "x"), "all")

    def test_int(self):
        self.assertEqual(parse_scenario_token("2", "x"), 2)

    def test_out_of_range(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_scenario_token("99", "x")

    def test_garbage(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_scenario_token("nope", "x")


class TestResolveScenario(unittest.TestCase):
    def test_fixed_int_returned_verbatim(self):
        self.assertEqual(resolve_scenario(2, 0), 2)
        self.assertEqual(resolve_scenario(2, 5), 2)

    def test_random_returns_none(self):
        self.assertIsNone(resolve_scenario("random", 0))
        self.assertIsNone(resolve_scenario("random", 7))

    def test_all_cycles_through_scenarios(self):
        from src.agents.customer_agent import CUSTOMER_SCENARIOS
        n = len(CUSTOMER_SCENARIOS)
        self.assertEqual(resolve_scenario("all", 0), 0)
        self.assertEqual(resolve_scenario("all", 1), 1 % n)
        self.assertEqual(resolve_scenario("all", n), 0)


if __name__ == "__main__":
    unittest.main()
