import argparse
import logging
import sys
from itertools import groupby

from .coffee_shop import CoffeeShop
from .config import CoffeeShopConfig
from .setups import list_setups, resolve_setup_name, resolve_setup_names, setup_dir
from .agents.customer_agent import CUSTOMER_SCENARIOS
from .trace_processing import TraceProcessor

coffee_shop_logger = logging.getLogger("coffee_shop")

STATUS_LEVEL = logging.CRITICAL + 10
logging.addLevelName(STATUS_LEVEL, "STATUS")


def log_status(message):
    coffee_shop_logger.log(STATUS_LEVEL, message)


def parse_scenario(value):
    if value == "all":
        return ("all", None)
    if value == "random":
        return ("random", None)
    try:
        idx = int(value)
        if 0 <= idx < len(CUSTOMER_SCENARIOS):
            return ("fixed", idx)
        coffee_shop_logger.error(
            f"scenario index must be 0-{len(CUSTOMER_SCENARIOS) - 1}"
        )
        sys.exit(1)
    except ValueError:
        coffee_shop_logger.error(
            f"'--scenario' must be 0-{len(CUSTOMER_SCENARIOS) - 1}, 'all', or 'random'"
        )
        sys.exit(1)


def pick_scenario_index(mode, fixed_index, trace_number):
    if mode == "fixed":
        return fixed_index
    if mode == "all":
        return trace_number % len(CUSTOMER_SCENARIOS)
    return None


def parse_batch_triple(raw: str) -> tuple[str, int, int]:
    """Parse one SETUP:SCENARIO:COUNT triple with per-part defaults.

    Missing or empty parts fall back to: setup=<default>, scenario=0, count=1.
    Accepts 1-3 colon-separated fields (`baseline`, `baseline:2`,
    `baseline:2:10`, `::10`, `:2:`, etc.).
    """
    parts = raw.split(":")
    if len(parts) > 3:
        raise argparse.ArgumentTypeError(
            f"--batches expects SETUP:SCENARIO:COUNT, got '{raw}' (too many ':' separators)"
        )
    parts = parts + [""] * (3 - len(parts))
    setup_str, scenario_str, count_str = (p.strip() for p in parts)

    setup = setup_str if setup_str else resolve_setup_name(None)

    if scenario_str:
        try:
            scenario = int(scenario_str)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--batches expects SETUP:SCENARIO:COUNT, got '{raw}' "
                f"(scenario must be an int)"
            )
    else:
        scenario = 0

    if count_str:
        try:
            count = int(count_str)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--batches expects SETUP:SCENARIO:COUNT, got '{raw}' "
                f"(count must be an int)"
            )
    else:
        count = 1

    if count <= 0:
        raise argparse.ArgumentTypeError(
            f"--batches expects SETUP:SCENARIO:COUNT, got '{raw}' "
            f"(count must be positive)"
        )
    if not (0 <= scenario < len(CUSTOMER_SCENARIOS)):
        raise argparse.ArgumentTypeError(
            f"--batches scenario {scenario} out of range "
            f"0-{len(CUSTOMER_SCENARIOS) - 1} in '{raw}'"
        )
    return setup, scenario, count


def parse_batches_arg(values: list[str]) -> list[tuple[str, int, int]]:
    triples: list[tuple[str, int, int]] = []
    for value in values:
        for chunk in value.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            triples.append(parse_batch_triple(chunk))
    if not triples:
        raise argparse.ArgumentTypeError("--batches requires at least one triple")
    return triples


def build_batches_from_simple_args(
    setup_names: list[str], scenario_mode: str, scenario_fixed: int | None, traces: int
) -> list[tuple[str, int | str, int]]:
    """Expand the simple `--setup/--scenario/--traces` shortcut into batches.

    Scenario position may be an int (fixed), 'all' (round-robin), or 'random'.
    """
    scenario_value: int | str
    if scenario_mode == "fixed":
        scenario_value = scenario_fixed  # type: ignore[assignment]
    else:
        scenario_value = scenario_mode  # 'all' or 'random'
    return [(name, scenario_value, traces) for name in setup_names]


def resolve_batch_scenario(scenario_value, trace_number):
    """Return a concrete scenario index (or None for random) for a batch item."""
    if scenario_value == "random":
        return None
    if scenario_value == "all":
        return trace_number % len(CUSTOMER_SCENARIOS)
    return scenario_value  # already an int


def make_on_message(args):
    if args.quiet:
        return None

    def on_message(role, content):
        prefix = "[Customer]" if role == "customer" else "[Agent]   "
        body = "\n" + content if args.full_messages else "\n" + content[:200]
        coffee_shop_logger.info(f"{prefix} {body}")

    return on_message


def main():
    parser = argparse.ArgumentParser(
        description="Run headless coffee shop simulations to generate traces"
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        metavar="SETUP:SCENARIO:COUNT",
        help=(
            "One or more SETUP:SCENARIO:COUNT triples (e.g. "
            "'baseline:0:50 unconstrained:2:50'). Missing parts fall back to "
            "setup=<default>, scenario=0, count=1 — so 'baseline' means one "
            "trace of scenario 0 under baseline, and '::10' means 10 traces of "
            "scenario 0 under the default setup. Mutually exclusive with "
            "--setup/--scenario/--traces."
        ),
    )
    parser.add_argument(
        "--traces",
        type=int,
        default=1,
        help="Number of conversation traces to run (default: 1)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="random",
        help="Scenario index (0-3), 'all' (round-robin), or 'random' (default: random)",
    )
    parser.add_argument(
        "--export-logs",
        action="store_true",
        help="Export event logs after simulation",
    )
    parser.add_argument(
        "--reset-inventory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset inventory before each trace (default: true). Use --no-reset-inventory to keep inventory state across traces.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output: only trace numbers, scenarios, and summary",
    )
    parser.add_argument(
        "--full-messages",
        action="store_true",
        help="Print full message content instead of truncating to 200 characters",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Set the logging level for the coffee_shop logger (default: info). Note: levels above info will not show progress messages, and debug/info may produce output even with --quiet.",
    )
    parser.add_argument(
        "--setup",
        type=str,
        default=None,
        action="append",
        help=(
            "Name of the setup under config/setups/ to load. Repeat the flag to run "
            "multiple setups sequentially (e.g. --setup baseline --setup all_handovers). "
            "Mutually exclusive with --batches."
        ),
    )
    parser.add_argument(
        "--list-setups",
        action="store_true",
        help="List available setups under config/setups/ and exit.",
    )
    parser.add_argument(
        "--process-supervisor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the process supervisor (default: true). Use --no-process-supervisor to run without it.",
    )
    args = parser.parse_args()

    if args.list_setups:
        names = list_setups()
        if not names:
            print("(no setups found in config/setups/)")
        else:
            for name in names:
                print(name)
        return 0

    if args.batches is not None and (
        args.setup is not None or args.traces != 1 or args.scenario != "random"
    ):
        parser.error(
            "--batches is mutually exclusive with --setup / --scenario / --traces"
        )

    if args.quiet and args.full_messages:
        coffee_shop_logger.warning(
            "'--full-messages' has no effect when '--quiet' is set"
        )
    if args.quiet and args.log_level.lower() in ["debug", "info"]:
        coffee_shop_logger.warning(
            "debug/info log levels may produce output even if '--quiet' is set"
        )
    coffee_shop_logger.setLevel(getattr(logging, args.log_level.upper()))

    if args.batches is not None:
        try:
            batches = parse_batches_arg(args.batches)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    else:
        setup_names = resolve_setup_names(args.setup)
        scenario_mode, scenario_fixed = parse_scenario(args.scenario)
        batches = build_batches_from_simple_args(
            setup_names, scenario_mode, scenario_fixed, args.traces
        )

    for setup_name, _, _ in batches:
        setup_dir(setup_name)

    total_traces = sum(count for _, _, count in batches)
    total_batches = len(batches)
    on_message = make_on_message(args)

    all_trace_ids: list[str] = []
    per_batch_counts: list[tuple[str, object, int, int]] = []
    batch_number = 0
    trace_number = 0

    for setup_name, group in groupby(batches, key=lambda b: b[0]):
        group_batches = list(group)
        coffee_shop_logger.info(f"=== Setup: {setup_name} ===")
        shop = CoffeeShop(
            CoffeeShopConfig(
                setup_name=setup_name,
                process_supervisor_enabled=args.process_supervisor,
            )
        )
        shop.open_shop(reset_inventory_first=args.reset_inventory)
        coffee_shop_logger.info(
            f"Resetting inventory before each trace: {args.reset_inventory}"
        )

        for _, scenario_value, count in group_batches:
            batch_number += 1
            scenario_label_key = (
                scenario_value if isinstance(scenario_value, int) else 0
            )
            label = CUSTOMER_SCENARIOS[scenario_label_key]
            log_status(
                f"=== Batch {batch_number}/{total_batches} | setup '{setup_name}' "
                f"| scenario {scenario_value}: {label[:60]} | {count} trace(s) ==="
            )

            batch_trace_ids: list[str] = []
            for i in range(count):
                trace_number += 1
                idx = resolve_batch_scenario(scenario_value, i)
                scenario_desc = (
                    CUSTOMER_SCENARIOS[idx] if idx is not None else "random"
                )
                log_status(
                    f"--- Trace {trace_number}/{total_traces} "
                    f"(batch {batch_number}/{total_batches}, conv {i + 1}/{count}) "
                    f"| Scenario {idx}: {scenario_desc[:60]} ---"
                )

                trace_ids = shop.run_conversation(
                    scenario_index=idx,
                    on_message=on_message,
                    reset_inventory_first=args.reset_inventory,
                )
                batch_trace_ids.extend(trace_ids)
                coffee_shop_logger.info(f"Trace IDs: {trace_ids}")

                feedback = shop.get_last_feedback()
                if feedback:
                    score = feedback["feedback_score"]
                    reason = feedback["feedback_reason"]
                    valid_marker = "" if feedback["valid"] else " (fallback)"
                    coffee_shop_logger.info(
                        f"Customer feedback [{score:.2f}{valid_marker}]: {reason}"
                    )

            all_trace_ids.extend(batch_trace_ids)
            per_batch_counts.append(
                (setup_name, scenario_value, count, len(batch_trace_ids))
            )

    coffee_shop_logger.info(
        f"=== Simulation complete: {total_batches} batch(es), "
        f"{len(all_trace_ids)} trace(s) generated ==="
    )
    for setup_name, scenario_value, requested, actual in per_batch_counts:
        coffee_shop_logger.info(
            f"  - {setup_name} / scenario {scenario_value}: "
            f"{actual} trace(s) (requested {requested})"
        )

    if args.export_logs:
        coffee_shop_logger.info("Exporting event logs...")
        TraceProcessor().process_all_traces()

    return 0


if __name__ == "__main__":
    sys.exit(main())
