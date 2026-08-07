import argparse
import logging
import sys
from dataclasses import dataclass
from itertools import groupby
from typing import Callable

from .coffee_shop import CoffeeShop
from .config import CoffeeShopConfig
from .setups import list_setups, resolve_setup_name, setup_dir
from .agents.customer_agent import CUSTOMER_SCENARIOS
from .trace_processing import TraceProcessor

coffee_shop_logger = logging.getLogger("coffee_shop")

STATUS_LEVEL = logging.CRITICAL + 10
logging.addLevelName(STATUS_LEVEL, "STATUS")


def log_status(message):
    coffee_shop_logger.log(STATUS_LEVEL, message)


ScenarioSpec = int | str  # int for fixed index, "random", or "all"

OnMessage = Callable[[str, str], None]


@dataclass(frozen=True)
class Batch:
    setup: str
    scenario: ScenarioSpec
    count: int


def parse_scenario_token(token: str, source: str) -> ScenarioSpec:
    """Accept an int-valued index, 'all', or 'random'."""
    if token in ("all", "random"):
        return token
    try:
        idx = int(token)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{source}: scenario must be an int 0-{len(CUSTOMER_SCENARIOS) - 1}, "
            f"'all', or 'random' (got {token!r})"
        )
    if not (0 <= idx < len(CUSTOMER_SCENARIOS)):
        raise argparse.ArgumentTypeError(
            f"{source}: scenario {idx} out of range 0-{len(CUSTOMER_SCENARIOS) - 1}"
        )
    return idx


def parse_batch_triple(raw: str) -> Batch:
    """Parse one SETUP:SCENARIO:COUNT triple with per-part defaults.

    Missing or empty parts fall back to: setup=<default>, scenario=0, count=1.
    Accepts up to 3 colon-separated fields.
    """
    parts = raw.split(":")
    if len(parts) > 3:
        raise argparse.ArgumentTypeError(
            f"--batches expects SETUP:SCENARIO:COUNT, got {raw!r} "
            "(too many ':' separators)"
        )
    parts.extend([""] * (3 - len(parts)))
    setup_str, scenario_str, count_str = (p.strip() for p in parts)

    setup = setup_str or resolve_setup_name(None)
    scenario = parse_scenario_token(scenario_str, raw) if scenario_str else 0

    if count_str:
        try:
            count = int(count_str)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--batches expects SETUP:SCENARIO:COUNT, got {raw!r} "
                "(count must be an int)"
            )
    else:
        count = 1
    if count <= 0:
        raise argparse.ArgumentTypeError(
            f"--batches expects SETUP:SCENARIO:COUNT, got {raw!r} "
            "(count must be positive)"
        )
    return Batch(setup, scenario, count)


def parse_batches_arg(values: list[str]) -> list[Batch]:
    """Parse `--batches` values into a list of Batch entries.

    Accepts both space-separated (via argparse `nargs="+"`) and
    comma-separated forms; empty chunks are ignored.
    """
    triples = [
        parse_batch_triple(chunk.strip())
        for value in values
        for chunk in value.split(",")
        if chunk.strip()
    ]
    if not triples:
        raise argparse.ArgumentTypeError("--batches requires at least one triple")
    return triples


def resolve_scenario(scenario: ScenarioSpec, trace_number: int) -> int | None:
    """Return the concrete scenario index for a trace (or None for random)."""
    if scenario == "random":
        return None
    if scenario == "all":
        return trace_number % len(CUSTOMER_SCENARIOS)
    assert isinstance(scenario, int)
    return scenario


def format_feedback(feedback: dict) -> str:
    """Render a feedback entry as '[score(fallback)]: reason'.

    `feedback_score` is None whenever the judge LLM returned unparseable JSON
    (CustomerAgent.get_feedback), so the score must never be format-spec'd
    directly — a TypeError here would abort a whole batch run.
    """
    score = feedback.get("feedback_score")
    score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
    marker = "" if feedback.get("valid") else " (fallback)"
    return f"[{score_text}{marker}]: {feedback.get('feedback_reason')}"


def make_on_message(quiet: bool, full_messages: bool) -> OnMessage | None:
    if quiet:
        return None

    def on_message(role: str, content: str) -> None:
        prefix = "[Customer]" if role == "customer" else "[Agent]   "
        body = "\n" + content if full_messages else "\n" + content[:200]
        coffee_shop_logger.info(f"{prefix} {body}")

    return on_message


def _configure_logging(log_level: str) -> None:
    """coffee_shop.py installs a StreamHandler on the 'coffee_shop' logger at
    import time, so we only need to set its level here and muzzle httpx."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    coffee_shop_logger.setLevel(getattr(logging, log_level.upper()))


def main():
    parser = argparse.ArgumentParser(
        description="Run headless coffee shop simulations to generate traces"
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        default=argparse.SUPPRESS,
        metavar="SETUP:SCENARIO:COUNT",
        help=(
            "One or more SETUP:SCENARIO:COUNT triples. Missing parts default to "
            "setup=<default>, scenario=0, count=1 (so 'baseline' means 1 trace of "
            "scenario 0 under baseline; '::10' means 10 traces of scenario 0 under "
            "the default setup). Scenario may also be 'all' or 'random'. Mutually "
            "exclusive with --setup / --scenario / --traces."
        ),
    )
    parser.add_argument(
        "--traces",
        type=int,
        default=argparse.SUPPRESS,
        help="Shortcut for a single batch: number of conversation traces (default: 1)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=argparse.SUPPRESS,
        help=(
            "Shortcut for a single batch: scenario index (0-6), 'all', or 'random' "
            "(default: random)"
        ),
    )
    parser.add_argument(
        "--setup",
        type=str,
        default=argparse.SUPPRESS,
        action="append",
        help=(
            "Shortcut for a single batch: setup under config/setups/ to load. "
            "Repeat to sequence multiple setups. Mutually exclusive with --batches."
        ),
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
        help="Reset inventory before each trace (default: true).",
    )
    parser.add_argument(
        "--on-error",
        type=str,
        default="abort",
        choices=["abort", "skip"],
        help=(
            "What to do when a conversation raises: 'abort' stops the run "
            "(default), 'skip' logs the failure and continues with the next "
            "trace — use it for long unattended sweeps."
        ),
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
        help="coffee_shop logger level (default: info).",
    )
    parser.add_argument(
        "--list-setups",
        action="store_true",
        help="List available setups under config/setups/ and exit.",
    )
    args = parser.parse_args()

    if args.list_setups:
        names = list_setups()
        print("\n".join(names) if names else "(no setups found in config/setups/)")
        return 0

    # `argparse.SUPPRESS` defaults keep these attributes absent unless the user
    # actually passed the flag — that's how the mutual-exclusion check
    # distinguishes explicit input from unset.
    batches_given = hasattr(args, "batches")
    shortcut_flags = [f for f in ("setup", "scenario", "traces") if hasattr(args, f)]
    if batches_given and shortcut_flags:
        parser.error(
            "--batches is mutually exclusive with --setup / --scenario / --traces"
        )

    if batches_given:
        try:
            batches = parse_batches_arg(args.batches)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    else:
        setups = getattr(args, "setup", None) or [resolve_setup_name(None)]
        try:
            scenario = parse_scenario_token(
                getattr(args, "scenario", "random"), "--scenario"
            )
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        traces = getattr(args, "traces", 1)
        batches = [
            Batch(setup=name, scenario=scenario, count=traces) for name in setups
        ]

    if args.quiet and args.full_messages:
        coffee_shop_logger.warning(
            "'--full-messages' has no effect when '--quiet' is set"
        )
    if args.quiet and args.log_level.lower() in ("debug", "info"):
        coffee_shop_logger.warning(
            "debug/info log levels may produce output even if '--quiet' is set"
        )

    _configure_logging(args.log_level)

    for batch in batches:
        setup_dir(batch.setup)

    total_traces = sum(b.count for b in batches)
    total_batches = len(batches)
    on_message = make_on_message(args.quiet, args.full_messages)

    all_trace_ids: list[str] = []
    trace_number = 0
    failed_traces = 0
    aborted_traces = 0

    # `groupby` merges consecutive same-setup batches so we open the shop once
    # per contiguous block; interleaved setups reopen the shop by design.
    batch_number = 0
    for setup_name, group in groupby(batches, key=lambda b: b.setup):
        coffee_shop_logger.info(f"=== Setup: {setup_name} ===")
        shop = CoffeeShop(
            CoffeeShopConfig(
                setup_name=setup_name,
            )
        )
        shop.open_shop(reset_inventory_first=args.reset_inventory)
        coffee_shop_logger.info(
            f"Resetting inventory before each trace: {args.reset_inventory}"
        )

        for batch in group:
            batch_number += 1
            log_status(
                f"=== Batch {batch_number}/{total_batches} | setup {setup_name!r} "
                f"| scenario {batch.scenario} | {batch.count} trace(s) ==="
            )
            for conv_number in range(1, batch.count + 1):
                trace_number += 1
                idx = resolve_scenario(batch.scenario, conv_number - 1)
                scenario_desc = CUSTOMER_SCENARIOS[idx] if idx is not None else "random"
                log_status(
                    f"--- Trace {trace_number}/{total_traces} "
                    f"(batch {batch_number}/{total_batches}, "
                    f"conv {conv_number}/{batch.count}) "
                    f"| Scenario {idx}: {scenario_desc[:60]} ---"
                )
                try:
                    trace_ids = shop.run_conversation(
                        scenario_index=idx,
                        on_message=on_message,
                        reset_inventory_first=args.reset_inventory,
                    )
                except Exception as exc:
                    if args.on_error == "abort":
                        raise
                    failed_traces += 1
                    coffee_shop_logger.error(
                        f"Trace {trace_number}/{total_traces} failed "
                        f"({type(exc).__name__}: {exc}) — skipping to the next trace"
                    )
                    continue
                all_trace_ids.extend(trace_ids)
                coffee_shop_logger.info(f"Trace IDs: {trace_ids}")

                feedback = shop.get_last_feedback()
                if feedback:
                    if feedback.get("aborted"):
                        aborted_traces += 1
                    coffee_shop_logger.info(
                        f"Customer feedback {format_feedback(feedback)}"
                    )

    log_status(
        f"=== Simulation complete: {total_batches} batch(es), "
        f"{len(all_trace_ids)} trace(s) generated, "
        f"{failed_traces} conversation(s) failed, "
        f"{aborted_traces} aborted (context overflow) ==="
    )
    for batch in batches:
        coffee_shop_logger.info(
            f"  - {batch.setup} / scenario {batch.scenario}: {batch.count} trace(s)"
        )

    if args.export_logs:
        coffee_shop_logger.info("Exporting event logs...")
        TraceProcessor().process_all_traces()

    return 0


if __name__ == "__main__":
    sys.exit(main())
