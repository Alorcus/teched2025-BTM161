"""Run a configured list of simulation batches.

Each batch is a (setup, scenario, count) tuple. Consecutive batches sharing
a setup reuse the same CoffeeShop instance, so keep same-setup entries next
to each other in the list to pay the init cost once per contiguous block.

The batch list and flags can be driven three ways (in precedence order):
  1. `--batches setup:scenario:count ...` on the command line
  2. `--config path/to/config.json` (mutually exclusive with `--batches`)
  3. The module-level `BATCHES` / `RESET_INVENTORY` / `PROCESS_SUPERVISOR`
     / `EXPORT_LOGS` defaults below

Run from the repo root:
    python -m scripts.run_batches
    python -m scripts.run_batches --batches baseline:0:1
    python -m scripts.run_batches --config path/to/batches.json
"""

import argparse
import json
import logging
import sys
from itertools import groupby
from pathlib import Path

from src.agents.customer_agent import CUSTOMER_SCENARIOS
from src.coffee_shop import CoffeeShop
from src.config import CoffeeShopConfig
from src.trace_processing import TraceProcessor


BATCHES: list[tuple[str, int, int]] = [
    ("baseline", 0, 50),
    ("baseline", 2, 50),
    ("baseline", 3, 50),
    ("unconstrained", 0, 50),
    ("unconstrained", 2, 50),
    ("unconstrained", 3, 50),
]

RESET_INVENTORY = True
PROCESS_SUPERVISOR = False
EXPORT_LOGS = False

logger = logging.getLogger("coffee_shop")


def _parse_triple(raw: str) -> tuple[str, int, int]:
    """Parse a single `setup:scenario:count` triple."""
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--batches expects setup:scenario:count triples, got '{raw}'"
        )
    setup, scenario_str, count_str = parts
    if not setup:
        raise argparse.ArgumentTypeError(
            f"--batches expects setup:scenario:count triples, got '{raw}' (empty setup)"
        )
    try:
        scenario = int(scenario_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--batches expects setup:scenario:count triples, got '{raw}' "
            f"(scenario must be an int)"
        )
    try:
        count = int(count_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--batches expects setup:scenario:count triples, got '{raw}' "
            f"(count must be an int)"
        )
    if count <= 0:
        raise argparse.ArgumentTypeError(
            f"--batches expects setup:scenario:count triples, got '{raw}' "
            f"(count must be positive)"
        )
    return setup, scenario, count


def _parse_batches_arg(values: list[str]) -> list[tuple[str, int, int]]:
    """Accept either repeated triples or a single comma-separated string."""
    triples: list[tuple[str, int, int]] = []
    for value in values:
        for chunk in value.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            triples.append(_parse_triple(chunk))
    return triples


def _load_config(path: Path) -> tuple[
    list[tuple[str, int, int]] | None,
    bool | None,
    bool | None,
    bool | None,
]:
    """Load a JSON config file. Returns (batches, reset, supervisor, export)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a JSON object")

    batches_raw = data.get("batches")
    batches: list[tuple[str, int, int]] | None = None
    if batches_raw is not None:
        if not isinstance(batches_raw, list):
            raise ValueError("config 'batches' must be a list of [setup, scenario, count]")
        batches = []
        for entry in batches_raw:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 3
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], int)
                or not isinstance(entry[2], int)
            ):
                raise ValueError(
                    f"config 'batches' entry must be [setup:str, scenario:int, count:int], "
                    f"got {entry!r}"
                )
            setup, scenario, count = entry
            if count <= 0:
                raise ValueError(f"config 'batches' count must be positive, got {count}")
            batches.append((setup, scenario, count))

    reset = data.get("reset_inventory")
    supervisor = data.get("process_supervisor")
    export = data.get("export_logs")
    for name, value in (
        ("reset_inventory", reset),
        ("process_supervisor", supervisor),
        ("export_logs", export),
    ):
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"config '{name}' must be a boolean, got {value!r}")

    return batches, reset, supervisor, export


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.run_batches",
        description=(
            "Run a configured list of simulation batches. "
            "With no flags, uses the module-level defaults."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--batches",
        nargs="+",
        metavar="SETUP:SCENARIO:COUNT",
        help=(
            "One or more setup:scenario:count triples "
            "(e.g. --batches baseline:0:50 baseline:2:50). "
            "May also be a single comma-separated string."
        ),
    )
    source.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help=(
            "Path to a JSON config file with schema "
            '{"batches": [["baseline", 0, 50], ...], "reset_inventory": true, '
            '"process_supervisor": false, "export_logs": false}. '
            "Mutually exclusive with --batches."
        ),
    )
    parser.add_argument(
        "--reset-inventory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reset inventory between conversations (default: True).",
    )
    parser.add_argument(
        "--process-supervisor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the process supervisor (default: False).",
    )
    parser.add_argument(
        "--export-logs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export event logs after all batches complete (default: False).",
    )
    return parser


def _resolve_settings(
    args: argparse.Namespace,
) -> tuple[list[tuple[str, int, int]], bool, bool, bool]:
    """Merge CLI args, optional config file, and module defaults."""
    batches: list[tuple[str, int, int]] = BATCHES
    reset = RESET_INVENTORY
    supervisor = PROCESS_SUPERVISOR
    export = EXPORT_LOGS

    if args.config is not None:
        cfg_batches, cfg_reset, cfg_supervisor, cfg_export = _load_config(args.config)
        if cfg_batches is not None:
            batches = cfg_batches
        if cfg_reset is not None:
            reset = cfg_reset
        if cfg_supervisor is not None:
            supervisor = cfg_supervisor
        if cfg_export is not None:
            export = cfg_export
    elif args.batches is not None:
        batches = _parse_batches_arg(args.batches)

    # CLI boolean flags override config + module defaults when supplied.
    if args.reset_inventory is not None:
        reset = args.reset_inventory
    if args.process_supervisor is not None:
        supervisor = args.process_supervisor
    if args.export_logs is not None:
        export = args.export_logs

    return batches, reset, supervisor, export


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = _build_parser()
    args = parser.parse_args(argv)
    batches_list, reset_inventory, process_supervisor, export_logs = _resolve_settings(
        args
    )

    for setup, scenario, count in batches_list:
        if not (0 <= scenario < len(CUSTOMER_SCENARIOS)):
            logger.error(
                f"scenario {scenario} out of range 0-{len(CUSTOMER_SCENARIOS) - 1}"
            )
            return 1

    all_trace_ids: list[str] = []
    per_batch_counts: list[tuple[str, int, int, int]] = []
    total_batches = len(batches_list)
    total_traces = sum(count for _, _, count in batches_list)
    batch_number = 0
    trace_number = 0

    for setup_name, batches in groupby(batches_list, key=lambda b: b[0]):
        batches = list(batches)
        logger.info(f"=== Setup: {setup_name} ===")
        shop = CoffeeShop(
            CoffeeShopConfig(
                setup_name=setup_name,
                process_supervisor_enabled=process_supervisor,
            )
        )
        shop.open_shop(reset_inventory_first=reset_inventory)

        for _, scenario, count in batches:
            batch_number += 1
            label = CUSTOMER_SCENARIOS[scenario]
            logger.info(
                f"=== Batch {batch_number}/{total_batches} | setup '{setup_name}' "
                f"| scenario {scenario}: {label[:60]} | {count} trace(s) ==="
            )
            batch_trace_ids: list[str] = []
            for i in range(count):
                trace_number += 1
                logger.info(
                    f"--- Batch {batch_number}/{total_batches} "
                    f"| conversation {i + 1}/{count} "
                    f"| trace {trace_number}/{total_traces} ---"
                )
                trace_ids = shop.run_conversation(
                    scenario_index=scenario,
                    on_message=None,
                    reset_inventory_first=reset_inventory,
                )
                batch_trace_ids.extend(trace_ids)
            all_trace_ids.extend(batch_trace_ids)
            per_batch_counts.append(
                (setup_name, scenario, count, len(batch_trace_ids))
            )

    logger.info(
        f"=== Complete: {len(batches_list)} batch(es), {len(all_trace_ids)} trace(s) ==="
    )
    for setup_name, scenario, requested, actual in per_batch_counts:
        logger.info(
            f"  - {setup_name} / scenario {scenario}: {actual} trace(s) "
            f"(requested {requested})"
        )

    if export_logs:
        logger.info("Exporting event logs...")
        TraceProcessor().process_all_traces()

    return 0


if __name__ == "__main__":
    sys.exit(main())
