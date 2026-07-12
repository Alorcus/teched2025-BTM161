"""Run a configured list of simulation batches.

Each batch is a (setup, scenario, count) tuple. Batches sharing a setup reuse
the same CoffeeShop instance so we only pay the init cost once per setup.

Edit BATCHES below, then run from the repo root:
    python -m scripts.run_batches
"""

import logging
import sys
from itertools import groupby

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


def main() -> int:
    logger.setLevel(logging.INFO)

    for setup, scenario, count in BATCHES:
        if not (0 <= scenario < len(CUSTOMER_SCENARIOS)):
            logger.error(
                f"scenario {scenario} out of range 0-{len(CUSTOMER_SCENARIOS) - 1}"
            )
            return 1

    all_trace_ids: list[str] = []
    per_batch_counts: list[tuple[str, int, int, int]] = []
    total_batches = len(BATCHES)
    total_traces = sum(count for _, _, count in BATCHES)
    batch_number = 0
    trace_number = 0

    for setup_name, batches in groupby(BATCHES, key=lambda b: b[0]):
        batches = list(batches)
        logger.info(f"=== Setup: {setup_name} ===")
        shop = CoffeeShop(
            CoffeeShopConfig(
                setup_name=setup_name,
                process_supervisor_enabled=PROCESS_SUPERVISOR,
            )
        )
        shop.open_shop(reset_inventory_first=RESET_INVENTORY)

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
                    reset_inventory_first=RESET_INVENTORY,
                )
                batch_trace_ids.extend(trace_ids)
            all_trace_ids.extend(batch_trace_ids)
            per_batch_counts.append(
                (setup_name, scenario, count, len(batch_trace_ids))
            )

    logger.info(
        f"=== Complete: {len(BATCHES)} batch(es), {len(all_trace_ids)} trace(s) ==="
    )
    for setup_name, scenario, requested, actual in per_batch_counts:
        logger.info(
            f"  - {setup_name} / scenario {scenario}: {actual} trace(s) "
            f"(requested {requested})"
        )

    if EXPORT_LOGS:
        logger.info("Exporting event logs...")
        TraceProcessor().process_all_traces()

    return 0


if __name__ == "__main__":
    sys.exit(main())
