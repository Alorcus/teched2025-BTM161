import csv
import logging
import threading
from pathlib import Path

logger = logging.getLogger("coffee_shop.coffee_machine.logger")

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "coffee_machine.csv"


FIXED_HEADER = [
    "case_id",
    "concept:name",
    "ocel_time",
    "duration",
    "org:resource",
    "job_id",
    "drink",
]
ALLOWED_ATTRS = {"job_id", "drink"}

_log_lock = threading.Lock()


def log_event(case_id: str, activity: str, timestamp: float, duration: float = None, **attrs):
    unknown = set(attrs) - ALLOWED_ATTRS
    if unknown:
        raise ValueError(
            f"log_event got unknown kwargs: {sorted(unknown)}. "
            f"Allowed: {sorted(ALLOWED_ATTRS)}."
        )

    with _log_lock:
        try:
            file_has_content = LOG_PATH.is_file() and LOG_PATH.stat().st_size > 0

            with open(LOG_PATH, "a", newline="") as f:
                writer = csv.writer(f)

                if not file_has_content:
                    writer.writerow(FIXED_HEADER)

                writer.writerow([
                    case_id,
                    activity,
                    timestamp,
                    duration,
                    "coffee_machine",
                    attrs.get("job_id", ""),
                    attrs.get("drink", ""),
                ])
        except OSError as e:
            logger.error("Failed to write OCEL event to CSV: %s", e)
