import csv
import os
import logging
import threading
from pathlib import Path

logger = logging.getLogger("coffee_shop.coffee_machine.logger")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "coffee_machine.csv"

_log_lock = threading.Lock()


def log_event(case_id: str, activity: str, timestamp: float, duration: float = None, **attrs):
    with _log_lock:
        try:
            file_exists = os.path.isfile(LOG_PATH)

            with open(LOG_PATH, "a", newline="") as f:
                writer = csv.writer(f)

                if not file_exists:
                    header = [
                        "case_id",
                        "concept:name",
                        "ocel_time",
                        "duration",
                        "org:resource"
                    ]
                    header += list(attrs.keys())
                    writer.writerow(header)

                row = [
                    case_id,
                    activity,
                    timestamp,
                    duration,
                    "coffee_machine"
                ]
                row += list(attrs.values())
                writer.writerow(row)
        except OSError as e:
            logger.error("Failed to write OCEL event to CSV: %s", e)
