import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("coffee_shop.control_plane.log_sink")


class JsonlLogSink:
    """Append-only JSONL log. Thread-safe via a process-local lock.

    Every record is stamped with `setup_name` so events from different
    experiment runs can be filtered apart in the log.
    """

    def __init__(self, path: str | os.PathLike, setup_name: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.setup_name = setup_name
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        record = {"ts": time.time(), "setup_name": self.setup_name, **event}
        line = json.dumps(record, default=str, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        if record.get("event_type") == "gateway_decision":
            logger.debug(
                f"logged {record.get('event_type')} - {record.get('final_decision')} - {record.get('tool_name')}: {record.get('tool_args')}",
            )
        elif record.get("event_type") == "tool_execution":
            logger.debug(
                f"logged {record.get('event_type')} - {record.get('tool_name')}: {record.get('tool_args')}",
            )


class NullLogSink:
    """Used when logging is disabled in tests."""

    def append(self, event: dict[str, Any]) -> None:  # pragma: no cover
        pass
