import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("coffee_shop.control_plane.log_sink")


class JsonlLogSink:
    """Append-only JSONL log. Thread-safe via a process-local lock."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        record = {"ts": time.time(), **event}
        line = json.dumps(record, default=str, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        logger.debug("logged %s", record.get("event_type"))


class NullLogSink:
    """Used when logging is disabled in tests."""

    def append(self, event: dict[str, Any]) -> None:  # pragma: no cover
        pass
