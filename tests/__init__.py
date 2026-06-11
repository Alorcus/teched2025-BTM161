"""Test package — ensures a clean database before the suite runs."""

from pathlib import Path

_DB_PATH = Path(__file__).resolve().parents[1] / "coffee_shop.db"

# SQLite in WAL mode leaves -shm and -wal sidecars next to the main DB.
# An interrupted run can leave these in a state that makes a later open
# fail with "disk I/O error" even after the main file is recreated.
for _path in (
    _DB_PATH,
    _DB_PATH.with_name(_DB_PATH.name + "-shm"),
    _DB_PATH.with_name(_DB_PATH.name + "-wal"),
):
    if _path.exists():
        try:
            _path.unlink()
        except OSError:
            try:
                _path.write_bytes(b"")
                _path.unlink()
            except OSError:
                pass
