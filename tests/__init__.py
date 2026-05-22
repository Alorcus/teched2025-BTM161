"""Test package — ensures a clean database before the suite runs."""
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parents[1] / "coffee_shop.db"

if _DB_PATH.exists():
    try:
        _DB_PATH.unlink()
    except OSError:
        try:
            _DB_PATH.write_bytes(b"")
            _DB_PATH.unlink()
        except OSError:
            pass
