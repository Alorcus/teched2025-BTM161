"""Wipe all trace, event-log, and conversation state from the working tree.

Removes MLflow tracking state, generated event logs / OCELs / visualizations,
the coffee-shop SQLite, and auxiliary log directories. Leaves source files,
configuration, and notebooks untouched.

Run with `poetry run reset`. Use `--yes` to skip confirmation; useful for CI
or scripted resets between test runs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import psutil


# Paths are relative to the current working directory (the repo root when
# invoked via `poetry run reset`). Order doesn't matter — each entry is
# removed independently.
_TARGETS: tuple[str, ...] = (
    "mlflow.db",
    "mlflow.db-shm",
    "mlflow.db-wal",
    "mlruns",
    "generated_event_log",
    "generated_ocel",
    "generated_visualizations",
    "feedback_store.json",
    "process_log",
    "guardrail_log",
    "retrospective_log",
    "coffee_shop.db",
    "coffee_shop.db-shm",
    "coffee_shop.db-wal",
    "services/coffee_machine/logs",
)


def _remove(path: Path) -> str:
    """Delete `path` whether file or directory. Returns a status word for
    the summary line: removed / dir-removed / missing."""
    if not path.exists() and not path.is_symlink():
        return "missing"
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return "dir-removed"
    path.unlink()
    return "removed"


def _dashboard_running(port: int = 5006) -> list[psutil.Process]:
    """Return any process that holds the dashboard port. We block on this
    because deleting open SQLite/WAL files only unlinks the inode — the
    running process keeps writing and the DB reappears at the next
    checkpoint, masking the reset."""
    seen_pids: set[int] = set()
    holders: list[psutil.Process] = []
    try:
        connections = psutil.net_connections(kind="tcp")
    except (psutil.AccessDenied, PermissionError):
        return holders
    for conn in connections:
        if (
            conn.status == psutil.CONN_LISTEN
            and conn.laddr
            and conn.laddr.port == port
            and conn.pid is not None
            and conn.pid not in seen_pids
        ):
            try:
                holders.append(psutil.Process(conn.pid))
                seen_pids.add(conn.pid)
            except psutil.NoSuchProcess:
                pass
    return holders


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reset trace and event-log state for the coffee-shop project."
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything.",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    targets = [(name, cwd / name) for name in _TARGETS]
    present = [(name, p) for name, p in targets if p.exists() or p.is_symlink()]

    if not present:
        print("Nothing to reset — no trace/event-log state found.")
        return 0

    holders = _dashboard_running()
    if holders:
        print(
            "Refusing to reset: the dashboard appears to be running on port 5006.",
            file=sys.stderr,
        )
        for proc in holders:
            try:
                print(f"  pid={proc.pid} cmd={' '.join(proc.cmdline())}", file=sys.stderr)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        print(
            "Stop it first (Ctrl-C in its terminal, or `kill <pid>`), then re-run.",
            file=sys.stderr,
        )
        return 1

    print(f"The following will be removed from {cwd}:")
    for name, _ in present:
        print(f"  - {name}")

    if args.dry_run:
        print("\n(dry run — nothing deleted)")
        return 0

    if not args.yes:
        try:
            answer = input("\nProceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return 1

    removed = 0
    for name, path in present:
        status = _remove(path)
        print(f"  {status:>12}  {name}")
        if status != "missing":
            removed += 1

    print(f"\nReset complete — {removed} item(s) removed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
