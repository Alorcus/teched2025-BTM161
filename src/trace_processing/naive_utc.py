"""Naive-UTC datetime helpers.

Event-log CSV timestamps are stored as naive-UTC ISO strings — the
wall-clock is UTC with no tzinfo attached. Downstream consumers
(`_load_combined_eventlog`, guardrail joins) all assume the string is UTC.

`datetime.fromtimestamp(epoch)` (no tz) returns naive-**LOCAL**, which
looks identical on inspection but silently shifts values by the machine's
timezone offset when re-parsed as UTC. `NaiveUTC` is a `NewType` guardrail
so callers must go through `to_naive_utc` / `from_epoch_naive_utc`.
"""
from datetime import datetime, timezone
from typing import NewType

NaiveUTC = NewType("NaiveUTC", datetime)


def to_naive_utc(dt: datetime) -> NaiveUTC:
    """Return `dt` as a naive-UTC datetime; naive inputs are assumed to
    already be UTC (caller contract)."""
    if dt.tzinfo is not None:
        return NaiveUTC(dt.astimezone(timezone.utc).replace(tzinfo=None))
    return NaiveUTC(dt)


def from_epoch_naive_utc(epoch: float) -> NaiveUTC:
    """Convert an epoch float to naive-UTC (safe replacement for
    `datetime.fromtimestamp(epoch)`, which returns naive-LOCAL)."""
    return NaiveUTC(
        datetime.fromtimestamp(epoch, tz=timezone.utc).replace(tzinfo=None)
    )
