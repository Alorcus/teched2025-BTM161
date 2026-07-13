"""Naive-UTC datetime helpers.

The event-log CSV timestamps are stored as **naive-UTC** ISO strings — the
wall-clock is UTC but no tzinfo is attached. That shape is load-bearing:
polars' `str.to_datetime()` yields naive-UTC values, downstream consumers
(`_load_combined_eventlog`, guardrail joins) all assume the string is UTC.

Using `datetime.fromtimestamp(epoch)` (no tz) returns naive-**LOCAL**, which
looks identical on inspection but silently shifts values by the machine's
timezone offset when re-parsed as UTC. `NaiveUTC` is a `NewType` guardrail:
functions that require the CSV shape can type-annotate their inputs, and
callers must go through `to_naive_utc` / `from_epoch_naive_utc` to produce
one, which forces the correct construction pattern.
"""
from datetime import datetime, timezone
from typing import NewType

NaiveUTC = NewType("NaiveUTC", datetime)


def to_naive_utc(dt: datetime) -> NaiveUTC:
    """Return `dt` as a naive-UTC datetime, converting from any timezone-aware
    input and asserting naive inputs are already UTC (caller contract)."""
    if dt.tzinfo is not None:
        return NaiveUTC(dt.astimezone(timezone.utc).replace(tzinfo=None))
    return NaiveUTC(dt)


def from_epoch_naive_utc(epoch: float) -> NaiveUTC:
    """Convert an epoch float to naive-UTC (safe replacement for the
    `datetime.fromtimestamp(epoch)` pattern which returns naive-LOCAL)."""
    return NaiveUTC(
        datetime.fromtimestamp(epoch, tz=timezone.utc).replace(tzinfo=None)
    )
