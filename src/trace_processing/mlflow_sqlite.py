"""Direct SQLite access for sqlite-backed MLflow tracking stores.

MLflow's `MlflowClient.search_traces()` paginates in 100-row pages; even
answering "how many traces are there?" costs a full walk. On a local
sqlite store the same information sits behind two SQL queries against
`trace_info` and `trace_request_metadata`. This module exposes those
queries so `trace_cache` can skip the paginated client path on the hot
loop.

Non-sqlite tracking URIs (mysql, postgres) are not handled here — callers
must fall back to the MLflow client.
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# MLflow accepts both `sqlite:///abs/path` (three slashes → absolute) and
# `sqlite:///relative` (three slashes → relative). urlsplit-with-scheme
# has no reliable way to distinguish those, so match by regex and
# probe both interpretations.
_SQLITE_URI_RE = re.compile(r"^sqlite:/{2,3}(?P<path>.+)$")


def sqlite_path_from_uri(tracking_uri: str) -> Path | None:
    """Return the .db path for a sqlite tracking URI, or None if not sqlite."""
    m = _SQLITE_URI_RE.match(tracking_uri)
    if not m:
        return None
    raw = m.group("path")
    if raw.startswith("/") and not Path(raw).exists() and Path(raw.lstrip("/")).exists():
        raw = raw.lstrip("/")
    return Path(raw)


@contextmanager
def _open_ro(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Read-only connection with a short busy timeout so a live MLflow
    writer never blocks us for long."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    try:
        yield conn
    finally:
        conn.close()


def sqlite_trace_count(db_path: Path) -> int:
    with _open_ro(db_path) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM trace_info").fetchone()
    return int(count)


def sqlite_request_id_thread_map(db_path: Path) -> dict[str, str | None]:
    """Return `{request_id: thread_id or None}` for every trace.

    thread_id comes from `trace_request_metadata` under the key
    `mlflow.trace.session` — MLflow's LangChain autolog stores the
    LangGraph thread_id there. A trace with no session metadata maps
    to None; the caller either materializes it to peek from the span
    tree or looks it up in a "no-session" ledger.
    """
    with _open_ro(db_path) as conn:
        rows = conn.execute(
            """
            SELECT ti.request_id, trm.value
              FROM trace_info ti
              LEFT JOIN trace_request_metadata trm
                ON ti.request_id = trm.request_id
               AND trm.key = 'mlflow.trace.session'
            """
        ).fetchall()
    return {rid: (tid if tid else None) for rid, tid in rows}


def sqlite_new_request_ids(
    db_path: Path,
    covered_case_ids: set[str],
    no_session_ledger: set[str],
) -> tuple[list[str], dict[str, str | None]]:
    """Return `(new_request_ids, request_id_to_thread_id_map)`.

    A request_id is "new" when:
      - it has a thread_id AND the thread_id isn't in `covered_case_ids`, OR
      - it has no thread_id AND the request_id isn't in `no_session_ledger`
        (we haven't already fetched and confirmed it produces no case_id).

    The ledger is the fix for the ~1292 non-LangGraph traces (feedback
    calls, standalone ChatAnthropic) that would otherwise be flagged
    "new" on every sync.
    """
    mapping = sqlite_request_id_thread_map(db_path)
    new_ids: list[str] = []
    for rid, tid in mapping.items():
        if tid is not None:
            if tid not in covered_case_ids:
                new_ids.append(rid)
        else:
            if rid not in no_session_ledger:
                new_ids.append(rid)
    return new_ids, mapping
