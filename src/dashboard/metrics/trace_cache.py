"""On-demand cache of all MLflow traces as a single CSV.

The Metrics Dashboard used to read whatever event-log CSVs the user had
manually exported from the (now-removed) export button on the Interaction
Observatory. With that button gone, the dashboard would otherwise be blind
to any conversation that wasn't explicitly exported. This module materializes
every MLflow trace into one canonical CSV (`_all_traces.csv`) on demand and
keeps it in sync with the MLflow store via a row-count check.

Trigger: `ensure_trace_cache()` runs on Metrics Dashboard page entry.
Staleness rule: MLflow trace count != distinct case_id count in the cache.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.trace_processing.trace_processor import TraceProcessor

# Fixed filename so the rest of the loader can ignore everything else in the
# directory — manually-exported CSVs are intentionally not part of the data
# source any more.
CACHE_FILENAME = "_all_traces.csv"
# Sidecar that records the MLflow trace count this cache was built from.
# We can't use distinct case_id count as the staleness signal because some
# traces produce zero events (feedback-only traces with no LangGraph root)
# and therefore contribute no case_id rows — comparing event-side cases to
# MLflow's trace count would always disagree and force a rebuild every load.
META_FILENAME = "_all_traces.meta"


def _mlflow_trace_count(tracking_uri: str) -> int:
    """Return the number of traces visible to the MLflow client. Microseconds
    against the local sqlite — cheaper than re-running the full export."""
    import mlflow

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiments = client.search_experiments()
    total = 0
    for exp in experiments:
        page_token = None
        while True:
            result = client.search_traces(
                experiment_ids=[exp.experiment_id],
                max_results=100,
                page_token=page_token,
            )
            total += len(result)
            if not result.token:
                break
            page_token = result.token
    return total


def _cached_trace_count(meta_path: Path) -> int:
    """Trace count this cache was last built from, read from the sidecar
    metadata file. Returns 0 if the file is missing or unreadable so the
    caller treats the cache as stale."""
    if not meta_path.exists():
        return 0
    try:
        return int(meta_path.read_text().strip())
    except (OSError, ValueError):
        return 0


def _write_cache_meta(meta_path: Path, trace_count: int) -> None:
    try:
        meta_path.write_text(f"{trace_count}\n")
    except OSError:
        pass


def _rebuild_cache(log_dir: Path, tracking_uri: str, trace_count: int) -> None:
    """Re-run TraceProcessor and consolidate its output into the single
    cache CSV. TraceProcessor writes a timestamped per-run CSV; we read every
    timestamped CSV in `log_dir`, union them, and write `_all_traces.csv`.
    The timestamped originals are removed so the directory only ever holds
    the canonical cache. Writes `_all_traces.meta` with the source trace
    count so the next staleness check has an accurate comparison point."""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot pre-existing timestamped CSVs so we can clean them up after the
    # rebuild. The cache itself is kept (we'll overwrite it).
    pre_existing = [
        p for p in log_dir.glob("*.csv") if p.name != CACHE_FILENAME
    ]

    processor = TraceProcessor(tracking_uri=tracking_uri)
    processor.process_all_traces()

    # Union every timestamped CSV plus any earlier cache.
    csvs = [p for p in log_dir.glob("*.csv") if p.name != CACHE_FILENAME]
    frames: list[pl.DataFrame] = []
    for path in csvs:
        try:
            frames.append(pl.read_csv(str(path), infer_schema_length=10_000))
        except Exception:
            continue
    cache_path = log_dir / CACHE_FILENAME
    meta_path = log_dir / META_FILENAME
    if not frames:
        # TraceProcessor produced nothing — record the source count anyway so
        # repeated entries with the same (empty) state don't keep rebuilding.
        _write_cache_meta(meta_path, trace_count)
        return

    combined = pl.concat(frames, how="diagonal_relaxed")
    combined.write_csv(str(cache_path))
    _write_cache_meta(meta_path, trace_count)

    # Remove the per-run CSVs (both the ones from before this rebuild and the
    # one TraceProcessor just produced) so the directory stays clean.
    for p in pre_existing + csvs:
        if p.name == CACHE_FILENAME:
            continue
        try:
            p.unlink()
        except OSError:
            pass


def ensure_trace_cache(
    log_dir: Path,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> Path | None:
    """Make sure `_all_traces.csv` reflects every trace currently in MLflow.

    Returns the cache path on success, or None if MLflow has no traces (the
    caller should render the empty-state template).

    Staleness compares MLflow's current trace count to the count recorded in
    `_all_traces.meta` at the last successful rebuild. Comparing against the
    cached *case* count would always disagree because some traces produce zero
    events (e.g. feedback-only traces with no LangGraph root), which would
    force a rebuild on every page entry.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_path = log_dir / CACHE_FILENAME
    meta_path = log_dir / META_FILENAME

    try:
        mlflow_count = _mlflow_trace_count(tracking_uri)
    except Exception:
        # MLflow unavailable / DB locked — fall back to whatever the cache
        # already has. Better to render stale data than nothing.
        return cache_path if cache_path.exists() else None

    if mlflow_count == 0:
        return None

    cached_count = _cached_trace_count(meta_path)
    if cached_count != mlflow_count:
        _rebuild_cache(log_dir, tracking_uri, mlflow_count)

    return cache_path if cache_path.exists() else None
