"""On-demand cache of MLflow traces as a single shareable CSV.

The Metrics Dashboard used to read whatever event-log CSVs the user had
manually exported from the (now-removed) export button on the Interaction
Observatory. With that button gone, the dashboard would otherwise be blind
to any conversation that wasn't explicitly exported. This module materializes
MLflow traces into one canonical CSV (`_all_traces.csv`) on demand.

The file is designed to be **shared**: a user can hand a curated CSV to
colleagues, who then get to explore those traces even though the underlying
MLflow store isn't on their machine. To make that work, the cache is
maintained in **append** mode:

- New MLflow traces whose `case_id` isn't in the CSV yet → their events are
  extracted and appended.
- MLflow traces whose `case_id` is already in the CSV → skipped (no re-work,
  no rewrite of existing rows).
- Rows in the CSV whose `case_id` doesn't exist in the local MLflow store
  (e.g. imported from a colleague) → preserved untouched.

Trigger: `ensure_trace_cache()` runs on Metrics Dashboard page entry.
Escape hatch: bumping `_SCHEMA_VERSION` forces a full rebuild, because a row
shape change would otherwise leave the file with mixed-version rows.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.trace_processing.trace_processor import TraceProcessor

# Fixed filename so the rest of the loader can ignore everything else in the
# directory — manually-exported CSVs are intentionally not part of the data
# source any more.
CACHE_FILENAME = "_all_traces.csv"
# Sidecar recording the extractor's schema version and (informational only)
# the MLflow trace count from the last sync. The schema version is the real
# staleness signal now that we append instead of rebuild.
META_FILENAME = "_all_traces.meta"
# Extractor schema version. Bump when LogGenerator's row shape changes so
# already-built caches from an older extractor are treated as stale even when
# nothing else moved. History:
#   1 → initial (pre-transfer-to-* fix)
#   2 → transfer_to_* handovers emitted as execute_tool rows
#   3 → modern MLflow LangChain autolog: `llm` spans recognised; call_llm
#       rows now populated (previously always zero for the current autolog)
#   4 → case_setup + case_scenario_index columns propagated from MLflow
#       trace tags so the metrics dashboard can filter by them
_SCHEMA_VERSION = 4


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


def _cached_meta(meta_path: Path) -> tuple[int, int]:
    """Return (trace_count, schema_version) recorded in the sidecar. Returns
    (0, 0) if the file is missing, unreadable, or in the old single-line
    format — either way the caller falls back to the schema-version rule."""
    if not meta_path.exists():
        return (0, 0)
    try:
        raw = meta_path.read_text().strip()
    except OSError:
        return (0, 0)
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return (0, 0)
    try:
        trace_count = int(lines[0].strip())
    except ValueError:
        return (0, 0)
    schema_version = 0
    if len(lines) >= 2:
        try:
            schema_version = int(lines[1].strip())
        except ValueError:
            schema_version = 0
    return (trace_count, schema_version)


def _write_cache_meta(meta_path: Path, trace_count: int) -> None:
    try:
        meta_path.write_text(f"{trace_count}\n{_SCHEMA_VERSION}\n")
    except OSError:
        pass


def _existing_case_ids(cache_path: Path) -> set[str]:
    """Read just the `case_id` column from the existing cache. Empty set if
    the file doesn't exist yet or the column is missing. Uses polars'
    lazy scan so the cost stays linear in the case_id column, not the row
    body — cheap even on large curated files."""
    if not cache_path.exists():
        return set()
    try:
        df = pl.scan_csv(str(cache_path)).select("case_id").collect()
    except Exception:
        # Corrupt cache or missing column → treat as empty so a fresh sync
        # can repopulate. We do NOT delete the file here; if the extraction
        # produces zero new rows the caller will leave the file alone.
        return set()
    return {str(v) for v in df["case_id"].drop_nulls().unique().to_list()}


def _sync_cache(log_dir: Path, tracking_uri: str, mlflow_count: int) -> None:
    """Append-mode sync: extract only MLflow traces whose case_id isn't
    already in the cache and union the result with what's already on disk.

    Rows for case_ids that exist in the CSV but not in MLflow (e.g. imported
    files) are preserved. Timestamped per-run CSVs from older versions of
    this module are cleaned up so the directory only holds the canonical
    cache.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_path = log_dir / CACHE_FILENAME
    meta_path = log_dir / META_FILENAME

    covered = _existing_case_ids(cache_path)

    processor = TraceProcessor(tracking_uri=tracking_uri)
    new_df, _tags, _new_ids = processor.extract_new_traces(covered)

    # Clean up per-run CSVs left over from older cache behavior. We match the
    # old rebuild path: anything in log_dir that's not the canonical file.
    for p in log_dir.glob("*.csv"):
        if p.name == CACHE_FILENAME:
            continue
        try:
            p.unlink()
        except OSError:
            pass

    if new_df.empty:
        # Nothing new to add; keep the sidecar current so future runs see the
        # latest MLflow count.
        _write_cache_meta(meta_path, mlflow_count)
        return

    new_frame = pl.from_pandas(new_df)

    if cache_path.exists():
        try:
            existing_frame = pl.read_csv(str(cache_path), infer_schema_length=10_000)
            combined = pl.concat(
                [existing_frame, new_frame], how="diagonal_relaxed"
            )
        except Exception:
            # If the existing file is unreadable, don't destroy it silently —
            # write the new slice under a sibling name and leave the original
            # for the user to inspect.
            fallback = log_dir / f"{CACHE_FILENAME}.new"
            new_frame.write_csv(str(fallback))
            _write_cache_meta(meta_path, mlflow_count)
            return
    else:
        combined = new_frame

    # Sort by timestamp so downstream readers still see chronological order.
    if "time:timestamp" in combined.columns:
        combined = combined.sort("time:timestamp")

    combined.write_csv(str(cache_path))
    _write_cache_meta(meta_path, mlflow_count)


def _full_rebuild(log_dir: Path, tracking_uri: str, mlflow_count: int) -> None:
    """Schema-version bump escape hatch. Delete the existing cache and re-run
    the sync with an empty covered set. Necessary because rows in a
    version-N file are shape-incompatible with rows produced by a later
    extractor; mixing them in one CSV breaks the dashboard."""
    cache_path = log_dir / CACHE_FILENAME
    if cache_path.exists():
        try:
            cache_path.unlink()
        except OSError:
            pass
    print(
        "ℹ️  Extractor schema changed; rebuilding _all_traces.csv from MLflow."
    )
    _sync_cache(log_dir, tracking_uri, mlflow_count)


def ensure_trace_cache(
    log_dir: Path,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> Path | None:
    """Make sure `_all_traces.csv` covers every trace currently in MLflow.

    Append semantics: new case_ids are added, existing rows are left alone,
    imported rows without a matching MLflow trace are preserved. Returns the
    cache path if it exists after the sync, or None when MLflow has no
    traces AND there is no imported cache to fall back on (caller renders
    the empty-state template).
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
        # Nothing to append. Preserve any imported CSV so users who received
        # a shared file can still explore it without a local MLflow store.
        return cache_path if cache_path.exists() else None

    _cached_count, cached_schema = _cached_meta(meta_path)
    if cached_schema and cached_schema != _SCHEMA_VERSION and cache_path.exists():
        _full_rebuild(log_dir, tracking_uri, mlflow_count)
    else:
        _sync_cache(log_dir, tracking_uri, mlflow_count)

    return cache_path if cache_path.exists() else None
