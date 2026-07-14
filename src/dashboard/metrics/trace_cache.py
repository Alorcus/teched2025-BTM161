"""On-demand cache of MLflow traces as a single shareable CSV.

Materializes MLflow traces into `_all_traces.csv` on demand, in append mode
so a curated CSV can be handed to colleagues without a local MLflow store:

- New MLflow case_ids → extracted and appended.
- case_ids already in the CSV → skipped.
- Rows without a matching MLflow trace (imported from a colleague) →
  preserved untouched.

Trigger: `ensure_trace_cache()` runs on Metrics Dashboard page entry.

Quarantine: on schema-version mismatch or missing sidecar with non-empty
cache, the existing CSV is renamed to a versioned backup rather than
overwritten, so imported rows are never silently destroyed. The user can
merge them back manually once compatibility is verified.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.trace_processing.trace_processor import TraceProcessor

logger = logging.getLogger(__name__)

CACHE_FILENAME = "_all_traces.csv"
META_FILENAME = "_all_traces.meta"
# Extractor schema version. Bump when LogGenerator's row shape changes so
# older caches are quarantined instead of appended to. History:
#   1 → initial (pre-transfer-to-* fix)
#   2 → transfer_to_* handovers emitted as execute_tool rows
#   3 → modern MLflow LangChain autolog: `llm` spans recognised; call_llm
#       rows now populated (previously always zero for the current autolog)
#   4 → case_setup + case_scenario_index columns propagated from MLflow
#       trace tags so the metrics dashboard can filter by them
#   5 → gateway_decision rows folded in from guardrail_log/events.jsonl
#   6 → gateway_decision rows written as naive-UTC (v5 wrote naive-LOCAL
#       and got double-shifted by _load_combined_eventlog's UTC→local
#       conversion on non-UTC hosts)
_SCHEMA_VERSION = 6


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
                locations=[exp.experiment_id],
                max_results=100,
                page_token=page_token,
            )
            total += len(result)
            if not result.token:
                break
            page_token = result.token
    return total


def _cached_schema_version(meta_path: Path) -> int:
    """Return the schema version recorded in the sidecar, or 0 if missing.

    Legacy sidecars had two lines (`{trace_count}\\n{schema_version}\\n`);
    current sidecars have one. When two lines are present we read line 2.
    """
    if not meta_path.exists():
        return 0
    try:
        raw = meta_path.read_text().strip()
    except OSError:
        return 0
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return 0
    candidate = lines[1] if len(lines) >= 2 else lines[0]
    try:
        return int(candidate.strip())
    except ValueError:
        return 0


def _write_cache_schema(meta_path: Path) -> None:
    try:
        meta_path.write_text(f"{_SCHEMA_VERSION}\n")
    except OSError:
        pass


def _quarantine_path(log_dir: Path, label: str) -> Path:
    """Return an unused path `_all_traces.{label}[.N].csv` — never overwrites
    an existing backup."""
    primary = log_dir / f"_all_traces.{label}.csv"
    if not primary.exists():
        return primary
    n = 1
    while True:
        candidate = log_dir / f"_all_traces.{label}.{n}.csv"
        if not candidate.exists():
            return candidate
        n += 1


def _existing_case_ids(cache_path: Path) -> set[str]:
    """Read just the `case_id` column from the existing cache; empty set on
    missing file or read failure."""
    if not cache_path.exists():
        return set()
    try:
        df = pl.scan_csv(str(cache_path)).select("case_id").collect()
    except (pl.exceptions.PolarsError, OSError) as e:
        # Corrupt/missing column → treat as empty, but don't delete: if the
        # extraction produces zero new rows we still want the original file
        # left in place for the user to inspect.
        logger.warning(
            "Could not read case_id column from %s: %s. "
            "Treating cache as empty for the purposes of this sync.",
            cache_path,
            e,
        )
        return set()
    return {str(v) for v in df["case_id"].drop_nulls().unique().to_list()}


def _sync_cache(log_dir: Path, tracking_uri: str, mlflow_count: int) -> None:
    """Append-mode sync: extract only MLflow traces whose case_id isn't
    already in the cache and union the result with what's already on disk."""
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_path = log_dir / CACHE_FILENAME
    meta_path = log_dir / META_FILENAME

    covered = _existing_case_ids(cache_path)

    processor = TraceProcessor(tracking_uri=tracking_uri)
    new_df, _tags, _new_ids = processor.extract_new_traces(covered)

    # Clean up per-run CSVs from older cache behavior. Must skip anything
    # starting with the cache stem so quarantine backups (`_all_traces.v5.csv`,
    # `_all_traces.unknown-schema.csv`, etc.) aren't collected as leftover.
    cache_stem = Path(CACHE_FILENAME).stem  # "_all_traces"
    for p in log_dir.glob("*.csv"):
        if p.name == CACHE_FILENAME:
            continue
        if p.name.startswith(f"{cache_stem}."):
            continue
        try:
            p.unlink()
        except OSError:
            pass

    if new_df.is_empty():
        _write_cache_schema(meta_path)
        return

    new_frame = new_df

    if cache_path.exists():
        try:
            existing_frame = pl.read_csv(str(cache_path), infer_schema_length=10_000)
            combined = pl.concat(
                [existing_frame, new_frame], how="diagonal_relaxed"
            )
        except (pl.exceptions.PolarsError, OSError) as e:
            # Existing file unreadable — write the new slice under a sibling
            # name rather than overwriting the original.
            logger.warning(
                "Existing cache %s is unreadable (%s); writing new slice to "
                "%s.new and leaving the original in place.",
                cache_path,
                e,
                cache_path,
            )
            fallback = log_dir / f"{CACHE_FILENAME}.new"
            new_frame.write_csv(str(fallback))
            _write_cache_schema(meta_path)
            return
    else:
        combined = new_frame

    if "time:timestamp" in combined.columns:
        combined = combined.sort("time:timestamp")

    combined.write_csv(str(cache_path))
    _write_cache_schema(meta_path)


def _quarantine_and_resync(
    log_dir: Path,
    tracking_uri: str,
    mlflow_count: int,
    label: str,
    reason: str,
) -> None:
    """Rename the existing cache under a quarantine name, then sync fresh.

    Rename happens BEFORE the sync so a subsequent sync failure can't lose
    imported rows — they're already safely under a backup filename.
    """
    cache_path = log_dir / CACHE_FILENAME
    if cache_path.exists():
        backup = _quarantine_path(log_dir, label)
        try:
            cache_path.rename(backup)
        except OSError as e:
            # Rename failed — leave the cache in place rather than deleting.
            logger.warning(
                "Could not quarantine %s to %s (%s); aborting resync to "
                "avoid destroying imported rows.",
                cache_path,
                backup,
                e,
            )
            return
        logger.warning(
            "%s Existing cache preserved as %s. Rows from that file are NOT "
            "loaded until you either (a) delete %s manually or (b) verify "
            "the older rows are compatible and merge them.",
            reason,
            backup.name,
            backup.name,
        )
    _sync_cache(log_dir, tracking_uri, mlflow_count)


def _full_rebuild(
    log_dir: Path, tracking_uri: str, mlflow_count: int, cached_schema: int
) -> None:
    """Quarantine an incompatible-schema cache under `_all_traces.v{N}.csv`
    and resync. Version-N rows are shape-incompatible with later extractors,
    so mixing them in one CSV would break the dashboard."""
    reason = (
        f"Extractor schema changed (v{cached_schema} → v{_SCHEMA_VERSION})."
    )
    logger.info(
        "Extractor schema changed (v%s → v%s); rebuilding _all_traces.csv "
        "from MLflow.",
        cached_schema,
        _SCHEMA_VERSION,
    )
    _quarantine_and_resync(
        log_dir,
        tracking_uri,
        mlflow_count,
        label=f"v{cached_schema}",
        reason=reason,
    )


def ensure_trace_cache(
    log_dir: Path,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> Path | None:
    """Make sure `_all_traces.csv` covers every trace currently in MLflow.

    Append semantics: new case_ids are added, existing rows are left alone,
    imported rows without a matching MLflow trace are preserved. Returns the
    cache path if it exists after the sync, or None when nothing is available.
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

    cached_schema = _cached_schema_version(meta_path)
    cache_has_rows = cache_path.exists() and cache_path.stat().st_size > 0

    if cached_schema == 0 and cache_has_rows:
        # Cache present but sidecar missing → schema unknown. Appending
        # current-version rows to a possibly-older file would mix shapes,
        # so quarantine and let the user opt back in manually.
        logger.warning(
            "Cache %s exists but sidecar %s is missing or unreadable; "
            "quarantining and rebuilding from MLflow.",
            cache_path,
            meta_path,
        )
        _quarantine_and_resync(
            log_dir,
            tracking_uri,
            mlflow_count,
            label="unknown-schema",
            reason=(
                "Cache present without a matching sidecar (schema version "
                "unknown)."
            ),
        )
    elif cached_schema and cached_schema != _SCHEMA_VERSION and cache_has_rows:
        _full_rebuild(log_dir, tracking_uri, mlflow_count, cached_schema)
    else:
        _sync_cache(log_dir, tracking_uri, mlflow_count)

    return cache_path if cache_path.exists() else None
