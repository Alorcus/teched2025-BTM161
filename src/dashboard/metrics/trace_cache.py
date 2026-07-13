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

Quarantine behavior — imported rows are never silently destroyed:
- Schema-version bump: the existing cache is renamed to
  `_all_traces.v{cached_schema}.csv` (with a `.v{cached_schema}.N.csv` suffix
  if the backup already exists) and a fresh sync runs at the current schema.
  A WARNING is logged so the user knows the old rows are quarantined but not
  loaded — merging them back is a manual step once compatibility is verified.
- Missing sidecar with a non-empty cache: schema version is unknown, so the
  cache is renamed to `_all_traces.unknown-schema.csv` (with a numeric suffix
  if that name is taken) and a fresh sync runs. If the user knows the rows
  are compatible, they can rename the quarantine file back to
  `_all_traces.csv` AND write a matching sidecar with the current schema
  version to opt back in.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.trace_processing.trace_processor import TraceProcessor

logger = logging.getLogger(__name__)

# Fixed filename so the rest of the loader can ignore everything else in the
# directory — manually-exported CSVs are intentionally not part of the data
# source any more.
CACHE_FILENAME = "_all_traces.csv"
# Sidecar recording the extractor's schema version. The schema version is the
# real staleness signal now that we append instead of rebuild.
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
#   5 → gateway_decision rows folded in from guardrail_log/events.jsonl so
#       a shared _all_traces.csv carries the guardrail signal end-to-end
#   6 → gateway_decision rows now written as naive-UTC (matching LogGenerator);
#       version-5 rows used naive-LOCAL and were double-shifted by
#       _load_combined_eventlog's UTC→local conversion, pushing gateway
#       events hours into the future on any non-UTC host
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
    """Return the schema version recorded in the sidecar, or 0 if the file is
    missing/unreadable.

    Backward-compat: earlier versions wrote a two-line sidecar
    `{trace_count}\\n{schema_version}\\n`; we now write only the schema
    version. On read, if two non-empty lines are present we ignore line 1
    (the stale count) and parse line 2. If one line is present we parse it
    as the schema version.
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
    # Two-line legacy format: line 1 is trace_count, line 2 is schema_version.
    # One-line current format: the single line is schema_version.
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
    """Return an available quarantine path of the form
    `_all_traces.{label}.csv`, adding a numeric suffix
    (`_all_traces.{label}.1.csv`, `.2.csv`, …) if the primary name is taken.
    Callers rely on this to never overwrite an existing backup."""
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
    """Read just the `case_id` column from the existing cache. Empty set if
    the file doesn't exist yet or the column is missing. Uses polars'
    lazy scan so the cost stays linear in the case_id column, not the row
    body — cheap even on large curated files."""
    if not cache_path.exists():
        return set()
    try:
        df = pl.scan_csv(str(cache_path)).select("case_id").collect()
    except (pl.exceptions.PolarsError, OSError) as e:
        # Corrupt cache or missing column → treat as empty so a fresh sync
        # can repopulate. We do NOT delete the file here; if the extraction
        # produces zero new rows the caller will leave the file alone.
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
    # NOTE: quarantine backups (e.g. `_all_traces.v5.csv`,
    # `_all_traces.unknown-schema.csv`) live in the same directory; we
    # deliberately skip anything that starts with the cache stem so the user's
    # quarantined rows aren't collected as "leftover".
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

    if new_df.empty:
        # Nothing new to add; keep the sidecar current so future runs see the
        # latest schema version.
        _write_cache_schema(meta_path)
        return

    new_frame = pl.from_pandas(new_df)

    if cache_path.exists():
        try:
            existing_frame = pl.read_csv(str(cache_path), infer_schema_length=10_000)
            combined = pl.concat(
                [existing_frame, new_frame], how="diagonal_relaxed"
            )
        except (pl.exceptions.PolarsError, OSError) as e:
            # If the existing file is unreadable, don't destroy it silently —
            # write the new slice under a sibling name and leave the original
            # for the user to inspect.
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

    # Sort by timestamp so downstream readers still see chronological order.
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
    """Rename the existing cache under a quarantine name, then run a fresh
    sync with an empty covered set. This is the shared implementation for
    both the schema-bump path (label=`v{old}`) and the missing-sidecar path
    (label=`unknown-schema`).

    Because the rename happens BEFORE the sync, a subsequent sync failure
    can't lose data — the imported rows are already safely under a backup
    filename. The user can rescue them by renaming the quarantine file back
    to `_all_traces.csv` AND writing a matching sidecar with the current
    schema version once they've verified compatibility.
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
    """Schema-version bump escape hatch. Quarantine the existing cache under
    `_all_traces.v{cached_schema}.csv` and re-run the sync with an empty
    covered set. The rows in a version-N file are shape-incompatible with
    rows produced by a later extractor; mixing them in one CSV breaks the
    dashboard. Preserving them under a versioned backup gives the user a
    manual escape hatch."""
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

    cached_schema = _cached_schema_version(meta_path)
    cache_has_rows = cache_path.exists() and cache_path.stat().st_size > 0

    if cached_schema == 0 and cache_has_rows:
        # Cache exists but sidecar is missing/unreadable — we can't know
        # which schema version wrote those rows, so silently appending v6
        # rows to a possibly-v5 file would mix shapes. Quarantine instead
        # and let the user opt back in manually.
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
