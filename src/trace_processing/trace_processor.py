import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from .log_generator import LogGenerator, is_langgraph_root
from .naive_utc import from_epoch_naive_utc
from ..config import CoffeeShopConfig
import polars as pl

logger = logging.getLogger(__name__)

# Anchor cwd-relative defaults to the project root so the trace processor
# behaves the same whether invoked from the repo root, a notebook, or a
# subprocess launched from elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

COFFEE_MACHINE_LOG = PROJECT_ROOT / "services/coffee_machine/logs/coffee_machine.csv"
GUARDRAIL_LOG = PROJECT_ROOT / "guardrail_log/events.jsonl"

# Activities the coffee machine is allowed to emit. Anything else in the CSV
# is a stale row from a previous version of the logger (e.g. pre-rename
# "user_prompt" rows that would otherwise collide with the agent-side
# user_prompt event type) and is dropped during merge.
_COFFEE_MACHINE_ACTIVITIES = {
    "job_created",
    "process_order",
    "brew_completed",
    "brew_failed",
    "clean_machine",
}


def _resolve_under_project_root(path: str | os.PathLike[str] | Path) -> Path:
    """Resolve `path` against `PROJECT_ROOT` when it is relative, so the
    default config values (which are cwd-relative) behave the same regardless
    of the invoking process's working directory."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_coffee_machine_rows(path: Path) -> pl.DataFrame:
    """Read the coffee machine's raw CSV and map it to the canonical schema.

    The expected column names mirror `FIXED_HEADER` in
    services/coffee_machine/logger.py — keep the two in sync.

    Returns an empty DataFrame if the file is missing or contains only the
    header. Optional canonical columns (message/model/tokens/tool/feedback_*)
    are left absent so the CSV write emits empty cells, which polars reads
    back as null. Do NOT fill them with "" here — the OCEL converter checks
    via is_not_null() and a literal empty string would slip past.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame()
    try:
        raw = pl.read_csv(path, infer_schema_length=10_000)
    except pl.exceptions.NoDataError:
        return pl.DataFrame()
    if raw.is_empty():
        return pl.DataFrame()

    # Drop rows whose activity isn't part of the current canonical set —
    # protects against stale CSV content from older logger versions.
    raw = raw.filter(pl.col("concept:name").is_in(_COFFEE_MACHINE_ACTIVITIES))
    if raw.is_empty():
        return pl.DataFrame()

    # Optional columns: fill in defaults when absent so the projection is
    # uniform. `drink` is used both to build `concept:instance` and to
    # populate its own column; `job_id` passes through untouched.
    if "drink" not in raw.columns:
        raw = raw.with_columns(pl.lit("").alias("drink"))
    if "job_id" not in raw.columns:
        raw = raw.with_columns(pl.lit("").alias("job_id"))
    # `duration` may be null (header-only or instantaneous events) → 0.
    raw = raw.with_columns(
        pl.col("drink").fill_null(""),
        pl.col("duration").fill_null(0.0).alias("_duration_s"),
    )

    ts_dt = pl.from_epoch(pl.col("ocel_time"), time_unit="s")
    # `_duration_s` is float seconds; multiply into nanoseconds for the
    # duration expression and for the canonical Int64 `duration` column.
    dur_ns = (pl.col("_duration_s") * 1_000_000_000).cast(pl.Int64)

    projected = raw.with_columns(
        pl.Series(
            "identity:id", [str(uuid.uuid4()) for _ in range(raw.height)]
        ),
        ts_dt.dt.strftime("%Y-%m-%dT%H:%M:%S%.3f").alias("time:timestamp"),
        (ts_dt + pl.duration(nanoseconds=dur_ns))
            .dt.strftime("%Y-%m-%dT%H:%M:%S%.3f")
            .alias("time_finished"),
        pl.concat_str(
            [
                pl.lit("coffee machine "),
                pl.col("concept:name").cast(pl.Utf8),
                pl.when(pl.col("drink") != "")
                    .then(pl.concat_str([pl.lit(" ("), pl.col("drink"), pl.lit(")")]))
                    .otherwise(pl.lit("")),
            ]
        ).alias("concept:instance"),
        pl.lit("coffee_machine").alias("org:resource"),
        dur_ns.alias("duration"),
        pl.col("case_id").cast(pl.Utf8),
    )

    return projected.select(
        "case_id",
        "identity:id",
        "time:timestamp",
        "time_finished",
        "concept:name",
        "concept:instance",
        "org:resource",
        "duration",
        "job_id",
        "drink",
    )


def _load_gateway_rows(path: Path) -> pl.DataFrame:
    """Read `guardrail_log/events.jsonl` and project each `gateway_decision`
    record into one canonical event-log row so the flat `_all_traces.csv`
    can carry the signal to users who don't have the JSONL on disk.

    `tool_args` and `verdicts` are stored as JSON-encoded strings — polars
    CSV doesn't support nested types, and the decoder round-trips both back
    to dicts/lists before feeding them to `project_decisions`.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame()
    rows: list[dict] = []
    bad_lines = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                if rec.get("event_type") != "gateway_decision":
                    continue
                thread_id = rec.get("thread_id")
                agent_id = rec.get("agent_id")
                setup_name = rec.get("setup_name")
                snapshot_id = rec.get("snapshot_id")
                tool_call_id = rec.get("tool_call_id")
                if not (thread_id and agent_id and setup_name and snapshot_id and tool_call_id):
                    continue
                ts_raw = rec.get("ts")
                try:
                    # `datetime.fromtimestamp(epoch)` without tz returns
                    # naive-LOCAL, which would double-shift these rows
                    # relative to LogGenerator's naive-UTC timestamps.
                    ts_dt = from_epoch_naive_utc(float(ts_raw))
                except (TypeError, ValueError):
                    continue
                ts_iso = ts_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
                rows.append({
                    "case_id":               str(thread_id),
                    "identity:id":           str(uuid.uuid4()),
                    "time:timestamp":        ts_iso,
                    "time_finished":         ts_iso,
                    "concept:name":          "gateway_decision",
                    "concept:instance":      f"gateway {rec.get('final_decision', 'allow')}: {rec.get('tool_name', '')}",
                    "org:resource":          str(agent_id),
                    "gateway_setup_name":    str(setup_name),
                    "gateway_snapshot_id":   str(snapshot_id),
                    "gateway_tool_name":     rec.get("tool_name", "") or "",
                    "gateway_tool_call_id":  str(tool_call_id),
                    "gateway_final_decision": rec.get("final_decision", "allow") or "allow",
                    "gateway_tool_args_json": json.dumps(rec.get("tool_args") or {}, sort_keys=True),
                    "gateway_verdicts_json":  json.dumps(rec.get("verdicts") or []),
                })
    except OSError:
        return pl.DataFrame()
    if bad_lines:
        logger.warning("gateway log: skipped %d malformed line(s) in %s", bad_lines, path)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


class TraceProcessor:
    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlflow.db",
        guardrail_log_path: str | os.PathLike[str] | None = None,
        coffee_machine_log_path: str | os.PathLike[str] | None = None,
    ):
        self.tracking_uri = tracking_uri
        if guardrail_log_path is None:
            guardrail_log_path = CoffeeShopConfig.__dataclass_fields__[
                "guardrail_log_path"
            ].default
        self.guardrail_log_path = _resolve_under_project_root(guardrail_log_path)
        if coffee_machine_log_path is None:
            self.coffee_machine_log_path = COFFEE_MACHINE_LOG
        else:
            self.coffee_machine_log_path = _resolve_under_project_root(
                coffee_machine_log_path
            )

    def _get_all_traces(self):
        """
        Retrieve all traces using the MLflow API.
        """
        import mlflow

        client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)

        experiments = client.search_experiments()
        experiment_ids = [exp.experiment_id for exp in experiments]

        if not experiment_ids:
            return []

        all_traces = []
        for exp_id in experiment_ids:
            page_token = None
            while True:
                result = client.search_traces(
                    locations=[exp_id],
                    max_results=100,
                    page_token=page_token,
                )
                all_traces.extend(result)
                if not result.token:
                    break
                page_token = result.token

        return all_traces

    def _get_traces_by_request_ids(self, request_ids: list[str]) -> list:
        """Fetch specific traces by request_id.

        Used by the SQLite fast path in `trace_cache`: SQL narrows the set
        of candidate request_ids to only those we haven't materialized
        yet, and we then pull each with a single `client.get_trace()`
        call. Skips traces the client refuses to return (deleted, or
        MLflow-schema mismatch) with a warning.
        """
        import mlflow

        client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
        traces = []
        for rid in request_ids:
            try:
                traces.append(client.get_trace(rid))
            except Exception as e:
                logger.warning("Could not fetch trace %s: %s", rid, e)
        return traces

    @staticmethod
    def _peek_case_id_from_info(trace_info) -> str | None:
        """Cheap read of the LangGraph thread_id from a `TraceInfo` object,
        BEFORE any expensive `Trace.to_dict()` serialization of the span
        payload.

        MLflow's LangChain autolog records the LangGraph `thread_id` on the
        trace as metadata under the key ``mlflow.trace.session`` (see
        ``mlflow/langchain/langchain_tracer.py``). Reading it from
        ``trace_info.trace_metadata`` avoids materializing the whole span
        tree just to learn the case_id.

        Returns None when the metadata is missing — the caller falls back
        to `_peek_case_id(trace_dict)` after paying for `to_dict()`.
        """
        try:
            metadata = getattr(trace_info, "trace_metadata", None)
        except AttributeError:
            metadata = None
        if metadata is None:
            return None
        cid = metadata.get("mlflow.trace.session")
        return str(cid) if cid else None

    @staticmethod
    def _peek_case_id(trace_dict: dict) -> str | None:
        """Fallback peek of the LangGraph thread_id from a serialized trace
        dict, used when `_peek_case_id_from_info` misses (legacy traces or
        test stubs without `trace_metadata`)."""
        # MLflow's client puts spans under trace_dict["data"]["spans"] when
        # this dict comes from Trace.to_dict(); LogGenerator also accepts a
        # bare "spans" key. Handle both shapes.
        spans = trace_dict.get("spans")
        if spans is None:
            spans = (trace_dict.get("data") or {}).get("spans")
        if not spans:
            return None
        for span in spans:
            if not is_langgraph_root(span.get("name", "")):
                continue
            raw_meta = (span.get("attributes") or {}).get("metadata")
            if not raw_meta:
                continue
            try:
                parsed = json.loads(raw_meta)
            except (TypeError, ValueError):
                continue
            cid = parsed.get("thread_id")
            return str(cid) if cid is not None else None
        return None

    def extract_new_traces(
        self, existing_case_ids: set[str] | None = None
    ) -> tuple[pl.DataFrame, dict[str, tuple[str | None, int]], set[str]]:
        """Run the trace-extraction pipeline for MLflow traces whose case_id
        is NOT already in `existing_case_ids`. Feedback and coffee-machine
        rows are joined in but scoped to the *new* case_ids only — the caller
        is responsible for preserving rows for already-covered cases.

        Returns `(combined_df, case_tags, new_case_ids)`. `combined_df` is
        empty (no columns) when there is nothing new to add.
        """
        if existing_case_ids is None:
            existing_case_ids = set()

        logger.info("🔍 Searching for traces...")
        traces = self._get_all_traces()
        return self._extract_from_traces(traces, existing_case_ids)

    def extract_new_traces_by_request_ids(
        self,
        request_ids: list[str],
        existing_case_ids: set[str] | None = None,
    ) -> tuple[pl.DataFrame, dict[str, tuple[str | None, int]], set[str], set[str]]:
        """Extract only the given request_ids.

        Same return shape as `extract_new_traces` plus a fourth element:
        the set of request_ids that were fetched but produced no
        LangGraph case_id (feedback / standalone ChatAnthropic). The
        caller adds them to the "no-session" ledger so we don't refetch
        them on subsequent syncs.
        """
        if existing_case_ids is None:
            existing_case_ids = set()

        logger.info("🔍 Fetching %d target trace(s)...", len(request_ids))
        traces = self._get_traces_by_request_ids(request_ids)
        combined, tags, new_ids, no_session = self._extract_from_traces(
            traces, existing_case_ids, return_no_session=True
        )
        return combined, tags, new_ids, no_session

    def _extract_from_traces(
        self,
        traces: list,
        existing_case_ids: set[str],
        return_no_session: bool = False,
    ):
        """Shared body of `extract_new_traces` and its request-id variant.

        When `return_no_session=True`, returns a 4-tuple with the set of
        request_ids that produced no case_id — used by the SQLite path's
        no-session ledger.
        """
        empty_result_3 = (pl.DataFrame(), {}, set())
        empty_result_4 = (pl.DataFrame(), {}, set(), set())

        if not traces:
            logger.error("❌ No traces found in MLflow")
            return empty_result_4 if return_no_session else empty_result_3

        logger.info("📁 Found %d traces", len(traces))

        build_start = time.perf_counter()

        feedback_store = {}
        feedback_path = Path("./feedback_store.json")
        if feedback_path.exists():
            with open(feedback_path) as f:
                feedback_store = json.load(f)

        successful_ingestions = 0
        failed_ingestions = 0
        skipped_ingestions = 0  # processed without error but produced 0 events
        skipped_already_covered = 0

        # Per-trace extraction accumulates into a list; a single pl.concat at
        # the end avoids the O(n²) copy cost of concatenating in the loop.
        trace_frames: list[pl.DataFrame] = []
        # case_id -> (setup_name, scenario_index) from MLflow trace tags.
        case_tags: dict[str, tuple[str | None, int]] = {}
        new_case_ids: set[str] = set()
        # Request-ids that were fetched but produced no case_id (feedback,
        # standalone ChatAnthropic). Only populated when the caller asked.
        no_session_request_ids: set[str] = set()

        per_trace_seconds: list[float] = []

        for i, trace in enumerate(traces, 1):
            trace_start = time.perf_counter()

            # Fast path: peek the case_id from `trace.info.trace_metadata`
            # BEFORE paying for `trace.to_dict()`, so already-covered traces
            # skip the full span-tree serialization.
            trace_info = getattr(trace, "info", None)
            peeked_case_id = (
                self._peek_case_id_from_info(trace_info)
                if trace_info is not None
                else None
            )
            if peeked_case_id is not None and peeked_case_id in existing_case_ids:
                skipped_already_covered += 1
                per_trace_seconds.append(time.perf_counter() - trace_start)
                continue

            trace_dict = trace.to_dict()
            trace_id = trace_dict.get('info', {}).get('trace_id', f'trace-{i}')
            trace_tags = trace_dict.get('info', {}).get('tags', {}) or {}

            # Fallback peek for legacy traces whose `TraceInfo` did not carry
            # `mlflow.trace.session`.
            if peeked_case_id is None:
                peeked_case_id = self._peek_case_id(trace_dict)
                if peeked_case_id is not None and peeked_case_id in existing_case_ids:
                    skipped_already_covered += 1
                    per_trace_seconds.append(time.perf_counter() - trace_start)
                    continue

            logger.info("\t📂 Processing trace %d/%d: %s", i, len(traces), trace_id)

            log_generator = LogGenerator()
            try:
                trace_event_log = log_generator.generate_event_log_df(trace_dict)
            except Exception as e:
                logger.error("❌ Failed to generate event log for %s: %s", trace_id, e)
                failed_ingestions += 1
                per_trace_seconds.append(time.perf_counter() - trace_start)
                continue

            if trace_event_log.is_empty():
                # LogGenerator returns an empty frame for traces with no
                # spans or no LangGraph root (e.g. standalone ChatOllama
                # calls from get_feedback()). These are legitimately skipped,
                # not successful — counting them as such hides real bugs.
                skipped_ingestions += 1
                # Ledger: request_id (aka trace_id) has no session-derived
                # case_id, so subsequent syncs shouldn't refetch it.
                rid = trace_dict.get('info', {}).get('request_id') or trace_id
                if rid:
                    no_session_request_ids.add(str(rid))
                per_trace_seconds.append(time.perf_counter() - trace_start)
                continue

            # A trace that produced ONLY user_prompt rows is a red flag:
            # some agent-side event failed extraction. Warn so future
            # extraction gaps stay visible.
            non_agent_names = {"user_prompt"}
            trace_event_types = set(trace_event_log["concept:name"].unique().to_list())
            if trace_event_types.issubset(non_agent_names):
                logger.warning(
                    "Trace %s produced no agent-side events (only %s). "
                    "Check for extraction gaps in LogGenerator.",
                    trace_id,
                    sorted(trace_event_types),
                )

            trace_frames.append(trace_event_log)
            successful_ingestions += 1

            # Stash the trace's tag values under this trace's case_id.
            # Missing tags → setup=None, scenario=-1 (the "unspecified"
            # sentinel used across the pipeline).
            case_ids_in_trace = (
                trace_event_log["case_id"].drop_nulls().unique().to_list()
            )
            if case_ids_in_trace:
                setup_tag = trace_tags.get("setup")
                scenario_tag_raw = trace_tags.get("scenario_index")
                try:
                    scenario_tag = int(scenario_tag_raw) if scenario_tag_raw not in (None, "", "None") else -1
                except (TypeError, ValueError):
                    scenario_tag = -1
                for cid in case_ids_in_trace:
                    case_tags[cid] = (setup_tag, scenario_tag)
                    new_case_ids.add(str(cid))
            per_trace_seconds.append(time.perf_counter() - trace_start)

        combined_logs = (
            pl.concat(trace_frames, how="diagonal_relaxed")
            if trace_frames
            else pl.DataFrame()
        )

        # combined_logs starts as an empty DataFrame with no columns; if every
        # trace either failed ingestion or returned an empty frame, there is
        # no "time:timestamp" column to sort on. Bail out cleanly so the
        # dashboard's export button doesn't surface a cryptic error.
        if combined_logs.is_empty() or "time:timestamp" not in combined_logs.columns:
            if skipped_already_covered:
                logger.warning(
                    "No new usable events extracted; %d trace(s) already covered.",
                    skipped_already_covered,
                )
            else:
                logger.warning("No usable events extracted from traces; nothing to export.")
            self._print_summary(
                total=len(traces),
                successful=successful_ingestions,
                skipped=skipped_ingestions,
                failed=failed_ingestions,
                already_covered=skipped_already_covered,
                total_seconds=time.perf_counter() - build_start,
                per_trace_seconds=per_trace_seconds,
            )
            if return_no_session:
                return pl.DataFrame(), {}, set(), no_session_request_ids
            return empty_result_3

        # Precompute the last timestamp per case in a single group_by so the
        # feedback loop below is O(F + R) instead of O(F * R).
        max_ts_agg = combined_logs.group_by("case_id").agg(
            pl.col("time:timestamp").max().alias("_max_ts")
        )
        max_ts_by_case = dict(
            zip(max_ts_agg["case_id"].to_list(), max_ts_agg["_max_ts"].to_list())
        )

        # Scoping to `new_case_ids` matters for append mode: feedback rows
        # for already-covered cases live in the caller's existing CSV and
        # must not be re-emitted here.
        feedback_rows = []
        for case_id, fb in feedback_store.items():
            if case_id not in new_case_ids:
                continue
            if case_id not in max_ts_by_case:
                continue
            last_ts = max_ts_by_case[case_id]
            feedback_ts = (
                datetime.strptime(last_ts, "%Y-%m-%dT%H:%M:%S.%f") + timedelta(milliseconds=1)
            ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            feedback_rows.append({
                "case_id": case_id,
                "identity:id": str(uuid.uuid4()),
                "time:timestamp": feedback_ts,
                "time_finished": feedback_ts,
                "concept:name": "user_feedback",
                "concept:instance": f"user rates: {fb['feedback_score']}",
                "org:resource": "user",
                "message": str(fb["feedback_score"]),
                "feedback_score": fb["feedback_score"],
                "feedback_reason": fb["feedback_reason"],
                "feedback_valid": fb["valid"],
                "scenario_index": fb.get("scenario_index"),
            })

        # Accumulate all supplementary streams and do a single concat+sort
        # at the end. Merging them one at a time forced the whole frame to
        # be re-sorted up to four times.
        supplementary: list[pl.DataFrame] = []
        if feedback_rows:
            supplementary.append(pl.DataFrame(feedback_rows))

        machine_rows = _load_coffee_machine_rows(self.coffee_machine_log_path)
        if not machine_rows.is_empty():
            before = machine_rows.height
            machine_rows = machine_rows.filter(pl.col("case_id").is_in(new_case_ids))
            dropped = before - machine_rows.height
            if dropped:
                logger.warning(
                    "Dropped %d coffee-machine row(s) with case_id not in the "
                    "new agent log slice (stale, mismatched, or belonging to "
                    "an already-covered case)",
                    dropped,
                )
            if not machine_rows.is_empty():
                supplementary.append(machine_rows)

        gateway_rows = _load_gateway_rows(self.guardrail_log_path)
        if not gateway_rows.is_empty():
            before = gateway_rows.height
            gateway_rows = gateway_rows.filter(pl.col("case_id").is_in(new_case_ids))
            dropped = before - gateway_rows.height
            if dropped:
                logger.warning(
                    "Dropped %d gateway_decision row(s) whose case_id is not "
                    "in the new agent log slice (stale, mismatched, or "
                    "belonging to an already-covered case)",
                    dropped,
                )
            if not gateway_rows.is_empty():
                supplementary.append(gateway_rows)

        if supplementary:
            combined_logs = pl.concat(
                [combined_logs, *supplementary], how="diagonal_relaxed"
            )
        combined_logs = combined_logs.sort("time:timestamp")

        # Broadcast per-case setup/scenario tags to every row so downstream
        # readers can filter without joining an extra table. A left join
        # against a small tag frame is dramatically faster than a per-row
        # python callback.
        if case_tags and not combined_logs.is_empty() and "case_id" in combined_logs.columns:
            tag_frame = pl.DataFrame({
                "case_id": list(case_tags.keys()),
                "case_setup": [t[0] for t in case_tags.values()],
                "case_scenario_index": [t[1] for t in case_tags.values()],
            })
            combined_logs = combined_logs.join(
                tag_frame, on="case_id", how="left"
            ).with_columns(pl.col("case_scenario_index").fill_null(-1))
        else:
            combined_logs = combined_logs.with_columns(
                pl.lit(None, dtype=pl.Utf8).alias("case_setup"),
                pl.lit(-1, dtype=pl.Int64).alias("case_scenario_index"),
            )

        self._print_summary(
            total=len(traces),
            successful=successful_ingestions,
            skipped=skipped_ingestions,
            failed=failed_ingestions,
            already_covered=skipped_already_covered,
            total_seconds=time.perf_counter() - build_start,
            per_trace_seconds=per_trace_seconds,
        )

        if successful_ingestions > 0:
            logger.info("Log generation process completed successfully!")

        if return_no_session:
            return combined_logs, case_tags, new_case_ids, no_session_request_ids
        return combined_logs, case_tags, new_case_ids

    def process_all_traces(self, export_as_json: bool = False):
        """Process all traces via the MLflow API and write a timestamped
        event-log file. Entrypoint for the headless
        `simulate --export-logs` path."""
        combined_logs, _tags, _new_ids = self.extract_new_traces(set())
        if combined_logs.is_empty():
            return
        self._generate_log_file(
            combined_logs, "./generated_event_log", json_format=export_as_json
        )
        return

    @staticmethod
    def _print_summary(
        *,
        total: int,
        successful: int,
        skipped: int,
        failed: int,
        total_seconds: float,
        per_trace_seconds: list[float],
        already_covered: int = 0,
    ) -> None:
        logger.info("📈 Processing Summary:")
        logger.info("   📊 Total traces processed: %d", total)
        logger.info("   ✅ Successful: %d", successful)
        if already_covered:
            logger.info("   ⏩ Already covered (skipped): %d", already_covered)
        logger.info("   ⏭️  Skipped (no events): %d", skipped)
        logger.info("   ❌ Failed: %d", failed)
        if per_trace_seconds:
            avg = sum(per_trace_seconds) / len(per_trace_seconds)
            logger.info(
                "   ⏱️  Build time: %.2fs total (avg %.1fms/trace, "
                "min %.1fms, max %.1fms)",
                total_seconds,
                avg * 1000,
                min(per_trace_seconds) * 1000,
                max(per_trace_seconds) * 1000,
            )
        else:
            logger.info("   ⏱️  Build time: %.2fs total", total_seconds)

    def _generate_log_file(self, dataframe: pl.DataFrame, output_path: str, json_format: bool = False):
        """
        Generate a log file from the given DataFrame.

        Args:
            dataframe: The DataFrame containing event log data
            output_path: The path to save the generated log file

        Returns:
            The written file path on success, or None on failure.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]  # UTC timestamp with ms
        filename = f"{timestamp}.eventlog"
        if json_format:
            filename += ".json"
        else:
            filename += ".csv"

        file_path = os.path.join(output_path, filename)

        try:
            if json_format:
                dataframe.write_json(file_path)
            else:
                dataframe.write_csv(file_path)
            logger.info("✅ Log file generated at %s", file_path)
            return file_path
        except Exception as e:
            logger.error("❌ Failed to generate log file at %s: %s", file_path, e)
            return None
