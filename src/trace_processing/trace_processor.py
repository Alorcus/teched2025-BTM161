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
import pandas as pd

logger = logging.getLogger(__name__)

# Anchor cwd-relative defaults to the project root so the trace processor
# behaves the same whether invoked from the repo root, a notebook, or a
# subprocess launched from elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Kept as module-level defaults for callers that want to override the paths
# on the CSV loaders directly (see tests/test_guardrail_csv_roundtrip.py).
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


def _load_coffee_machine_rows(path: Path) -> pd.DataFrame:
    """Read the coffee machine's raw CSV and map it to the canonical schema.

    The expected column names mirror `FIXED_HEADER` in
    services/coffee_machine/logger.py — keep the two in sync.

    Returns an empty DataFrame if the file is missing or contains only the
    header. Optional canonical columns (message/model/tokens/tool/feedback_*)
    are left absent so pandas → CSV writes them as empty cells, which polars
    reads back as null. Do NOT fillna("") here — the OCEL converter checks
    via is_not_null() and a literal empty string would slip past.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()

    # Drop rows whose activity isn't part of the current canonical set —
    # protects against stale CSV content from older logger versions.
    raw = raw[raw["concept:name"].isin(_COFFEE_MACHINE_ACTIVITIES)]
    if raw.empty:
        return pd.DataFrame()
    raw = raw.reset_index(drop=True)

    # epoch float seconds → ISO-8601 ms strings
    ts = pd.to_datetime(raw["ocel_time"], unit="s")
    ts_str = ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3]

    # duration may be NaN (header-only or instantaneous events) → 0
    dur_seconds = raw["duration"].fillna(0.0)
    finish = ts + pd.to_timedelta(dur_seconds, unit="s")
    finish_str = finish.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3]

    drink_safe = raw.get("drink", pd.Series([""] * len(raw))).fillna("")
    instance = (
        "coffee machine " + raw["concept:name"].astype(str)
        + drink_safe.apply(lambda d: f" ({d})" if d else "")
    )

    return pd.DataFrame({
        "case_id":          raw["case_id"].astype(str),
        "identity:id":      [str(uuid.uuid4()) for _ in range(len(raw))],
        "time:timestamp":   ts_str,
        "time_finished":    finish_str,
        "concept:name":     raw["concept:name"],
        "concept:instance": instance,
        "org:resource":     "coffee_machine",
        "duration":         (dur_seconds * 1e9).astype("int64"),
        "job_id":           raw.get("job_id", pd.Series([""] * len(raw))),
        "drink":            drink_safe,
    })


def _load_gateway_rows(path: Path) -> pd.DataFrame:
    """Read `guardrail_log/events.jsonl` and project each `gateway_decision`
    record into one canonical event-log row so the flat `_all_traces.csv`
    can carry the signal to users who don't have the JSONL on disk.

    The row shape mirrors what `guardrail_log_loader.load_guardrail_events_from_eventlog`
    expects to decode back into a `GuardrailOcelExtension`. `tool_args` and
    `verdicts` are stored as JSON-encoded strings — polars CSV doesn't
    support nested types, and the decoder round-trips both back to dicts/lists
    before feeding them to the shared `project_decisions` function.

    Rows for `gateway_decision` records missing any of the required
    identifiers (`thread_id`, `agent_id`, `setup_name`, `snapshot_id`,
    `tool_call_id`) are dropped — same rule the JSONL loader applies at
    `guardrail_log_loader.py:192-193`.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
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
                    # `rec["ts"]` is an epoch float. `from_epoch_naive_utc`
                    # produces the same naive-UTC shape LogGenerator writes
                    # (from OpenTelemetry `start_time_unix_nano`), which is
                    # what `_load_combined_eventlog` interprets every CSV
                    # timestamp as before converting to local. Using the
                    # bare `datetime.fromtimestamp(epoch)` (naive-LOCAL)
                    # would double-shift these rows by the local offset.
                    ts_dt = from_epoch_naive_utc(float(ts_raw))
                except (TypeError, ValueError):
                    continue
                # Full microsecond precision — the JSONL loader parses `ts`
                # via `datetime.fromtimestamp(float)` which keeps microseconds,
                # so trimming to milliseconds here would leave the CSV-embedded
                # extension slightly off from the JSONL-derived one and break
                # dashboards that filter/sort by exact timestamp.
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
        return pd.DataFrame()
    if bad_lines:
        logger.warning("gateway log: skipped %d malformed line(s) in %s", bad_lines, path)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


class TraceProcessor:
    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlflow.db",
        guardrail_log_path: str | os.PathLike[str] | None = None,
        coffee_machine_log_path: str | os.PathLike[str] | None = None,
    ):
        self.tracking_uri = tracking_uri
        # Guardrail path default comes from CoffeeShopConfig so the trace
        # processor and the dashboard read the same file. Coffee-machine
        # path has no config field yet, so fall back to the module-level
        # PROJECT_ROOT-anchored default.
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

    @staticmethod
    def _peek_case_id_from_info(trace_info) -> str | None:
        """Cheap read of the LangGraph thread_id from a `TraceInfo` object,
        BEFORE any expensive `Trace.to_dict()` serialization of the span
        payload.

        MLflow's LangChain autolog records the LangGraph `thread_id` on the
        trace as metadata under the key ``mlflow.trace.session`` (see
        ``mlflow/langchain/langchain_tracer.py`` — the tracer calls
        ``mlflow.update_current_trace(metadata={TraceMetadataKey.TRACE_SESSION: thread_id})``
        for every chain span carrying a `thread_id`). Reading it from
        ``trace_info.trace_metadata`` avoids materializing the whole span
        tree just to learn the case_id, which is what the info-first peek
        exists to prevent.

        Returns None when the metadata is missing (feedback-only traces
        with no LangGraph root, legacy traces predating this MLflow behavior,
        or when the caller passes a stub that has no `.trace_metadata`
        attribute) — the caller falls back to `_peek_case_id(trace_dict)`
        after paying for `to_dict()`.
        """
        # Accept both a full TraceInfo (with .trace_metadata) and a mapping
        # shape; guard AttributeError for stubs that expose neither.
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
        """Cheap read of the LangGraph thread_id (= case_id) without running
        LogGenerator over the whole trace. Returns None if the trace has no
        LangGraph root or no metadata (e.g. feedback-only traces).

        Kept as the fallback path for legacy traces whose ``TraceInfo`` does
        not carry ``mlflow.trace.session`` — see `_peek_case_id_from_info`
        for the fast path that avoids paying for `to_dict()` at all."""
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
    ) -> tuple[pd.DataFrame, dict[str, tuple[str | None, int]], set[str]]:
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

        empty_result: tuple[pd.DataFrame, dict[str, tuple[str | None, int]], set[str]] = (
            pd.DataFrame(),
            {},
            set(),
        )

        if not traces:
            logger.error("❌ No traces found in MLflow")
            return empty_result

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

        # Per-trace extraction accumulates into a list; a single pd.concat at
        # the end avoids the O(n²) copy cost of concatenating in the loop.
        trace_frames: list[pd.DataFrame] = []
        # case_id -> (setup_name, scenario_index) from MLflow trace tags. Filled
        # while iterating traces; broadcast onto every event row (all sources)
        # after all concats complete, so filter/aggregate queries can slice by
        # setup or scenario without joining an extra table.
        case_tags: dict[str, tuple[str | None, int]] = {}
        new_case_ids: set[str] = set()

        per_trace_seconds: list[float] = []

        for i, trace in enumerate(traces, 1):
            trace_start = time.perf_counter()

            # Fast path: peek the case_id from `trace.info.trace_metadata`
            # BEFORE paying for `trace.to_dict()`. On warm caches with N
            # already-covered traces this saves N full span-tree
            # serializations. Real MLflow traces expose `.info`; test stubs
            # that only implement `to_dict()` fall through to the dict path
            # below.
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
            # `mlflow.trace.session` (or test stubs that skipped the info
            # path). Feedback-only traces (no LangGraph root) peek as None
            # here too and fall through to LogGenerator so the existing
            # empty-frame branch keeps handling them.
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

            if trace_event_log.empty:
                # LogGenerator returns an empty frame for traces with no
                # spans, no LangGraph root, etc. (e.g. standalone ChatOllama
                # calls from get_feedback()). These are legitimately-skipped,
                # not successful — counting them as such hides real bugs.
                skipped_ingestions += 1
                per_trace_seconds.append(time.perf_counter() - trace_start)
                continue

            # A trace that produced ONLY user_prompt rows and nothing else is
            # a red flag: the conversation ran, the user turned up, but no
            # agent-side event survived extraction. Handover-only threads used
            # to look like this before the transfer_to_* fix landed; keeping
            # the warning around means future extraction gaps stay visible
            # instead of silently collapsing threads to user prompts + feedback.
            non_agent_names = {"user_prompt"}
            trace_event_types = set(trace_event_log["concept:name"].unique())
            if trace_event_types.issubset(non_agent_names):
                logger.warning(
                    "Trace %s produced no agent-side events (only %s). "
                    "Check for extraction gaps in LogGenerator.",
                    trace_id,
                    sorted(trace_event_types),
                )

            trace_frames.append(trace_event_log)
            successful_ingestions += 1

            # Stash the trace's tag values under this trace's case_id. Multiple
            # traces share a case_id (each user turn = one MLflow trace); every
            # tagged trace in the same case carries the same setup/scenario, so
            # last-write-wins is fine. Missing tags → setup=None, scenario=-1
            # (the "unspecified" sentinel used across the pipeline).
            case_ids_in_trace = trace_event_log["case_id"].dropna().unique()
            if len(case_ids_in_trace) > 0:
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
            pd.concat(trace_frames, ignore_index=True)
            if trace_frames
            else pd.DataFrame()
        )

        # combined_logs starts as an empty DataFrame with no columns; if every
        # trace either failed ingestion or returned an empty frame (e.g.
        # feedback-only traces with no LangGraph root — see log_generator.py:33),
        # there is no "time:timestamp" column to sort on and pandas raises
        # KeyError. Bail out cleanly instead so the dashboard's export button
        # doesn't surface a cryptic "❌ time:timestamp".
        if combined_logs.empty or "time:timestamp" not in combined_logs.columns:
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
            return empty_result

        # Precompute the last timestamp per case in a single groupby so the
        # feedback loop below is O(F + R) instead of O(F * R) (see todo 021).
        # .max() over a Series doesn't require sorted input, so we can defer
        # the sort until after all supplementary streams merge in.
        max_ts_by_case = combined_logs.groupby("case_id")["time:timestamp"].max()

        # Append exactly one user_feedback event per NEW case, after all other
        # events. Scoping to `new_case_ids` matters for append mode: feedback
        # rows for already-covered cases live in the caller's existing CSV
        # and must not be re-emitted here.
        feedback_rows = []
        for case_id, fb in feedback_store.items():
            if case_id not in new_case_ids:
                continue
            if case_id not in max_ts_by_case.index:
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

        # Accumulate all supplementary streams (feedback, coffee-machine,
        # gateway) and do a single concat+sort at the end. Merging them one
        # at a time forced the whole frame to be re-sorted up to four times.
        supplementary: list[pd.DataFrame] = []
        if feedback_rows:
            supplementary.append(pd.DataFrame(feedback_rows))

        # Merge coffee machine rows (stream 2). Filter to NEW case_ids so
        # rows keyed to already-covered cases stay in the caller's existing
        # CSV rather than getting duplicated.
        machine_rows = _load_coffee_machine_rows(self.coffee_machine_log_path)
        if not machine_rows.empty:
            before = len(machine_rows)
            machine_rows = machine_rows[machine_rows["case_id"].isin(new_case_ids)]
            dropped = before - len(machine_rows)
            if dropped:
                logger.warning(
                    "Dropped %d coffee-machine row(s) with case_id not in the "
                    "new agent log slice (stale, mismatched, or belonging to "
                    "an already-covered case)",
                    dropped,
                )
            if not machine_rows.empty:
                supplementary.append(machine_rows)

        # Merge gateway decisions (stream 3). This is what lets a shared
        # `_all_traces.csv` reproduce the guardrail dashboard panel on a
        # machine that never had `guardrail_log/events.jsonl`.
        gateway_rows = _load_gateway_rows(self.guardrail_log_path)
        if not gateway_rows.empty:
            before = len(gateway_rows)
            gateway_rows = gateway_rows[gateway_rows["case_id"].isin(new_case_ids)]
            dropped = before - len(gateway_rows)
            if dropped:
                logger.warning(
                    "Dropped %d gateway_decision row(s) whose case_id is not "
                    "in the new agent log slice (stale, mismatched, or "
                    "belonging to an already-covered case)",
                    dropped,
                )
            if not gateway_rows.empty:
                supplementary.append(gateway_rows)

        if supplementary:
            combined_logs = pd.concat(
                [combined_logs, *supplementary], ignore_index=True
            )
        combined_logs.sort_values(by="time:timestamp", inplace=True)

        # Broadcast per-case setup/scenario tags to every row so downstream
        # readers (e.g. the metrics dashboard) can filter without joining an
        # extra table. Cases without tagged traces show setup=None (rendered
        # as an "(unknown)" bucket in the dashboard) and scenario=-1.
        if case_tags and not combined_logs.empty and "case_id" in combined_logs.columns:
            combined_logs["case_setup"] = combined_logs["case_id"].map(
                lambda cid: case_tags.get(cid, (None, -1))[0]
            )
            combined_logs["case_scenario_index"] = combined_logs["case_id"].map(
                lambda cid: case_tags.get(cid, (None, -1))[1]
            )
        else:
            combined_logs["case_setup"] = None
            combined_logs["case_scenario_index"] = -1

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

        return combined_logs, case_tags, new_case_ids

    def process_all_traces(self, export_as_json: bool = False):
        """
        Process all traces found via the MLflow API. Kept as the entrypoint
        for the headless `simulate --export-logs` path — writes one
        timestamped event-log file with every trace's events.
        """
        combined_logs, _tags, _new_ids = self.extract_new_traces(set())
        if combined_logs.empty:
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
        # Emit line-by-line via the module logger so callers (dashboard,
        # simulate) can silence or redirect the summary through standard
        # logging configuration rather than intercepting stdout.
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

    def _generate_log_file(self, dataframe: pd.DataFrame, output_path: str, json_format: bool = False):
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
                dataframe.to_json(file_path, orient="index")
            else:
                dataframe.to_csv(file_path, index=False)
            logger.info("✅ Log file generated at %s", file_path)
            return file_path
        except Exception as e:
            logger.error("❌ Failed to generate log file at %s: %s", file_path, e)
            return None
