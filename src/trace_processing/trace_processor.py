import json
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from .log_generator import LogGenerator, _is_langgraph_root
import pandas as pd

COFFEE_MACHINE_LOG = Path("services/coffee_machine/logs/coffee_machine.csv")
GUARDRAIL_LOG = Path("guardrail_log/events.jsonl")

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
            for line in f:
                line = line.strip()
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
                    ts_dt = datetime.fromtimestamp(float(ts_raw))
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
        print(f"   ⚠️  gateway log: skipped {bad_lines} malformed line(s) in {path}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


class TraceProcessor:
    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db"):
        self.tracking_uri = tracking_uri

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
                    experiment_ids=[exp_id],
                    max_results=100,
                    page_token=page_token,
                )
                all_traces.extend(result)
                if not result.token:
                    break
                page_token = result.token

        return all_traces

    @staticmethod
    def _peek_case_id(trace_dict: dict) -> str | None:
        """Cheap read of the LangGraph thread_id (= case_id) without running
        LogGenerator over the whole trace. Returns None if the trace has no
        LangGraph root or no metadata (e.g. feedback-only traces)."""
        info = trace_dict.get("info", {}) or {}
        # MLflow's client puts spans under trace_dict["data"]["spans"] when
        # this dict comes from Trace.to_dict(); LogGenerator also accepts a
        # bare "spans" key. Handle both shapes.
        spans = trace_dict.get("spans")
        if spans is None:
            spans = (trace_dict.get("data") or {}).get("spans")
        if not spans:
            return None
        for span in spans:
            if _is_langgraph_root(span.get("name", "")):
                raw_meta = (span.get("attributes") or {}).get("metadata")
                if not raw_meta:
                    return None
                try:
                    parsed = json.loads(raw_meta)
                except (TypeError, ValueError):
                    return None
                cid = parsed.get("thread_id")
                return str(cid) if cid is not None else None
        _ = info  # kept for future debug logging; explicit read silences linters.
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

        print("🔍 Searching for traces...")
        traces = self._get_all_traces()

        empty_result: tuple[pd.DataFrame, dict[str, tuple[str | None, int]], set[str]] = (
            pd.DataFrame(),
            {},
            set(),
        )

        if not traces:
            print("❌ No traces found in MLflow")
            return empty_result

        print(f"📁 Found {len(traces)} traces")

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
            trace_dict = trace.to_dict()
            trace_id = trace_dict.get('info', {}).get('trace_id', f'trace-{i}')
            trace_tags = trace_dict.get('info', {}).get('tags', {}) or {}

            # Cheap peek before the expensive LogGenerator pass: if the
            # trace's thread_id is already covered by the caller's CSV, we
            # skip it entirely. Feedback-only traces (no LangGraph root) peek
            # as None and fall through to LogGenerator so the existing empty-
            # frame branch keeps handling them.
            peeked_case_id = self._peek_case_id(trace_dict)
            if peeked_case_id is not None and peeked_case_id in existing_case_ids:
                skipped_already_covered += 1
                per_trace_seconds.append(time.perf_counter() - trace_start)
                continue

            print(f"\t📂 Processing trace {i}/{len(traces)}: {trace_id}")

            log_generator = LogGenerator()
            try:
                trace_event_log = log_generator.generate_event_log_df(trace_dict)
            except Exception as e:
                print(f"   ❌ Failed to generate event log for {trace_id}: {e}")
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
                print(
                    f"   ⚠️  Trace {trace_id} produced no agent-side events "
                    f"(only {sorted(trace_event_types)}). Check for extraction "
                    f"gaps in LogGenerator."
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

        # Sort combined logs by timestamp. combined_logs starts as an empty
        # DataFrame with no columns; if every trace either failed ingestion or
        # returned an empty frame (e.g. feedback-only traces with no LangGraph
        # root — see log_generator.py:33), there is no "time:timestamp" column
        # to sort by and pandas raises KeyError. Bail out cleanly instead so
        # the dashboard's export button doesn't surface a cryptic
        # "❌ time:timestamp".
        if combined_logs.empty or "time:timestamp" not in combined_logs.columns:
            if skipped_already_covered:
                print(
                    f"⚠️  No new usable events extracted; "
                    f"{skipped_already_covered} trace(s) already covered."
                )
            else:
                print("⚠️  No usable events extracted from traces; nothing to export.")
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
        combined_logs.sort_values(by="time:timestamp", inplace=True)

        # Append exactly one user_feedback event per NEW case, after all other
        # events. Scoping to `new_case_ids` matters for append mode: feedback
        # rows for already-covered cases live in the caller's existing CSV
        # and must not be re-emitted here.
        feedback_rows = []
        for case_id, fb in feedback_store.items():
            if case_id not in new_case_ids:
                continue
            case_mask = combined_logs["case_id"] == case_id
            if not case_mask.any():
                continue
            last_ts = combined_logs.loc[case_mask, "time:timestamp"].max()
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

        if feedback_rows:
            combined_logs = pd.concat(
                [combined_logs, pd.DataFrame(feedback_rows)], ignore_index=True
            ).sort_values(by="time:timestamp")

        # Merge coffee machine rows (stream 2). Same shape as the feedback
        # injection above: read the source CSV, map to canonical columns,
        # filter to NEW case_ids, concat, re-sort. Rows keyed to already-
        # covered cases remain in the caller's existing CSV.
        machine_rows = _load_coffee_machine_rows(COFFEE_MACHINE_LOG)
        if not machine_rows.empty:
            before = len(machine_rows)
            machine_rows = machine_rows[machine_rows["case_id"].isin(new_case_ids)]
            dropped = before - len(machine_rows)
            if dropped:
                print(
                    f"   ⚠️  Dropped {dropped} coffee-machine row(s) with "
                    f"case_id not in the new agent log slice "
                    f"(stale, mismatched, or belonging to an already-covered case)"
                )
            if not machine_rows.empty:
                combined_logs = pd.concat(
                    [combined_logs, machine_rows], ignore_index=True
                ).sort_values(by="time:timestamp")

        # Merge gateway decisions (stream 3). Same shape as the two blocks
        # above: read the JSONL source, project to canonical columns, filter
        # to NEW case_ids, concat, re-sort. This is what lets a shared
        # `_all_traces.csv` reproduce the guardrail dashboard panel on a
        # machine that never had `guardrail_log/events.jsonl`.
        gateway_rows = _load_gateway_rows(GUARDRAIL_LOG)
        if not gateway_rows.empty:
            before = len(gateway_rows)
            gateway_rows = gateway_rows[gateway_rows["case_id"].isin(new_case_ids)]
            dropped = before - len(gateway_rows)
            if dropped:
                print(
                    f"   ⚠️  Dropped {dropped} gateway_decision row(s) whose "
                    f"case_id is not in the new agent log slice "
                    f"(stale, mismatched, or belonging to an already-covered case)"
                )
            if not gateway_rows.empty:
                combined_logs = pd.concat(
                    [combined_logs, gateway_rows], ignore_index=True
                ).sort_values(by="time:timestamp")

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
            print("\nLog generation process completed successfully!")

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
        print("\n📈 Processing Summary:")
        print(f"   📊 Total traces processed: {total}")
        print(f"   ✅ Successful: {successful}")
        if already_covered:
            print(f"   ⏩ Already covered (skipped): {already_covered}")
        print(f"   ⏭️  Skipped (no events): {skipped}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏱️  Build time: {total_seconds:.2f}s total", end="")
        if per_trace_seconds:
            avg = sum(per_trace_seconds) / len(per_trace_seconds)
            print(
                f" (avg {avg * 1000:.1f}ms/trace, "
                f"min {min(per_trace_seconds) * 1000:.1f}ms, "
                f"max {max(per_trace_seconds) * 1000:.1f}ms)"
            )
        else:
            print("")

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
            print(f"\n✅ Log file generated at {file_path}")
            return file_path
        except Exception as e:
            print(f"\n″❌ Failed to generate log file at {file_path}: {e}")
            return None
