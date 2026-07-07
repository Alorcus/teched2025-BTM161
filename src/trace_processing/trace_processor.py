import json
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from .log_generator import LogGenerator
import pandas as pd

COFFEE_MACHINE_LOG = Path("services/coffee_machine/logs/coffee_machine.csv")

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

    def process_all_traces(self, export_as_json: bool = False):
        """
        Process all traces found via the MLflow API.
        """

        print("🔍 Searching for traces...")
        traces = self._get_all_traces()

        if not traces:
            print("❌ No traces found in MLflow")
            return {"total": 0, "successful": 0, "failed": 0}

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

        # Per-trace extraction accumulates into a list; a single pd.concat at
        # the end avoids the O(n²) copy cost of concatenating in the loop.
        trace_frames: list[pd.DataFrame] = []
        # case_id -> (setup_name, scenario_index) from MLflow trace tags. Filled
        # while iterating traces; broadcast onto every event row (all sources)
        # after all concats complete, so filter/aggregate queries can slice by
        # setup or scenario without joining an extra table.
        case_tags: dict[str, tuple[str | None, int]] = {}

        per_trace_seconds: list[float] = []

        for i, trace in enumerate(traces, 1):
            trace_start = time.perf_counter()
            trace_dict = trace.to_dict()
            trace_id = trace_dict.get('info', {}).get('trace_id', f'trace-{i}')
            trace_tags = trace_dict.get('info', {}).get('tags', {}) or {}
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
            print("⚠️  No usable events extracted from traces; nothing to export.")
            self._print_summary(
                total=len(traces),
                successful=successful_ingestions,
                skipped=skipped_ingestions,
                failed=failed_ingestions,
                total_seconds=time.perf_counter() - build_start,
                per_trace_seconds=per_trace_seconds,
            )
            return
        combined_logs.sort_values(by="time:timestamp", inplace=True)

        # Append exactly one user_feedback event per case, after all other events
        feedback_rows = []
        for case_id, fb in feedback_store.items():
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
        # filter to known case_ids, concat, re-sort.
        machine_rows = _load_coffee_machine_rows(COFFEE_MACHINE_LOG)
        if not machine_rows.empty:
            valid_case_ids = set(combined_logs["case_id"].unique())
            before = len(machine_rows)
            machine_rows = machine_rows[machine_rows["case_id"].isin(valid_case_ids)]
            dropped = before - len(machine_rows)
            if dropped:
                print(
                    f"   ⚠️  Dropped {dropped} coffee-machine row(s) with "
                    f"case_id not in agent log (stale or correlation_id mismatch)"
                )
            if not machine_rows.empty:
                combined_logs = pd.concat(
                    [combined_logs, machine_rows], ignore_index=True
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

        self._generate_log_file(
            combined_logs, "./generated_event_log", json_format=export_as_json
        )

        self._print_summary(
            total=len(traces),
            successful=successful_ingestions,
            skipped=skipped_ingestions,
            failed=failed_ingestions,
            total_seconds=time.perf_counter() - build_start,
            per_trace_seconds=per_trace_seconds,
        )

        if successful_ingestions > 0:
            print("\nLog generation process completed successfully!")
        if len(traces) == 0:
            print("\nNo trace files found. Make sure you have completed some coffee shop interactions first.")
            print("💡 Go back to step 4 and create some orders to generate trace data.")

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
    ) -> None:
        print("\n📈 Processing Summary:")
        print(f"   📊 Total traces processed: {total}")
        print(f"   ✅ Successful: {successful}")
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
