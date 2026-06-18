import csv
import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from .log_generator import LogGenerator
import pandas as pd

COFFEE_MACHINE_LOG = Path("services/coffee_machine/logs/coffee_machine.csv")


def _load_coffee_machine_rows(path: Path) -> pd.DataFrame:
    """Read the coffee machine's raw CSV and map it to the canonical schema.

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

        feedback_store = {}
        feedback_path = Path("./feedback_store.json")
        if feedback_path.exists():
            with open(feedback_path) as f:
                feedback_store = json.load(f)

        successful_ingestions = 0
        failed_ingestions = 0
        skipped_ingestions = 0  # processed without error but produced 0 events

        combined_logs = pd.DataFrame()

        for i, trace in enumerate(traces, 1):
            trace_dict = trace.to_dict()
            trace_id = trace_dict.get('info', {}).get('trace_id', f'trace-{i}')
            print(f"\t📂 Processing trace {i}/{len(traces)}: {trace_id}")

            log_generator = LogGenerator()
            try:
                trace_event_log = log_generator.generate_event_log_df(trace_dict)
            except Exception as e:
                print(f"   ❌ Failed to generate event log for {trace_id}: {e}")
                failed_ingestions += 1
                continue

            if trace_event_log.empty:
                # LogGenerator returns an empty frame for traces with no
                # spans, no LangGraph root, etc. (e.g. standalone ChatOllama
                # calls from get_feedback()). These are legitimately-skipped,
                # not successful — counting them as such hides real bugs.
                skipped_ingestions += 1
                continue

            combined_logs = pd.concat([combined_logs, trace_event_log], ignore_index=True)
            successful_ingestions += 1

        # Sort combined logs by timestamp. combined_logs starts as an empty
        # DataFrame with no columns; if every trace either failed ingestion or
        # returned an empty frame (e.g. feedback-only traces with no LangGraph
        # root — see log_generator.py:33), there is no "time:timestamp" column
        # to sort by and pandas raises KeyError. Bail out cleanly instead so
        # the dashboard's export button doesn't surface a cryptic
        # "❌ time:timestamp".
        if combined_logs.empty or "time:timestamp" not in combined_logs.columns:
            print("⚠️  No usable events extracted from traces; nothing to export.")
            print("\n📈 Processing Summary:")
            print(f"   📊 Total traces processed: {len(traces)}")
            print(f"   ✅ Successful: {successful_ingestions}")
            print(f"   ⏭️  Skipped (no events): {skipped_ingestions}")
            print(f"   ❌ Failed: {failed_ingestions}")
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
            })

        if feedback_rows:
            combined_logs = pd.concat(
                [combined_logs, pd.DataFrame(feedback_rows)], ignore_index=True
            ).sort_values(by="time:timestamp")

        # Merge coffee machine rows (stream 2). Same shape as the feedback
        # injection above: read the source CSV, map to canonical columns,
        # filter to known case_ids, concat, re-sort.
        machine_rows = _load_coffee_machine_rows(COFFEE_MACHINE_LOG)
        merged_machine_rows = False
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
                merged_machine_rows = True

        written_path = self._generate_log_file(
            combined_logs, "./generated_event_log", json_format=export_as_json
        )

        # Only truncate the source CSV if the merge actually pulled rows from
        # it AND the unified log was written successfully. Otherwise a write
        # failure would lose source data.
        if merged_machine_rows and written_path:
            self._truncate_coffee_machine_log(COFFEE_MACHINE_LOG)

        print("\n📈 Processing Summary:")
        print(f"   📊 Total traces processed: {len(traces)}")
        print(f"   ✅ Successful: {successful_ingestions}")
        print(f"   ⏭️  Skipped (no events): {skipped_ingestions}")
        print(f"   ❌ Failed: {failed_ingestions}")

        if successful_ingestions > 0:
            print("\nLog generation process completed successfully!")
        if len(traces) == 0:
            print("\nNo trace files found. Make sure you have completed some coffee shop interactions first.")
            print("💡 Go back to step 4 and create some orders to generate trace data.")

        return

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

    def _truncate_coffee_machine_log(self, path: Path) -> None:
        """Reset the coffee machine CSV after a successful merge.

        Writes only the header row so subsequent appends from the FastAPI
        service are valid. The header MUST mirror logger.FIXED_HEADER in
        services/coffee_machine/logger.py — keep the two in sync.

        Assumes the FastAPI worker is not actively writing during this call
        (true today: process_all_traces runs after the session ends).
        """
        header = [
            "case_id", "concept:name", "ocel_time", "duration",
            "org:resource", "job_id", "drink",
        ]
        try:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)
            print(f"   🧹 Truncated {path}")
        except OSError as e:
            print(f"   ⚠️  Failed to truncate {path}: {e}")


def _extract_case_id(trace_dict: dict) -> str | None:
    """Extract the thread_id (case_id) from a raw MLflow trace dict."""
    spans = trace_dict.get("data", {}).get("spans", trace_dict.get("spans", []))
    root = next((s for s in spans if s.get("name") == "LangGraph"), None)
    if not root:
        return None
    try:
        return json.loads(root["attributes"]["metadata"]).get("thread_id")
    except (KeyError, json.JSONDecodeError):
        return None
