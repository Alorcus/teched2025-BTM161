import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from .log_generator import LogGenerator
import pandas as pd

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

        combined_logs = pd.DataFrame()

        for i, trace in enumerate(traces, 1):
            trace_dict = trace.to_dict()
            trace_id = trace_dict.get('info', {}).get('trace_id', f'trace-{i}')
            print(f"\t📂 Processing trace {i}/{len(traces)}: {trace_id}")

            log_generator = LogGenerator()
            try:
                trace_event_log = log_generator.generate_event_log_df(trace_dict)
                combined_logs = pd.concat([combined_logs, trace_event_log], ignore_index=True)
                successful_ingestions += 1
            except Exception as e:
                print(f"   ❌ Failed to generate event log for {trace_id}: {e}")
                failed_ingestions += 1
                continue

        # Sort combined logs by timestamp
        combined_logs.sort_values(by="time:timestamp", inplace=True)

        # Append exactly one customer_feedback event per case, after all other events
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
                "concept:name": "customer_feedback",
                "concept:instance": f"customer rates: {fb['feedback_score']}",
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

        self._generate_log_file(combined_logs, "./generated_event_log", json_format=export_as_json)

        print("\n📈 Processing Summary:")
        print(f"   📊 Total traces processed: {len(traces)}")
        print(f"   ✅ Successful: {successful_ingestions}")
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
        except Exception as e:
            print(f"\n″❌ Failed to generate log file at {file_path}: {e}")


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
