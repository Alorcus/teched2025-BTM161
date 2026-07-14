import json
import uuid
import polars as pl


def is_langgraph_root(span_name: str) -> bool:
    """MLflow autolog appends '_<n>' to the root span when the same LangGraph
    instance is invoked multiple times in one process (e.g. 'LangGraph_1').
    Accept the bare name and any numeric suffix."""
    if not span_name:
        return False
    if span_name == "LangGraph":
        return True
    return (
        span_name.startswith("LangGraph_") and span_name[len("LangGraph_") :].isdigit()
    )


class LogGenerator:
    def generate_event_log_df(self, trace_source) -> pl.DataFrame:
        self.process_events = []
        self.case_id = None
        self.spans = None
        self.langgraph_root_span = None

        if isinstance(trace_source, str):
            try:
                with open(trace_source, "r") as f:
                    trace_data = json.load(f)
            except Exception as e:
                raise Exception(f"Error loading trace file {trace_source}: {e}")
        elif isinstance(trace_source, dict):
            trace_data = trace_source
        else:
            raise ValueError(
                f"Expected file path (str) or trace data (dict), got {type(trace_source)}"
            )

        if "spans" in trace_data:
            self.spans = trace_data["spans"]
        elif "data" in trace_data and "spans" in trace_data["data"]:
            self.spans = trace_data["data"]["spans"]
        else:
            raise Exception("Cannot locate spans in trace data!")

        langgraph_roots = [
            span for span in self.spans if is_langgraph_root(span["name"])
        ]
        if not langgraph_roots:
            return pl.DataFrame()
        self.langgraph_root_span = langgraph_roots[0]
        self.case_id = json.loads(self.langgraph_root_span["attributes"]["metadata"])[
            "thread_id"
        ]

        self._process_root_span()

        if self._is_agent_span(self.langgraph_root_span):
            self._process_agent_span(self.langgraph_root_span)
        else:
            agent_spans = [
                span
                for span in self.spans
                if span["parent_span_id"] == self.langgraph_root_span["span_id"]
            ]

            for agent_span in agent_spans:
                self._process_agent_span(agent_span)

        dataframe = pl.DataFrame(self.process_events).sort("time:timestamp")
        # Trace may have been canceled before any LLM answer was recorded.
        if "duration" not in dataframe.columns:
            dataframe = dataframe.with_columns(
                pl.lit(None, dtype=pl.Int64).alias("duration")
            )
        # `time:timestamp` is nanosecond epoch (Int64) at this point; format
        # both it and `time_finished` as ms-precision naive ISO strings — the
        # shape trace_processor and the dashboard read downstream. Polars'
        # `%.3f` yields milliseconds directly, matching pandas' `%f[:-3]`.
        ts_dt = pl.from_epoch(pl.col("time:timestamp"), time_unit="ns")
        dataframe = dataframe.with_columns(
            (ts_dt + pl.duration(nanoseconds=pl.col("duration").fill_null(0)))
                .dt.strftime("%Y-%m-%dT%H:%M:%S%.3f")
                .alias("time_finished"),
            ts_dt.dt.strftime("%Y-%m-%dT%H:%M:%S%.3f").alias("time:timestamp"),
        )

        return dataframe

    def _get_span_metadata(self, span):
        # Some spans (e.g. RunnableSequence, ChatOllama) have no `metadata`
        # attribute — return an empty dict instead of raising KeyError so
        # callers can uniformly do `.get('langgraph_node')`.
        raw = span.get('attributes', {}).get('metadata')
        if raw is None:
            return {}
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return {}


    def _is_agent_span(self, span):
        child_spans = [s for s in self.spans if s["parent_span_id"] == span["span_id"]]
        for child in child_spans:
            # Legacy create_react_agent shape: agent_* span with a call_model grandchild.
            if child['name'].startswith('agent_'):
                grandchild_spans = [s for s in self.spans if s['parent_span_id'] == child['span_id']]
                if any(g['name'].startswith('call_model') for g in grandchild_spans):
                    return True

            # New guardrail control-plane subgraph: children are llm_N / tools / gateway.
            child_metadata = self._get_span_metadata(child)
            node = child_metadata.get('langgraph_node')
            if node in ("llm", "tools"):
                return True
            if child['name'].startswith('llm') or child['name'].startswith('tools'):
                return True

        return False

    def _process_llm_span(self, span, agent_name):
        call_model_child_spans = [s for s in self.spans if s['parent_span_id'] == span['span_id'] and s['name'].startswith('call_model')]
        if len(call_model_child_spans) == 1:
            # Legacy create_react_agent path: the LLM call is a call_model
            # grandchild under an agent_* span.
            span = call_model_child_spans[0]
        elif len(call_model_child_spans) == 0:
            # New guardrail control-plane subgraph AND modern MLflow LangChain
            # autolog: the llm span itself (named `llm` or `llm_N`) carries
            # mlflow.spanOutputs in the shape we need.
            pass
        else:
            print(f'Unexpected number of call_model children for {span["name"]}: {len(call_model_child_spans)}')
            return

        raw_output = span['attributes'].get('mlflow.spanOutputs')
        # prevent keyError if the simulation froze during an LLM call and was interrupted
        if raw_output is None:
            return
        span_output = json.loads(raw_output)["messages"][0]

        model_name = span_output.get("response_metadata", {}).get("model_name", None)

        response_message = None
        returned_contents = span_output.get("content", [])

        if isinstance(returned_contents, str):
            response_message = returned_contents
        elif isinstance(returned_contents, list):
            for content in returned_contents:
                if content.get("type", None) == "text":
                    response_message = content.get("text", None)

        usage_metadata = span_output.get("usage_metadata", {})

        self.process_events.append(
            {
                "case_id": self.case_id,
                "identity:id": str(uuid.uuid4()),
                "time:timestamp": span["start_time_unix_nano"],
                "time_finished": span["end_time_unix_nano"],
                "duration": span["end_time_unix_nano"] - span["start_time_unix_nano"],
                "concept:instance": f"{agent_name} calls llm",
                "concept:name": "call_llm",
                "org:resource": agent_name,
                "model": model_name,
                "input_tokens": usage_metadata.get("input_tokens", None),
                "response_tokens": usage_metadata.get("output_tokens", None),
                "message": response_message,
            }
        )

    def _process_tool_span(self, span, agent_name):
        parsed = json.loads(span["attributes"].get("mlflow.spanInputs", "[]"))
        tool_input = None
        if isinstance(parsed, list):
            tool_input = parsed[0] if parsed else None
        elif isinstance(parsed, dict) and "messages" in parsed:
            for msg in reversed(parsed["messages"]):
                if msg.get("type") == "ai" and msg.get("tool_calls"):
                    tool_input = msg["tool_calls"][0]
                    break

        if tool_input is None:
            return

        tool_name = "unknown_tool"
        if tool_input.get("type", None) == "tool_call":
            tool_name = tool_input.get("name", "unknown_tool")

        # Emit transfer_to_* calls as execute_tool rows. If the only tool
        # call in a conversation was a transfer, dropping it would collapse
        # the trace to user_prompt + user_feedback. The gateway can also
        # flag transfers, and its tool_call object (keyed by tool_call_id)
        # would dangle if the event row didn't exist. The event_type here
        # is the tool name itself (e.g. `transfer_to_customer_service_agent`),
        # distinct from the synthesised `<from>_handover_<to>` type.

        self.process_events.append(
            {
                "case_id": self.case_id,
                "identity:id": str(uuid.uuid4()),
                "time:timestamp": span["start_time_unix_nano"],
                "time_finished": span["end_time_unix_nano"],
                "duration": span["end_time_unix_nano"] - span["start_time_unix_nano"],
                "concept:name": "execute_tool",
                "concept:instance": f"{agent_name} uses tool {tool_name}",
                "org:resource": agent_name,
                "tool": tool_name,
                "tool_call_id": tool_input.get("id"),
            }
        )

    def _process_agent_span(self, agent_span):
        agent_span_id = agent_span["span_id"]
        agent_metadata = json.loads(agent_span["attributes"]["metadata"])

        agent_name = None

        if is_langgraph_root(agent_span["name"]):
            agent_name = "root_agent"
        else:
            agent_name = agent_metadata["langgraph_node"]

        if agent_name in ["__start__"]:
            return

        agent_child_spans = [
            span for span in self.spans if span["parent_span_id"] == agent_span_id
        ]

        if not self._is_agent_span(agent_span):
            if len(agent_child_spans) != 1:
                raise Exception(
                    f"Expected exactly one child span for non-agent span {agent_span['name']}, found {len(agent_child_spans)}:\n{[child['name'] for child in agent_child_spans]}"
                )
            sub_agent_span = agent_child_spans[0]
            agent_child_spans = [
                span
                for span in self.spans
                if span["parent_span_id"] == sub_agent_span["span_id"]
            ]

        for child_span in agent_child_spans:
            child_meta = self._get_span_metadata(child_span)
            node = child_meta.get('langgraph_node')
            name = child_span['name']

            if node == "llm" or name.startswith('agent') or name.startswith('llm'):
                self._process_llm_span(child_span, agent_name)
            elif node == "tools" or name.startswith('tools'):
                self._process_tool_span(child_span, agent_name)
            # gateway / route_after_* / __start__ / __end__: skip silently

    def _process_root_span(self):
        user_input = None
        span_inputs = json.loads(
            self.langgraph_root_span["attributes"]["mlflow.spanInputs"]
        )["messages"]
        for message in span_inputs:
            if (
                message.get("role", "") == "user"
                or message.get("type", None) == "human"
            ):
                user_input = message.get("content", None)

        self.process_events.append(
            {
                "case_id": self.case_id,
                "identity:id": str(uuid.uuid4()),
                "time:timestamp": self.langgraph_root_span["start_time_unix_nano"],
                "concept:instance": "prompt",
                "concept:name": "user_prompt",
                "org:resource": "user",
                "message": user_input,
            }
        )
