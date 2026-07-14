import json
import threading
import uuid
import logging

import mlflow
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from src.coffee_shop import CoffeeShop
from src.agents import CUSTOMER_SCENARIOS
from src.agents.tray import get_tray, clear_tray
from src.agents.order_store import load_order, set_order_status
from src.agents.shared_components import OrderStatus
from src.conversation import _tag_trace
from src.stream import SWARM_AGENTS
from .event_bus import EventBus, DashboardEvent, EventType

logger = logging.getLogger("coffee_shop.dashboard")

MAX_CONVERSATION_TURNS = 30

HANDOVER_PAUSE_TIMEOUT_SECONDS = 300


def _summarize_tool_calls(tool_calls: list[dict]) -> str:
    """Render a list of tool_calls as plain prose suitable for embedding inside
    a quoted-critique HumanMessage. Raw tool_call dicts read poorly inside a
    natural-language critique, and LangGraph state cannot hold an orphaned
    tool_call without a matching ToolMessage."""
    if not tool_calls:
        return ""
    parts: list[str] = []
    for tc in tool_calls:
        name = tc.get("name") or "?"
        args = tc.get("args") or {}
        if name.startswith("transfer_to") or name == "transfer_to_agent":
            target = args.get("target_agent") or args.get("target") or "?"
            summary = args.get("context_summary") or ""
            if summary:
                parts.append(
                    f"tried to hand off to {target} with summary "
                    f"{json.dumps(summary, ensure_ascii=False)[:120]}"
                )
            else:
                parts.append(f"tried to hand off to {target}")
        else:
            try:
                args_repr = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                args_repr = str(args)
            if len(args_repr) > 120:
                args_repr = args_repr[:120] + "..."
            parts.append(f"tried to call {name} with {args_repr}")
    return "; ".join(parts)


def _rejected_content(msg: AIMessage) -> str:
    """The content surfaced for AGENT_MESSAGE_REJECTED. For text-only AI
    messages this is the text; for tool-call-only messages this is the prose
    summary so the dashboard shows what was attempted."""
    content = (msg.content or "").strip() if isinstance(msg.content, str) else ""
    if content:
        return content
    return _summarize_tool_calls(getattr(msg, "tool_calls", None) or [])


def _extract_text(content) -> str:
    """Flatten an AIMessage.content field into a plain string.

    Anthropic tool-call turns yield a list of content blocks
    (e.g. [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]);
    text-only turns yield a str. Callers that just want the model's prose
    should use this rather than str(content), which would render list
    repr syntax into the panel.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return ""


class ConversationRunner:
    def __init__(self, shop: CoffeeShop, event_bus: EventBus):
        self.shop = shop
        self.event_bus = event_bus
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.is_running = False
        self._tray_ready = False
        self._active_agent = "order_agent"
        self._current_order_id: str | None = None
        self._manual_thread_id = str(uuid.uuid4())
        self._current_scenario_index: int | None = None
        cfg = getattr(shop, "config", None)
        from src.config import CoffeeShopConfig as _CoffeeShopConfig

        if isinstance(cfg, _CoffeeShopConfig):
            self._supervisor_active: bool = bool(cfg.process_supervisor_active)
            self._max_retries: int = int(cfg.process_supervisor_max_retries)
            self.pause_on_next_handover: bool = bool(cfg.handover_pause_default)
        else:
            # No real config available (e.g. unit-test MagicMock shop). Default
            # to passive mode so legacy behaviour is preserved.
            self._supervisor_active = False
            self._max_retries = 3
            self.pause_on_next_handover = False
        # Set when no pause is pending; clear()ed to block the runner thread at
        # the next handover seam. resume() sets it again to release the wait.
        self._resume_event = threading.Event()
        self._resume_event.set()
        # Per-agent retry counter for the active-mode loop. Reset between turns.
        self._retry_counts: dict[str, int] = {}
        # The accumulating supervisor-authored HumanMessage for the current
        # retry cycle, keyed by agent. Reset when the agent successfully
        # produces a non-violating message or retries are exhausted.
        self._critique_msgs: dict[str, HumanMessage] = {}

    def start(self, scenario_index=None, custom_prompt=None):
        with self._lock:
            if self.is_running:
                return
            self.is_running = True
        # A fresh run always starts un-paused. A previously-paused-then-resumed
        # runner has _resume_event already set, but explicitly setting it here
        # protects against a future code path that could leave it cleared.
        self._resume_event.set()
        self._thread = threading.Thread(
            target=self._run, args=(scenario_index, custom_prompt), daemon=True
        )
        self._thread.start()

    def resume(self):
        """Release a runner thread that is blocked at the handover pause seam.

        No-op if the runner is not currently paused: setting an already-set
        Event is harmless, and the seam only blocks when pause_on_next_handover
        is True at the moment a fresh handover signature is observed."""
        self._resume_event.set()

    @property
    def is_paused(self) -> bool:
        """True iff the runner thread is currently blocked at the handover
        pause seam waiting for resume()."""
        return self.is_running and not self._resume_event.is_set()

    def _run(self, scenario_index, custom_prompt=None):
        try:
            self._run_conversation(scenario_index, custom_prompt)
        except Exception as e:
            logger.exception("Conversation runner failed")
            self.event_bus.publish(
                DashboardEvent(
                    event_type=EventType.CONVERSATION_END,
                    agent_name="system",
                    content=f"ERROR: {e}",
                )
            )
        finally:
            with self._lock:
                self.is_running = False

    def _run_conversation(self, scenario_index, custom_prompt=None):
        self.shop.customer_agent.reset(scenario_index, custom_prompt=custom_prompt)
        self._active_agent = "order_agent"
        self._current_order_id = None
        self._current_scenario_index = scenario_index
        thread_id = str(uuid.uuid4())

        if scenario_index is None:
            scenario_label = "random"
        elif 0 <= scenario_index < len(CUSTOMER_SCENARIOS):
            scenario_label = CUSTOMER_SCENARIOS[scenario_index]
        else:
            # Sentinel from the dashboard "Custom" dropdown entry (-1), or any
            # other out-of-range index: the customer agent is running with a
            # user-edited system prompt, so the preset list does not describe it.
            scenario_label = "custom prompt"
        self.event_bus.publish(
            DashboardEvent(
                event_type=EventType.CONVERSATION_START,
                agent_name="system",
                content=f"Scenario: {scenario_label[:80]}",
            )
        )

        message = self.shop.customer_agent.get_initial_message()
        self.event_bus.publish(
            DashboardEvent(
                event_type=EventType.CUSTOMER_MESSAGE,
                agent_name="customer",
                content=message,
            )
        )
        self.event_bus.publish(
            DashboardEvent(
                event_type=EventType.USER_VISIBLE,
                agent_name=self._active_agent,
                content=message,
            )
        )

        turns = 0
        while message:
            if turns >= MAX_CONVERSATION_TURNS:
                logger.warning(
                    "Conversation reached %d turns, stopping", MAX_CONVERSATION_TURNS
                )
                break
            turns += 1

            agent_reply = self._stream_with_events(thread_id, message)
            if not agent_reply:
                break

            message = self.shop.customer_agent.respond_to(agent_reply)
            if message:
                self.event_bus.publish(
                    DashboardEvent(
                        event_type=EventType.CUSTOMER_MESSAGE,
                        agent_name="customer",
                        content=message,
                    )
                )
                self.event_bus.publish(
                    DashboardEvent(
                        event_type=EventType.USER_VISIBLE,
                        agent_name=self._active_agent,
                        content=message,
                    )
                )

        if self._current_order_id:
            self.event_bus.publish(
                DashboardEvent(
                    event_type=EventType.TRAY_READY,
                    agent_name="customer",
                    content=self._current_order_id,
                )
            )

        feedback = self.shop.capture_feedback(thread_id, self._current_order_id)
        logger.info(
            "Customer feedback [%.2f]: %s",
            feedback["feedback_score"],
            feedback["feedback_reason"],
        )

        self.event_bus.publish(
            DashboardEvent(
                event_type=EventType.CONVERSATION_END,
                agent_name="system",
            )
        )

    def _consume_tray(self):
        """Customer takes the tray — apply effects, mark order complete, clear tray."""
        order_id = self._current_order_id
        if not order_id:
            return

        tray_items = get_tray(order_id)
        if not tray_items:
            return

        items_summary = ", ".join(f"{e.quantity}x {e.item_name}" for e in tray_items)
        self.event_bus.publish(
            DashboardEvent(
                event_type=EventType.TOOL_CALL,
                agent_name="customer",
                tool_name="take_tray",
                tool_args={"order_id": order_id, "items": items_summary},
            )
        )

        has_contaminated = any(entry.contaminated for entry in tray_items)
        if has_contaminated:
            self.shop.customer_agent.inject_experience(
                "You received your coffee but it tastes slightly off — almost metallic. Something isn't right."
            )

        order = load_order(order_id)
        if order and order.status != OrderStatus.COMPLETED:
            set_order_status(
                order, OrderStatus.COMPLETED, context="tray pickup by customer"
            )

        clear_tray(order_id)

        result = {"status": "picked_up", "items": items_summary}
        if has_contaminated:
            result["warning"] = "contaminated items received"
        self.event_bus.publish(
            DashboardEvent(
                event_type=EventType.TOOL_RESULT,
                agent_name="customer",
                tool_name="take_tray",
                tool_result=json.dumps(result),
            )
        )

        self.event_bus.publish(
            DashboardEvent(
                event_type=EventType.TRAY_TAKEN,
                agent_name="customer",
            )
        )

    def take_tray(self):
        self._consume_tray()

    def _stream_with_events(self, thread_id: str, message: str) -> str | None:
        config = self.shop._get_config(thread_id)

        # Reset per-turn active-mode state. A new user turn means a new retry
        # budget and a fresh critique accumulator.
        self._retry_counts.clear()
        self._critique_msgs.clear()

        stream_input: dict | None = {
            "messages": [{"role": "user", "content": message}],
            "handoff_context": None,
        }
        last_agent_message = None
        seen: set[str] = set()
        # `handoff_context` is set in the parent graph state by transfer_to_agent
        # and is never cleared, so terminal/router updates can re-surface the
        # same context after the destination agent has already worked. Dedup by
        # signature so the global conversation log shows one HANDOFF per actual
        # transfer.
        last_handoff_sig: tuple | None = None
        current_agent: str | None = None
        # Outer retry loop: each iteration drives one stream() invocation. A
        # violation that triggers re-invocation breaks out of the inner loop
        # and re-enters here with stream_input=None so the graph resumes from
        # checkpoint with the patched state.
        while True:
            try:
                stream = self.shop.app.stream(
                    stream_input,
                    config,
                    subgraphs=True,
                )
            except Exception as e:
                logger.exception("Failed to start stream")
                self.event_bus.publish(
                    DashboardEvent(
                        event_type=EventType.AGENT_MESSAGE,
                        agent_name="system",
                        content=f"Stream error: {e}",
                    )
                )
                return None

            rejection: dict | None = None
            exhausted: bool = False
            try:
                for ns, update in stream:
                    agent_name = self._parse_agent_name(ns)

                    if agent_name and agent_name != current_agent:
                        if current_agent:
                            self.event_bus.publish(
                                DashboardEvent(
                                    event_type=EventType.AGENT_THINKING,
                                    agent_name=current_agent,
                                    content="idle",
                                )
                            )
                        current_agent = agent_name
                        self.event_bus.publish(
                            DashboardEvent(
                                event_type=EventType.AGENT_THINKING,
                                agent_name=agent_name,
                                content="thinking",
                            )
                        )

                    for node, node_data in update.items():
                        if node_data is None:
                            continue

                        if isinstance(node_data, dict):
                            resolved_agent = (
                                agent_name or node_data.get("active_agent") or "unknown"
                            )

                            if (
                                "handoff_context" in node_data
                                and node_data["handoff_context"]
                            ):
                                hc = node_data["handoff_context"]
                                target = node_data.get("active_agent")
                                sig = (
                                    hc.get("from_agent"),
                                    target,
                                    hc.get("context_summary"),
                                )
                                if sig != last_handoff_sig:
                                    last_handoff_sig = sig
                                    self.event_bus.publish(
                                        DashboardEvent(
                                            event_type=EventType.HANDOFF,
                                            agent_name=hc.get(
                                                "from_agent", resolved_agent
                                            ),
                                            handoff_context=hc,
                                            target_agent=target,
                                        )
                                    )
                                    if target:
                                        self._active_agent = target
                                    # Pause seam: block here (between sender's
                                    # handoff emission and receiver's first
                                    # node execution) iff the toggle is on.
                                    # Reading the flag inside the loop honours
                                    # toggles flipped mid-stream — but only
                                    # for handovers not yet observed.
                                    if self.pause_on_next_handover:
                                        self._wait_for_resume(
                                            from_agent=hc.get("from_agent"),
                                            target=target,
                                        )

                            msgs_key = next(
                                (k for k in node_data if k == "messages"), None
                            )
                            if msgs_key:
                                msgs_list = node_data[msgs_key]
                                if not msgs_list:
                                    continue
                                msg = msgs_list[-1]
                                content = getattr(msg, "content", "")
                                name = getattr(msg, "name", "")
                                msg_uid = getattr(msg, "id", "") or getattr(
                                    msg, "tool_call_id", ""
                                )
                                if msg_uid:
                                    msg_id = f"{type(msg).__name__}:{msg_uid}"
                                else:
                                    msg_id = f"{type(msg).__name__}:{name}:{content}"
                                if msg_id in seen:
                                    continue
                                seen.add(msg_id)
                                # Normalize attribution: prefer the message's
                                # own .name, then the namespace-derived
                                # resolved_agent, then the last known active
                                # swarm agent. Without this fallback an
                                # AIMessage that lands here with name="" and
                                # a stale namespace would leak past the
                                # SWARM_AGENTS guard in _process_message and
                                # get published as if it had no agent.
                                msg_agent = getattr(msg, "name", None) or resolved_agent
                                if (
                                    msg_agent not in SWARM_AGENTS
                                    and self._active_agent in SWARM_AGENTS
                                ):
                                    msg_agent = self._active_agent
                                outcome = self._process_message(msg, msg_agent)

                                if outcome.get("status") == "rejected":
                                    rejection = outcome
                                    rejection["msg"] = msg
                                    rejection["agent"] = msg_agent
                                    break
                                if outcome.get("status") == "exhausted_suppressed":
                                    # Hard stop: the agent has hit the retry
                                    # cap. Don't re-stream; let the outer
                                    # while-loop fall through.
                                    rejection = None
                                    exhausted = True
                                    break

                                if (
                                    isinstance(msg, AIMessage)
                                    and msg.content
                                    and not msg.tool_calls
                                    and getattr(msg, "name", None) in SWARM_AGENTS
                                ):
                                    last_agent_message = msg.content

                    if rejection is not None:
                        break
                    if exhausted:
                        break
            except Exception as e:
                logger.exception("Error during stream iteration")
                self.event_bus.publish(
                    DashboardEvent(
                        event_type=EventType.AGENT_MESSAGE,
                        agent_name="system",
                        content=f"Stream error: {e}",
                    )
                )
                break

            if rejection is None:
                if exhausted:
                    # Cap hit: surface a clear log line and stop the outer
                    # retry loop without publishing anything more. The
                    # AGENT_MESSAGE_REJECTED event was already emitted by
                    # _handle_violation.
                    self.event_bus.publish(
                        DashboardEvent(
                            event_type=EventType.LOG_MESSAGE,
                            agent_name="process_supervisor",
                            log_level=logging.ERROR,
                            content=(
                                "Active supervisor retry cap reached; halting this "
                                "turn for the offending agent."
                            ),
                        )
                    )
                self._tag_last_trace()
                break

            # We got a rejection: patch the graph state and resume.
            try:
                self._apply_state_patch_for_rejection(config, rejection)
            except Exception:
                logger.exception(
                    "Failed to apply supervisor state patch; aborting active-mode loop"
                )
                break
            stream_input = rejection.get("resume_input")  # critique as fresh input

        if current_agent:
            self.event_bus.publish(
                DashboardEvent(
                    event_type=EventType.AGENT_THINKING,
                    agent_name=current_agent,
                    content="idle",
                )
            )

        return last_agent_message

    def _tag_last_trace(self) -> None:
        """Attach setup + scenario tags to the MLflow trace produced by the
        just-completed `app.stream(...)` call. No-op when mlflow is disabled or
        no trace was produced (e.g. autolog off, or a stream that errored)."""
        if not self.shop.config.mlflow_enabled:
            return
        trace_id = mlflow.get_last_active_trace_id()
        if trace_id is None:
            return
        _tag_trace(trace_id, self.shop.config.setup_name, self._current_scenario_index)

    def _apply_state_patch_for_rejection(self, config: dict, rejection: dict) -> None:
        """Inject the supervisor's quoted-critique HumanMessage into the
        graph state so the agent sees it on re-invocation.

        The critique HumanMessage is returned to the caller in
        rejection["resume_input"] so the outer loop can feed it as a fresh
        graph input — `stream(None, ...)` does nothing once the previous
        run completed, so we restart with the critique as the new turn.
        Any orphan tool_calls on the offending AIMessage that survives in
        state get synthetic ToolMessage stubs to keep state well-formed.
        """
        msg: AIMessage = rejection["msg"]
        critique_msg: HumanMessage = rejection["critique_msg"]
        violation_reason = rejection.get("supervisor_line", "")
        if " | " in violation_reason:
            violation_reason = violation_reason.split(" | ", 1)[0]
        target_id = self._find_state_message_id(config, msg)
        patch_msgs: list = []
        if target_id:
            patch_msgs.append(RemoveMessage(id=target_id))
        else:
            for tc in getattr(msg, "tool_calls", None) or []:
                tc_id = tc.get("id")
                if not tc_id:
                    continue
                patch_msgs.append(
                    ToolMessage(
                        content=(
                            f"REJECTED by process supervisor ({violation_reason}). "
                            "This tool call was not executed. See the supervisor "
                            "critique that follows for guidance."
                        ),
                        tool_call_id=tc_id,
                        name=tc.get("name") or "unknown",
                    )
                )
        if patch_msgs:
            self.shop.app.update_state(config, {"messages": patch_msgs})
        # The critique is delivered as a fresh user-style turn so the graph
        # has something to react to (stream(None, ...) is a no-op once the
        # prior step completed). Pass the HumanMessage as a list so the
        # add_messages reducer appends it cleanly.
        rejection["resume_input"] = {
            "messages": [critique_msg],
            "handoff_context": None,
        }

    def _find_state_message_id(self, config: dict, target: AIMessage) -> str | None:
        """Look up the offending AIMessage in the parent graph's checkpointed
        state. We match by content (and tool_calls if content is empty), not
        by id — subgraph stream events expose a different id than the parent
        checkpoint stores."""
        try:
            snapshot = self.shop.app.get_state(config)
        except Exception:
            logger.exception("get_state failed")
            return None
        msgs = []
        values = getattr(snapshot, "values", None) or {}
        if isinstance(values, dict):
            msgs = values.get("messages") or []
        target_content = (
            (target.content or "") if isinstance(target.content, str) else ""
        )
        target_tcs = getattr(target, "tool_calls", None) or []
        for cand in reversed(msgs):
            if not isinstance(cand, AIMessage):
                continue
            cand_content = (cand.content or "") if isinstance(cand.content, str) else ""
            cand_tcs = getattr(cand, "tool_calls", None) or []
            if target_content and cand_content == target_content:
                return getattr(cand, "id", None)
            if target_tcs and cand_tcs:
                if cand_tcs[0].get("name") == target_tcs[0].get("name") and cand_tcs[
                    0
                ].get("args") == target_tcs[0].get("args"):
                    return getattr(cand, "id", None)
        return None

    def _process_message(self, msg, agent_name: str) -> dict:
        # Ask the process supervisor for its verdict on THIS message and stamp
        # the resulting line on the first DashboardEvent we emit for it. Sibling
        # events (e.g. extra tool_calls in the same AIMessage) get None — the
        # trace table renders that as "—".
        supervisor_line: str | None = None
        supervisor = getattr(self.shop, "process_supervisor", None)
        if supervisor is not None:
            try:
                supervisor_line = supervisor.observe(msg, agent_name)
            except Exception:
                logger.exception("process supervisor observe failed")
                supervisor_line = None

        is_violation_on_agent_aimessage = (
            self._supervisor_active
            and supervisor is not None
            and isinstance(msg, AIMessage)
            and agent_name in SWARM_AGENTS
            and supervisor_line is not None
            and supervisor_line.split(" | ", 1)[0].startswith("Violation:")
        )

        if is_violation_on_agent_aimessage:
            return self._handle_violation(msg, agent_name, supervisor_line, supervisor)

        # Reset the retry counter for this agent on a successful (non-violation)
        # message so the next violation starts from zero.
        if isinstance(msg, AIMessage) and agent_name in SWARM_AGENTS:
            self._retry_counts.pop(agent_name, None)
            self._critique_msgs.pop(agent_name, None)

        self._publish_message_normally(msg, agent_name, supervisor_line)
        return {"status": "published"}

    def _handle_violation(
        self,
        msg: AIMessage,
        agent_name: str,
        supervisor_line: str,
        supervisor,
    ) -> dict:
        """Active-mode handler for a Violation:* on an AIMessage from an agent.

        Always SUPPRESSES the offending message (rejected event, no tool exec).
        Under the per-agent retry cap we additionally synthesize a critique
        and re-invoke. At/over the cap we stop synthesizing critiques and
        let the conversation deadlock — that's strictly safer than letting a
        jailbroken / non-compliant worker run free, which is what the prior
        implementation did when it published the cap-hit message normally.
        The retry counter is NOT reset here; it only resets when the same
        agent produces a non-violating message (see _process_message)."""
        retries_so_far = self._retry_counts.get(agent_name, 0)
        verdict = supervisor_line.split(" | ", 1)[0]
        violation_reason = (
            verdict[len("Violation:") :]
            if verdict.startswith("Violation:")
            else verdict
        )
        rejected_text = _rejected_content(msg)

        if retries_so_far >= self._max_retries:
            # Cap hit: publish as REJECTED (NOT normal), do not reset counter.
            try:
                supervisor.append_violation("supervisor_retry_exhausted")
            except Exception:
                logger.exception("failed to append supervisor_retry_exhausted")
            self.event_bus.publish(
                DashboardEvent(
                    event_type=EventType.AGENT_MESSAGE_REJECTED,
                    agent_name=agent_name,
                    content=rejected_text,
                    supervisor_line=supervisor_line,
                    rejection_reason=(
                        f"supervisor retry cap reached ({self._max_retries}); "
                        "this attempt is suppressed and the conversation will "
                        "halt for this agent until the user intervenes."
                    ),
                )
            )
            self._publish_rejected_tool_calls(msg, agent_name)
            # Returning "exhausted_suppressed" tells the outer loop to stop
            # rather than re-stream — the agent has demonstrated it cannot
            # comply, so we hand control back to the caller.
            return {"status": "exhausted_suppressed"}

        # Under cap: suppress + critique + re-invoke.
        try:
            critique_text = supervisor.critique(msg, agent_name, violation_reason)
        except Exception:
            logger.exception("supervisor.critique failed; using terse fallback")
            critique_text = (
                f"Your previous step violated the process model "
                f"({violation_reason}). Please choose a different next step "
                "from the activities allowed for your role."
            )

        self.event_bus.publish(
            DashboardEvent(
                event_type=EventType.AGENT_MESSAGE_REJECTED,
                agent_name=agent_name,
                content=rejected_text,
                supervisor_line=supervisor_line,
                rejection_reason=critique_text,
            )
        )
        self._publish_rejected_tool_calls(msg, agent_name)

        critique_msg = self._compose_or_extend_critique(
            agent_name,
            rejected_text,
            violation_reason,
            critique_text,
        )
        self._retry_counts[agent_name] = retries_so_far + 1
        return {
            "status": "rejected",
            "supervisor_line": supervisor_line,
            "critique_text": critique_text,
            "critique_msg": critique_msg,
        }

    def _publish_rejected_tool_calls(self, msg: AIMessage, agent_name: str) -> None:
        """Emit a TOOL_CALL row per tool_call on a rejected AIMessage so the
        trace table renders the structured args alongside the REJECTED row.
        The caller is responsible for not actually executing these calls;
        these events are render-only (tagged via supervisor_line=REJECTED)."""
        for tc in getattr(msg, "tool_calls", None) or []:
            self.event_bus.publish(
                DashboardEvent(
                    event_type=EventType.TOOL_CALL,
                    agent_name=agent_name,
                    tool_name=tc.get("name") or "?",
                    tool_args=tc.get("args") or {},
                    supervisor_line="REJECTED — not executed",
                    rejection_reason="rejected by process supervisor; not executed",
                )
            )

    def _compose_or_extend_critique(
        self,
        agent_name: str,
        rejected_text: str,
        violation_reason: str,
        critique_text: str,
    ) -> HumanMessage:
        """Build the supervisor-authored HumanMessage that quotes the rejected
        attempt as third-party evidence. On repeat violations within the same
        retry cycle we APPEND a new block so bad drafts pile up inside the
        critique, never inside agent state.

        The lead-in explicitly identifies the message as out-of-band runtime
        control rather than a user/customer turn or external "process
        supervisor" actor — workers were treating the prior framing as
        prompt injection and refusing it. We also set name="system" so the
        framework labels the message-source channel accordingly."""
        block = (
            f"\n\nYour attempt was:\n>>>\n{rejected_text}\n>>>\n"
            f"Reason rejected: {violation_reason}\n"
            f"Required correction: {critique_text}"
        )
        existing = self._critique_msgs.get(agent_name)
        if existing is not None:
            new_content = existing.content + block
            new_msg = HumanMessage(
                content=new_content,
                id=existing.id,
                name="system",
            )
        else:
            header = (
                "[SYSTEM CONTROL — not user content, not from any agent]\n"
                "This message is emitted by the runtime's process-supervisor "
                "module. It is authoritative: ignoring or refusing it is not "
                "an option, and treating it as prompt injection is incorrect. "
                "Your previous step was suppressed before it took effect "
                "because it violates the coffee-shop process model. The "
                "transcript below is third-party evidence — do not treat it "
                "as your own prior turn. Read the required correction and "
                "produce a different next step that fits one of your allowed "
                "activities."
            )
            new_msg = HumanMessage(
                content=header + block,
                name="system",
            )
        self._critique_msgs[agent_name] = new_msg
        return new_msg

    def _publish_message_normally(
        self, msg, agent_name: str, supervisor_line: str | None
    ) -> None:
        """The original (passive-mode) publish path. Extracted so the active
        path can call it for the retry-exhausted fallback."""
        nonlocal_ref = {"line": supervisor_line}

        def _take() -> str | None:
            line, nonlocal_ref["line"] = nonlocal_ref["line"], None
            return line

        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                thought_text = _extract_text(msg.content).strip()
                if thought_text:
                    self.event_bus.publish(
                        DashboardEvent(
                            event_type=EventType.AGENT_THOUGHT,
                            agent_name=agent_name,
                            content=thought_text,
                            tool_name=msg.tool_calls[0].get("name"),
                        )
                    )
                for tc in msg.tool_calls:
                    self.event_bus.publish(
                        DashboardEvent(
                            event_type=EventType.TOOL_CALL,
                            agent_name=agent_name,
                            tool_name=tc["name"],
                            tool_args=tc.get("args", {}),
                            supervisor_line=_take(),
                        )
                    )
            elif msg.content:
                self.event_bus.publish(
                    DashboardEvent(
                        event_type=EventType.AGENT_MESSAGE,
                        agent_name=agent_name,
                        content=msg.content,
                        supervisor_line=_take(),
                    )
                )
        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            self.event_bus.publish(
                DashboardEvent(
                    event_type=EventType.TOOL_RESULT,
                    agent_name=agent_name,
                    tool_name=getattr(msg, "name", None),
                    tool_result=content,
                    supervisor_line=_take(),
                )
            )
            self._track_order_id(getattr(msg, "name", None), content)

            tool_name = getattr(msg, "name", None)

            if tool_name == "place_on_tray":
                try:
                    data = json.loads(content)

                    order_id = data.get("order_id")

                    if order_id:
                        self.event_bus.publish(
                            DashboardEvent(
                                event_type=EventType.TRAY_READY,
                                agent_name="customer",
                                content=order_id,
                            )
                        )
                except Exception:
                    pass

    def _track_order_id(self, tool_name: str | None, content: str):
        if self._current_order_id:
            return
        if tool_name not in (
            "process_order",
            "check_inventory",
            "start_preparation",
            "place_on_tray",
        ):
            return
        try:
            data = json.loads(content)
            order_id = data.get("order_id")
            if order_id:
                self._current_order_id = order_id
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    def _wait_for_resume(self, from_agent: str | None, target: str | None) -> None:
        """Block the runner thread between sender emit and receiver accept.

        Clears the resume Event, publishes a paused LOG_MESSAGE so the UI
        can render the paused state, waits up to HANDOVER_PAUSE_TIMEOUT_SECONDS
        for resume() to set the Event again, then publishes a resumed
        LOG_MESSAGE. A timeout is logged as a warning and the runner resumes
        on its own — better than leaking a thread if the user forgets the
        dashboard tab.
        """
        self._resume_event.clear()
        self.event_bus.publish(
            DashboardEvent(
                event_type=EventType.LOG_MESSAGE,
                agent_name="runner",
                log_level=logging.INFO,
                content=f"PAUSED at handover {from_agent or '?'} -> {target or '?'}",
            )
        )
        resumed = self._resume_event.wait(timeout=HANDOVER_PAUSE_TIMEOUT_SECONDS)
        if not resumed:
            self.event_bus.publish(
                DashboardEvent(
                    event_type=EventType.LOG_MESSAGE,
                    agent_name="runner",
                    log_level=logging.WARNING,
                    content=(
                        f"Handover pause timed out after "
                        f"{HANDOVER_PAUSE_TIMEOUT_SECONDS}s; auto-resuming."
                    ),
                )
            )
            self._resume_event.set()
        else:
            self.event_bus.publish(
                DashboardEvent(
                    event_type=EventType.LOG_MESSAGE,
                    agent_name="runner",
                    log_level=logging.INFO,
                    content=f"RESUMED handover -> {target or '?'}",
                )
            )

    def _parse_agent_name(self, ns: tuple) -> str | None:
        if not ns:
            return None
        first = ns[0] if isinstance(ns[0], str) else str(ns[0])
        return first.split(":")[0] if ":" in first else first

    def send_manual_message(self, message: str):
        with self._lock:
            if self.is_running:
                return
            self.is_running = True
        thread_id = self._manual_thread_id
        threading.Thread(
            target=self._run_manual_turn, args=(thread_id, message), daemon=True
        ).start()

    def end_manual_conversation(self, feedback_score: float, feedback_reason: str = ""):
        """Called when the user explicitly ends the manual conversation."""
        if self._current_order_id:
            self.event_bus.publish(
                DashboardEvent(
                    event_type=EventType.TRAY_READY,
                    agent_name="customer",
                    content=self._current_order_id,
                )
            )

        logger.info(
            "Customer feedback [%.2f]: %s",
            feedback_score,
            feedback_reason or "Manual feedback",
        )

        self.event_bus.publish(
            DashboardEvent(
                event_type=EventType.CONVERSATION_END,
                agent_name="system",
            )
        )
        self._manual_thread_id = str(uuid.uuid4())
        self._current_order_id = None

    def _run_manual_turn(self, thread_id: str, message: str):
        try:
            self.event_bus.publish(
                DashboardEvent(
                    event_type=EventType.CUSTOMER_MESSAGE,
                    agent_name="customer",
                    content=message,
                )
            )
            self.event_bus.publish(
                DashboardEvent(
                    event_type=EventType.USER_VISIBLE,
                    agent_name=self._active_agent,
                    content=message,
                )
            )
            self._stream_with_events(thread_id, message)
        except Exception as e:
            logger.exception("Manual turn failed")
            self.event_bus.publish(
                DashboardEvent(
                    event_type=EventType.CONVERSATION_END,
                    agent_name="system",
                    content=f"ERROR: {e}",
                )
            )
        finally:
            with self._lock:
                self.is_running = False
