import json
import threading
import uuid
import logging

import mlflow
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.coffee_shop import CoffeeShop
from src.agents import CUSTOMER_SCENARIOS
from src.agents.tray import get_tray, clear_tray
from src.agents.order_store import load_order, set_order_status
from src.agents.shared_components import OrderStatus
from src.control_plane.subgraph import (
    CORRECTION_KWARG,
    REJECTED_AGENT_KWARG,
    REJECTED_CONTENT_KWARG,
    REJECTING_GUARDRAIL_KWARG,
    REJECTION_REASON_KWARG,
)
from src.conversation import _tag_trace
from src.stream import SWARM_AGENTS
from .event_bus import EventBus, DashboardEvent, EventType

logger = logging.getLogger("coffee_shop.dashboard")

MAX_CONVERSATION_TURNS = 30

HANDOVER_PAUSE_TIMEOUT_SECONDS = 300


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
            self.pause_on_next_handover: bool = bool(cfg.handover_pause_default)
        else:
            # No real config available (e.g. unit-test MagicMock shop).
            self.pause_on_next_handover = False
        # Set when no pause is pending; clear()ed to block the runner thread at
        # the next handover seam. resume() sets it again to release the wait.
        self._resume_event = threading.Event()
        self._resume_event.set()

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
        if order and order.status == OrderStatus.IN_PREPARATION:
            set_order_status(
                order, OrderStatus.COMPLETED, context="tray pickup by customer"
            )
        elif order and order.status != OrderStatus.COMPLETED:
            logger.warning(
                "Tray pickup skipped completion for %s: status=%s not IN_PREPARATION",
                order_id,
                order.status.value,
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

        stream_input = {
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
                            self._publish_message(msg, msg_agent)

                            if isinstance(msg, HumanMessage) and (
                                (msg.additional_kwargs or {}).get(CORRECTION_KWARG) is True
                            ):
                                rejected_content = (msg.additional_kwargs or {}).get(
                                    REJECTED_CONTENT_KWARG, ""
                                )
                                if last_agent_message == rejected_content:
                                    last_agent_message = None

                            if (
                                isinstance(msg, AIMessage)
                                and msg.content
                                and not msg.tool_calls
                                and getattr(msg, "name", None) in SWARM_AGENTS
                            ):
                                last_agent_message = msg.content
        except Exception as e:
            logger.exception("Error during stream iteration")
            self.event_bus.publish(
                DashboardEvent(
                    event_type=EventType.AGENT_MESSAGE,
                    agent_name="system",
                    content=f"Stream error: {e}",
                )
            )

        # Tag the MLflow trace produced by the just-completed stream.
        if self.shop.config.mlflow_enabled:
            trace_id = mlflow.get_last_active_trace_id()
            if trace_id is not None:
                _tag_trace(
                    trace_id, self.shop.config.setup_name, self._current_scenario_index
                )

        if current_agent:
            self.event_bus.publish(
                DashboardEvent(
                    event_type=EventType.AGENT_THINKING,
                    agent_name=current_agent,
                    content="idle",
                )
            )

        return last_agent_message

    def _publish_message(self, msg, agent_name: str) -> None:
        if isinstance(msg, HumanMessage):
            kwargs = msg.additional_kwargs or {}
            if kwargs.get(CORRECTION_KWARG) is True:
                rejected_agent = kwargs.get(REJECTED_AGENT_KWARG) or agent_name
                self.event_bus.publish(
                    DashboardEvent(
                        event_type=EventType.AGENT_MESSAGE_REJECTED,
                        agent_name=rejected_agent,
                        content=kwargs.get(REJECTED_CONTENT_KWARG, ""),
                        rejection_reason=kwargs.get(REJECTION_REASON_KWARG, ""),
                        rejecting_guardrail=kwargs.get(REJECTING_GUARDRAIL_KWARG, ""),
                    )
                )
            return
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
                        )
                    )
            elif msg.content:
                self.event_bus.publish(
                    DashboardEvent(
                        event_type=EventType.AGENT_MESSAGE,
                        agent_name=agent_name,
                        content=msg.content,
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
