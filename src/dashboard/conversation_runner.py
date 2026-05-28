import json
import threading
import time
import uuid
import logging

from langchain_core.messages import AIMessage, ToolMessage

from src.coffee_shop import CoffeeShop
from src.agents import CUSTOMER_SCENARIOS
from src.agents.tray import get_tray, clear_tray
from src.agents.order_store import load_order, save_order
from src.agents.shared_components import OrderStatus
from src.stream import SWARM_AGENTS
from .event_bus import EventBus, DashboardEvent, EventType

logger = logging.getLogger("coffee_shop.dashboard")

MAX_CONVERSATION_TURNS = 30


class ConversationRunner:
    def __init__(self, shop: CoffeeShop, event_bus: EventBus):
        self.shop = shop
        self.event_bus = event_bus
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.is_running = False
        self._active_agent = "order_agent"
        self._current_order_id: str | None = None

    def start(self, scenario_index=None, custom_prompt=None):
        with self._lock:
            if self.is_running:
                return
            self.is_running = True
        self._thread = threading.Thread(
            target=self._run, args=(scenario_index, custom_prompt), daemon=True
        )
        self._thread.start()

    def _run(self, scenario_index, custom_prompt=None):
        try:
            self._run_conversation(scenario_index, custom_prompt)
        except Exception as e:
            logger.exception("Conversation runner failed")
            self.event_bus.publish(DashboardEvent(
                event_type=EventType.CONVERSATION_END,
                agent_name="system",
                content=f"ERROR: {e}",
            ))
        finally:
            with self._lock:
                self.is_running = False

    def _run_conversation(self, scenario_index, custom_prompt=None):
        self.shop.customer_agent.reset(scenario_index, custom_prompt=custom_prompt)
        self._active_agent = "order_agent"
        self._current_order_id = None
        thread_id = str(uuid.uuid4())

        scenario_label = (
            CUSTOMER_SCENARIOS[scenario_index]
            if scenario_index is not None
            else "random"
        )
        self.event_bus.publish(DashboardEvent(
            event_type=EventType.CONVERSATION_START,
            agent_name="system",
            content=f"Scenario: {scenario_label[:80]}",
        ))

        message = self.shop.customer_agent.get_initial_message()
        self.event_bus.publish(DashboardEvent(
            event_type=EventType.CUSTOMER_MESSAGE,
            agent_name="customer",
            content=message,
        ))
        self.event_bus.publish(DashboardEvent(
            event_type=EventType.USER_VISIBLE,
            agent_name=self._active_agent,
            content=message,
        ))

        turns = 0
        while message:
            if turns >= MAX_CONVERSATION_TURNS:
                logger.warning("Conversation reached %d turns, stopping", MAX_CONVERSATION_TURNS)
                break
            turns += 1

            agent_reply = self._stream_with_events(thread_id, message)
            if not agent_reply:
                break

            message = self.shop.customer_agent.respond_to(agent_reply)
            if message:
                self.event_bus.publish(DashboardEvent(
                    event_type=EventType.CUSTOMER_MESSAGE,
                    agent_name="customer",
                    content=message,
                ))
                self.event_bus.publish(DashboardEvent(
                    event_type=EventType.USER_VISIBLE,
                    agent_name=self._active_agent,
                    content=message,
                ))

        self._consume_tray()

        feedback = self.shop.capture_feedback(thread_id, self._current_order_id)
        logger.info(
            "Customer feedback [%s]: %s", feedback["feedback_label"], feedback["feedback_reason"]
        )

        self.event_bus.publish(DashboardEvent(
            event_type=EventType.CONVERSATION_END,
            agent_name="system",
        ))

    def _consume_tray(self):
        """Customer takes the tray — apply effects, mark order complete, clear tray."""
        order_id = self._current_order_id
        if not order_id:
            return

        tray_items = get_tray(order_id)
        if not tray_items:
            return

        items_summary = ", ".join(
            f"{e.quantity}x {e.item_name}" for e in tray_items
        )
        self.event_bus.publish(DashboardEvent(
            event_type=EventType.TOOL_CALL,
            agent_name="customer",
            tool_name="take_tray",
            tool_args={"order_id": order_id, "items": items_summary},
        ))

        has_contaminated = any(entry.contaminated for entry in tray_items)
        if has_contaminated:
            self.shop.customer_agent.inject_experience(
                "You received your coffee but it tastes slightly off — almost metallic. Something isn't right."
            )

        order = load_order(order_id)
        if order and order.status != OrderStatus.COMPLETED:
            order.status = OrderStatus.COMPLETED
            save_order(order)

        clear_tray(order_id)

        result = {"status": "picked_up", "items": items_summary}
        if has_contaminated:
            result["warning"] = "contaminated items received"
        self.event_bus.publish(DashboardEvent(
            event_type=EventType.TOOL_RESULT,
            agent_name="customer",
            tool_name="take_tray",
            tool_result=json.dumps(result),
        ))

    def _stream_with_events(self, thread_id: str, message: str) -> str | None:
        config = self.shop._get_config(thread_id)

        try:
            stream = self.shop.app.stream(
                {"messages": [{"role": "user", "content": message}], "handoff_context": None},
                config,
                subgraphs=True,
            )
        except Exception as e:
            logger.exception("Failed to start stream")
            self.event_bus.publish(DashboardEvent(
                event_type=EventType.AGENT_MESSAGE,
                agent_name="system",
                content=f"Stream error: {e}",
            ))
            return None

        last_agent_message = None
        seen = set()
        current_agent = None

        try:
            for ns, update in stream:
                agent_name = self._parse_agent_name(ns)

                if agent_name and agent_name != current_agent:
                    if current_agent:
                        self.event_bus.publish(DashboardEvent(
                            event_type=EventType.AGENT_THINKING,
                            agent_name=current_agent,
                            content="idle",
                        ))
                    current_agent = agent_name
                    self.event_bus.publish(DashboardEvent(
                        event_type=EventType.AGENT_THINKING,
                        agent_name=agent_name,
                        content="thinking",
                    ))

                for node, node_data in update.items():
                    if node_data is None:
                        continue

                    if isinstance(node_data, dict):
                        resolved_agent = agent_name or node_data.get("active_agent") or "unknown"

                        if "handoff_context" in node_data and node_data["handoff_context"]:
                            hc = node_data["handoff_context"]
                            target = node_data.get("active_agent")
                            self.event_bus.publish(DashboardEvent(
                                event_type=EventType.HANDOFF,
                                agent_name=hc.get("from_agent", resolved_agent),
                                handoff_context=hc,
                                target_agent=target,
                            ))
                            if target:
                                self._active_agent = target

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
                            msg_uid = getattr(msg, "id", "") or getattr(msg, "tool_call_id", "")
                            if msg_uid:
                                msg_id = f"{type(msg).__name__}:{msg_uid}"
                            else:
                                msg_id = f"{type(msg).__name__}:{name}:{content}"
                            if msg_id in seen:
                                continue
                            seen.add(msg_id)
                            msg_agent = getattr(msg, "name", None) or resolved_agent
                            self._process_message(msg, msg_agent)

                            if (
                                isinstance(msg, AIMessage)
                                and msg.content
                                and not msg.tool_calls
                                and getattr(msg, "name", None) in SWARM_AGENTS
                            ):
                                last_agent_message = msg.content
        except Exception as e:
            logger.exception("Error during stream iteration")
            self.event_bus.publish(DashboardEvent(
                event_type=EventType.AGENT_MESSAGE,
                agent_name="system",
                content=f"Stream error: {e}",
            ))

        if current_agent:
            self.event_bus.publish(DashboardEvent(
                event_type=EventType.AGENT_THINKING,
                agent_name=current_agent,
                content="idle",
            ))

        return last_agent_message

    def _process_message(self, msg, agent_name: str):
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    self.event_bus.publish(DashboardEvent(
                        event_type=EventType.TOOL_CALL,
                        agent_name=agent_name,
                        tool_name=tc["name"],
                        tool_args=tc.get("args", {}),
                    ))
            elif msg.content:
                self.event_bus.publish(DashboardEvent(
                    event_type=EventType.AGENT_MESSAGE,
                    agent_name=agent_name,
                    content=msg.content,
                ))
        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            self.event_bus.publish(DashboardEvent(
                event_type=EventType.TOOL_RESULT,
                agent_name=agent_name,
                tool_name=getattr(msg, "name", None),
                tool_result=content,
            ))
            self._track_order_id(getattr(msg, "name", None), content)

    def _track_order_id(self, tool_name: str | None, content: str):
        """Extract order_id from tool results to track the current order."""
        if self._current_order_id:
            return
        if tool_name not in ("process_order", "check_inventory", "start_preparation"):
            return
        try:
            data = json.loads(content)
            order_id = data.get("order_id")
            if order_id:
                self._current_order_id = order_id
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    def _parse_agent_name(self, ns: tuple) -> str | None:
        if not ns:
            return None
        first = ns[0] if isinstance(ns[0], str) else str(ns[0])
        return first.split(":")[0] if ":" in first else first

