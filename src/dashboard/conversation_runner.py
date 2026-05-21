import threading
import time
import uuid
import logging

from langchain_core.messages import AIMessage, ToolMessage

from src.coffee_shop import CoffeeShop
from src.agents import reset_inventory, CUSTOMER_SCENARIOS
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

    def start(self, scenario_index=None):
        with self._lock:
            if self.is_running:
                return
            self.is_running = True
        self._thread = threading.Thread(
            target=self._run, args=(scenario_index,), daemon=True
        )
        self._thread.start()

    def _run(self, scenario_index):
        try:
            self._run_conversation(scenario_index)
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

    def _run_conversation(self, scenario_index):
        reset_inventory()
        self.shop.customer_agent.reset(scenario_index)
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
            event_type=EventType.CONVERSATION_END,
            agent_name="system",
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
                            self.event_bus.publish(DashboardEvent(
                                event_type=EventType.HANDOFF,
                                agent_name=hc.get("from_agent", resolved_agent),
                                handoff_context=hc,
                                target_agent=node_data.get("active_agent"),
                            ))

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
                                and getattr(msg, "name", None)
                                in ("order_agent", "inventory_agent", "barista_agent", "customer_service_agent")
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
            self.event_bus.publish(DashboardEvent(
                event_type=EventType.TOOL_RESULT,
                agent_name=agent_name,
                tool_name=getattr(msg, "name", None),
                tool_result=msg.content if isinstance(msg.content, str) else str(msg.content),
            ))

    def _parse_agent_name(self, ns: tuple) -> str | None:
        if not ns:
            return None
        first = ns[0] if isinstance(ns[0], str) else str(ns[0])
        return first.split(":")[0] if ":" in first else first
