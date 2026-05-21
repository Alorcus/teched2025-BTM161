from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Queue, Empty
import time


class EventType(Enum):
    CONVERSATION_START = auto()
    CONVERSATION_END = auto()
    CUSTOMER_MESSAGE = auto()
    AGENT_THINKING = auto()
    AGENT_MESSAGE = auto()
    TOOL_CALL = auto()
    TOOL_RESULT = auto()
    HANDOFF = auto()


@dataclass
class DashboardEvent:
    event_type: EventType
    agent_name: str
    timestamp: float = field(default_factory=time.time)
    content: str = ""
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: str | None = None
    handoff_context: dict | None = None
    target_agent: str | None = None


class EventBus:
    def __init__(self):
        self._queue: Queue[DashboardEvent] = Queue()

    def publish(self, event: DashboardEvent):
        self._queue.put_nowait(event)

    def drain(self) -> list[DashboardEvent]:
        events = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except Empty:
                break
        return events
