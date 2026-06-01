"""
Module to save dashboard events as OCEL-compatible CSV files.
"""
from datetime import datetime
from pathlib import Path
import uuid

import polars as pl

from .event_bus import DashboardEvent, EventType, EventBus


class DashboardLogSaver:
    """
    Captures dashboard events and saves them as OCEL-compatible CSV files.

    Limitations:
    - Dashboard events don't include token counts, model names, or precise durations
    - These fields will be null in the generated CSV
    - For complete metrics, use MLflow-based export via `simulate --export-logs`
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.events: list[DashboardEvent] = []
        self.case_id = str(uuid.uuid4())

    def capture_events(self) -> list[DashboardEvent]:
        """
        Capture new events from the event bus.
        Returns the list of captured events (for further processing).
        """
        new_events = self.event_bus.drain()
        self.events.extend(new_events)
        return new_events

    def reset(self):
        """Clear captured events and generate new case ID."""
        self.events.clear()
        self.case_id = str(uuid.uuid4())

    def save_to_csv(self, filepath: str | Path) -> Path:
        """
        Convert captured events to OCEL-compatible CSV format and save.

        Args:
            filepath: Path to save the CSV file

        Returns:
            Path object of the saved file

        Raises:
            ValueError: If no events have been captured
        """
        if not self.events:
            raise ValueError("No events to save")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for event in self.events:
            # Skip internal log messages (they clutter the event log)
            if event.event_type == EventType.LOG_MESSAGE:
                continue

            row = {
                'case_id': self.case_id,
                'identity:id': str(uuid.uuid4()),
                'time:timestamp': datetime.fromtimestamp(event.timestamp).isoformat(),
                'time_finished': datetime.fromtimestamp(event.timestamp).isoformat(),
                'concept:name': self._map_event_type(event),
                'concept:instance': self._get_instance(event),
                'org:resource': event.agent_name,
                'duration': None,  # Not tracked in dashboard events
                'model': None,     # Not tracked in dashboard events
                'input_tokens': None,
                'response_tokens': None,
                'tool': event.tool_name if event.event_type == EventType.TOOL_CALL else None,
                'message': event.content if event.event_type == EventType.AGENT_MESSAGE else None,
            }
            rows.append(row)

        if not rows:
            raise ValueError("No non-log events to save")

        df = pl.DataFrame(rows)
        df.write_csv(filepath)

        return filepath

    def _map_event_type(self, event: DashboardEvent) -> str:
        """Map EventType to OCEL concept:name."""
        mapping = {
            EventType.CONVERSATION_START: 'user_prompt',
            EventType.CONVERSATION_END: 'conversation_end',
            EventType.CUSTOMER_MESSAGE: 'user_prompt',
            EventType.USER_VISIBLE: 'user_prompt',
            EventType.AGENT_THINKING: 'agent_thinking',
            EventType.AGENT_MESSAGE: 'call_llm',
            EventType.TOOL_CALL: 'execute_tool',
            EventType.TOOL_RESULT: 'execute_tool',
            EventType.HANDOFF: 'transfer_to_agent',
        }
        return mapping.get(event.event_type, 'unknown')

    def _get_instance(self, event: DashboardEvent) -> str:
        """Get human-readable instance description."""
        if event.event_type == EventType.TOOL_CALL:
            return f"{event.tool_name or 'unknown_tool'} call"
        elif event.event_type == EventType.TOOL_RESULT:
            return f"{event.tool_name or 'unknown_tool'} result"
        elif event.event_type == EventType.HANDOFF:
            return f"handoff to {event.target_agent or 'unknown'}"
        elif event.event_type == EventType.AGENT_MESSAGE:
            return "response"
        elif event.event_type in (EventType.CUSTOMER_MESSAGE, EventType.USER_VISIBLE, EventType.CONVERSATION_START):
            return "prompt"
        else:
            return event.event_type.name.lower()
