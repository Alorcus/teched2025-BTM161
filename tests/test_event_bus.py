"""Tests 29-30: Dashboard EventBus.

Validates publish/drain ordering and thread safety.
"""
import threading
import unittest

from src.dashboard.event_bus import EventBus, DashboardEvent, EventType


class TestEventBusPublishAndDrain(unittest.TestCase):
    """Test 29: Events are queued and drained in FIFO order."""

    def test_fifo_ordering(self):
        bus = EventBus()
        events = [
            DashboardEvent(event_type=EventType.CONVERSATION_START, agent_name="system", content="start"),
            DashboardEvent(event_type=EventType.AGENT_MESSAGE, agent_name="order_agent", content="hello"),
            DashboardEvent(event_type=EventType.TOOL_CALL, agent_name="order_agent", tool_name="process_order"),
        ]
        for ev in events:
            bus.publish(ev)

        drained = bus.drain()
        self.assertEqual(len(drained), 3)
        self.assertEqual(drained[0].content, "start")
        self.assertEqual(drained[1].content, "hello")
        self.assertEqual(drained[2].tool_name, "process_order")

    def test_second_drain_returns_empty(self):
        bus = EventBus()
        bus.publish(DashboardEvent(event_type=EventType.AGENT_MESSAGE, agent_name="x", content="hi"))
        bus.drain()
        self.assertEqual(bus.drain(), [])

    def test_drain_empty_bus(self):
        bus = EventBus()
        self.assertEqual(bus.drain(), [])


class TestEventBusThreadSafety(unittest.TestCase):
    """Test 30: Concurrent publishers don't lose events."""

    def test_concurrent_publish(self):
        bus = EventBus()
        num_threads = 20
        events_per_thread = 50
        barrier = threading.Barrier(num_threads)

        def publish_many(thread_id):
            barrier.wait()
            for i in range(events_per_thread):
                bus.publish(DashboardEvent(
                    event_type=EventType.AGENT_MESSAGE,
                    agent_name=f"agent_{thread_id}",
                    content=f"msg_{i}",
                ))

        threads = [threading.Thread(target=publish_many, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        drained = bus.drain()
        self.assertEqual(len(drained), num_threads * events_per_thread)


if __name__ == "__main__":
    unittest.main()
