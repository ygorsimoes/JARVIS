import pytest

from jarvis.bus import EventBus
from jarvis.models.events import Event, EventType


@pytest.mark.asyncio
class TestEventBus:
    async def test_bus_tracks_dropped_events(self):
        bus = EventBus(max_queue_size=1)
        await bus.subscribe([EventType.USER_TURN])

        await bus.publish(Event(event_type=EventType.USER_TURN, payload={"text": "um"}))
        await bus.publish(
            Event(event_type=EventType.USER_TURN, payload={"text": "dois"})
        )

        assert bus.dropped_events_count == 1
        assert bus.dropped_events_by_type[EventType.USER_TURN] == 1

    async def test_bus_only_delivers_matching_event_types(self):
        bus = EventBus()
        subscription = await bus.subscribe([EventType.USER_TURN])

        await bus.publish(Event(event_type=EventType.ASSISTANT_COMPLETED, payload={}))
        await bus.publish(Event(event_type=EventType.USER_TURN, payload={"text": "ok"}))

        event = await subscription.get()
        assert event.event_type == EventType.USER_TURN
        assert subscription.queue.empty()

    async def test_bus_unsubscribe_stops_future_delivery(self):
        bus = EventBus()
        subscription = await bus.subscribe([EventType.USER_TURN])
        await bus.unsubscribe(subscription)

        await bus.publish(Event(event_type=EventType.USER_TURN, payload={"text": "oi"}))

        assert subscription.queue.empty()

    async def test_bus_preserves_publish_order_per_subscription(self):
        bus = EventBus()
        subscription = await bus.subscribe([EventType.USER_TURN])

        await bus.publish(Event(event_type=EventType.USER_TURN, payload={"text": "um"}))
        await bus.publish(
            Event(event_type=EventType.USER_TURN, payload={"text": "dois"})
        )
        await bus.publish(
            Event(event_type=EventType.USER_TURN, payload={"text": "tres"})
        )

        events = [await subscription.get() for _ in range(3)]

        assert [event.payload["text"] for event in events] == ["um", "dois", "tres"]
