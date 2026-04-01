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
