import pytest

from jarvis.bus import EventBus
from jarvis.core.state_machine import InvalidStateTransitionError, StateMachine
from jarvis.models.events import EventType
from jarvis.models.state import JarvisState


@pytest.mark.asyncio
class TestStateMachine:
    async def test_valid_transition_sequence(self):
        bus = EventBus()
        machine = StateMachine(event_bus=bus)
        subscription = await bus.subscribe([EventType.STATE_CHANGED])

        await machine.transition(JarvisState.ARMED, "activation")
        await machine.transition(JarvisState.LISTENING, "speech detected")
        await machine.transition(JarvisState.TRANSCRIBING, "turn ended")
        await machine.transition(JarvisState.THINKING, "stt complete")

        assert machine.state == JarvisState.THINKING
        assert len(machine.history) == 4

        event = await subscription.get()
        assert event.event_type == EventType.STATE_CHANGED

    async def test_invalid_transition_raises(self):
        machine = StateMachine()
        with pytest.raises(InvalidStateTransitionError):
            await machine.transition(JarvisState.SPEAKING, "cannot skip ahead")
