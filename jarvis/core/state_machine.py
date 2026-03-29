from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, Optional, Set

from ..bus import EventBus
from ..models.events import Event, EventType
from ..models.state import JarvisState, StateTransition


class InvalidStateTransitionError(RuntimeError):
    pass


ALLOWED_TRANSITIONS: Dict[JarvisState, Set[JarvisState]] = {
    JarvisState.IDLE: {JarvisState.ARMED},
    JarvisState.ARMED: {JarvisState.LISTENING, JarvisState.IDLE, JarvisState.FAILED},
    JarvisState.LISTENING: {
        JarvisState.TRANSCRIBING,
        JarvisState.INTERRUPTED,
        JarvisState.FAILED,
        JarvisState.IDLE,
    },
    JarvisState.TRANSCRIBING: {JarvisState.THINKING, JarvisState.FAILED, JarvisState.IDLE},
    JarvisState.THINKING: {
        JarvisState.ACTING,
        JarvisState.SPEAKING,
        JarvisState.INTERRUPTED,
        JarvisState.FAILED,
        JarvisState.IDLE,
    },
    JarvisState.ACTING: {JarvisState.THINKING, JarvisState.SPEAKING, JarvisState.FAILED, JarvisState.IDLE},
    JarvisState.SPEAKING: {JarvisState.IDLE, JarvisState.LISTENING, JarvisState.INTERRUPTED, JarvisState.FAILED},
    JarvisState.INTERRUPTED: {JarvisState.LISTENING, JarvisState.IDLE, JarvisState.FAILED},
    JarvisState.FAILED: {JarvisState.IDLE, JarvisState.ARMED},
}


class StateMachine:
    def __init__(
        self,
        initial_state: JarvisState = JarvisState.IDLE,
        event_bus: Optional[EventBus] = None,
        history_limit: int = 256,
    ) -> None:
        self._state = initial_state
        self._event_bus = event_bus
        self._history: Deque[StateTransition] = deque(maxlen=history_limit)

    @property
    def state(self) -> JarvisState:
        return self._state

    @property
    def history(self) -> list[StateTransition]:
        return list(self._history)

    def can_transition(self, new_state: JarvisState) -> bool:
        return new_state in ALLOWED_TRANSITIONS[self._state]

    async def transition(self, new_state: JarvisState, reason: str, metadata: Optional[dict] = None) -> StateTransition:
        if not self.can_transition(new_state):
            raise InvalidStateTransitionError(
                "cannot transition from %s to %s" % (self._state.value, new_state.value)
            )
        transition = StateTransition(
            previous_state=self._state,
            new_state=new_state,
            reason=reason,
            metadata=metadata or {},
        )
        self._state = new_state
        self._history.append(transition)
        if self._event_bus is not None:
            await self._event_bus.publish(
                Event(
                    event_type=EventType.STATE_CHANGED,
                    payload={
                        "previous_state": transition.previous_state.value,
                        "new_state": transition.new_state.value,
                        "reason": transition.reason,
                        "metadata": transition.metadata,
                    },
                )
            )
        return transition

    async def reset(self, reason: str = "reset") -> StateTransition:
        if self._state == JarvisState.IDLE:
            transition = StateTransition(JarvisState.IDLE, JarvisState.IDLE, reason)
            self._history.append(transition)
            return transition
        return await self.force_transition(JarvisState.IDLE, reason)

    async def force_transition(
        self,
        new_state: JarvisState,
        reason: str,
        metadata: Optional[dict] = None,
    ) -> StateTransition:
        transition = StateTransition(
            previous_state=self._state,
            new_state=new_state,
            reason=reason,
            metadata=metadata or {},
        )
        self._state = new_state
        self._history.append(transition)
        if self._event_bus is not None:
            await self._event_bus.publish(
                Event(
                    event_type=EventType.STATE_CHANGED,
                    payload={
                        "previous_state": transition.previous_state.value,
                        "new_state": transition.new_state.value,
                        "reason": transition.reason,
                        "metadata": transition.metadata,
                        "forced": True,
                    },
                )
            )
        return transition

    def allowed_from(self, state: JarvisState) -> Iterable[JarvisState]:
        return tuple(ALLOWED_TRANSITIONS[state])
