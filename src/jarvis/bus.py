from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, Optional, Set

from .models.events import Event, EventType


@dataclass
class EventSubscription:
    queue: asyncio.Queue
    event_types: Optional[Set[EventType]] = None

    async def get(self) -> Event:
        return await self.queue.get()


class EventBus:
    def __init__(self, max_queue_size: int = 128) -> None:
        self._max_queue_size = max_queue_size
        self._subscriptions: list[EventSubscription] = []
        self._lock = asyncio.Lock()
        self._dropped_events_count = 0
        self._dropped_events_by_type: dict[EventType, int] = {}

    async def publish(self, event: Event) -> None:
        async with self._lock:
            subscriptions = list(self._subscriptions)
        for subscription in subscriptions:
            if (
                subscription.event_types
                and event.event_type not in subscription.event_types
            ):
                continue
            if subscription.queue.full():
                try:
                    subscription.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self._dropped_events_count += 1
                self._dropped_events_by_type[event.event_type] = (
                    self._dropped_events_by_type.get(event.event_type, 0) + 1
                )
            subscription.queue.put_nowait(event)

    async def subscribe(
        self, event_types: Optional[Iterable[EventType]] = None
    ) -> EventSubscription:
        normalized = set(event_types) if event_types else None
        subscription = EventSubscription(
            queue=asyncio.Queue(maxsize=self._max_queue_size),
            event_types=normalized,
        )
        async with self._lock:
            self._subscriptions.append(subscription)
        return subscription

    async def unsubscribe(self, subscription: EventSubscription) -> None:
        async with self._lock:
            self._subscriptions = [
                item for item in self._subscriptions if item is not subscription
            ]

    @property
    def dropped_events_count(self) -> int:
        return self._dropped_events_count

    @property
    def dropped_events_by_type(self) -> dict[EventType, int]:
        return dict(self._dropped_events_by_type)
