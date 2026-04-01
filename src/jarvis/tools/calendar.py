import asyncio
from typing import Any, Dict, List


class CalendarTool:
    def __init__(self):
        self._store = None

    def _ensure_store(self):
        if self._store is None:
            try:
                import EventKit

                self._store = EventKit.EKEventStore.alloc().init()
                self._EKEvent = EventKit.EKEvent
                self._EKEntityTypeEvent = EventKit.EKEntityTypeEvent
            except ImportError as exc:
                raise RuntimeError(
                    "pyobjc-framework-EventKit nao esta instalado. Use pip install pyobjc-framework-EventKit."
                ) from exc
        return self._store

    async def _request_access(self):
        store = self._ensure_store()
        status = store.authorizationStatusForEntityType_(self._EKEntityTypeEvent)
        # EM_AuthorizationStatusAuthorized = 3 based on Apple Docs
        if status == 3:
            return True

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def completion(granted, error):
            if not future.done():
                if error is not None:
                    loop.call_soon_threadsafe(future.set_exception, Exception(error))
                else:
                    loop.call_soon_threadsafe(future.set_result, granted)

        # Requires macOS 10.9 or newer
        store.requestAccessToEntityType_completion_(self._EKEntityTypeEvent, completion)
        granted = await future
        return granted

    async def list_events(self, days: int = 1) -> Dict[str, Any]:
        """
        Lista eventos do calendário pelo framework nativo do macOS nas próximas 'days' horas/dias.
        """
        try:
            granted = await self._request_access()
            if not granted:
                return {
                    "status": "error",
                    "error": "Permissao ao Calendario negada pelo usuario do macOS.",
                }

            import Foundation

            now = Foundation.NSDate.date()
            # Seconds in a day * days
            end_date = now.dateByAddingTimeInterval_(days * 24 * 60 * 60)

            predicate = self._store.predicateForEventsWithStartDate_endDate_calendars_(
                now, end_date, None
            )
            events = self._store.eventsMatchingPredicate_(predicate)

            result: List[Dict[str, str]] = []
            if events:
                for event in events:
                    result.append(
                        {
                            "title": str(event.title()),
                            "start": str(event.startDate().description()),
                            "end": str(event.endDate().description()),
                            "location": str(event.location())
                            if event.location()
                            else "Sem local",
                        }
                    )

            return {"status": "success", "events": result, "count": len(result)}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
