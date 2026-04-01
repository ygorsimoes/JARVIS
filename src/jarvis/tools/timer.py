from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass
class TimerRecord:
    timer_id: str
    label: str
    duration_seconds: int
    created_at: datetime = field(default_factory=utc_now)
    task: asyncio.Task | None = None

    @property
    def ends_at(self) -> datetime:
        return self.created_at + timedelta(seconds=self.duration_seconds)

    @property
    def remaining_seconds(self) -> int:
        return max(0, int((self.ends_at - utc_now()).total_seconds()))


class TimerTool:
    def __init__(self) -> None:
        self._timers: Dict[str, TimerRecord] = {}

    async def start(self, duration_seconds: int, label: Optional[str] = None) -> dict:
        timer = TimerRecord(
            timer_id=str(uuid.uuid4()),
            label=label or "Timer",
            duration_seconds=duration_seconds,
        )
        self._timers[timer.timer_id] = timer

        # Start expiration task
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            self._wait_and_notify(timer.timer_id, duration_seconds, timer.label)
        )
        timer.task = task

        return {
            "timer_id": timer.timer_id,
            "label": timer.label,
            "duration_seconds": timer.duration_seconds,
            "ends_at": timer.ends_at.isoformat(),
        }

    async def _wait_and_notify(self, timer_id: str, duration_seconds: int, label: str):
        try:
            await asyncio.sleep(duration_seconds)
            if timer_id in self._timers:
                self._timers.pop(timer_id, None)
                msg = f"Timer '{label}' finalizado."
                cmd = f'display notification "{msg}" with title "J.A.R.V.I.S." sound name "Glass"'
                process = await asyncio.create_subprocess_exec("osascript", "-e", cmd)
                await process.wait()
        except asyncio.CancelledError:
            pass

    async def list(self) -> list:
        return sorted(
            [
                {
                    "timer_id": timer.timer_id,
                    "label": timer.label,
                    "remaining_seconds": timer.remaining_seconds,
                }
                for timer in self._timers.values()
            ],
            key=lambda timer: (timer["remaining_seconds"], timer["label"]),
        )

    async def cancel(self, timer_id: str) -> dict:
        timer = self._timers.pop(timer_id, None)
        if timer:
            if timer.task is not None and not timer.task.done():
                timer.task.cancel()
            return {"timer_id": timer.timer_id, "cancelled": True}
        return {"timer_id": timer_id, "cancelled": False}


def parse_duration_seconds(text: str) -> Optional[int]:
    normalized = text.lower()
    total_seconds = 0
    patterns = (
        (re.compile(r"(\d+)\s*h(?:ora|oras)?"), 3600),
        (re.compile(r"(\d+)\s*min(?:uto|utos)?"), 60),
        (re.compile(r"(\d+)\s*s(?:egundo|egundos)?"), 1),
        (re.compile(r"(\d+)\s*h\s*(\d+)\s*min"), None),
    )

    combined_match = patterns[3][0].search(normalized)
    if combined_match:
        return int(combined_match.group(1)) * 3600 + int(combined_match.group(2)) * 60

    for pattern, multiplier in patterns[:3]:
        for match in pattern.finditer(normalized):
            total_seconds += int(match.group(1)) * int(multiplier)

    if total_seconds:
        return total_seconds

    compact = re.search(r"(\d+)\s*m(?:in)?\b", normalized)
    if compact:
        return int(compact.group(1)) * 60
    return None
