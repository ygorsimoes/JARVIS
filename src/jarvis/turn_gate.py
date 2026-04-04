from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from pipecat.frames.frames import CancelFrame, EndFrame, Frame, LLMContextFrame, StartFrame
from pipecat.processors.aggregators.gated_llm_context import GatedLLMContextAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.sync.event_notifier import EventNotifier


class SafeGatedLLMContextAggregator(GatedLLMContextAggregator):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await FrameProcessor.process_frame(self, frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._start()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMContextFrame):
            if self._start_open:
                self._start_open = False
                await self.push_frame(frame, direction)
            else:
                self._last_context_frame = frame
        else:
            await self.push_frame(frame, direction)


class TrailingUserMessagesNormalizer(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMContextFrame):
            _merge_trailing_user_messages(frame.context.messages)

        await self.push_frame(frame, direction)


class TurnGateController:
    def __init__(
        self,
        *,
        notifier: EventNotifier,
        delay_secs: float,
    ) -> None:
        self._notifier = notifier
        self._delay_secs = delay_secs
        self._pending_task: asyncio.Task | None = None
        self._context_released = False

    async def cancel_pending(self) -> None:
        if self._pending_task:
            self._pending_task.cancel()
            try:
                await self._pending_task
            except asyncio.CancelledError:
                pass
            self._pending_task = None

    async def schedule_release(self, *, owner: FrameProcessor) -> None:
        await self.cancel_pending()
        logger.debug("[turngate] liberando contexto em {:.2f}s (debounce)", self._delay_secs)
        self._pending_task = owner.create_task(self._notify_after_delay())

    def should_interrupt_on_resume(self) -> bool:
        return self._context_released

    def mark_assistant_started(self) -> None:
        self._context_released = False

    def reset(self) -> None:
        self._context_released = False

    async def _notify_after_delay(self) -> None:
        try:
            await asyncio.sleep(self._delay_secs)
            self._context_released = True
            await self._notifier.notify()
        except asyncio.CancelledError:
            raise
        finally:
            self._pending_task = None


def _merge_trailing_user_messages(messages: list[Any]) -> None:
    trailing_messages: list[dict[str, Any]] = []
    trailing_start_index = len(messages)

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            break
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            break
        trailing_messages.append(message)
        trailing_start_index = index

    if len(trailing_messages) < 2:
        return

    trailing_messages.reverse()
    merged_content = " ".join(message["content"].strip() for message in trailing_messages)
    messages[trailing_start_index:] = [{"role": "user", "content": merged_content}]
