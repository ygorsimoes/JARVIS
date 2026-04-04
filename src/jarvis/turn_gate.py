from __future__ import annotations

import asyncio
import re
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
        settle_delay_secs: float,
        trailing_delay_secs: float,
        incomplete_delay_secs: float,
    ) -> None:
        self._notifier = notifier
        self._settle_delay_secs = settle_delay_secs
        self._trailing_delay_secs = trailing_delay_secs
        self._incomplete_delay_secs = incomplete_delay_secs
        self._pending_task: asyncio.Task | None = None

    async def cancel_pending(self) -> None:
        if self._pending_task:
            self._pending_task.cancel()
            try:
                await self._pending_task
            except asyncio.CancelledError:
                pass
            self._pending_task = None

    async def schedule_release(self, text: str, *, owner: FrameProcessor) -> None:
        await self.cancel_pending()
        delay_secs, reason = compute_turn_gate_delay(
            text,
            settle_delay_secs=self._settle_delay_secs,
            trailing_delay_secs=self._trailing_delay_secs,
            incomplete_delay_secs=self._incomplete_delay_secs,
        )
        logger.debug("[turngate] liberando contexto em {:.2f}s ({})", delay_secs, reason)
        self._pending_task = owner.create_task(self._notify_after_delay(delay_secs))

    async def _notify_after_delay(self, delay_secs: float) -> None:
        try:
            await asyncio.sleep(delay_secs)
            await self._notifier.notify()
        except asyncio.CancelledError:
            raise
        finally:
            self._pending_task = None


def compute_turn_gate_delay(
    text: str,
    *,
    settle_delay_secs: float,
    trailing_delay_secs: float,
    incomplete_delay_secs: float,
) -> tuple[float, str]:
    normalized = _normalize_text(text)
    if not normalized:
        return settle_delay_secs, "empty"
    if _ends_terminal_sentence(normalized):
        return min(settle_delay_secs, 0.35), "terminal"
    if _looks_incomplete(normalized):
        return incomplete_delay_secs, "incomplete"
    if _looks_trailing(normalized):
        return trailing_delay_secs, "trailing"
    return settle_delay_secs, "complete"


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


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _looks_incomplete(text: str) -> bool:
    if not text:
        return False
    if _ends_terminal_sentence(text):
        return False
    if text.endswith(("...", "…")):
        return True
    if len(text) < 28 and not text.endswith(("?", "!", ".")):
        return True
    incomplete_endings = (
        "eu queria",
        "eu gostaria",
        "eu queria saber",
        "eu gostaria de",
        "talvez",
        "bom",
        "entao",
        "então",
        "sobre",
        "que é",
        "que eu",
    )
    if any(text.endswith(ending) for ending in incomplete_endings):
        return True
    return False


def _looks_trailing(text: str) -> bool:
    if not text:
        return False
    if _ends_terminal_sentence(text):
        return False
    if text.endswith((",", ":", ";")):
        return True
    trailing_words = {
        "e",
        "mas",
        "porque",
        "que",
        "sobre",
        "para",
        "com",
        "de",
        "do",
        "da",
    }
    return text.rsplit(" ", 1)[-1] in trailing_words


def _ends_terminal_sentence(text: str) -> bool:
    return text.endswith(("?", "!", "."))
