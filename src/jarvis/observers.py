from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    InterruptionFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection


class TerminalDebugObserver(BaseObserver):
    def __init__(self) -> None:
        super().__init__()
        self._seen_frame_ids: set[int] = set()

    async def on_push_frame(self, data: FramePushed):
        if data.frame.id in self._seen_frame_ids:
            return
        self._seen_frame_ids.add(data.frame.id)

        time_sec = data.timestamp / 1_000_000_000
        arrow = "->" if data.direction == FrameDirection.DOWNSTREAM else "<-"

        if isinstance(data.frame, TranscriptionFrame):
            logger.info("[stt] usuario> {}", data.frame.text.strip())
        elif isinstance(data.frame, BotStartedSpeakingFrame):
            logger.info(
                "[tts] assistente comecou a falar | {} {} {} | {:.2f}s",
                data.source,
                arrow,
                data.destination,
                time_sec,
            )
        elif isinstance(data.frame, BotStoppedSpeakingFrame):
            logger.info(
                "[tts] assistente terminou de falar | {} {} {} | {:.2f}s",
                data.source,
                arrow,
                data.destination,
                time_sec,
            )
        elif isinstance(data.frame, InterruptionFrame):
            logger.warning(
                "[turn] interrupcao detectada | {} {} {} | {:.2f}s",
                data.source,
                arrow,
                data.destination,
                time_sec,
            )
        elif isinstance(data.frame, ErrorFrame):
            logger.error("[pipeline] erro: {}", data.frame.error)
