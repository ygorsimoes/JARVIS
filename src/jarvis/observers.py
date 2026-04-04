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
        self._assistant_speaking_started_at: float | None = None

    async def on_push_frame(self, data: FramePushed):
        if data.frame.id in self._seen_frame_ids:
            return
        self._seen_frame_ids.add(data.frame.id)

        time_sec = data.timestamp / 1_000_000_000
        arrow = "->" if data.direction == FrameDirection.DOWNSTREAM else "<-"

        if isinstance(data.frame, TranscriptionFrame):
            logger.info("[stt] usuario> {}", data.frame.text.strip())
        elif isinstance(data.frame, BotStartedSpeakingFrame):
            self._assistant_speaking_started_at = time_sec
            logger.info(
                "[tts] assistente comecou a falar | {} {} {} | t={:.1f}ms",
                data.source,
                arrow,
                data.destination,
                _to_ms(time_sec),
            )
        elif isinstance(data.frame, BotStoppedSpeakingFrame):
            duration = None
            if self._assistant_speaking_started_at is not None:
                duration = time_sec - self._assistant_speaking_started_at
                self._assistant_speaking_started_at = None

            logger.info(
                "[tts] assistente terminou de falar | {} {} {} | t={:.1f}ms{}",
                data.source,
                arrow,
                data.destination,
                _to_ms(time_sec),
                f" | duracao={_to_ms(duration):.1f}ms" if duration is not None else "",
            )
        elif isinstance(data.frame, InterruptionFrame):
            logger.warning(
                "[turn] interrupcao detectada | {} {} {} | t={:.1f}ms",
                data.source,
                arrow,
                data.destination,
                _to_ms(time_sec),
            )
        elif isinstance(data.frame, ErrorFrame):
            logger.error("[pipeline] erro: {}", data.frame.error)


def _to_ms(seconds: float) -> float:
    return seconds * 1000.0
