from __future__ import annotations

from collections import deque

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
    _MAX_TRACKED_FRAME_IDS = 4096

    def __init__(self, *, log_transcription_segments: bool = True) -> None:
        super().__init__()
        self._seen_frame_ids: set[int] = set()
        self._recent_frame_ids: deque[int] = deque(maxlen=self._MAX_TRACKED_FRAME_IDS)
        self._assistant_speaking_started_at: float | None = None
        self._log_transcription_segments = log_transcription_segments

    async def on_push_frame(self, data: FramePushed):
        if not self._remember_frame_id(data.frame.id):
            return

        time_sec = data.timestamp / 1_000_000_000
        arrow = "->" if data.direction == FrameDirection.DOWNSTREAM else "<-"

        if isinstance(data.frame, TranscriptionFrame):
            if self._log_transcription_segments:
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

    def _remember_frame_id(self, frame_id: int) -> bool:
        if frame_id in self._seen_frame_ids:
            return False

        if len(self._recent_frame_ids) == self._recent_frame_ids.maxlen:
            oldest_frame_id = self._recent_frame_ids.popleft()
            self._seen_frame_ids.discard(oldest_frame_id)

        self._recent_frame_ids.append(frame_id)
        self._seen_frame_ids.add(frame_id)
        return True


def _to_ms(seconds: float) -> float:
    return seconds * 1000.0
