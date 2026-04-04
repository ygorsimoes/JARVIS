from __future__ import annotations

import time

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class LocalAudioEchoGate(FrameProcessor):
    def __init__(self, *, release_ms: int = 350, **kwargs) -> None:
        super().__init__(**kwargs)
        self._release_ms = release_ms
        self._assistant_speaking = False
        self._mute_input_until_ns = 0
        self._suppressed_frames = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        now_ns = time.monotonic_ns()

        if isinstance(frame, BotStartedSpeakingFrame):
            self._assistant_speaking = True
            self._mute_input_until_ns = 0
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._assistant_speaking = False
            self._mute_input_until_ns = now_ns + (self._release_ms * 1_000_000)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InputAudioRawFrame):
            if self._assistant_speaking or now_ns < self._mute_input_until_ns:
                self._suppressed_frames += 1
                return

            if self._suppressed_frames:
                logger.debug(
                    "[audio] gate liberou o microfone apos suprimir {} frames",
                    self._suppressed_frames,
                )
                self._suppressed_frames = 0

        await self.push_frame(frame, direction)
