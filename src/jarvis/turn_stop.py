from __future__ import annotations

import asyncio

from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer
from pipecat.frames.frames import TranscriptionFrame, VADUserStoppedSpeakingFrame
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy


class ConservativeTurnAnalyzerUserTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    def __init__(
        self,
        *,
        turn_analyzer: BaseTurnAnalyzer,
        resume_delay_secs: float,
        **kwargs,
    ) -> None:
        super().__init__(turn_analyzer=turn_analyzer, **kwargs)
        self._resume_delay_secs = resume_delay_secs

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        await super()._handle_vad_user_stopped_speaking(frame)

        if not self._turn_complete or self._timeout_task is None:
            return

        await self.task_manager.cancel_task(self._timeout_task)
        timeout = self._calculate_commit_delay()
        logger.debug(
            "[turn] candidato a fim de turno; aguardando {:.2f}s para permitir retomada",
            timeout,
        )
        self._timeout_task = self.task_manager.create_task(
            self._timeout_handler(timeout),
            f"{self}::_timeout_handler",
        )
        await asyncio.sleep(0)

    async def _handle_transcription(self, frame: TranscriptionFrame):
        self._text = frame.text
        if frame.finalized:
            self._transcript_finalized = True

        if not self._vad_user_speaking and self._vad_stopped_time is None:
            if self._timeout_task:
                await self.task_manager.cancel_task(self._timeout_task)

            self._turn_complete = True
            timeout = self._calculate_commit_delay()
            logger.debug(
                "[turn] transcricao final sem VAD stop; aguardando {:.2f}s antes de fechar",
                timeout,
            )
            self._timeout_task = self.task_manager.create_task(
                self._timeout_handler(timeout),
                f"{self}::_timeout_handler",
            )
            await asyncio.sleep(0)

    async def _maybe_trigger_user_turn_stopped(self):
        if not self._text or not self._turn_complete:
            return

        if self._timeout_task is None:
            await self.trigger_user_turn_stopped()

    def _calculate_commit_delay(self) -> float:
        effective_stt_wait = max(0.0, self._stt_timeout - self._stop_secs)
        return max(effective_stt_wait, self._resume_delay_secs)
