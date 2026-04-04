from __future__ import annotations

import json

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class JsonReplyExtractor(FrameProcessor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._buffer: list[str] = []
        self._buffering = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = []
            self._buffering = True
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame) and self._buffering:
            self._buffer.append(frame.text)
            return

        if isinstance(frame, LLMFullResponseEndFrame) and self._buffering:
            reply = _extract_reply("".join(self._buffer))
            self._buffer = []
            self._buffering = False

            if reply:
                logger.info("[llm] assistente> {}", reply)
                await self.push_frame(LLMTextFrame(reply), direction)

            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)


def _extract_reply(payload: str) -> str:
    text = payload.strip()
    if not text:
        return ""

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("[llm] resposta JSON invalida; usando texto bruto")
        return text

    reply = data.get("reply")
    if not isinstance(reply, str):
        logger.warning("[llm] JSON sem chave reply; usando payload bruto")
        return text

    return reply.strip()
