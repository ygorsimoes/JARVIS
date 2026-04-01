from __future__ import annotations

import asyncio
from typing import AsyncIterator, List

from ...models.conversation import Message, Role


class FakeLLMAdapter:
    def __init__(self, mode: str = "hot_path", chunk_words: int = 2) -> None:
        self.mode = mode
        self.chunk_words = max(1, chunk_words)

    async def chat_stream(
        self,
        messages: List[Message],
        tools: List[dict],
        max_kv_size: int,
        tool_invoker=None,
    ) -> AsyncIterator[str]:
        del tools, max_kv_size, tool_invoker
        user_text = self._last_user_message(messages)
        response = self._build_response(user_text)
        words = response.split()
        for index in range(0, len(words), self.chunk_words):
            chunk = " ".join(words[index : index + self.chunk_words])
            if index + self.chunk_words < len(words):
                chunk += " "
            await asyncio.sleep(0)
            yield chunk

    def _last_user_message(self, messages: List[Message]) -> str:
        for message in reversed(messages):
            if message.role == Role.USER:
                return message.content
        return ""

    def _build_response(self, user_text: str) -> str:
        if self.mode == "deliberative":
            return (
                "Analisei o pedido: %s. Segue um plano curto e objetivo."
            ) % user_text.strip()
        return (
            "Resposta rapida para %s. Estou pronto para o proximo passo."
            % user_text.strip()
        )

    async def cancel_current_response(self) -> bool:
        return False

    async def close_session(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None
