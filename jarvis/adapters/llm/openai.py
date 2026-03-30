from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncIterator, List

from ...models.conversation import Message, Role


class OpenAIAdapter:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._cancel_event = asyncio.Event()
        self._generation_active = False

    @property
    def last_generation_stats(self) -> dict[str, object]:
        return {}

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "openai no esta instalado. Instale usando pip install openai."
                ) from exc
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY nao esta definida no ambiente.")
            self._client = AsyncOpenAI(api_key=api_key)
        return self._client

    async def chat_stream(
        self,
        messages: List[Message],
        tools: List[dict],
        max_kv_size: int,
        tool_invoker=None,
    ) -> AsyncIterator[str]:
        del max_kv_size, tool_invoker
        client = self._ensure_client()
        self._cancel_event.clear()
        self._generation_active = True

        openai_messages = []
        for msg in messages:
            role = msg.role.value if msg.role != Role.SYSTEM else "system"
            openai_messages.append({"role": role, "content": msg.content})

        kwargs = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        # Simplified direct tools mapping could go here if needed
        # For now, it streams text response.

        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if self._cancel_event.is_set():
                    break
                
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as exc:
            raise exc
        finally:
            self._generation_active = False

    async def cancel_current_response(self) -> bool:
        was_active = self._generation_active
        self._cancel_event.set()
        return was_active

    async def close_session(self) -> None:
        self._cancel_event.set()

    async def shutdown(self) -> None:
        self._cancel_event.set()
        await asyncio.sleep(0)
