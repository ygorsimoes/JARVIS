from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator, List

from ...models.conversation import Message, Role


class AnthropicAdapter:
    def __init__(
        self,
        model_name: str = "claude-3-7-sonnet-20250219",
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
                from anthropic import AsyncAnthropic
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic no esta instalado. Instale usando pip install anthropic."
                ) from exc

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY nao esta definida no ambiente.")
            self._client = AsyncAnthropic(api_key=api_key)
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

        system_message = ""
        anthropic_messages = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_message += msg.content + "\n"
                continue

            role = "user" if msg.role == Role.USER else "assistant"
            anthropic_messages.append({"role": role, "content": msg.content})

        kwargs = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "system": system_message.strip(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for chunk in stream.text_stream:
                    if self._cancel_event.is_set():
                        break
                    if chunk:
                        yield chunk
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
