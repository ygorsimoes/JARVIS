from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    List,
    Protocol,
    runtime_checkable,
)

from ..models.conversation import Message
from ..models.memory import Memory


@runtime_checkable
class STTSession(Protocol):
    async def iter_events(self) -> AsyncIterator[dict]: ...

    async def stop(self) -> None: ...


@runtime_checkable
class STTAdapter(Protocol):
    async def start_live_session(self) -> STTSession: ...

    async def transcribe_stream(self) -> AsyncIterator[str]: ...

    async def shutdown(self) -> None: ...


@runtime_checkable
class LLMAdapter(Protocol):
    async def chat_stream(
        self,
        messages: List[Message],
        tools: List[dict],
        max_kv_size: int,
        tool_invoker: Callable[[str, dict], Awaitable[object]] | None = None,
    ) -> AsyncIterator[str]: ...

    async def cancel_current_response(self) -> bool: ...

    async def close_session(self) -> None: ...

    async def shutdown(self) -> None: ...


@runtime_checkable
class TTSAdapter(Protocol):
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]: ...

    async def shutdown(self) -> None: ...


@runtime_checkable
class WakeWordAdapter(Protocol):
    async def listen(self) -> bool: ...


@runtime_checkable
class MemoryAdapter(Protocol):
    def search(self, query: str, top_k: int) -> List[Memory]: ...

    def save(self, content: str, metadata: dict) -> None: ...

    def should_persist(self, turn: Any) -> bool: ...
