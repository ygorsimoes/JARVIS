from __future__ import annotations

from typing import AsyncIterator


class NoOpTTSAdapter:
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        yield text.encode("utf-8")

    async def shutdown(self) -> None:
        return None
