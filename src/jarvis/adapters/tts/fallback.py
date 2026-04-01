from __future__ import annotations

from collections.abc import AsyncIterator

from ...observability import get_logger

logger = get_logger(__name__)


class FallbackTTSAdapter:
    def __init__(
        self, primary, fallback, *, primary_name: str, fallback_name: str
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name
        self._last_successful = primary
        self._fallback_only = False

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        if self._fallback_only:
            async for chunk in self.fallback.synthesize_stream(text):
                self._last_successful = self.fallback
                yield chunk
            return

        emitted_any_chunk = False
        try:
            async for chunk in self.primary.synthesize_stream(text):
                emitted_any_chunk = True
                self._last_successful = self.primary
                yield chunk
            return
        except Exception as exc:
            if emitted_any_chunk:
                raise
            self._fallback_only = True
            logger.debug(
                "Primary TTS backend failed, falling back",
                primary=self.primary_name,
                fallback=self.fallback_name,
                error=str(exc),
            )

        async for chunk in self.fallback.synthesize_stream(text):
            self._last_successful = self.fallback
            yield chunk

    async def cancel_current_synthesis(self) -> bool:
        cancelled = False
        for adapter in (self.primary, self.fallback):
            cancel = getattr(adapter, "cancel_current_synthesis", None)
            if cancel is None:
                continue
            try:
                cancelled = await cancel() or cancelled
            except Exception:
                continue
        return cancelled

    async def shutdown(self) -> None:
        for adapter in (self.primary, self.fallback):
            shutdown = getattr(adapter, "shutdown", None)
            if shutdown is not None:
                await shutdown()
