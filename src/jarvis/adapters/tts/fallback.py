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
        self._effective_backend_name = primary_name
        self._last_error_message: str | None = None

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        if self._fallback_only:
            self._effective_backend_name = self.fallback_name
            async for chunk in self.fallback.synthesize_stream(text):
                self._last_successful = self.fallback
                yield chunk
            return

        emitted_any_chunk = False
        try:
            async for chunk in self.primary.synthesize_stream(text):
                emitted_any_chunk = True
                self._last_successful = self.primary
                self._effective_backend_name = self.primary_name
                self._last_error_message = None
                yield chunk
            return
        except Exception as exc:
            if emitted_any_chunk:
                raise
            self._fallback_only = True
            self._effective_backend_name = self.fallback_name
            self._last_error_message = str(exc)
            logger.warning(
                "Primary TTS backend failed, falling back",
                primary=self.primary_name,
                fallback=self.fallback_name,
                error=str(exc),
            )

        async for chunk in self.fallback.synthesize_stream(text):
            self._last_successful = self.fallback
            yield chunk

    def trace_backend_state(self) -> dict[str, object]:
        return {
            "tts_primary_backend": self.primary_name,
            "tts_fallback_backend": self.fallback_name,
            "tts_effective_backend": self._effective_backend_name,
            "tts_fallback_active": self._effective_backend_name == self.fallback_name,
            "tts_last_error": self._last_error_message,
        }

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

    async def prewarm(self) -> None:
        primary_prewarm = getattr(self.primary, "prewarm", None)
        fallback_prewarm = getattr(self.fallback, "prewarm", None)

        if primary_prewarm is not None and not self._fallback_only:
            try:
                await primary_prewarm()
                self._effective_backend_name = self.primary_name
                self._last_error_message = None
                return
            except Exception as exc:
                self._fallback_only = True
                self._effective_backend_name = self.fallback_name
                self._last_error_message = str(exc)
                logger.warning(
                    "Primary TTS backend failed during prewarm, falling back",
                    primary=self.primary_name,
                    fallback=self.fallback_name,
                    error=str(exc),
                )

        if fallback_prewarm is not None:
            await fallback_prewarm()
