from __future__ import annotations

import asyncio
import importlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class PlaybackBackend(Protocol):
    async def play(self, audio_bytes: bytes, sample_rate_hz: int) -> None: ...

    async def stop(self) -> None: ...

    async def shutdown(self) -> None: ...


class NoOpPlaybackBackend:
    async def play(self, audio_bytes: bytes, sample_rate_hz: int) -> None:
        del audio_bytes, sample_rate_hz
        await asyncio.sleep(0)

    async def stop(self) -> None:
        await asyncio.sleep(0)

    async def shutdown(self) -> None:
        await asyncio.sleep(0)


class SoundDevicePlaybackBackend:
    def __init__(self) -> None:
        self._sounddevice = None
        self._numpy = None

    async def play(self, audio_bytes: bytes, sample_rate_hz: int) -> None:
        if not audio_bytes:
            return
        sd, np = await asyncio.to_thread(self._ensure_dependencies)
        array = np.frombuffer(audio_bytes, dtype=np.float32)
        if array.size == 0:
            return
        await asyncio.to_thread(sd.play, array, sample_rate_hz, blocking=True)

    async def stop(self) -> None:
        try:
            sd, _ = await asyncio.to_thread(self._ensure_dependencies)
        except RuntimeError:
            return
        await asyncio.to_thread(sd.stop)

    async def shutdown(self) -> None:
        try:
            await self.stop()
        except RuntimeError:
            return

    def _ensure_dependencies(self):
        if self._sounddevice is None or self._numpy is None:
            try:
                np = importlib.import_module("numpy")
                sd = importlib.import_module("sounddevice")
            except ImportError as exc:
                raise RuntimeError(
                    "sounddevice playback requires the 'audio' dependency extra"
                ) from exc
            self._sounddevice = sd
            self._numpy = np
        return self._sounddevice, self._numpy
