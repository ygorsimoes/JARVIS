from __future__ import annotations

import asyncio
import importlib
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PlaybackBackend(Protocol):
    async def play(self, audio_bytes: bytes, sample_rate_hz: int) -> None: ...

    async def flush(self) -> None: ...

    async def stop(self) -> None: ...

    async def shutdown(self) -> None: ...


class NoOpPlaybackBackend:
    async def play(self, audio_bytes: bytes, sample_rate_hz: int) -> None:
        del audio_bytes, sample_rate_hz
        await asyncio.sleep(0)

    async def flush(self) -> None:
        await asyncio.sleep(0)

    async def stop(self) -> None:
        await asyncio.sleep(0)

    async def shutdown(self) -> None:
        await asyncio.sleep(0)


class SoundDevicePlaybackBackend:
    def __init__(self) -> None:
        self._sounddevice = None
        self._numpy = None
        self._stream = None
        self._sample_rate_hz: int | None = None
        self._write_queue: asyncio.Queue | None = None
        self._worker_task: asyncio.Task | None = None
        self._pending_writes = 0
        self._drain_waiters: list[asyncio.Future[None]] = []
        self._legacy_mode = False

    async def play(self, audio_bytes: bytes, sample_rate_hz: int) -> None:
        if not audio_bytes:
            return
        sd, np = await asyncio.to_thread(self._ensure_dependencies)
        array = np.frombuffer(audio_bytes, dtype=np.float32)
        if array.size == 0:
            return

        await self._ensure_output(sample_rate_hz)
        if self._legacy_mode:
            await asyncio.to_thread(sd.play, array, sample_rate_hz, blocking=True)
            return

        if hasattr(array, "copy"):
            array = array.copy()
        self._pending_writes += 1
        assert self._write_queue is not None
        await self._write_queue.put(array)

    async def flush(self) -> None:
        if self._legacy_mode or self._pending_writes == 0:
            return

        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        self._drain_waiters.append(waiter)
        self._resolve_drain_waiters_if_idle()
        await waiter

    async def stop(self) -> None:
        try:
            sd, _ = await asyncio.to_thread(self._ensure_dependencies)
        except RuntimeError:
            return

        if self._worker_task is None and self._stream is None and not self._legacy_mode:
            return

        if self._write_queue is not None:
            while True:
                try:
                    self._write_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        self._pending_writes = 0
        self._resolve_drain_waiters_if_idle()

        worker = self._worker_task
        self._worker_task = None
        if worker is not None and self._write_queue is not None:
            await self._write_queue.put(None)

        stream = self._stream
        self._stream = None
        self._sample_rate_hz = None

        if stream is not None and not self._legacy_mode:
            abort = getattr(stream, "abort", None)
            stop_stream = getattr(stream, "stop", None)
            close = getattr(stream, "close", None)
            if abort is not None:
                await asyncio.to_thread(abort)
            elif stop_stream is not None:
                await asyncio.to_thread(stop_stream)
            if close is not None:
                await asyncio.to_thread(close)
        else:
            await asyncio.to_thread(sd.stop)

        if worker is not None:
            results = await asyncio.gather(worker, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    raise result

    async def shutdown(self) -> None:
        try:
            await self.stop()
        except RuntimeError:
            return

    async def _ensure_output(self, sample_rate_hz: int) -> None:
        if self._worker_task is not None and self._sample_rate_hz == sample_rate_hz:
            return
        if self._worker_task is not None:
            await self.stop()

        sd, _ = await asyncio.to_thread(self._ensure_dependencies)
        if not hasattr(sd, "OutputStream"):
            self._legacy_mode = True
            return

        self._legacy_mode = False
        self._sample_rate_hz = sample_rate_hz
        self._write_queue = asyncio.Queue()
        self._stream = await asyncio.to_thread(self._open_stream, sample_rate_hz)
        self._worker_task = asyncio.create_task(
            self._playback_worker(),
            name="jarvis-sounddevice-playback",
        )

    async def _playback_worker(self) -> None:
        assert self._write_queue is not None
        assert self._stream is not None

        while True:
            item = await self._write_queue.get()
            if item is None:
                self._resolve_drain_waiters_if_idle()
                return
            await asyncio.to_thread(self._stream.write, item)
            self._pending_writes = max(0, self._pending_writes - 1)
            self._resolve_drain_waiters_if_idle()

    def _open_stream(self, sample_rate_hz: int):
        sd, _ = self._ensure_dependencies()
        stream = sd.OutputStream(
            samplerate=sample_rate_hz,
            channels=1,
            dtype="float32",
        )
        start = getattr(stream, "start", None)
        if start is not None:
            start()
        return stream

    def _resolve_drain_waiters_if_idle(self) -> None:
        if self._pending_writes != 0:
            return
        pending = [waiter for waiter in self._drain_waiters if not waiter.done()]
        self._drain_waiters.clear()
        for waiter in pending:
            waiter.set_result(None)

    def _ensure_dependencies(self) -> tuple[Any, Any]:
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
