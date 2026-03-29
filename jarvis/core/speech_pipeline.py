from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Optional

from ..audio.playback import PlaybackBackend
from ..models.events import Event, EventType


@dataclass(frozen=True)
class QueuedSentence:
    index: int
    text: str


@dataclass(frozen=True)
class RenderedAudioChunk:
    index: int
    text: str
    audio_bytes: bytes
    is_first_chunk: bool
    is_last_chunk: bool


class SpeechPipeline:
    def __init__(
        self,
        tts_adapter,
        playback_backend: PlaybackBackend,
        sample_rate_hz: int,
        event_bus=None,
    ) -> None:
        self.tts_adapter = tts_adapter
        self.playback_backend = playback_backend
        self.sample_rate_hz = sample_rate_hz
        self.event_bus = event_bus
        self.utterance_id = str(uuid.uuid4())

        self._sentence_queue: asyncio.Queue = asyncio.Queue()
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._tts_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None
        self._started = False
        self._stopped = False
        self._error: Optional[Exception] = None

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._tts_task = asyncio.create_task(self._tts_worker(), name="jarvis-tts-worker")
        self._playback_task = asyncio.create_task(
            self._playback_worker(), name="jarvis-playback-worker"
        )

    async def enqueue_sentence(self, index: int, text: str) -> None:
        if self._stopped:
            return
        if not self._started:
            await self.start()
        await self._sentence_queue.put(QueuedSentence(index=index, text=text))
        await self._publish(EventType.TTS_SENTENCE_QUEUED, {"utterance_id": self.utterance_id, "index": index, "text": text})

    async def finish(self) -> None:
        if not self._started:
            await self.start()
        await self._sentence_queue.put(None)
        await self._await_workers()
        if self._error is not None:
            raise self._error

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._tts_task is not None:
            self._tts_task.cancel()
        if self._playback_task is not None:
            self._playback_task.cancel()
        await self.playback_backend.stop()
        await self._await_workers()

    async def shutdown(self) -> None:
        await self.stop()
        await self.playback_backend.shutdown()

    async def _tts_worker(self) -> None:
        try:
            while True:
                item = await self._sentence_queue.get()
                if item is None:
                    break
                await self._publish(
                    EventType.TTS_STARTED,
                    {"utterance_id": self.utterance_id, "index": item.index, "text": item.text},
                )
                total_size = 0
                emitted_any_chunk = False
                async for rendered in self._render_sentence_chunks(item):
                    emitted_any_chunk = True
                    total_size += len(rendered.audio_bytes)
                    await self._audio_queue.put(rendered)
                if not emitted_any_chunk:
                    await self._audio_queue.put(
                        RenderedAudioChunk(
                            index=item.index,
                            text=item.text,
                            audio_bytes=b"",
                            is_first_chunk=True,
                            is_last_chunk=True,
                        )
                    )
                await self._publish(
                    EventType.TTS_COMPLETED,
                    {"utterance_id": self.utterance_id, "index": item.index, "size_bytes": total_size},
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._error = exc
            await self._audio_queue.put(exc)
        finally:
            await self._audio_queue.put(None)

    async def _playback_worker(self) -> None:
        try:
            while True:
                item = await self._audio_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    self._error = item
                    break
                if item.is_first_chunk:
                    await self._publish(
                        EventType.PLAYBACK_STARTED,
                        {"utterance_id": self.utterance_id, "index": item.index},
                    )
                if item.audio_bytes:
                    await self.playback_backend.play(item.audio_bytes, self.sample_rate_hz)
                if item.is_last_chunk:
                    await self._publish(
                        EventType.PLAYBACK_COMPLETED,
                        {"utterance_id": self.utterance_id, "index": item.index},
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._error = exc

    async def _render_sentence_chunks(self, item: QueuedSentence):
        pending_chunk: bytes | None = None
        chunk_index = 0

        async for audio_chunk in self.tts_adapter.synthesize_stream(item.text):
            if pending_chunk is not None:
                yield RenderedAudioChunk(
                    index=item.index,
                    text=item.text,
                    audio_bytes=pending_chunk,
                    is_first_chunk=chunk_index == 0,
                    is_last_chunk=False,
                )
                chunk_index += 1
            pending_chunk = audio_chunk

        if pending_chunk is None:
            return

        yield RenderedAudioChunk(
            index=item.index,
            text=item.text,
            audio_bytes=pending_chunk,
            is_first_chunk=chunk_index == 0,
            is_last_chunk=True,
        )

    async def _await_workers(self) -> None:
        tasks = [task for task in (self._tts_task, self._playback_task) if task is not None]
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                self._error = self._error or result

    async def _publish(self, event_type: EventType, payload: dict) -> None:
        if self.event_bus is None:
            return
        await self.event_bus.publish(Event(event_type=event_type, payload=payload))
