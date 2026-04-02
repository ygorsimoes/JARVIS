from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

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
        event_context: Optional[
            dict[str, object] | Callable[[], dict[str, object]]
        ] = None,
    ) -> None:
        self.tts_adapter = tts_adapter
        self.playback_backend = playback_backend
        self.sample_rate_hz = sample_rate_hz
        self.event_bus = event_bus
        self.event_context = event_context or {}
        self.utterance_id = str(uuid.uuid4())

        self._sentence_queue: asyncio.Queue = asyncio.Queue()
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._tts_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None
        self._started = False
        self._stopped = False
        self._shutdown = False
        self._error: Optional[Exception] = None

    async def start(self) -> None:
        if self._started or self._shutdown:
            return
        self._started = True
        self._tts_task = asyncio.create_task(
            self._tts_worker(), name="jarvis-tts-worker"
        )
        self._playback_task = asyncio.create_task(
            self._playback_worker(), name="jarvis-playback-worker"
        )

    async def enqueue_sentence(self, index: int, text: str) -> None:
        if self._stopped or self._shutdown:
            return
        if not self._started:
            await self.start()
        await self._sentence_queue.put(QueuedSentence(index=index, text=text))
        await self._publish(
            EventType.TTS_SENTENCE_QUEUED,
            {"utterance_id": self.utterance_id, "index": index, "text": text},
        )

    async def finish(self) -> None:
        if self._shutdown:
            return
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

        cancel_synthesis = getattr(self.tts_adapter, "cancel_current_synthesis", None)
        if cancel_synthesis is not None:
            try:
                await cancel_synthesis()
            except Exception:
                pass

        self._drain_queue(self._sentence_queue)
        self._drain_queue(self._audio_queue)

        if self._tts_task is not None:
            self._tts_task.cancel()
        if self._playback_task is not None:
            self._playback_task.cancel()

        await self.playback_backend.stop()
        await self._await_workers()

    async def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        await self.stop()
        await self.playback_backend.shutdown()

    async def _tts_worker(self) -> None:
        try:
            while True:
                item = await self._sentence_queue.get()
                if item is None or self._stopped:
                    break
                await self._publish(
                    EventType.TTS_STARTED,
                    {
                        "utterance_id": self.utterance_id,
                        "index": item.index,
                        "text": item.text,
                    },
                )
                total_size = 0
                emitted_any_chunk = False
                async for rendered in self._render_sentence_chunks(item):
                    if self._stopped:
                        break
                    emitted_any_chunk = True
                    total_size += len(rendered.audio_bytes)
                    await self._audio_queue.put(rendered)
                if self._stopped:
                    break
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
                    {
                        "utterance_id": self.utterance_id,
                        "index": item.index,
                        "size_bytes": total_size,
                    },
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
                if item is None or self._stopped:
                    break
                if isinstance(item, Exception):
                    self._error = item
                    break
                if item.is_first_chunk:
                    await self._publish(
                        EventType.PLAYBACK_STARTED,
                        {"utterance_id": self.utterance_id, "index": item.index},
                    )
                if item.audio_bytes and not self._stopped:
                    await self.playback_backend.play(
                        item.audio_bytes, self.sample_rate_hz
                    )
                if item.is_last_chunk and not self._stopped:
                    await self.playback_backend.flush()
                    await self._publish(
                        EventType.PLAYBACK_COMPLETED,
                        {"utterance_id": self.utterance_id, "index": item.index},
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._error = exc

    async def _render_sentence_chunks(self, item: QueuedSentence):
        emitted_any_chunk = False
        chunk_index = 0

        async for audio_chunk in self.tts_adapter.synthesize_stream(item.text):
            if self._stopped:
                break
            emitted_any_chunk = True
            yield RenderedAudioChunk(
                index=item.index,
                text=item.text,
                audio_bytes=audio_chunk,
                is_first_chunk=chunk_index == 0,
                is_last_chunk=False,
            )
            chunk_index += 1

        if not emitted_any_chunk or self._stopped:
            return

        yield RenderedAudioChunk(
            index=item.index,
            text=item.text,
            audio_bytes=b"",
            is_first_chunk=False,
            is_last_chunk=True,
        )

    async def _await_workers(self) -> None:
        tasks = [
            task for task in (self._tts_task, self._playback_task) if task is not None
        ]
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self._tts_task = None
        self._playback_task = None
        for result in results:
            if isinstance(result, Exception) and not isinstance(
                result, asyncio.CancelledError
            ):
                self._error = self._error or result

    def _drain_queue(self, queue: asyncio.Queue) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def _publish(self, event_type: EventType, payload: dict) -> None:
        if self.event_bus is None:
            return
        context = (
            self.event_context() if callable(self.event_context) else self.event_context
        )
        enriched = dict(context)
        enriched.update(payload)
        await self.event_bus.publish(Event(event_type=event_type, payload=enriched))
