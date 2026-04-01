import asyncio

import pytest

from jarvis.bus import EventBus
from jarvis.core.speech_pipeline import SpeechPipeline
from jarvis.models.events import EventType


class FakeTTSAdapter:
    def __init__(self, suffix: bytes = b"-audio"):
        self.suffix = suffix
        self.calls = []

    async def synthesize_stream(self, text: str):
        self.calls.append(text)
        yield text.encode("utf-8") + self.suffix


class SlowFakeTTSAdapter(FakeTTSAdapter):
    async def synthesize_stream(self, text: str):
        self.calls.append(text)
        await asyncio.sleep(1)
        yield text.encode("utf-8") + self.suffix


class CancellableSlowFakeTTSAdapter(SlowFakeTTSAdapter):
    def __init__(self, suffix: bytes = b"-audio"):
        super().__init__(suffix=suffix)
        self.cancel_calls = 0

    async def cancel_current_synthesis(self) -> bool:
        self.cancel_calls += 1
        return True


class FakePlaybackBackend:
    def __init__(self):
        self.played = []
        self.stop_calls = 0
        self.shutdown_calls = 0

    async def play(self, audio_bytes: bytes, sample_rate_hz: int) -> None:
        self.played.append((audio_bytes, sample_rate_hz))

    async def stop(self) -> None:
        self.stop_calls += 1

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


class ChunkedFakeTTSAdapter(FakeTTSAdapter):
    async def synthesize_stream(self, text: str):
        self.calls.append(text)
        yield text.encode("utf-8") + b"-chunk-1"
        yield text.encode("utf-8") + b"-chunk-2"


class FailingFakeTTSAdapter(FakeTTSAdapter):
    async def synthesize_stream(self, text: str):
        self.calls.append(text)
        raise RuntimeError("tts exploded")
        yield text.encode("utf-8")


class FailingPlaybackBackend(FakePlaybackBackend):
    async def play(self, audio_bytes: bytes, sample_rate_hz: int) -> None:
        await super().play(audio_bytes, sample_rate_hz)
        raise RuntimeError("playback exploded")


@pytest.mark.asyncio
class TestSpeechPipeline:
    async def test_pipeline_synthesizes_and_plays_in_order(self):
        bus = EventBus()
        subscription = await bus.subscribe(
            [EventType.TTS_STARTED, EventType.PLAYBACK_COMPLETED]
        )
        tts = FakeTTSAdapter()
        playback = FakePlaybackBackend()
        pipeline = SpeechPipeline(tts, playback, sample_rate_hz=24000, event_bus=bus)

        await pipeline.start()
        await pipeline.enqueue_sentence(0, "Primeira frase.")
        await pipeline.enqueue_sentence(1, "Segunda frase.")
        await pipeline.finish()

        assert tts.calls == ["Primeira frase.", "Segunda frase."]
        assert len(playback.played) == 2
        assert playback.played[0][1] == 24000

        events = [await subscription.get() for _ in range(4)]
        event_types = [event.event_type for event in events]
        assert EventType.TTS_STARTED in event_types
        assert EventType.PLAYBACK_COMPLETED in event_types

    async def test_pipeline_stop_cancels_pending_work(self):
        tts = CancellableSlowFakeTTSAdapter()
        playback = FakePlaybackBackend()
        pipeline = SpeechPipeline(tts, playback, sample_rate_hz=24000)

        await pipeline.start()
        await pipeline.enqueue_sentence(0, "Frase lenta.")
        await asyncio.sleep(0.05)
        await pipeline.stop()

        assert playback.stop_calls == 1
        assert playback.played == []
        assert tts.cancel_calls == 1

    async def test_pipeline_shutdown_is_idempotent_and_shuts_down_playback(self):
        tts = FakeTTSAdapter()
        playback = FakePlaybackBackend()
        pipeline = SpeechPipeline(tts, playback, sample_rate_hz=24000)

        await pipeline.start()
        await pipeline.shutdown()
        await pipeline.shutdown()

        assert playback.shutdown_calls == 1

    async def test_pipeline_plays_multiple_chunks_in_order_for_one_sentence(self):
        bus = EventBus()
        subscription = await bus.subscribe(
            [EventType.PLAYBACK_STARTED, EventType.PLAYBACK_COMPLETED]
        )
        tts = ChunkedFakeTTSAdapter()
        playback = FakePlaybackBackend()
        pipeline = SpeechPipeline(tts, playback, sample_rate_hz=24000, event_bus=bus)

        await pipeline.enqueue_sentence(0, "Primeira frase.")
        await pipeline.finish()

        assert tts.calls == ["Primeira frase."]
        assert playback.played == [
            (b"Primeira frase.-chunk-1", 24000),
            (b"Primeira frase.-chunk-2", 24000),
        ]
        started_event = await subscription.get()
        completed_event = await subscription.get()
        assert started_event.event_type == EventType.PLAYBACK_STARTED
        assert completed_event.event_type == EventType.PLAYBACK_COMPLETED
        assert started_event.payload["index"] == 0
        assert completed_event.payload["index"] == 0

    async def test_pipeline_finish_propagates_tts_errors(self):
        tts = FailingFakeTTSAdapter()
        playback = FakePlaybackBackend()
        pipeline = SpeechPipeline(tts, playback, sample_rate_hz=24000)

        await pipeline.enqueue_sentence(0, "Frase com erro.")

        with pytest.raises(RuntimeError, match="tts exploded"):
            await pipeline.finish()

        assert playback.played == []

    async def test_pipeline_finish_propagates_playback_errors(self):
        tts = FakeTTSAdapter()
        playback = FailingPlaybackBackend()
        pipeline = SpeechPipeline(tts, playback, sample_rate_hz=24000)

        await pipeline.enqueue_sentence(0, "Frase com playback falho.")

        with pytest.raises(RuntimeError, match="playback exploded"):
            await pipeline.finish()

        assert playback.played == [(b"Frase com playback falho.-audio", 24000)]
