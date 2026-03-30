import asyncio
import unittest

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


class SpeechPipelineTests(unittest.IsolatedAsyncioTestCase):
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

        self.assertEqual(tts.calls, ["Primeira frase.", "Segunda frase."])
        self.assertEqual(len(playback.played), 2)
        self.assertEqual(playback.played[0][1], 24000)

        events = [await subscription.get() for _ in range(4)]
        event_types = [event.event_type for event in events]
        self.assertIn(EventType.TTS_STARTED, event_types)
        self.assertIn(EventType.PLAYBACK_COMPLETED, event_types)

    async def test_pipeline_stop_cancels_pending_work(self):
        tts = CancellableSlowFakeTTSAdapter()
        playback = FakePlaybackBackend()
        pipeline = SpeechPipeline(tts, playback, sample_rate_hz=24000)

        await pipeline.start()
        await pipeline.enqueue_sentence(0, "Frase lenta.")
        await asyncio.sleep(0.05)
        await pipeline.stop()

        self.assertEqual(playback.stop_calls, 1)
        self.assertEqual(playback.played, [])
        self.assertEqual(tts.cancel_calls, 1)

    async def test_pipeline_shutdown_is_idempotent_and_shuts_down_playback(self):
        tts = FakeTTSAdapter()
        playback = FakePlaybackBackend()
        pipeline = SpeechPipeline(tts, playback, sample_rate_hz=24000)

        await pipeline.start()
        await pipeline.shutdown()
        await pipeline.shutdown()

        self.assertEqual(playback.shutdown_calls, 1)


if __name__ == "__main__":
    unittest.main()
