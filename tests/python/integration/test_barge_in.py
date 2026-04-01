import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from jarvis.config import JarvisConfig
from jarvis.core.speech_pipeline import SpeechPipeline
from jarvis.runtime import JarvisRuntime


class FakeSTTSession:
    def __init__(self, events):
        self._events = list(events)
        self.stopped = False

    async def iter_events(self):
        for event in self._events:
            yield event
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        self.stopped = True


class FakeSTTAdapter:
    def __init__(self, events):
        self.session = FakeSTTSession(events)

    async def start_live_session(self):
        return self.session


class FakeVADAdapter:
    def classify_event(self, event):
        return event.get("classified")


@pytest.mark.asyncio
class TestBargeInWatcher:
    @pytest_asyncio.fixture(autouse=True)
    async def _runtime(self):
        self.runtime = JarvisRuntime.from_config(
            JarvisConfig(allowed_file_roots=["/tmp"]),
            enable_native_backends=False,
        )
        yield
        await self.runtime.shutdown()

    async def test_watcher_cancels_turn_on_speech(self):
        self.runtime.stt_adapter = FakeSTTAdapter(
            [
                {"type": "other_event"},
                {"type": "speech", "classified": {"speech_detected": True}},
            ]
        )
        self.runtime.vad_adapter = FakeVADAdapter()

        self.runtime.interrupt_current_turn = AsyncMock(return_value=True)

        async def dummy_response():
            await asyncio.sleep(1.0)

        task = asyncio.create_task(dummy_response())
        pipeline = MagicMock(spec=SpeechPipeline)

        await self.runtime._barge_in_watcher(task, pipeline)

        self.runtime.interrupt_current_turn.assert_called_once_with(reason="barge-in")
        task.cancel()

    async def test_watcher_exits_if_response_completes(self):
        self.runtime.stt_adapter = FakeSTTAdapter(
            [
                {"type": "other_event"},
                {"type": "other_event"},
                {"type": "speech", "classified": {"speech_detected": True}},
            ]
        )
        self.runtime.vad_adapter = FakeVADAdapter()

        self.runtime.interrupt_current_turn = AsyncMock(return_value=True)

        async def dummy_response():
            # Completes before the speech event
            pass

        task = asyncio.create_task(dummy_response())
        await task  # wait for it to complete

        pipeline = MagicMock(spec=SpeechPipeline)

        await self.runtime._barge_in_watcher(task, pipeline)

        self.runtime.interrupt_current_turn.assert_not_called()
