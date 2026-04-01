import asyncio
import types
from unittest.mock import patch

import pytest

from jarvis.adapters.activation import PushToTalkActivationAdapter
from jarvis.adapters.vad import SpeechDetectorAdapter
from jarvis.audio.playback import NoOpPlaybackBackend, SoundDevicePlaybackBackend
from jarvis.config import JarvisConfig
from jarvis.core.resource_governor import ResourceGovernor


@pytest.mark.asyncio
class TestPlaybackBackend:
    async def test_noop_backend_methods_complete(self):
        backend = NoOpPlaybackBackend()
        await backend.play(b"audio", 24000)
        await backend.stop()
        await backend.shutdown()

    async def test_sounddevice_backend_skips_empty_audio(self):
        backend = SoundDevicePlaybackBackend()
        with patch.object(backend, "_ensure_dependencies") as ensure_dependencies:
            await backend.play(b"", 24000)
        ensure_dependencies.assert_not_called()

    async def test_sounddevice_backend_reports_missing_dependencies(self):
        backend = SoundDevicePlaybackBackend()
        with patch(
            "jarvis.audio.playback.importlib.import_module", side_effect=ImportError
        ):
            with pytest.raises(RuntimeError):
                await backend.play(b"audio", 24000)

    async def test_sounddevice_backend_plays_stops_and_shuts_down_with_fake_dependencies(
        self,
    ):
        backend = SoundDevicePlaybackBackend()
        events = []

        class FakeArray:
            size = 2

        fake_numpy = types.SimpleNamespace(
            frombuffer=lambda audio_bytes, dtype: FakeArray(), float32="float32"
        )
        fake_sounddevice = types.SimpleNamespace(
            play=lambda array, sample_rate_hz, blocking: events.append(
                ("play", sample_rate_hz, blocking, array.size)
            ),
            stop=lambda: events.append(("stop",)),
        )

        with patch.object(
            backend, "_ensure_dependencies", return_value=(fake_sounddevice, fake_numpy)
        ):
            await backend.play(b"audio", 24000)
            await backend.stop()
            await backend.shutdown()

        assert events == [("play", 24000, True, 2), ("stop",), ("stop",)]


@pytest.mark.asyncio
class TestRuntimeSupport:
    async def test_push_to_talk_waits_for_input(self):
        adapter = PushToTalkActivationAdapter(
            prompt="Pressione", backend="push_to_talk_terminal"
        )
        with patch("builtins.input", return_value="") as fake_input:
            activated = await adapter.listen()

        fake_input.assert_called_once_with("Pressione")
        assert activated

    async def test_push_to_talk_uses_hotkey_when_available(self):
        adapter = PushToTalkActivationAdapter()

        def fake_ensure_listener():
            adapter._activation_queue = asyncio.Queue(maxsize=1)
            adapter._activation_queue.put_nowait(True)

        with patch.object(
            adapter, "_ensure_hotkey_listener", side_effect=fake_ensure_listener
        ):
            with patch("builtins.input") as fake_input:
                activated = await adapter.listen()

        fake_input.assert_not_called()
        assert activated

    async def test_push_to_talk_shutdown_stops_hotkey_listener(self):
        adapter = PushToTalkActivationAdapter()
        stop_calls = []
        adapter.__dict__["_listener"] = types.SimpleNamespace(
            stop=lambda: stop_calls.append("stop")
        )

        await adapter.shutdown()

        assert stop_calls == ["stop"]
        assert adapter._listener is None

    async def test_speech_detector_classifies_speech_and_silence(self):
        assert SpeechDetectorAdapter.event_has_speech({"type": "speech_started"})
        assert SpeechDetectorAdapter.event_has_speech(
            {"type": "speech_detector_result", "speech_detected": True}
        )
        assert SpeechDetectorAdapter.event_is_silence({"type": "speech_ended"})
        assert SpeechDetectorAdapter.event_is_silence(
            {"type": "speech_detector_result", "speech_detected": False}
        )
        assert (
            SpeechDetectorAdapter.signal_reason(
                {"type": "speech_detector_result", "speech_detected": True}
            )
            == "speech_detector_result"
        )

    async def test_resource_governor_returns_unavailable_without_mlx(self):
        governor = ResourceGovernor(JarvisConfig())
        with patch(
            "jarvis.core.resource_governor.importlib.import_module",
            side_effect=ImportError,
        ):
            status = governor.apply()

        assert not status.applied
        assert status.backend == "unavailable"
        assert status.limits["max_kv_size"] == 4096

    async def test_resource_governor_applies_limits_when_mlx_is_available(self):
        applied = []
        fake_mx = types.SimpleNamespace(
            set_memory_limit=lambda value: applied.append(("memory", value)),
            set_wired_limit=lambda value: applied.append(("wired", value)),
            set_cache_limit=lambda value: applied.append(("cache", value)),
        )
        governor = ResourceGovernor(JarvisConfig())
        with patch(
            "jarvis.core.resource_governor.importlib.import_module",
            return_value=fake_mx,
        ):
            status = governor.apply()

        assert status.applied
        assert status.backend == "mlx"
        assert [item[0] for item in applied] == ["memory", "wired", "cache"]
