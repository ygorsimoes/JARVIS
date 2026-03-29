import unittest

from jarvis.adapters.activation import PushToTalkActivationAdapter
from jarvis.adapters.factory import build_runtime_adapters
from jarvis.adapters.llm import (
    FakeLLMAdapter,
    FoundationModelsBridgeAdapter,
    MLXLMAdapter,
)
from jarvis.adapters.stt import SpeechAnalyzerSTTAdapter
from jarvis.adapters.tts import MLXAudioKokoroAdapter, NoOpTTSAdapter
from jarvis.adapters.vad import SpeechDetectorAdapter
from jarvis.audio import NoOpPlaybackBackend, SoundDevicePlaybackBackend
from jarvis.config import JarvisConfig


class RuntimeAdaptersFactoryTests(unittest.TestCase):
    def test_defaults_use_fake_and_noop_backends_without_native_flag(self):
        adapters = build_runtime_adapters(JarvisConfig(), enable_native_backends=False)

        self.assertIsInstance(adapters.activation, PushToTalkActivationAdapter)
        self.assertIsInstance(adapters.stt, SpeechAnalyzerSTTAdapter)
        self.assertIsInstance(adapters.vad, SpeechDetectorAdapter)
        self.assertIsInstance(adapters.hot_path_llm, FakeLLMAdapter)
        self.assertIsInstance(adapters.deliberative_llm, FakeLLMAdapter)
        self.assertIsInstance(adapters.tts, NoOpTTSAdapter)
        self.assertIsInstance(adapters.playback, NoOpPlaybackBackend)

    def test_native_flag_builds_real_backends_when_requested(self):
        config = JarvisConfig(
            llm_hot_path="foundation_models",
            llm_deliberative="mlx_lm",
            tts_backend="mlx_audio_kokoro",
            playback_backend="sounddevice",
        )

        adapters = build_runtime_adapters(config, enable_native_backends=True)

        self.assertIsInstance(adapters.hot_path_llm, FoundationModelsBridgeAdapter)
        self.assertIsInstance(adapters.deliberative_llm, MLXLMAdapter)
        self.assertIsInstance(adapters.tts, MLXAudioKokoroAdapter)
        self.assertIsInstance(adapters.playback, SoundDevicePlaybackBackend)


if __name__ == "__main__":
    unittest.main()
