from jarvis.adapters.activation import PushToTalkActivationAdapter
from jarvis.adapters.factory import build_runtime_adapters
from jarvis.adapters.llm import (
    FakeLLMAdapter,
    FoundationModelsBridgeAdapter,
    MLXLMAdapter,
)
from jarvis.adapters.stt import SpeechAnalyzerSTTAdapter
from jarvis.adapters.tts import (
    AVSpeechAdapter,
    FallbackTTSAdapter,
    MLXAudioKokoroAdapter,
    NoOpTTSAdapter,
)
from jarvis.adapters.vad import SpeechDetectorAdapter
from jarvis.audio import NoOpPlaybackBackend, SoundDevicePlaybackBackend
from jarvis.config import JarvisConfig


class TestRuntimeAdaptersFactory:
    def test_defaults_use_fake_and_noop_backends_without_native_flag(self):
        adapters = build_runtime_adapters(JarvisConfig(), enable_native_backends=False)

        assert isinstance(adapters.activation, PushToTalkActivationAdapter)
        assert isinstance(adapters.stt, SpeechAnalyzerSTTAdapter)
        assert isinstance(adapters.vad, SpeechDetectorAdapter)
        assert isinstance(adapters.hot_path_llm, FakeLLMAdapter)
        assert isinstance(adapters.deliberative_llm, FakeLLMAdapter)
        assert isinstance(adapters.tts, NoOpTTSAdapter)
        assert isinstance(adapters.playback, NoOpPlaybackBackend)

    def test_defaults_build_real_backends_when_native_mode_is_enabled(self):
        adapters = build_runtime_adapters(JarvisConfig())

        assert isinstance(adapters.activation, PushToTalkActivationAdapter)
        assert isinstance(adapters.stt, SpeechAnalyzerSTTAdapter)
        assert isinstance(adapters.vad, SpeechDetectorAdapter)
        assert isinstance(adapters.hot_path_llm, FoundationModelsBridgeAdapter)
        assert isinstance(adapters.deliberative_llm, MLXLMAdapter)
        assert isinstance(adapters.fallback_llm, MLXLMAdapter)
        assert adapters.fallback_backend_name == "mlx_lm_fallback"
        assert isinstance(adapters.tts, AVSpeechAdapter)
        assert adapters.tts_backend_name == "avspeech"
        assert isinstance(adapters.playback, SoundDevicePlaybackBackend)

    def test_native_flag_builds_real_backends_when_requested(self):
        config = JarvisConfig(
            llm_hot_path="foundation_models",
            llm_deliberative="mlx_lm",
            llm_hot_path_fallback="mlx_lm",
            tts_backend="mlx_audio_kokoro",
            playback_backend="sounddevice",
        )

        adapters = build_runtime_adapters(config, enable_native_backends=True)

        assert isinstance(adapters.hot_path_llm, FoundationModelsBridgeAdapter)
        assert isinstance(adapters.deliberative_llm, MLXLMAdapter)
        assert isinstance(adapters.fallback_llm, MLXLMAdapter)
        assert isinstance(adapters.tts, FallbackTTSAdapter)
        assert isinstance(adapters.tts.primary, MLXAudioKokoroAdapter)
        assert isinstance(adapters.tts.fallback, AVSpeechAdapter)
        assert isinstance(adapters.playback, SoundDevicePlaybackBackend)

    def test_native_flag_can_build_avspeech_fallback(self):
        config = JarvisConfig(tts_backend="avspeech")

        adapters = build_runtime_adapters(config, enable_native_backends=True)

        assert isinstance(adapters.tts, AVSpeechAdapter)
        assert adapters.tts_backend_name == "avspeech"
