from __future__ import annotations

from dataclasses import dataclass

from ..audio import NoOpPlaybackBackend, SoundDevicePlaybackBackend
from ..config import JarvisConfig
from .activation import PushToTalkActivationAdapter
from .llm import FakeLLMAdapter, FoundationModelsBridgeAdapter, MLXLMAdapter
from .stt import SpeechAnalyzerSTTAdapter
from .tts import MLXAudioKokoroAdapter, NoOpTTSAdapter
from .vad import SpeechDetectorAdapter


@dataclass
class RuntimeAdapters:
    activation: object
    hot_path_llm: object
    deliberative_llm: object
    stt: object
    tts: object
    vad: object
    playback: object


def build_runtime_adapters(config: JarvisConfig, enable_native_backends: bool = False) -> RuntimeAdapters:
    activation = PushToTalkActivationAdapter()
    stt = SpeechAnalyzerSTTAdapter(config.stt_bridge_bin, locale=config.stt_locale)
    vad = SpeechDetectorAdapter()
    tts = NoOpTTSAdapter()
    playback = NoOpPlaybackBackend()

    hot_path_llm = FakeLLMAdapter(mode="hot_path")
    deliberative_llm = FakeLLMAdapter(mode="deliberative")

    if enable_native_backends and config.llm_hot_path == "foundation_models":
        hot_path_llm = FoundationModelsBridgeAdapter(
            base_url=config.llm_hot_path_url,
            instructions=config.system_prompt,
            bridge_binary_path=config.llm_hot_path_bridge_bin,
        )
    elif config.llm_hot_path == "fake":
        hot_path_llm = FakeLLMAdapter(mode="hot_path")

    if enable_native_backends and config.llm_deliberative == "mlx_lm":
        deliberative_llm = MLXLMAdapter(
            model_repo=config.llm_deliberative_model,
            max_tokens=config.llm_response_max_tokens,
        )
    elif config.llm_deliberative == "fake":
        deliberative_llm = FakeLLMAdapter(mode="deliberative")

    if enable_native_backends and config.tts_backend == "mlx_audio_kokoro":
        tts = MLXAudioKokoroAdapter(
            model_repo=config.tts_model,
            voice=config.tts_voice,
            lang_code=config.tts_lang_code,
        )
    elif config.tts_backend == "noop":
        tts = NoOpTTSAdapter()

    if enable_native_backends and config.playback_backend == "sounddevice":
        playback = SoundDevicePlaybackBackend()
    elif config.playback_backend == "noop":
        playback = NoOpPlaybackBackend()

    return RuntimeAdapters(
        activation=activation,
        hot_path_llm=hot_path_llm,
        deliberative_llm=deliberative_llm,
        stt=stt,
        tts=tts,
        vad=vad,
        playback=playback,
    )
