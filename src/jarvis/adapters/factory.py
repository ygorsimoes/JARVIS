from __future__ import annotations

from dataclasses import dataclass

from ..audio import NoOpPlaybackBackend, SoundDevicePlaybackBackend
from ..config import JarvisConfig
from .activation import PushToTalkActivationAdapter
from .activation.porcupine import PorcupineActivationAdapter
from .llm import FakeLLMAdapter, FoundationModelsBridgeAdapter, MLXLMAdapter
from .llm.anthropic import AnthropicAdapter
from .llm.openai import OpenAIAdapter
from .stt import SpeechAnalyzerSTTAdapter
from .tts import (
    AVSpeechAdapter,
    FallbackTTSAdapter,
    MLXAudioKokoroAdapter,
    NoOpTTSAdapter,
)
from .tts.mlx_audio_qwen3 import MLXAudioQwen3Adapter
from .vad import SpeechDetectorAdapter


@dataclass
class RuntimeAdapters:
    activation: object
    activation_backend_name: str
    hot_path_llm: object
    hot_path_backend_name: str
    deliberative_llm: object
    deliberative_backend_name: str
    fallback_llm: object
    fallback_backend_name: str
    stt: object
    stt_backend_name: str
    tts: object
    tts_backend_name: str
    vad: object
    vad_backend_name: str
    playback: object
    playback_backend_name: str


def build_runtime_adapters(
    config: JarvisConfig, enable_native_backends: bool = True
) -> RuntimeAdapters:
    if not enable_native_backends:
        return _build_test_adapters(config)

    if config.activation_backend == "porcupine":
        activation = PorcupineActivationAdapter(keyword=config.activation_keyword)
    else:
        activation = PushToTalkActivationAdapter(
            backend=config.activation_backend,
            hotkey=config.activation_hotkey,
            terminal_fallback=config.activation_terminal_fallback,
        )
    activation_backend_name = config.activation_backend

    if config.stt_backend != "speech_analyzer":
        raise RuntimeError(
            "unsupported stt backend %s for macOS runtime" % config.stt_backend
        )
    stt = SpeechAnalyzerSTTAdapter(config.stt_bridge_bin, locale=config.stt_locale)
    stt_backend_name = config.stt_backend

    vad = SpeechDetectorAdapter()
    vad_backend_name = "speech_detector"

    if config.llm_hot_path == "foundation_models":
        hot_path_llm = FoundationModelsBridgeAdapter(
            base_url=config.llm_hot_path_url,
            instructions=config.system_prompt,
            bridge_binary_path=config.llm_hot_path_bridge_bin,
        )
        hot_path_backend_name = "foundation_models"
    else:
        raise RuntimeError(
            "unsupported hot path backend %s for macOS runtime" % config.llm_hot_path
        )

    if config.llm_deliberative == "mlx_lm":
        deliberative_llm = MLXLMAdapter(
            model_repo=config.llm_deliberative_model,
            max_tokens=config.llm_response_max_tokens,
            temperature=config.llm_deliberative_temperature,
            top_p=config.llm_deliberative_top_p,
            repetition_penalty=config.llm_deliberative_repetition_penalty,
        )
        deliberative_backend_name = "mlx_lm"
    else:
        raise RuntimeError(
            "unsupported deliberative backend %s for macOS runtime"
            % config.llm_deliberative
        )

    if config.llm_hot_path_fallback == "mlx_lm":
        fallback_llm = MLXLMAdapter(
            model_repo=config.llm_hot_path_fallback_model,
            max_tokens=config.llm_response_max_tokens,
            temperature=config.llm_deliberative_temperature,
            top_p=config.llm_deliberative_top_p,
            repetition_penalty=config.llm_deliberative_repetition_penalty,
        )
        fallback_backend_name = "mlx_lm_fallback"
    elif config.llm_hot_path_fallback == "anthropic":
        fallback_llm = AnthropicAdapter(
            model_name=config.llm_hot_path_fallback_model,
            temperature=config.llm_deliberative_temperature,
            max_tokens=config.llm_response_max_tokens,
        )
        fallback_backend_name = "anthropic"
    elif config.llm_hot_path_fallback == "openai":
        fallback_llm = OpenAIAdapter(
            model_name=config.llm_hot_path_fallback_model,
            temperature=config.llm_deliberative_temperature,
            max_tokens=config.llm_response_max_tokens,
        )
        fallback_backend_name = "openai"
    else:
        raise RuntimeError(
            "unsupported hot path fallback backend %s for macOS runtime"
            % config.llm_hot_path_fallback
        )

    if config.tts_backend == "mlx_audio_kokoro":
        primary_tts = MLXAudioKokoroAdapter(
            model_repo=config.tts_model,
            voice=config.tts_voice,
            lang_code=config.tts_lang_code,
        )
        fallback_tts = AVSpeechAdapter(
            voice=config.tts_avspeech_voice,
            sample_rate_hz=config.tts_sample_rate_hz,
            rate=config.tts_avspeech_rate,
        )
        tts = FallbackTTSAdapter(
            primary_tts,
            fallback_tts,
            primary_name="mlx_audio_kokoro",
            fallback_name="avspeech",
        )
        tts_backend_name = "mlx_audio_kokoro+avspeech"
    elif config.tts_backend == "mlx_audio_qwen3":
        tts = MLXAudioQwen3Adapter(
            model_repo=config.tts_model,
            speaker_id=config.tts_voice,
        )
        tts_backend_name = "mlx_audio_qwen3"
    elif config.tts_backend == "avspeech":
        tts = AVSpeechAdapter(
            voice=config.tts_avspeech_voice,
            sample_rate_hz=config.tts_sample_rate_hz,
            rate=config.tts_avspeech_rate,
        )
        tts_backend_name = "avspeech"
    else:
        raise RuntimeError(
            "unsupported tts backend %s for macOS runtime" % config.tts_backend
        )

    if config.playback_backend == "sounddevice":
        playback = SoundDevicePlaybackBackend()
        playback_backend_name = "sounddevice"
    else:
        raise RuntimeError(
            "unsupported playback backend %s for macOS runtime"
            % config.playback_backend
        )

    return RuntimeAdapters(
        activation=activation,
        activation_backend_name=activation_backend_name,
        hot_path_llm=hot_path_llm,
        hot_path_backend_name=hot_path_backend_name,
        deliberative_llm=deliberative_llm,
        deliberative_backend_name=deliberative_backend_name,
        fallback_llm=fallback_llm,
        fallback_backend_name=fallback_backend_name,
        stt=stt,
        stt_backend_name=stt_backend_name,
        tts=tts,
        tts_backend_name=tts_backend_name,
        vad=vad,
        vad_backend_name=vad_backend_name,
        playback=playback,
        playback_backend_name=playback_backend_name,
    )


def _build_test_adapters(config: JarvisConfig) -> RuntimeAdapters:
    if config.activation_backend == "porcupine":
        activation = PorcupineActivationAdapter(keyword=config.activation_keyword)
    else:
        activation = PushToTalkActivationAdapter(
            backend=config.activation_backend,
            hotkey=config.activation_hotkey,
            terminal_fallback=config.activation_terminal_fallback,
        )

    return RuntimeAdapters(
        activation=activation,
        activation_backend_name=config.activation_backend,
        hot_path_llm=FakeLLMAdapter(mode="hot_path"),
        hot_path_backend_name="fake",
        deliberative_llm=FakeLLMAdapter(mode="deliberative"),
        deliberative_backend_name="fake",
        fallback_llm=FakeLLMAdapter(mode="fallback"),
        fallback_backend_name="fake",
        stt=SpeechAnalyzerSTTAdapter(config.stt_bridge_bin, locale=config.stt_locale),
        stt_backend_name=config.stt_backend,
        tts=NoOpTTSAdapter(),
        tts_backend_name="noop",
        vad=SpeechDetectorAdapter(),
        vad_backend_name="speech_detector",
        playback=NoOpPlaybackBackend(),
        playback_backend_name="noop",
    )
