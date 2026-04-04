from __future__ import annotations

import json
from dataclasses import replace
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from loguru import logger
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from .config import AppConfig
from .observers import TerminalDebugObserver
from .prompt import SYSTEM_PROMPT


def build_transcribe_task(config: AppConfig) -> PipelineTask:
    transport = _build_local_audio_transport(config, audio_in=True, audio_out=False)
    vad_analyzer = _build_vad_analyzer()
    stt = _build_whisper_stt(config)

    pipeline = Pipeline([transport.input(), VADProcessor(vad_analyzer=vad_analyzer), stt])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=config.audio_in_sample_rate,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        enable_rtvi=False,
        enable_turn_tracking=False,
        idle_timeout_secs=None,
        observers=[MetricsLogObserver(), TerminalDebugObserver()],
    )

    _attach_task_logging(task, mode="transcribe")
    return task


def build_chat_task(config: AppConfig) -> PipelineTask:
    resolved_model = resolve_ollama_model(config)
    chat_config = replace(config, ollama_model=resolved_model)

    transport = _build_local_audio_transport(chat_config, audio_in=True, audio_out=True)
    vad_analyzer = _build_vad_analyzer()
    stt = _build_whisper_stt(chat_config)
    llm = _build_ollama_llm(chat_config)
    tts = _build_kokoro_tts(chat_config)

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad_analyzer,
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy(enable_interruptions=True)],
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(),
                    )
                ],
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=chat_config.audio_in_sample_rate,
            audio_out_sample_rate=chat_config.audio_out_sample_rate,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        enable_rtvi=False,
        idle_timeout_secs=None,
        observers=[MetricsLogObserver(), TerminalDebugObserver()],
    )

    _attach_task_logging(task, mode="chat")
    logger.info(
        "[chat] modelo ativo: {} | fallback: {} | voz: {}",
        chat_config.ollama_model,
        chat_config.ollama_fallback_model,
        chat_config.kokoro_voice,
    )
    return task


def resolve_ollama_model(config: AppConfig) -> str:
    tags = _fetch_ollama_tags(config.ollama_base_url)
    preferred = _match_model_name(config.ollama_model, tags)
    if preferred:
        return preferred

    fallback = _match_model_name(config.ollama_fallback_model, tags)
    if fallback:
        logger.warning(
            "[ollama] modelo {} nao encontrado; usando fallback {}",
            config.ollama_model,
            fallback,
        )
        return fallback

    available = ", ".join(sorted(tags)) if tags else "nenhum"
    raise RuntimeError(
        "Nenhum modelo Ollama compativel foi encontrado. "
        f"Esperado: {config.ollama_model} ou {config.ollama_fallback_model}. "
        f"Disponiveis: {available}. Rode `ollama pull {config.ollama_model}`."
    )


def _build_local_audio_transport(
    config: AppConfig,
    *,
    audio_in: bool,
    audio_out: bool,
) -> LocalAudioTransport:
    return LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=audio_in,
            audio_out_enabled=audio_out,
            input_device_index=config.input_device_index,
            output_device_index=config.output_device_index,
            audio_in_channels=1,
            audio_out_channels=1,
        )
    )


def _build_vad_analyzer() -> SileroVADAnalyzer:
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=0.7,
            start_secs=0.2,
            stop_secs=0.2,
        )
    )


def _build_whisper_stt(config: AppConfig) -> WhisperSTTServiceMLX:
    return WhisperSTTServiceMLX(
        settings=WhisperSTTServiceMLX.Settings(
            model=config.whisper_model,
            language=config.language,
            temperature=config.whisper_temperature,
            no_speech_prob=config.whisper_no_speech_prob,
        )
    )


def _build_ollama_llm(config: AppConfig) -> OLLamaLLMService:
    return OLLamaLLMService(
        base_url=config.ollama_base_url,
        settings=OLLamaLLMService.Settings(
            model=config.ollama_model,
            temperature=config.ollama_temperature,
        ),
    )


def _build_kokoro_tts(config: AppConfig) -> KokoroTTSService:
    return KokoroTTSService(
        model_path=str(config.kokoro_model_path) if config.kokoro_model_path else None,
        voices_path=str(config.kokoro_voices_path) if config.kokoro_voices_path else None,
        settings=KokoroTTSService.Settings(
            voice=config.kokoro_voice,
            language=Language.PT,
        ),
    )


def _attach_task_logging(task: PipelineTask, *, mode: str) -> None:
    @task.event_handler("on_pipeline_started")
    async def on_pipeline_started(task: PipelineTask, frame):
        logger.info("[{}] pipeline iniciado", mode)

    @task.event_handler("on_pipeline_finished")
    async def on_pipeline_finished(task: PipelineTask, frame):
        logger.info("[{}] pipeline finalizado com {}", mode, frame)

    @task.event_handler("on_pipeline_error")
    async def on_pipeline_error(task: PipelineTask, frame):
        logger.error("[{}] erro de pipeline: {}", mode, frame.error)


def _fetch_ollama_tags(base_url: str) -> set[str]:
    tags_url = f"{base_url.removesuffix('/v1')}/api/tags"
    try:
        with urlopen(tags_url, timeout=3) as response:
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(
            "Nao foi possivel acessar o Ollama em "
            f"{tags_url}. Verifique se `ollama serve` esta ativo."
        ) from exc

    return {item.get("name", "") for item in payload.get("models", []) if item.get("name")}


def _match_model_name(name: str, available: set[str]) -> str | None:
    if name in available:
        return name

    candidates = [candidate for candidate in available if candidate.startswith(f"{name}:")]
    if len(candidates) == 1:
        return candidates[0]

    return None
