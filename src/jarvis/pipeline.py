from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import mlx_whisper
import numpy as np
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from loguru import logger
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
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

from .audio_gate import LocalAudioEchoGate
from .config import AppConfig
from .llm_postprocess import JsonReplyExtractor
from .observers import TerminalDebugObserver
from .prompt import QWEN3_FALLBACK_JSON_SYSTEM_PROMPT, SYSTEM_PROMPT


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
    latency_observer = UserBotLatencyObserver()

    transport = _build_local_audio_transport(config, audio_in=True, audio_out=True)
    vad_analyzer = _build_vad_analyzer()
    stt = _build_whisper_stt(config)
    llm = _build_ollama_llm(config)
    tts = _build_kokoro_tts(config)

    gate = None
    interruptions_enabled = True
    if config.echo_suppression_enabled:
        gate = LocalAudioEchoGate(release_ms=config.echo_suppression_release_ms)
        interruptions_enabled = False

    context = LLMContext(messages=[])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad_analyzer,
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy(enable_interruptions=interruptions_enabled)],
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(),
                    )
                ],
            ),
        ),
    )

    processors = [transport.input()]
    if gate is not None:
        processors.append(gate)
    processors.extend(
        [
            stt,
            user_aggregator,
            llm,
            JsonReplyExtractor() if _uses_json_reply_mode(config.ollama_model) else None,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    processors = [processor for processor in processors if processor is not None]

    pipeline = Pipeline(processors)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=config.audio_in_sample_rate,
            audio_out_sample_rate=config.audio_out_sample_rate,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        enable_rtvi=False,
        idle_timeout_secs=None,
        observers=[MetricsLogObserver(), TerminalDebugObserver(), latency_observer],
    )

    _attach_task_logging(task, mode="chat")
    _attach_turn_logging(task, latency_observer)
    logger.info(
        "[chat] modelo ativo: {} | fallback: {} | voz: {}",
        config.ollama_model,
        config.ollama_fallback_model,
        config.kokoro_voice,
    )
    if _needs_reasoning_none(config.ollama_model):
        logger.info("[ollama] forcing reasoning.effort=none para reduzir latencia")
    if _uses_json_reply_mode(config.ollama_model):
        logger.info("[ollama] fallback qwen3 em modo JSON restrito para evitar resposta meta")
    if config.echo_suppression_enabled:
        logger.warning(
            "[audio] supressao de eco local ativa; interrupcao por voz "
            "durante a fala do assistente fica desativada sem fones"
        )
    return task


def prepare_chat_config(config: AppConfig) -> AppConfig:
    prepared = prepare_stt_config(config)
    resolved_model = resolve_ollama_model(prepared)
    prepared = replace(prepared, ollama_model=resolved_model)

    if prepared.prewarm_enabled:
        prewarm_whisper_model(prepared)

    if prepared.ollama_prewarm_enabled:
        prewarm_ollama_model(prepared)

    if prepared.ollama_model.startswith("qwen3.5:"):
        logger.info("[ollama] preparando qwen3.5 em modo de baixa latencia")
    elif prepared.ollama_model.startswith("qwen3:"):
        logger.info("[ollama] preparando fallback qwen3 em modo de compatibilidade")

    return prepared


def prepare_stt_config(config: AppConfig) -> AppConfig:
    resolved_whisper_model = _resolve_whisper_model_path(config.whisper_model)
    if resolved_whisper_model == config.whisper_model:
        return config

    logger.info("[stt] usando snapshot local do whisper")
    return replace(config, whisper_model=resolved_whisper_model)


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
    extra: dict[str, Any] = {}
    if _needs_reasoning_none(config.ollama_model):
        extra["extra_body"] = {"reasoning": {"effort": "none"}}
    if _uses_json_reply_mode(config.ollama_model):
        extra["response_format"] = {"type": "json_object"}

    system_instruction = (
        QWEN3_FALLBACK_JSON_SYSTEM_PROMPT
        if _uses_json_reply_mode(config.ollama_model)
        else SYSTEM_PROMPT
    )

    return OLLamaLLMService(
        base_url=config.ollama_base_url,
        settings=OLLamaLLMService.Settings(
            model=config.ollama_model,
            temperature=config.ollama_temperature,
            max_tokens=config.ollama_max_tokens,
            system_instruction=system_instruction,
            extra=extra,
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


def _attach_turn_logging(task: PipelineTask, latency_observer: UserBotLatencyObserver) -> None:
    turn_observer = task.turn_tracking_observer

    if turn_observer is not None:

        @turn_observer.event_handler("on_turn_started")
        async def on_turn_started(observer, turn_number: int):
            logger.info("[turn] turno {} iniciado", turn_number)

        @turn_observer.event_handler("on_turn_ended")
        async def on_turn_ended(
            observer,
            turn_number: int,
            duration: float,
            was_interrupted: bool,
        ):
            status = "interrompido" if was_interrupted else "concluido"
            logger.info(
                "[turn] turno {} {} | duracao={:.1f}ms",
                turn_number,
                status,
                _to_ms(duration),
            )

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency_measured(observer, latency_seconds: float):
        logger.info("[latency] usuario->assistente {:.1f}ms", _to_ms(latency_seconds))


def prewarm_whisper_model(config: AppConfig) -> None:
    started_at = time.perf_counter()
    silence = np.zeros(config.audio_in_sample_rate, dtype=np.float32)
    mlx_whisper.transcribe(
        silence,
        path_or_hf_repo=config.whisper_model,
        temperature=config.whisper_temperature,
        language=config.language,
    )
    logger.info("[warmup] whisper pronto | {:.1f}ms", _to_ms(time.perf_counter() - started_at))


def _resolve_whisper_model_path(model_ref: str) -> str:
    model_path = Path(model_ref).expanduser()
    if model_path.exists():
        return str(model_path)

    try:
        return snapshot_download(model_ref, local_files_only=True)
    except LocalEntryNotFoundError:
        logger.info("[stt] baixando assets do whisper pela primeira vez")
        return snapshot_download(model_ref)


def prewarm_ollama_model(config: AppConfig) -> None:
    payload = {
        "model": config.ollama_model,
        "messages": [{"role": "user", "content": "Responda apenas com ok."}],
        "stream": False,
        "keep_alive": config.ollama_keep_alive,
    }

    if _needs_reasoning_none(config.ollama_model):
        payload["options"] = {}
        payload["reasoning"] = {"effort": "none"}

    request = Request(
        f"{config.ollama_base_url.removesuffix('/v1')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    started_at = time.perf_counter()
    try:
        with urlopen(request, timeout=180) as response:
            json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError("Falha ao aquecer o modelo Ollama local.") from exc

    logger.info("[warmup] ollama pronto | {:.1f}ms", _to_ms(time.perf_counter() - started_at))


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


def _is_qwen35_model(model_name: str) -> bool:
    return model_name.startswith("qwen3.5:")


def _is_qwen3_model(model_name: str) -> bool:
    return model_name.startswith("qwen3:")


def _needs_reasoning_none(model_name: str) -> bool:
    return _is_qwen35_model(model_name) or _is_qwen3_model(model_name)


def _uses_json_reply_mode(model_name: str) -> bool:
    return _is_qwen3_model(model_name)


def _to_ms(seconds: float) -> float:
    return seconds * 1000.0
