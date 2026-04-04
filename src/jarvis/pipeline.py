from __future__ import annotations

from dataclasses import replace

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
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from .audio_gate import LocalAudioEchoGate
from .config import AppConfig
from .observers import TerminalDebugObserver
from .ollama import (
    build_ollama_extra,
    is_qwen35_model,
    resolve_ollama_model,
    uses_reasoning_effort_none,
)
from .prompt import (
    INCOMPLETE_LONG_PROMPT_PT_BR,
    INCOMPLETE_SHORT_PROMPT_PT_BR,
    SYSTEM_PROMPT,
    TURN_COMPLETION_INSTRUCTIONS_PT_BR,
)
from .warmup import prepare_stt_config, prewarm_ollama_model, prewarm_whisper_model


def build_transcribe_task(config: AppConfig) -> PipelineTask:
    transport = _build_local_audio_transport(config, audio_in=True, audio_out=False)
    vad_analyzer = _build_vad_analyzer(config)
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
        observers=[MetricsLogObserver(), TerminalDebugObserver(log_transcription_segments=True)],
    )

    _attach_task_logging(task, mode="transcribe")
    return task


def build_chat_task(config: AppConfig) -> PipelineTask:
    latency_observer = UserBotLatencyObserver()

    transport = _build_local_audio_transport(config, audio_in=True, audio_out=True)
    vad_analyzer = _build_vad_analyzer(config)
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
            filter_incomplete_user_turns=config.filter_incomplete_user_turns,
            user_turn_completion_config=_build_turn_completion_config(config),
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy(enable_interruptions=interruptions_enabled)],
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())],
            ),
        ),
    )

    processors = [transport.input()]
    if gate is not None:
        processors.append(gate)
    processors.extend([stt, user_aggregator, llm, tts, transport.output(), assistant_aggregator])

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
        observers=[
            MetricsLogObserver(),
            TerminalDebugObserver(log_transcription_segments=False),
            latency_observer,
        ],
    )

    _attach_task_logging(task, mode="chat")
    _attach_turn_logging(task, latency_observer)
    _attach_transcript_logging(user_aggregator, assistant_aggregator)
    logger.info(
        "[chat] modelo ativo: {} | fallback: {} | voz: {}",
        config.ollama_model,
        config.ollama_fallback_model,
        config.kokoro_voice,
    )
    if uses_reasoning_effort_none(config.ollama_model):
        logger.info("[ollama] forcing reasoning.effort=none para reduzir latencia")
    if config.filter_incomplete_user_turns:
        logger.info("[turn] filtro de fala incompleta ativado")
    if config.echo_suppression_enabled:
        logger.warning(
            "[audio] supressao de eco local ativa; interrupcao por voz "
            "durante a fala do assistente fica desativada sem fones"
        )
    return task


def prepare_chat_config(config: AppConfig) -> AppConfig:
    prepared = prepare_stt_config(config)
    resolved_model = resolve_ollama_model(
        prepared.ollama_base_url,
        prepared.ollama_model,
        prepared.ollama_fallback_model,
    )
    prepared = replace(prepared, ollama_model=resolved_model)

    if prepared.prewarm_enabled:
        prewarm_whisper_model(prepared)

    if prepared.ollama_prewarm_enabled:
        prewarm_ollama_model(prepared)

    if is_qwen35_model(prepared.ollama_model):
        logger.info("[ollama] preparando qwen3.5 em modo de baixa latencia")

    return prepared


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


def _build_vad_analyzer(config: AppConfig) -> SileroVADAnalyzer:
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=0.7,
            start_secs=config.vad_start_secs,
            stop_secs=config.vad_stop_secs,
        )
    )


def _build_whisper_stt(config: AppConfig) -> WhisperSTTServiceMLX:
    return WhisperSTTServiceMLX(
        ttfs_p99_latency=config.whisper_ttfs_p99_latency,
        settings=WhisperSTTServiceMLX.Settings(
            model=config.whisper_model,
            language=config.language,
            temperature=config.whisper_temperature,
            no_speech_prob=config.whisper_no_speech_prob,
        ),
    )


def _build_ollama_llm(config: AppConfig) -> OLLamaLLMService:
    return OLLamaLLMService(
        base_url=config.ollama_base_url,
        settings=OLLamaLLMService.Settings(
            model=config.ollama_model,
            temperature=config.ollama_temperature,
            max_tokens=config.ollama_max_tokens,
            system_instruction=SYSTEM_PROMPT,
            extra=build_ollama_extra(config.ollama_model),
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


def _attach_transcript_logging(user_aggregator, assistant_aggregator) -> None:
    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        if message.content:
            logger.info("[user] turno final> {}", message.content)

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        if message.content:
            logger.info("[llm] assistente> {}", message.content)


def _to_ms(seconds: float) -> float:
    return seconds * 1000.0


def _build_turn_completion_config(config: AppConfig) -> UserTurnCompletionConfig:
    return UserTurnCompletionConfig(
        instructions=TURN_COMPLETION_INSTRUCTIONS_PT_BR,
        incomplete_short_timeout=config.incomplete_short_timeout,
        incomplete_long_timeout=config.incomplete_long_timeout,
        incomplete_short_prompt=INCOMPLETE_SHORT_PROMPT_PT_BR,
        incomplete_long_prompt=INCOMPLETE_LONG_PROMPT_PT_BR,
    )
