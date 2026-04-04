from __future__ import annotations

from dataclasses import replace

from loguru import logger
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionConfig
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.context.llm_context_summarization import (
    LLMAutoContextSummarizationConfig,
    LLMContextSummaryConfig,
)
from pipecat.utils.sync.event_notifier import EventNotifier

from .audio_gate import LocalAudioEchoGate
from .config import AppConfig
from .observers import TerminalDebugObserver
from .ollama import (
    build_ollama_extra,
    is_qwen35_model,
    resolve_ollama_model,
    select_smaller_qwen_model,
    uses_reasoning_effort_none,
)
from .prompt import (
    SYSTEM_PROMPT,
    TURN_COMPLETION_INSTRUCTIONS,
    TURN_COMPLETION_LONG_PROMPT,
    TURN_COMPLETION_SHORT_PROMPT,
)
from .turn_gate import (
    SafeGatedLLMContextAggregator,
    TrailingUserMessagesNormalizer,
    TurnGateController,
)
from .turn_stop import ConservativeTurnAnalyzerUserTurnStopStrategy
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
    context_notifier = EventNotifier()
    gated_context = SafeGatedLLMContextAggregator(notifier=context_notifier)
    turn_gate = TurnGateController(
        notifier=context_notifier,
        delay_secs=config.context_settle_secs,
    )

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
            user_turn_stop_timeout=_build_user_turn_stop_timeout(config),
            filter_incomplete_user_turns=True,
            user_turn_completion_config=_build_user_turn_completion_config(config),
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy(enable_interruptions=interruptions_enabled)],
                stop=[
                    ConservativeTurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(
                            params=SmartTurnParams(stop_secs=config.smart_turn_stop_secs)
                        ),
                        resume_delay_secs=config.turn_resume_delay_secs,
                    )
                ],
            ),
        ),
        assistant_params=_build_assistant_params(config),
    )

    processors = [transport.input()]
    if gate is not None:
        processors.append(gate)
    processors.extend(
        [
            stt,
            user_aggregator,
            TrailingUserMessagesNormalizer(),
            gated_context,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

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
    _attach_transcript_logging(user_aggregator, assistant_aggregator, turn_gate)
    logger.info(
        "[chat] modelo ativo: {} | fallback: {} | voz: {}",
        config.ollama_model,
        config.ollama_fallback_model,
        config.kokoro_voice,
    )
    if uses_reasoning_effort_none(config.ollama_model):
        logger.info("[ollama] forcing reasoning.effort=none para reduzir latencia")
    if config.context_summarization_enabled:
        logger.info("[context] sumarizacao automatica ativada")
    logger.info(
        "[turn] preset={} | retomada={:.1f}s | debounce={:.1f}s | "
        "incomplete_curta={:.1f}s | incomplete_longa={:.1f}s | "
        "smart_turn_stop={:.1f}s | stop_timeout={:.1f}s",
        config.turn_preset,
        config.turn_resume_delay_secs,
        config.context_settle_secs,
        config.context_trailing_secs,
        config.context_incomplete_secs,
        config.smart_turn_stop_secs,
        _build_user_turn_stop_timeout(config),
    )
    logger.info("[turn] filtro nativo de turno incompleto ativado")
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


def _build_user_turn_completion_config(config: AppConfig) -> UserTurnCompletionConfig:
    return UserTurnCompletionConfig(
        instructions=TURN_COMPLETION_INSTRUCTIONS,
        incomplete_short_timeout=config.context_trailing_secs,
        incomplete_long_timeout=config.context_incomplete_secs,
        incomplete_short_prompt=TURN_COMPLETION_SHORT_PROMPT,
        incomplete_long_prompt=TURN_COMPLETION_LONG_PROMPT,
    )


def _build_user_turn_stop_timeout(config: AppConfig) -> float:
    return max(
        config.user_speech_timeout,
        config.smart_turn_stop_secs + 2.0,
        config.context_incomplete_secs + 2.0,
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
            extra=build_ollama_extra(config.ollama_model, keep_alive=config.ollama_keep_alive),
        ),
    )


def _build_assistant_params(config: AppConfig) -> LLMAssistantAggregatorParams:
    params = LLMAssistantAggregatorParams()
    if not config.context_summarization_enabled:
        return params

    summary_llm = _build_summary_llm(config)
    params.enable_auto_context_summarization = True
    params.auto_context_summarization_config = LLMAutoContextSummarizationConfig(
        max_context_tokens=config.context_summary_max_context_tokens,
        max_unsummarized_messages=config.context_summary_max_unsummarized_messages,
        summary_config=LLMContextSummaryConfig(
            target_context_tokens=config.context_summary_target_context_tokens,
            min_messages_after_summary=config.context_summary_min_messages_after_summary,
            summary_message_template="Resumo da conversa: {summary}",
            llm=summary_llm,
        ),
    )
    return params


def _build_summary_llm(config: AppConfig) -> OLLamaLLMService:
    summary_model = select_smaller_qwen_model(config.ollama_model, config.ollama_fallback_model)
    return OLLamaLLMService(
        base_url=config.ollama_base_url,
        settings=OLLamaLLMService.Settings(
            model=summary_model,
            temperature=0.0,
            max_tokens=config.context_summary_target_context_tokens,
            extra=build_ollama_extra(summary_model, keep_alive=config.ollama_keep_alive),
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


def _attach_transcript_logging(
    user_aggregator,
    assistant_aggregator,
    turn_gate: TurnGateController,
) -> None:
    @user_aggregator.event_handler("on_user_turn_started")
    async def on_user_turn_started(aggregator, strategy) -> None:
        should_interrupt = turn_gate.should_interrupt_on_resume()
        await turn_gate.cancel_pending()
        if should_interrupt:
            turn_gate.reset()
            logger.info("[turn] usuario retomou a fala; cancelando resposta em andamento")
            await aggregator.broadcast_interruption()

    @assistant_aggregator.event_handler("on_assistant_turn_started")
    async def on_assistant_turn_started(aggregator) -> None:
        turn_gate.mark_assistant_started()

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(
        aggregator,
        strategy,
        message: UserTurnStoppedMessage,
    ) -> None:
        if message.content:
            logger.info("[user] turno final> {}", message.content)
            await turn_gate.schedule_release(owner=aggregator)

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(
        aggregator,
        message: AssistantTurnStoppedMessage,
    ) -> None:
        if message.content:
            logger.info("[llm] assistente> {}", message.content)

    @assistant_aggregator.event_handler("on_summary_applied")
    async def on_summary_applied(aggregator, summarizer, event) -> None:
        logger.info(
            "[context] resumo aplicado | mensagens={} -> {} | comprimidas={}",
            event.original_message_count,
            event.new_message_count,
            event.summarized_message_count,
        )


def _to_ms(seconds: float) -> float:
    return seconds * 1000.0
