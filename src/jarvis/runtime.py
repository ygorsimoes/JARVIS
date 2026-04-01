from __future__ import annotations

import asyncio
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Callable, Iterable, List, Optional

from .adapters import build_runtime_adapters
from .adapters.memory.sqlite_vec import SQLiteVecMemoryAdapter
from .adapters.stt.speech_analyzer import SpeechAnalyzerStreamError
from .bus import EventBus
from .config import JarvisConfig
from .core.action_broker import ActionBroker, ActionRequest
from .core.capability_broker import (
    Capability,
    CapabilityBroker,
    ConfirmationRequiredError,
    RiskLevel,
)
from .core.complexity_router import ComplexityRouter
from .core.dialogue_manager import DialogueManager
from .core.policy_engine import PolicyEngine
from .core.resource_governor import ResourceGovernor
from .core.sentence_streamer import SentenceStreamer, SentenceStreamerConfig
from .core.speech_pipeline import SpeechPipeline
from .core.state_machine import StateMachine
from .core.turn_manager import CompletedTurn, TurnManager, TurnManagerConfig
from .memory import MemorySystem
from .models.actions import (
    BrowserFetchURLAction,
    BrowserSearchAction,
    CancelTimerAction,
    GetTimeAction,
    ListCalendarEventsAction,
    ListTimersAction,
    OpenAppAction,
    SetVolumeAction,
    StartTimerAction,
)
from .models.conversation import Role, RouteDecision, RouteTarget
from .models.events import Event, EventType
from .models.memory import Memory
from .models.state import JarvisState
from .observability import VoiceTraceReporter, get_logger
from .tools import ToolRegistry, build_default_registry
from .tools.timer import parse_duration_seconds

logger = get_logger(__name__)


@dataclass
class JarvisResponse:
    input_text: str
    route: RouteDecision
    sentences: List[str]
    full_text: str


@dataclass
class JarvisTurnChunk:
    sentence: str
    route: RouteDecision
    index: int


@dataclass
class CapturedVoiceTurn:
    turn_id: str
    text: str
    reason: str
    used_partial: bool
    metadata: dict


class VoiceCaptureError(RuntimeError):
    pass


@dataclass
class ActiveTurnExecution:
    response_task: asyncio.Task
    mode: str
    turn_id: str
    adapter: object | None = None
    speech_pipeline: SpeechPipeline | None = None
    route: RouteDecision | None = None
    backend_name: str | None = None
    interrupted: bool = False


class JarvisRuntime:
    def __init__(
        self,
        config: JarvisConfig,
        event_bus: EventBus,
        state_machine: StateMachine,
        dialogue_manager: DialogueManager,
        sentence_streamer: SentenceStreamer,
        router: ComplexityRouter,
        policy_engine: PolicyEngine,
        action_broker: ActionBroker,
        tool_registry: ToolRegistry,
        turn_manager_config: TurnManagerConfig,
        activation_backend_name: str,
        hot_path_backend_name: str,
        deliberative_backend_name: str,
        fallback_backend_name: str,
        stt_backend_name: str,
        tts_backend_name: str,
        vad_backend_name: str,
        playback_backend_name: str,
        activation_adapter,
        hot_path_adapter,
        deliberative_adapter,
        fallback_adapter,
        stt_adapter,
        tts_adapter,
        vad_adapter,
        playback_backend,
        memory_adapter: Optional[SQLiteVecMemoryAdapter] = None,
        memory_system: Optional[MemorySystem] = None,
    ) -> None:
        self.config = config
        self.session_id = str(uuid.uuid4())
        self.event_bus = event_bus
        self.state_machine = state_machine
        self.dialogue_manager = dialogue_manager
        self.sentence_streamer = sentence_streamer
        self.router = router
        self.policy_engine = policy_engine
        self.action_broker = action_broker
        self.tool_registry = tool_registry
        self.turn_manager_config = turn_manager_config
        self.activation_backend_name = activation_backend_name
        self.hot_path_backend_name = hot_path_backend_name
        self.deliberative_backend_name = deliberative_backend_name
        self.fallback_backend_name = fallback_backend_name
        self.stt_backend_name = stt_backend_name
        self.tts_backend_name = tts_backend_name
        self.vad_backend_name = vad_backend_name
        self.playback_backend_name = playback_backend_name
        self.activation_adapter = activation_adapter
        self.hot_path_adapter = hot_path_adapter
        self.deliberative_adapter = deliberative_adapter
        self.fallback_adapter = fallback_adapter
        self.stt_adapter = stt_adapter
        self.tts_adapter = tts_adapter
        self.vad_adapter = vad_adapter
        self.playback_backend = playback_backend
        self.memory_adapter: Optional[SQLiteVecMemoryAdapter] = memory_adapter
        self.memory_system: Optional[MemorySystem] = memory_system
        self._active_turn: ActiveTurnExecution | None = None
        self._pending_voice_pipeline: SpeechPipeline | None = None
        self._pending_voice_relisten = False
        self._trace_reporter: VoiceTraceReporter | None = None

    @classmethod
    def from_config(
        cls,
        config: Optional[JarvisConfig] = None,
        enable_native_backends: bool = True,
    ) -> "JarvisRuntime":
        config = config or JarvisConfig()
        policy_engine = PolicyEngine(
            config, enable_native_backends=enable_native_backends
        )
        if policy_engine.requires_resource_governor():
            ResourceGovernor(config).apply()
        event_bus = EventBus(config.event_bus_queue_size)
        state_machine = StateMachine(event_bus=event_bus)
        dialogue_manager = DialogueManager(
            system_prompt=config.system_prompt,
            working_memory_turns=config.working_memory_turns,
        )
        sentence_streamer = SentenceStreamer(
            SentenceStreamerConfig(
                min_dispatch_tokens=config.sentence_min_tokens,
                min_soft_boundary_chars=config.sentence_min_soft_boundary_chars,
                max_pending_segments=config.sentence_max_pending_segments,
                backpressure_poll_interval_s=config.sentence_backpressure_poll_ms
                / 1000.0,
            )
        )
        tool_registry = build_default_registry(config)
        capability_broker = CapabilityBroker(cls._default_capabilities(config))
        action_broker = ActionBroker(capability_broker, tool_registry)
        adapters = build_runtime_adapters(
            config, enable_native_backends=enable_native_backends
        )
        turn_manager_config = TurnManagerConfig(
            silence_timeout_ms=config.turn_silence_timeout_ms,
            partial_commit_min_chars=config.turn_partial_commit_min_chars,
            partial_stability_ms=config.turn_partial_stability_ms,
            tick_interval_ms=config.turn_tick_interval_ms,
            max_turn_duration_s=config.turn_max_duration_s,
        )
        # Memory adapter — stored in ~/.jarvis/memory.db by default.
        memory_db_path = Path.home() / ".jarvis" / "memory.db"
        memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        embedding_backend = "auto" if enable_native_backends else "stub"
        memory_adapter = SQLiteVecMemoryAdapter(
            db_path=memory_db_path,
            embedding_backend=embedding_backend,
        )
        memory_system = MemorySystem(memory_adapter)

        runtime = cls(
            config=config,
            event_bus=event_bus,
            state_machine=state_machine,
            dialogue_manager=dialogue_manager,
            sentence_streamer=sentence_streamer,
            router=ComplexityRouter(),
            policy_engine=policy_engine,
            action_broker=action_broker,
            tool_registry=tool_registry,
            turn_manager_config=turn_manager_config,
            activation_backend_name=adapters.activation_backend_name,
            hot_path_backend_name=adapters.hot_path_backend_name,
            deliberative_backend_name=adapters.deliberative_backend_name,
            fallback_backend_name=adapters.fallback_backend_name,
            stt_backend_name=adapters.stt_backend_name,
            tts_backend_name=adapters.tts_backend_name,
            vad_backend_name=adapters.vad_backend_name,
            playback_backend_name=adapters.playback_backend_name,
            activation_adapter=adapters.activation,
            hot_path_adapter=adapters.hot_path_llm,
            deliberative_adapter=adapters.deliberative_llm,
            fallback_adapter=adapters.fallback_llm,
            stt_adapter=adapters.stt,
            tts_adapter=adapters.tts,
            vad_adapter=adapters.vad,
            playback_backend=adapters.playback,
            memory_adapter=memory_adapter,
            memory_system=memory_system,
        )
        logger.info(
            "Runtime backends configured", **runtime.trace_configuration_payload()
        )
        return runtime

    async def run_voice_foreground(self, turn_limit: Optional[int] = None) -> None:
        await self._ensure_trace_reporter_started()
        handled_turns = 0
        carry_activation = False
        try:
            while turn_limit is None or handled_turns < turn_limit:
                handoff_capture = carry_activation
                carry_activation = False
                if not handoff_capture:
                    activated = await self.activation_adapter.listen()
                    if not activated:
                        continue

                voice_turn_id = str(uuid.uuid4())
                if self._trace_reporter is not None:
                    self._trace_reporter.register_turn_start(voice_turn_id)

                partial_state = {"text": "", "printed": False}

                def on_partial_transcript(text: str) -> None:
                    normalized = text.strip()
                    if not normalized or normalized == partial_state["text"]:
                        return
                    partial_state["text"] = normalized
                    partial_state["printed"] = True
                    if self._trace_reporter is not None:
                        print(
                            self._trace_reporter.format_conversation_line(
                                "voce~>",
                                normalized,
                                turn_id=voice_turn_id,
                            )
                        )
                        return
                    if sys.stdout.isatty():
                        print(
                            "voce~> %s" % normalized,
                            end="\r",
                            flush=True,
                        )
                    else:
                        print("voce~> %s" % normalized)

                await self._publish_event(
                    EventType.ACTIVATION_TRIGGERED,
                    {
                        "source": (
                            "barge_in_handoff" if handoff_capture else "foreground"
                        )
                    },
                    turn_id=voice_turn_id,
                    mode="voice",
                    backend=self.activation_backend_name,
                )

                try:
                    voice_turn = await self.capture_voice_turn(
                        on_partial_transcript=on_partial_transcript,
                        turn_id=voice_turn_id,
                    )
                except VoiceCaptureError as exc:
                    if (
                        partial_state["printed"]
                        and sys.stdout.isatty()
                        and self._trace_reporter is None
                    ):
                        print("", flush=True)
                    self._print_conversation_line(
                        "jarvis>",
                        str(exc),
                        turn_id=voice_turn_id,
                    )
                    if handoff_capture:
                        continue
                    handled_turns += 1
                    continue

                if (
                    partial_state["printed"]
                    and sys.stdout.isatty()
                    and self._trace_reporter is None
                ):
                    print("", flush=True)
                self._print_conversation_line(
                    "voce>",
                    voice_turn.text,
                    turn_id=voice_turn.turn_id,
                )
                pipeline = SpeechPipeline(
                    tts_adapter=self.tts_adapter,
                    playback_backend=self.playback_backend,
                    sample_rate_hz=self.config.tts_sample_rate_hz,
                    event_bus=self.event_bus,
                    event_context=self._voice_pipeline_event_context(
                        voice_turn.turn_id
                    ),
                )
                self._pending_voice_pipeline = pipeline
                await pipeline.start()
                response_queue: asyncio.Queue = asyncio.Queue()
                response_task = asyncio.create_task(
                    self._pump_turn_chunks_to_queue(
                        self.stream_transcribed_turn(
                            voice_turn.text,
                            turn_id=voice_turn.turn_id,
                        ),
                        response_queue,
                    ),
                    name="jarvis-voice-response",
                )
                barge_in_task = asyncio.create_task(
                    self._barge_in_watcher(response_task),
                    name="jarvis-barge-in-watcher",
                )
                try:
                    while True:
                        item = await response_queue.get()
                        if item is None:
                            break
                        if isinstance(item, Exception):
                            raise item
                        self._print_conversation_line(
                            "jarvis>",
                            item.sentence,
                            turn_id=voice_turn.turn_id,
                        )
                        await pipeline.enqueue_sentence(item.index, item.sentence)
                    await pipeline.finish()
                finally:
                    self._pending_voice_pipeline = None
                    if not response_task.done():
                        response_task.cancel()
                    try:
                        await response_task
                    except asyncio.CancelledError:
                        pass
                    if not barge_in_task.done():
                        barge_in_task.cancel()
                    try:
                        await barge_in_task
                    except asyncio.CancelledError:
                        pass
                    await pipeline.shutdown()

                if self._consume_pending_voice_relisten():
                    carry_activation = True
                    continue
                handled_turns += 1
        except asyncio.CancelledError:
            pass
        finally:
            if self._trace_reporter is not None:
                await self._trace_reporter.shutdown(self.event_bus)
                self._trace_reporter = None

    async def _barge_in_watcher(
        self,
        response_task: asyncio.Task,
        pipeline: SpeechPipeline | None = None,
    ) -> None:
        """Monitora o VAD em paralelo durante SPEAKING/THINKING.
        Cancela response_task ao detectar fala."""
        del pipeline
        try:
            async for event in self._iter_barge_in_events():
                if response_task.done():
                    return
                classified = self.vad_adapter.classify_event(event)
                if classified and classified["speech_detected"]:
                    await self.interrupt_current_turn(reason="barge-in")
                    return
        except SpeechAnalyzerStreamError:
            # O watcher de barge-in eh auxiliar: se o bridge cair ou o binario
            # nao suportar o modo reduzido, preservamos o turno atual.
            return

    async def capture_voice_turn(
        self,
        on_partial_transcript: Callable[[str], None] | None = None,
        turn_id: Optional[str] = None,
    ) -> CapturedVoiceTurn:
        turn_id = turn_id or str(uuid.uuid4())
        turn_manager = TurnManager(self.turn_manager_config)
        saw_speech_signal = False
        saw_partial_transcript = False
        await self.state_machine.transition(JarvisState.ARMED, "activation accepted")
        await self.state_machine.transition(
            JarvisState.LISTENING, "voice capture started"
        )

        session = await self.stt_adapter.start_live_session()
        completed_turn: Optional[CompletedTurn] = None
        pending_event_task: Optional[asyncio.Task] = None
        try:
            event_iterator = session.iter_events().__aiter__()
            while completed_turn is None:
                try:
                    if pending_event_task is None:
                        pending_event_task = asyncio.create_task(
                            event_iterator.__anext__(),
                            name="jarvis-stt-next-event",
                        )
                    event = await asyncio.wait_for(
                        asyncio.shield(pending_event_task),
                        timeout=self.turn_manager_config.tick_interval_ms / 1000.0,
                    )
                    pending_event_task = None
                except SpeechAnalyzerStreamError as exc:
                    raise VoiceCaptureError(
                        "erro no reconhecimento de voz: %s" % exc
                    ) from exc
                except asyncio.TimeoutError:
                    completed_turn = turn_manager.tick()
                    continue
                except StopAsyncIteration:
                    completed_turn = turn_manager.finalize(reason="speech_stream_ended")
                    break

                await self._publish_stt_event(event, turn_id=turn_id)
                event_type = event.get("type")
                if event_type == "error":
                    message = str(
                        event.get("message") or "speech analyzer reported an error"
                    )
                    raise VoiceCaptureError(
                        "erro no reconhecimento de voz: %s" % message
                    )
                if event_type in {"speech_started", "final_transcript"}:
                    saw_speech_signal = True
                if event_type == "speech_detector_result" and event.get(
                    "speech_detected"
                ):
                    saw_speech_signal = True
                if (
                    event_type == "partial_transcript"
                    and str(event.get("text") or "").strip()
                ):
                    if on_partial_transcript is not None:
                        on_partial_transcript(str(event.get("text") or ""))
                    saw_partial_transcript = True
                    saw_speech_signal = True
                completed_turn = self._consume_turn_signal(turn_manager, event)

            if completed_turn is None or not completed_turn.text:
                if saw_partial_transcript:
                    raise VoiceCaptureError(
                        "nao consegui fechar uma transcricao final. Tente falar uma frase completa e aguarde um instante em silencio."
                    )
                if saw_speech_signal:
                    raise VoiceCaptureError(
                        "o microfone captou voz, mas nao houve uma transcricao utilizavel. Tente novamente falando um pouco mais devagar."
                    )
                raise VoiceCaptureError(
                    "nenhuma fala utilizavel foi capturada. Verifique o microfone e tente novamente."
                )

            await self._publish_event(
                EventType.TURN_READY,
                {
                    "text": completed_turn.text,
                    "reason": completed_turn.reason,
                    "used_partial": completed_turn.used_partial,
                },
                turn_id=turn_id,
                mode="voice",
                backend=self.stt_backend_name,
            )
            await self.state_machine.transition(
                JarvisState.TRANSCRIBING,
                completed_turn.reason,
                {
                    "used_partial": completed_turn.used_partial,
                    "turn_duration_ms": completed_turn.metadata.get("turn_duration_ms"),
                    "silence_duration_ms": completed_turn.metadata.get(
                        "silence_duration_ms"
                    ),
                },
            )
            return CapturedVoiceTurn(
                turn_id=turn_id,
                text=completed_turn.text,
                reason=completed_turn.reason,
                used_partial=completed_turn.used_partial,
                metadata=dict(completed_turn.metadata),
            )
        except Exception as exc:
            await self._handle_turn_failure(exc, input_text=None)
            raise
        finally:
            if pending_event_task is not None and not pending_event_task.done():
                pending_event_task.cancel()
                try:
                    await pending_event_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
            await session.stop()

    async def stream_transcribed_turn(
        self,
        text: str,
        recalled_memories: Optional[Iterable[Memory]] = None,
        turn_id: Optional[str] = None,
    ) -> AsyncIterator[JarvisTurnChunk]:
        execution = self._bind_active_turn(
            mode="voice", turn_id=turn_id or str(uuid.uuid4())
        )
        try:
            await self.state_machine.transition(
                JarvisState.THINKING, "transcript ready"
            )
            async for chunk in self._stream_response_chunks(
                text, recalled_memories=recalled_memories
            ):
                yield chunk
        except asyncio.CancelledError:
            if execution.interrupted:
                await self._handle_turn_interrupted()
                return
            raise
        except Exception as exc:
            await self._handle_turn_failure(exc, input_text=text)
            raise
        finally:
            self._clear_active_turn(execution)

    async def stream_text_turn(
        self,
        text: str,
        recalled_memories: Optional[Iterable[Memory]] = None,
    ) -> AsyncIterator[JarvisTurnChunk]:
        explicit_recalled_memories = (
            list(recalled_memories) if recalled_memories is not None else None
        )
        turn_id = str(uuid.uuid4())
        route: Optional[RouteDecision] = None
        response_sentences: List[str] = []
        execution = self._bind_active_turn(mode="text", turn_id=turn_id)

        try:
            await self.state_machine.transition(JarvisState.ARMED, "text turn received")
            await self.state_machine.transition(
                JarvisState.LISTENING, "virtual listening for text mode"
            )
            await self.state_machine.transition(
                JarvisState.TRANSCRIBING, "text mode bypass"
            )
            await self.state_machine.transition(JarvisState.THINKING, "routing request")
            async for chunk in self._stream_response_chunks(
                text, recalled_memories=explicit_recalled_memories
            ):
                route = chunk.route
                response_sentences.append(chunk.sentence)
                yield chunk
        except asyncio.CancelledError:
            if execution.interrupted:
                await self._handle_turn_interrupted()
                return
            raise
        except Exception as exc:
            await self._handle_turn_failure(exc, input_text=text, route=route)
            raise
        finally:
            self._clear_active_turn(execution)

    async def respond_text(
        self,
        text: str,
        recalled_memories: Optional[Iterable[Memory]] = None,
    ) -> JarvisResponse:
        route: Optional[RouteDecision] = None
        sentences: List[str] = []
        async for chunk in self.stream_text_turn(
            text, recalled_memories=recalled_memories
        ):
            route = chunk.route
            sentences.append(chunk.sentence)

        if route is None:
            raise RuntimeError("turn completed without a route")

        full_text = " ".join(sentence.strip() for sentence in sentences).strip()
        return JarvisResponse(
            input_text=text,
            route=route,
            sentences=sentences,
            full_text=full_text,
        )

    async def shutdown(self) -> None:
        if self._pending_voice_pipeline is not None:
            await self._pending_voice_pipeline.shutdown()
            self._pending_voice_pipeline = None
        if self._trace_reporter is not None:
            await self._trace_reporter.shutdown(self.event_bus)
            self._trace_reporter = None
        for adapter in (
            self.activation_adapter,
            self.hot_path_adapter,
            self.deliberative_adapter,
            self.stt_adapter,
            self.tts_adapter,
            self.playback_backend,
        ):
            shutdown = getattr(adapter, "shutdown", None)
            if shutdown is not None:
                await shutdown()
        if self.memory_adapter is not None:
            await self.memory_adapter.close()

    async def interrupt_current_turn(self, reason: str = "barge-in") -> bool:
        execution = self._active_turn
        if execution is None:
            return False
        if execution.interrupted:
            return True

        execution.interrupted = True
        if execution.mode == "voice" and reason == "barge-in":
            self._pending_voice_relisten = True
        await self._publish_event(
            EventType.INTERRUPTION_REQUESTED,
            {"reason": reason},
        )
        await self.state_machine.force_transition(JarvisState.INTERRUPTED, reason)

        cancel_current_response = getattr(
            execution.adapter, "cancel_current_response", None
        )
        if cancel_current_response is not None:
            try:
                await cancel_current_response()
            except Exception:
                pass

        if execution.speech_pipeline is not None:
            await execution.speech_pipeline.stop()

        if not execution.response_task.done():
            execution.response_task.cancel()
        return True

    async def _stream_llm_sentences(self, adapter, messages) -> AsyncIterator[str]:
        sentence_streamer = SentenceStreamer(self._sentence_streamer_config())
        queue: asyncio.Queue = asyncio.Queue()
        pump_task = asyncio.create_task(
            sentence_streamer.pump(
                adapter.chat_stream(
                    messages=messages,
                    tools=self.action_broker.describe_available_tools(),
                    max_kv_size=self.config.llm_max_kv_size,
                    tool_invoker=self._invoke_llm_tool,
                ),
                queue,
            )
        )
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            await pump_task

    def _sentence_streamer_config(self) -> SentenceStreamerConfig:
        config = self.sentence_streamer.config
        if self._active_turn is None or self._active_turn.mode != "voice":
            return config
        return SentenceStreamerConfig(
            min_dispatch_tokens=min(config.min_dispatch_tokens, 4),
            min_soft_boundary_chars=min(config.min_soft_boundary_chars, 24),
            max_pending_segments=config.max_pending_segments,
            backpressure_poll_interval_s=config.backpressure_poll_interval_s,
        )

    async def _stream_response_chunks(
        self,
        text: str,
        recalled_memories: Optional[Iterable[Memory]] = None,
    ) -> AsyncIterator[JarvisTurnChunk]:
        explicit_recalled_memories = (
            list(recalled_memories) if recalled_memories is not None else None
        )
        initial_route = self.router.route(
            text,
            recalled_memories=len(explicit_recalled_memories or []),
            recent_turns=len(self.dialogue_manager.working_memory),
        )
        pre_routed_memories: list[Memory] = []
        if (
            explicit_recalled_memories is None
            and self.memory_system is not None
            and initial_route.target != RouteTarget.DIRECT_TOOL
        ):
            pre_routed_memories = await self._recall_with_budget(
                text,
                RouteTarget.HOT_PATH,
                top_k=2,
            )
        route = initial_route
        if pre_routed_memories:
            route = self.router.route(
                text,
                recalled_memories=len(
                    explicit_recalled_memories or pre_routed_memories
                ),
                recent_turns=len(self.dialogue_manager.working_memory),
            )

        if (
            explicit_recalled_memories is None
            and self.memory_system is not None
            and route.target != RouteTarget.DIRECT_TOOL
        ):
            if route.target == RouteTarget.HOT_PATH and pre_routed_memories:
                explicit_recalled_memories = pre_routed_memories
            else:
                explicit_recalled_memories = await self._recall_with_budget(
                    text,
                    route.target,
                    fallback=pre_routed_memories,
                )
        recalled_memories = list(explicit_recalled_memories or [])

        llm_plan = None
        backend_name = route.tool_name
        messages = None
        if route.target != RouteTarget.DIRECT_TOOL:
            llm_plan = await self.policy_engine.select_llm(
                route=route,
                hot_path_adapter=self.hot_path_adapter,
                hot_path_backend_name=self.hot_path_backend_name,
                deliberative_adapter=self.deliberative_adapter,
                deliberative_backend_name=self.deliberative_backend_name,
                fallback_adapter=self.fallback_adapter,
                fallback_backend_name=self.fallback_backend_name,
            )
            backend_name = llm_plan.backend_name
            messages = self.dialogue_manager.compose_messages(
                text,
                recalled_memories,
                route_target=route.target,
            )

        self._set_active_turn_context(route=route, backend_name=backend_name)

        self.dialogue_manager.record_turn(
            Role.USER,
            text,
            {"route": route.target.value, "backend": backend_name},
        )
        await self._publish_event(
            EventType.USER_TURN,
            {"text": text},
            route=route,
            backend=backend_name,
        )
        await self._publish_event(
            EventType.ROUTE_SELECTED,
            {
                "input_text": text,
                "target": route.target.value,
                "tool_name": route.tool_name,
                "reason": route.reason,
                "effective_target": (
                    llm_plan.effective_target.value
                    if llm_plan is not None
                    else route.target.value
                ),
                "policy_reason": llm_plan.reason if llm_plan is not None else None,
                "fallback_used": (
                    llm_plan.fallback_used if llm_plan is not None else False
                ),
                "backend_detail": self._backend_detail_for_name(backend_name),
            },
            route=route,
            backend=backend_name,
        )

        response_sentences: List[str] = []
        if route.target == RouteTarget.DIRECT_TOOL:
            await self.state_machine.transition(
                JarvisState.ACTING, "executing direct tool"
            )
            sentence = await self._execute_direct_tool(text, route)
            response_sentences.append(sentence)
            await self.state_machine.transition(
                JarvisState.SPEAKING, "tool response ready"
            )
            await self._publish_event(
                EventType.ASSISTANT_SENTENCE,
                {"sentence": sentence, "index": 0},
                route=route,
                backend=backend_name,
            )
            yield JarvisTurnChunk(sentence=sentence, route=route, index=0)
        else:
            assert llm_plan is not None
            assert messages is not None
            adapter = llm_plan.adapter
            if self._active_turn is not None:
                self._active_turn.adapter = adapter
            speaking_started = False
            index = 0
            async for sentence in self._stream_llm_sentences(adapter, messages):
                if not speaking_started:
                    await self.state_machine.transition(
                        JarvisState.SPEAKING, "first sentence ready"
                    )
                    speaking_started = True
                response_sentences.append(sentence)
                await self._publish_event(
                    EventType.ASSISTANT_SENTENCE,
                    {"sentence": sentence, "index": index},
                    route=route,
                    backend=backend_name,
                )
                yield JarvisTurnChunk(sentence=sentence, route=route, index=index)
                index += 1

        full_text = " ".join(
            sentence.strip() for sentence in response_sentences
        ).strip()
        self.dialogue_manager.record_turn(
            Role.ASSISTANT,
            full_text,
            {"route": route.target.value, "backend": backend_name},
        )
        await self._publish_event(
            EventType.ASSISTANT_COMPLETED,
            {"text": full_text},
            route=route,
            backend=backend_name,
        )
        await self.state_machine.transition(JarvisState.IDLE, "turn completed")
        # --- Memory persistence (fire-and-forget, never blocks) -------
        if self.memory_system is not None and text and full_text:
            asyncio.create_task(
                self.memory_system.maybe_persist_turn(text, full_text),
                name="jarvis-memory-persist",
            )

    async def _publish_stt_event(
        self, event: dict, turn_id: Optional[str] = None
    ) -> None:
        event_type = event.get("type")
        if event_type == "ready":
            mapped = EventType.STT_READY
        elif event_type == "speech_started":
            mapped = EventType.SPEECH_STARTED
        elif event_type == "speech_ended":
            mapped = EventType.SPEECH_ENDED
        elif event_type == "partial_transcript":
            mapped = EventType.PARTIAL_TRANSCRIPT
        elif event_type == "final_transcript":
            mapped = EventType.FINAL_TRANSCRIPT
        elif event_type == "speech_detector_result":
            mapped = EventType.VAD_ACTIVITY
        else:
            return

        await self._publish_event(
            mapped,
            dict(event),
            turn_id=turn_id,
            mode="voice",
            backend=self.stt_backend_name,
        )

    def _consume_turn_signal(
        self, turn_manager: TurnManager, event: dict
    ) -> Optional[CompletedTurn]:
        classified_event = self.vad_adapter.classify_event(event)
        if classified_event is not None:
            return turn_manager.consume_vad_signal(
                bool(classified_event["speech_detected"]),
                reason=self.vad_adapter.signal_reason(event) or "vad_adapter",
            )
        event_type = event.get("type")
        if event_type == "speech_started":
            return turn_manager.consume_vad_signal(True, reason="speech_started")
        if event_type == "speech_ended":
            return turn_manager.consume_vad_signal(False, reason="speech_ended")
        if event_type == "partial_transcript":
            return turn_manager.consume_partial_transcript(event.get("text", ""))
        if event_type == "final_transcript":
            return turn_manager.consume_final_transcript(event.get("text", ""))
        return turn_manager.consume_event(event)

    async def _handle_turn_failure(
        self,
        exc: Exception,
        input_text: Optional[str],
        route: Optional[RouteDecision] = None,
    ) -> None:
        await self._publish_event(
            EventType.ERROR,
            {
                "message": str(exc),
                "input_text": input_text,
            },
            route=route,
        )
        await self.state_machine.force_transition(
            JarvisState.FAILED, "turn failed", {"message": str(exc)}
        )
        await self.state_machine.force_transition(
            JarvisState.IDLE, "recovered after failure"
        )

    async def _handle_turn_interrupted(self) -> None:
        await self.state_machine.force_transition(
            JarvisState.IDLE, "interruption handled"
        )

    async def _iter_barge_in_events(self) -> AsyncIterator[dict]:
        start_vad_session = getattr(self.stt_adapter, "start_vad_session", None)
        if start_vad_session is not None:
            try:
                async for event in self._iter_session_events(start_vad_session):
                    yield event
                return
            except SpeechAnalyzerStreamError as exc:
                if not self._is_unsupported_vad_only_error(exc):
                    raise

        async for event in self._iter_session_events(
            self.stt_adapter.start_live_session
        ):
            yield event

    async def _iter_session_events(self, start_session) -> AsyncIterator[dict]:
        session = await start_session()
        try:
            async for event in session.iter_events():
                yield event
        finally:
            await session.stop()

    @staticmethod
    def _is_unsupported_vad_only_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "invalid_arguments" in message or "usage:" in message

    async def _recall_with_budget(
        self,
        query: str,
        route_target: RouteTarget,
        *,
        top_k: int | None = None,
        fallback: Iterable[Memory] | None = None,
    ) -> list[Memory]:
        if self.memory_system is None or route_target == RouteTarget.DIRECT_TOOL:
            return list(fallback or [])

        timeout_s = self.config.memory_recall_timeout_ms / 1000.0
        try:
            if timeout_s <= 0:
                return await self.memory_system.recall(query, route_target, top_k=top_k)
            return await asyncio.wait_for(
                self.memory_system.recall(query, route_target, top_k=top_k),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            return list(fallback or [])

    def _consume_pending_voice_relisten(self) -> bool:
        pending = self._pending_voice_relisten
        self._pending_voice_relisten = False
        return pending

    async def _pump_turn_chunks_to_queue(
        self,
        source: AsyncIterator[JarvisTurnChunk],
        queue: asyncio.Queue,
    ) -> None:
        try:
            async for chunk in source:
                await queue.put(chunk)
        except Exception as exc:
            await queue.put(exc)
        finally:
            await queue.put(None)

    def _bind_active_turn(self, mode: str, turn_id: str) -> ActiveTurnExecution:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("active turn requires a running asyncio task")
        execution = ActiveTurnExecution(
            response_task=task,
            mode=mode,
            turn_id=turn_id,
            speech_pipeline=self._pending_voice_pipeline if mode == "voice" else None,
        )
        self._active_turn = execution
        return execution

    def _clear_active_turn(self, execution: ActiveTurnExecution) -> None:
        if self._active_turn is execution:
            self._active_turn = None

    async def _execute_direct_tool(self, text: str, route: RouteDecision) -> str:
        action = self._build_direct_action(text, route)
        request = ActionRequest.from_action(action, source="direct_tool")
        try:
            result = await self._execute_action_request(
                request,
                acting_reason="executing direct tool",
            )
        except ConfirmationRequiredError as exc:
            return self._format_confirmation_required(exc)
        return self._format_tool_output(result.tool_name, result.output)

    async def _invoke_llm_tool(self, tool_name: str, arguments: dict) -> object:
        if not isinstance(arguments, dict):
            raise TypeError("llm tool arguments must be an object")

        try:
            result = await self._execute_action_request(
                ActionRequest(
                    tool_name=tool_name,
                    arguments=arguments,
                    source="llm_tool",
                ),
                acting_reason="executing llm-requested tool",
            )
            return result.output
        except ConfirmationRequiredError as exc:
            return exc.to_payload()

    async def _execute_action_request(
        self,
        request: ActionRequest,
        *,
        acting_reason: str,
    ):
        transitioned_to_acting = False
        if self.state_machine.state == JarvisState.THINKING:
            await self.state_machine.transition(
                JarvisState.ACTING,
                acting_reason,
                {"tool_name": request.tool_name},
            )
            transitioned_to_acting = True

        try:
            result = await self.action_broker.execute(request)
            await self._publish_event(
                EventType.TOOL_EXECUTED,
                {
                    "tool_name": result.tool_name,
                    "output": result.output,
                    "scope": result.scope,
                    "confirmed": result.confirmed,
                    "side_effects": list(result.side_effects),
                    "audit_logged": result.audit_logged,
                },
            )
            return result
        finally:
            if (
                transitioned_to_acting
                and self.state_machine.state == JarvisState.ACTING
            ):
                await self.state_machine.transition(
                    JarvisState.THINKING,
                    "tool execution completed",
                    {"tool_name": request.tool_name},
                )

    def _build_direct_action(self, text: str, route: RouteDecision):
        if route.tool_name == "system.get_time":
            return GetTimeAction()
        if route.tool_name == "system.set_volume":
            return SetVolumeAction(level=self._extract_volume_level(text))
        if route.tool_name == "timer.start":
            duration_seconds = parse_duration_seconds(text)
            if duration_seconds is None:
                raise ValueError("nao consegui interpretar a duracao do timer")
            return StartTimerAction(
                duration_seconds=duration_seconds, label=text.strip()
            )
        if route.tool_name == "timer.list":
            return ListTimersAction()
        if route.tool_name == "timer.cancel":
            return CancelTimerAction(timer_id=self._extract_timer_id(text))
        if route.tool_name == "browser.search":
            query = self._extract_search_query(text)
            return BrowserSearchAction(query=query)
        if route.tool_name == "browser.fetch_url":
            return BrowserFetchURLAction(url=self._extract_url(text))
        if route.tool_name == "calendar.list_events":
            return ListCalendarEventsAction(days=self._extract_days_window(text))
        if route.tool_name == "system.open_app":
            app_name = self._extract_app_name(text)
            return OpenAppAction(app_name=app_name)
        raise KeyError("unsupported direct tool route %s" % route.tool_name)

    @staticmethod
    def _extract_search_query(text: str) -> str:
        lowered = text.lower()
        for prefix in ("pesquise ", "procure ", "busque na web ", "pesquisa na web "):
            if lowered.startswith(prefix):
                return text[len(prefix) :].strip()
        return text.strip()

    @staticmethod
    def _extract_app_name(text: str) -> str:
        lowered = text.lower().strip()
        for prefix in ("abre o ", "abre a ", "abre ", "abrir o ", "abrir a ", "abrir "):
            if lowered.startswith(prefix):
                return text[len(prefix) :].strip()
        return text.strip()

    @staticmethod
    def _extract_volume_level(text: str) -> int:
        import re

        match = re.search(r"(\d{1,3})", text)
        if match is None:
            raise ValueError("nao consegui interpretar o nivel de volume")
        level = int(match.group(1))
        if level < 0 or level > 100:
            raise ValueError("o nivel de volume deve estar entre 0 e 100")
        return level

    @staticmethod
    def _extract_timer_id(text: str) -> str:
        import re

        match = re.search(
            r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
            text,
        )
        if match is None:
            raise ValueError("nao consegui identificar o timer para cancelar")
        return match.group(1)

    @staticmethod
    def _extract_url(text: str) -> str:
        import re

        match = re.search(r"https?://\S+", text)
        if match is None:
            raise ValueError("nao encontrei uma URL HTTP(s) no pedido")
        return match.group(0).rstrip(".,)")

    @staticmethod
    def _extract_days_window(text: str) -> int:
        import re

        match = re.search(r"(\d+)\s*dias?", text.lower())
        if match is None:
            return 1
        return max(1, int(match.group(1)))

    @staticmethod
    def _format_tool_output(tool_name: str, output) -> str:
        if tool_name == "system.get_time":
            return "Agora sao %s." % output["time"]
        if tool_name == "system.set_volume":
            return "Confirmado. Volume em %d por cento." % output["volume"]
        if tool_name == "timer.start":
            duration_seconds = int(output["duration_seconds"])
            return "Confirmado. Timer para %s." % JarvisRuntime._format_duration(
                duration_seconds
            )
        if tool_name == "timer.list":
            count = len(output)
            if count == 0:
                return "Nao ha timers ativos."
            nearest = output[0]
            remaining = int(nearest.get("remaining_seconds", 0))
            label = str(nearest.get("label") or "Timer")
            if count == 1:
                return "Ha 1 timer ativo. %s termina em %s." % (
                    label,
                    JarvisRuntime._format_duration(remaining),
                )
            return "Ha %d timers ativos. O mais proximo, %s, termina em %s." % (
                count,
                label,
                JarvisRuntime._format_duration(remaining),
            )
        if tool_name == "timer.cancel":
            if output.get("cancelled"):
                return "Confirmado. Timer cancelado."
            return "Nao encontrei esse timer."
        if tool_name == "browser.search":
            return "Abri a busca por %s." % output["query"]
        if tool_name == "browser.fetch_url":
            if output.get("status") == "success":
                return "Carreguei o conteudo de %s." % output["url"]
            return "Nao consegui carregar a URL informada."
        if tool_name == "calendar.list_events":
            if output.get("status") == "error":
                return "Nao consegui consultar seu calendario."
            count = int(output.get("count", 0))
            if count <= 0:
                return "Nao encontrei eventos no periodo consultado."
            events = output.get("events") or []
            first_event = events[0] if events else {}
            title = str(first_event.get("title") or "Sem titulo")
            start = JarvisRuntime._format_calendar_start(first_event.get("start"))
            if count == 1:
                return "Encontrei 1 evento. O proximo e %s%s." % (title, start)
            return "Encontrei %d eventos. O proximo e %s%s." % (
                count,
                title,
                start,
            )
        if tool_name == "system.open_app":
            return "Confirmado. Abri o %s." % output["app_name"]
        return str(output)

    @staticmethod
    def _format_confirmation_required(exc: ConfirmationRequiredError) -> str:
        if exc.side_effects:
            return "Preciso da sua confirmacao para executar %s. Efeitos: %s." % (
                exc.tool_name,
                ", ".join(exc.side_effects),
            )
        return "Preciso da sua confirmacao para executar %s." % exc.tool_name

    @staticmethod
    def _format_duration(duration_seconds: int) -> str:
        duration_seconds = max(0, int(duration_seconds))
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        parts: list[str] = []
        if hours:
            parts.append("%d hora%s" % (hours, "s" if hours != 1 else ""))
        if minutes:
            parts.append("%d minuto%s" % (minutes, "s" if minutes != 1 else ""))
        if seconds and not hours:
            parts.append("%d segundo%s" % (seconds, "s" if seconds != 1 else ""))
        if not parts:
            return "menos de 1 segundo"
        if len(parts) == 1:
            return parts[0]
        return "%s e %s" % (", ".join(parts[:-1]), parts[-1])

    @staticmethod
    def _format_calendar_start(value: object) -> str:
        if not value:
            return ""
        raw = str(value).strip()
        formats = (
            "%Y-%m-%d %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S",
        )
        for fmt in formats:
            try:
                parsed = datetime.strptime(raw, fmt)
                return " as %s" % parsed.strftime("%H:%M")
            except ValueError:
                continue
        return " em %s" % raw

    async def _ensure_trace_reporter_started(self) -> None:
        if self.config.trace_mode == "off":
            return
        if self._trace_reporter is None:
            self._trace_reporter = VoiceTraceReporter(
                session_id=self.session_id,
                mode=self.config.trace_mode,
                jsonl_path=self.config.trace_jsonl_path,
            )
            await self._trace_reporter.start(self.event_bus)
            self._trace_reporter.emit_session_configuration(
                self.trace_configuration_payload()
            )

    def _print_conversation_line(
        self,
        prefix: str,
        text: str,
        *,
        turn_id: str | None = None,
    ) -> None:
        if self._trace_reporter is not None:
            print(
                self._trace_reporter.format_conversation_line(
                    prefix,
                    text,
                    turn_id=turn_id,
                )
            )
            return
        print("%s %s" % (prefix, text))

    def trace_configuration_payload(self) -> dict[str, object]:
        return {
            "activation_backend": self.activation_backend_name,
            "activation_hotkey": self.config.activation_hotkey,
            "stt_backend": self.stt_backend_name,
            "stt_locale": self.config.stt_locale,
            "stt_binary": self.config.stt_bridge_bin,
            "hot_path_backend": self.hot_path_backend_name,
            "hot_path_url": self.config.llm_hot_path_url,
            "hot_path_bridge_bin": self.config.llm_hot_path_bridge_bin,
            "deliberative_backend": self.deliberative_backend_name,
            "deliberative_model": self.config.llm_deliberative_model,
            "fallback_backend": self.fallback_backend_name,
            "fallback_model": self.config.llm_hot_path_fallback_model,
            "tts_backend": self.tts_backend_name,
            "tts_configured_backend": self.config.tts_backend,
            "tts_model": self.config.tts_model,
            "tts_voice": self.config.tts_voice,
            "tts_lang_code": self.config.tts_lang_code,
            "tts_avspeech_voice": self.config.tts_avspeech_voice,
            "playback_backend": self.playback_backend_name,
        }

    def _voice_pipeline_event_context(self, turn_id: str) -> dict[str, object]:
        def build_context() -> dict[str, object]:
            context: dict[str, object] = {
                "session_id": self.session_id,
                "turn_id": turn_id,
                "mode": "voice",
                "backend": self.tts_backend_name,
                "tts_backend": self.tts_backend_name,
                "tts_model": self.config.tts_model,
                "tts_voice": self.config.tts_voice,
                "playback_backend": self.playback_backend_name,
            }
            trace_backend_state = getattr(self.tts_adapter, "trace_backend_state", None)
            if callable(trace_backend_state):
                try:
                    extra = trace_backend_state()
                except Exception:
                    extra = None
                if isinstance(extra, dict):
                    context.update(extra)
            return context

        return build_context

    def _backend_detail_for_name(
        self, backend_name: Optional[str]
    ) -> dict[str, object] | None:
        if backend_name is None:
            return None
        if backend_name == self.hot_path_backend_name:
            return {
                "backend": backend_name,
                "model": "apple.foundation_models",
                "bridge_url": self.config.llm_hot_path_url,
                "bridge_bin": self.config.llm_hot_path_bridge_bin,
            }
        if backend_name == self.deliberative_backend_name:
            return {
                "backend": backend_name,
                "model": self.config.llm_deliberative_model,
            }
        if backend_name == self.fallback_backend_name:
            return {
                "backend": backend_name,
                "model": self.config.llm_hot_path_fallback_model,
            }
        if backend_name == self.tts_backend_name:
            return {
                "backend": backend_name,
                "model": self.config.tts_model,
                "voice": self.config.tts_voice,
                "fallback_voice": self.config.tts_avspeech_voice,
            }
        if backend_name == self.stt_backend_name:
            return {
                "backend": backend_name,
                "locale": self.config.stt_locale,
                "binary": self.config.stt_bridge_bin,
            }
        if backend_name == self.activation_backend_name:
            return {
                "backend": backend_name,
                "hotkey": self.config.activation_hotkey,
            }
        if backend_name == self.playback_backend_name:
            return {"backend": backend_name}
        return {"backend": backend_name}

    def _set_active_turn_context(
        self, route: RouteDecision, backend_name: Optional[str]
    ) -> None:
        if self._active_turn is None:
            return
        self._active_turn.route = route
        self._active_turn.backend_name = backend_name

    async def _publish_event(
        self,
        event_type: EventType,
        payload: Optional[dict[str, object]] = None,
        *,
        turn_id: Optional[str] = None,
        route: Optional[RouteDecision] = None,
        backend: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> None:
        await self.event_bus.publish(
            Event(
                event_type=event_type,
                payload=self._build_event_payload(
                    payload=payload,
                    turn_id=turn_id,
                    route=route,
                    backend=backend,
                    mode=mode,
                ),
            )
        )

    def _build_event_payload(
        self,
        payload: Optional[dict[str, object]] = None,
        *,
        turn_id: Optional[str] = None,
        route: Optional[RouteDecision] = None,
        backend: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> dict[str, object]:
        active_turn = self._active_turn
        resolved_route = route or (
            active_turn.route if active_turn is not None else None
        )
        resolved_backend = (
            backend
            if backend is not None
            else (active_turn.backend_name if active_turn is not None else None)
        )
        resolved_turn_id = (
            turn_id
            if turn_id is not None
            else (active_turn.turn_id if active_turn is not None else None)
        )
        resolved_mode = (
            mode
            if mode is not None
            else (active_turn.mode if active_turn is not None else None)
        )

        enriched: dict[str, object] = {"session_id": self.session_id}
        if resolved_turn_id is not None:
            enriched["turn_id"] = resolved_turn_id
        if resolved_mode is not None:
            enriched["mode"] = resolved_mode
        if resolved_route is not None:
            enriched["route"] = resolved_route.target.value
            enriched["route_reason"] = resolved_route.reason
            enriched["route_confidence"] = resolved_route.confidence
            if resolved_route.tool_name is not None:
                enriched["tool_name"] = resolved_route.tool_name
        if resolved_backend is not None:
            enriched["backend"] = resolved_backend
            enriched.setdefault(
                "backend_detail", self._backend_detail_for_name(resolved_backend)
            )
        if payload:
            enriched.update(payload)
        return enriched

    @staticmethod
    def _default_capabilities(config: JarvisConfig) -> list:
        files_enabled = bool(config.allowed_file_roots)
        return [
            Capability("system.get_time", enabled=True, risk_level=RiskLevel.READ_ONLY),
            Capability(
                "system.open_app", enabled=True, risk_level=RiskLevel.WRITE_SAFE
            ),
            Capability(
                "system.set_volume",
                enabled=False,
                risk_level=RiskLevel.WRITE_SAFE,
                requires_confirmation=True,
                side_effects=["system_volume"],
            ),
            Capability("timer.start", enabled=True, risk_level=RiskLevel.WRITE_SAFE),
            Capability("timer.list", enabled=True, risk_level=RiskLevel.READ_ONLY),
            Capability("timer.cancel", enabled=True, risk_level=RiskLevel.WRITE_SAFE),
            Capability("browser.search", enabled=True, risk_level=RiskLevel.READ_ONLY),
            Capability(
                "browser.fetch_url", enabled=True, risk_level=RiskLevel.READ_ONLY
            ),
            Capability(
                "files.list", enabled=files_enabled, risk_level=RiskLevel.READ_ONLY
            ),
            Capability(
                "files.read", enabled=files_enabled, risk_level=RiskLevel.READ_ONLY
            ),
            Capability(
                "files.move",
                enabled=False,
                risk_level=RiskLevel.WRITE_SAFE,
                requires_confirmation=True,
                side_effects=["filesystem_write"],
            ),
            Capability(
                "calendar.list_events", enabled=True, risk_level=RiskLevel.READ_ONLY
            ),
            Capability("shell.execute", enabled=False, risk_level=RiskLevel.WRITE_SAFE),
        ]
