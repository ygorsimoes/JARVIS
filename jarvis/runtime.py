from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, List, Optional

from .adapters import build_runtime_adapters
from .bus import EventBus
from .config import JarvisConfig
from .core.action_broker import ActionBroker, ActionRequest
from .core.capability_broker import Capability, CapabilityBroker, RiskLevel
from .core.complexity_router import ComplexityRouter
from .core.dialogue_manager import DialogueManager
from .core.resource_governor import ResourceGovernor
from .core.sentence_streamer import SentenceStreamer, SentenceStreamerConfig
from .core.speech_pipeline import SpeechPipeline
from .core.state_machine import StateMachine
from .core.turn_manager import CompletedTurn, TurnManager, TurnManagerConfig
from .models.actions import (
    BrowserSearchAction,
    GetTimeAction,
    OpenAppAction,
    StartTimerAction,
)
from .models.conversation import RouteDecision, RouteTarget, Role
from .models.events import Event, EventType
from .models.memory import Memory
from .models.state import JarvisState
from .tools import ToolRegistry, build_default_registry
from .tools.timer import parse_duration_seconds


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
    text: str
    reason: str
    used_partial: bool
    metadata: dict


@dataclass
class ActiveTurnExecution:
    response_task: asyncio.Task
    mode: str
    adapter: object | None = None
    speech_pipeline: SpeechPipeline | None = None
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
        action_broker: ActionBroker,
        tool_registry: ToolRegistry,
        turn_manager_config: TurnManagerConfig,
        activation_adapter,
        hot_path_adapter,
        deliberative_adapter,
        stt_adapter,
        tts_adapter,
        vad_adapter,
        playback_backend,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.state_machine = state_machine
        self.dialogue_manager = dialogue_manager
        self.sentence_streamer = sentence_streamer
        self.router = router
        self.action_broker = action_broker
        self.tool_registry = tool_registry
        self.turn_manager_config = turn_manager_config
        self.activation_adapter = activation_adapter
        self.hot_path_adapter = hot_path_adapter
        self.deliberative_adapter = deliberative_adapter
        self.stt_adapter = stt_adapter
        self.tts_adapter = tts_adapter
        self.vad_adapter = vad_adapter
        self.playback_backend = playback_backend
        self._active_turn: ActiveTurnExecution | None = None
        self._pending_voice_pipeline: SpeechPipeline | None = None

    @classmethod
    def from_config(
        cls,
        config: Optional[JarvisConfig] = None,
        enable_native_backends: bool = False,
    ) -> "JarvisRuntime":
        config = config or JarvisConfig()
        if enable_native_backends:
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
        capability_broker = CapabilityBroker(cls._default_capabilities())
        action_broker = ActionBroker(capability_broker, tool_registry)
        adapters = build_runtime_adapters(
            config, enable_native_backends=enable_native_backends
        )
        turn_manager_config = TurnManagerConfig(
            silence_timeout_ms=config.turn_silence_timeout_ms,
            partial_commit_min_chars=config.turn_partial_commit_min_chars,
            tick_interval_ms=config.turn_tick_interval_ms,
            max_turn_duration_s=config.turn_max_duration_s,
        )

        return cls(
            config=config,
            event_bus=event_bus,
            state_machine=state_machine,
            dialogue_manager=dialogue_manager,
            sentence_streamer=sentence_streamer,
            router=ComplexityRouter(),
            action_broker=action_broker,
            tool_registry=tool_registry,
            turn_manager_config=turn_manager_config,
            activation_adapter=adapters.activation,
            hot_path_adapter=adapters.hot_path_llm,
            deliberative_adapter=adapters.deliberative_llm,
            stt_adapter=adapters.stt,
            tts_adapter=adapters.tts,
            vad_adapter=adapters.vad,
            playback_backend=adapters.playback,
        )

    async def run_voice_foreground(self, turn_limit: Optional[int] = None) -> None:
        handled_turns = 0
        while turn_limit is None or handled_turns < turn_limit:
            activated = await self.activation_adapter.listen()
            if not activated:
                continue

            await self.event_bus.publish(
                Event(
                    event_type=EventType.ACTIVATION_TRIGGERED,
                    payload={"source": "foreground"},
                )
            )

            voice_turn = await self.capture_voice_turn()
            print("voce> %s" % voice_turn.text)
            pipeline = SpeechPipeline(
                tts_adapter=self.tts_adapter,
                playback_backend=self.playback_backend,
                sample_rate_hz=self.config.tts_sample_rate_hz,
                event_bus=self.event_bus,
            )
            self._pending_voice_pipeline = pipeline
            await pipeline.start()
            response_queue: asyncio.Queue = asyncio.Queue()
            response_task = asyncio.create_task(
                self._pump_turn_chunks_to_queue(
                    self.stream_transcribed_turn(voice_turn.text),
                    response_queue,
                ),
                name="jarvis-voice-response",
            )
            try:
                while True:
                    item = await response_queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    print("jarvis> %s" % item.sentence)
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
                await pipeline.shutdown()
            handled_turns += 1

    async def capture_voice_turn(self) -> CapturedVoiceTurn:
        turn_manager = TurnManager(self.turn_manager_config)
        await self.state_machine.transition(JarvisState.ARMED, "activation accepted")
        await self.state_machine.transition(
            JarvisState.LISTENING, "voice capture started"
        )

        session = await self.stt_adapter.start_live_session()
        completed_turn: Optional[CompletedTurn] = None
        try:
            event_iterator = session.iter_events().__aiter__()
            while completed_turn is None:
                try:
                    event = await asyncio.wait_for(
                        event_iterator.__anext__(),
                        timeout=self.turn_manager_config.tick_interval_ms / 1000.0,
                    )
                except asyncio.TimeoutError:
                    completed_turn = turn_manager.tick()
                    continue
                except StopAsyncIteration:
                    completed_turn = turn_manager.finalize(reason="speech_stream_ended")
                    break

                await self._publish_stt_event(event)
                completed_turn = self._consume_turn_signal(turn_manager, event)

            if completed_turn is None or not completed_turn.text:
                raise RuntimeError("voice turn ended without a usable transcript")

            await self.event_bus.publish(
                Event(
                    event_type=EventType.TURN_READY,
                    payload={
                        "text": completed_turn.text,
                        "reason": completed_turn.reason,
                        "used_partial": completed_turn.used_partial,
                    },
                )
            )
            await self.state_machine.transition(
                JarvisState.TRANSCRIBING,
                completed_turn.reason,
                {"used_partial": completed_turn.used_partial},
            )
            return CapturedVoiceTurn(
                text=completed_turn.text,
                reason=completed_turn.reason,
                used_partial=completed_turn.used_partial,
                metadata=dict(completed_turn.metadata),
            )
        except Exception as exc:
            await self._handle_turn_failure(exc, input_text=None)
            raise
        finally:
            await session.stop()

    async def stream_transcribed_turn(
        self,
        text: str,
        recalled_memories: Optional[Iterable[Memory]] = None,
    ) -> AsyncIterator[JarvisTurnChunk]:
        execution = self._bind_active_turn(mode="voice")
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
        recalled_memories = list(recalled_memories or [])
        route: Optional[RouteDecision] = None
        response_sentences: List[str] = []
        execution = self._bind_active_turn(mode="text")

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
                text, recalled_memories=recalled_memories
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
        for adapter in (
            self.hot_path_adapter,
            self.deliberative_adapter,
            self.stt_adapter,
            self.tts_adapter,
            self.playback_backend,
        ):
            shutdown = getattr(adapter, "shutdown", None)
            if shutdown is not None:
                await shutdown()

    async def interrupt_current_turn(self, reason: str = "barge-in") -> bool:
        execution = self._active_turn
        if execution is None:
            return False
        if execution.interrupted:
            return True

        execution.interrupted = True
        await self.event_bus.publish(
            Event(
                event_type=EventType.INTERRUPTION_REQUESTED, payload={"reason": reason}
            )
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
        sentence_streamer = SentenceStreamer(self.sentence_streamer.config)
        queue: asyncio.Queue = asyncio.Queue()
        pump_task = asyncio.create_task(
            sentence_streamer.pump(
                adapter.chat_stream(
                    messages=messages,
                    tools=self.tool_registry.describe(),
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

    async def _stream_response_chunks(
        self,
        text: str,
        recalled_memories: Optional[Iterable[Memory]] = None,
    ) -> AsyncIterator[JarvisTurnChunk]:
        recalled_memories = list(recalled_memories or [])
        route = self.router.route(
            text,
            recalled_memories=len(recalled_memories),
            recent_turns=len(self.dialogue_manager.working_memory),
        )

        self.dialogue_manager.record_turn(
            Role.USER, text, {"route": route.target.value}
        )
        await self.event_bus.publish(
            Event(event_type=EventType.USER_TURN, payload={"text": text})
        )
        await self.event_bus.publish(
            Event(
                event_type=EventType.ROUTE_SELECTED,
                payload={
                    "input_text": text,
                    "target": route.target.value,
                    "tool_name": route.tool_name,
                    "reason": route.reason,
                },
            )
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
            await self.event_bus.publish(
                Event(
                    event_type=EventType.ASSISTANT_SENTENCE,
                    payload={
                        "sentence": sentence,
                        "route": route.target.value,
                        "index": 0,
                    },
                )
            )
            yield JarvisTurnChunk(sentence=sentence, route=route, index=0)
        else:
            messages = self.dialogue_manager.compose_messages(text, recalled_memories)
            adapter = (
                self.hot_path_adapter
                if route.target == RouteTarget.HOT_PATH
                else self.deliberative_adapter
            )
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
                await self.event_bus.publish(
                    Event(
                        event_type=EventType.ASSISTANT_SENTENCE,
                        payload={
                            "sentence": sentence,
                            "route": route.target.value,
                            "index": index,
                        },
                    )
                )
                yield JarvisTurnChunk(sentence=sentence, route=route, index=index)
                index += 1

        full_text = " ".join(
            sentence.strip() for sentence in response_sentences
        ).strip()
        self.dialogue_manager.record_turn(
            Role.ASSISTANT, full_text, {"route": route.target.value}
        )
        await self.event_bus.publish(
            Event(
                event_type=EventType.ASSISTANT_COMPLETED,
                payload={"text": full_text, "route": route.target.value},
            )
        )
        await self.state_machine.transition(JarvisState.IDLE, "turn completed")

    async def _publish_stt_event(self, event: dict) -> None:
        event_type = event.get("type")
        if event_type == "speech_started":
            mapped = EventType.SPEECH_STARTED
        elif event_type == "speech_ended":
            mapped = EventType.SPEECH_ENDED
        elif event_type == "partial_transcript":
            mapped = EventType.PARTIAL_TRANSCRIPT
        elif event_type == "final_transcript":
            mapped = EventType.FINAL_TRANSCRIPT
        else:
            return

        await self.event_bus.publish(Event(event_type=mapped, payload=dict(event)))

    def _consume_turn_signal(
        self, turn_manager: TurnManager, event: dict
    ) -> Optional[CompletedTurn]:
        event_type = event.get("type")
        if event_type == "speech_detector_result":
            speech_detected = bool(event.get("speech_detected"))
            return turn_manager.consume_vad_signal(
                speech_detected, reason="speech_detector"
            )
        if event_type == "speech_started":
            return turn_manager.consume_vad_signal(True, reason="speech_started")
        if event_type == "speech_ended":
            return turn_manager.consume_vad_signal(False, reason="speech_ended")
        if event_type == "partial_transcript":
            return turn_manager.consume_partial_transcript(event.get("text", ""))
        if event_type == "final_transcript":
            return turn_manager.consume_final_transcript(event.get("text", ""))
        event_has_speech = getattr(self.vad_adapter, "event_has_speech", None)
        if event_has_speech is not None and event_has_speech(event):
            return turn_manager.consume_vad_signal(True, reason="vad_adapter")
        event_is_silence = getattr(self.vad_adapter, "event_is_silence", None)
        if event_is_silence is not None and event_is_silence(event):
            return turn_manager.consume_vad_signal(False, reason="vad_adapter")
        return turn_manager.consume_event(event)

    async def _handle_turn_failure(
        self,
        exc: Exception,
        input_text: Optional[str],
        route: Optional[RouteDecision] = None,
    ) -> None:
        await self.event_bus.publish(
            Event(
                event_type=EventType.ERROR,
                payload={
                    "message": str(exc),
                    "route": route.target.value if route else None,
                    "input_text": input_text,
                },
            )
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

    def _bind_active_turn(self, mode: str) -> ActiveTurnExecution:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("active turn requires a running asyncio task")
        execution = ActiveTurnExecution(
            response_task=task,
            mode=mode,
            speech_pipeline=self._pending_voice_pipeline if mode == "voice" else None,
        )
        self._active_turn = execution
        return execution

    def _clear_active_turn(self, execution: ActiveTurnExecution) -> None:
        if self._active_turn is execution:
            self._active_turn = None

    async def _execute_direct_tool(self, text: str, route: RouteDecision) -> str:
        action = self._build_direct_action(text, route)
        request = ActionRequest.from_action(action)
        result = await self.action_broker.execute(request)
        await self.event_bus.publish(
            Event(
                event_type=EventType.TOOL_EXECUTED,
                payload={"tool_name": result.tool_name, "output": result.output},
            )
        )
        return self._format_tool_output(result.tool_name, result.output)

    async def _invoke_llm_tool(self, tool_name: str, arguments: dict) -> object:
        if not isinstance(arguments, dict):
            raise TypeError("llm tool arguments must be an object")

        transitioned_to_acting = False
        if self.state_machine.state == JarvisState.THINKING:
            await self.state_machine.transition(
                JarvisState.ACTING,
                "executing llm-requested tool",
                {"tool_name": tool_name},
            )
            transitioned_to_acting = True

        try:
            result = await self.action_broker.execute(
                ActionRequest(tool_name=tool_name, arguments=arguments)
            )
            await self.event_bus.publish(
                Event(
                    event_type=EventType.TOOL_EXECUTED,
                    payload={"tool_name": result.tool_name, "output": result.output},
                )
            )
            return result.output
        finally:
            if (
                transitioned_to_acting
                and self.state_machine.state == JarvisState.ACTING
            ):
                await self.state_machine.transition(
                    JarvisState.THINKING,
                    "llm-requested tool completed",
                    {"tool_name": tool_name},
                )

    def _build_direct_action(self, text: str, route: RouteDecision):
        if route.tool_name == "system.get_time":
            return GetTimeAction()
        if route.tool_name == "timer.start":
            duration_seconds = parse_duration_seconds(text)
            if duration_seconds is None:
                raise ValueError("nao consegui interpretar a duracao do timer")
            return StartTimerAction(
                duration_seconds=duration_seconds, label=text.strip()
            )
        if route.tool_name == "browser.search":
            query = self._extract_search_query(text)
            return BrowserSearchAction(query=query)
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
    def _format_tool_output(tool_name: str, output) -> str:
        if tool_name == "system.get_time":
            return "Agora sao %s." % output["time"]
        if tool_name == "timer.start":
            duration_seconds = int(output["duration_seconds"])
            minutes = duration_seconds // 60
            if minutes and duration_seconds % 60 == 0:
                return "Timer definido para %d minutos." % minutes
            return "Timer definido para %d segundos." % duration_seconds
        if tool_name == "browser.search":
            return "Busca aberta para %s." % output["query"]
        if tool_name == "system.open_app":
            return "%s aberto." % output["app_name"]
        return str(output)

    @staticmethod
    def _default_capabilities() -> list:
        return [
            Capability("system.get_time", enabled=True, risk_level=RiskLevel.READ_ONLY),
            Capability(
                "system.open_app", enabled=True, risk_level=RiskLevel.WRITE_SAFE
            ),
            Capability("timer.start", enabled=True, risk_level=RiskLevel.WRITE_SAFE),
            Capability("timer.list", enabled=True, risk_level=RiskLevel.READ_ONLY),
            Capability("timer.cancel", enabled=True, risk_level=RiskLevel.WRITE_SAFE),
            Capability("browser.search", enabled=True, risk_level=RiskLevel.READ_ONLY),
            Capability("files.list", enabled=True, risk_level=RiskLevel.READ_ONLY),
            Capability("files.read", enabled=True, risk_level=RiskLevel.READ_ONLY),
        ]
