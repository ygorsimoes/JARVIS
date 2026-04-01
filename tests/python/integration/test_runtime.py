import asyncio
from typing import Any, cast
from unittest.mock import patch

import pytest
import pytest_asyncio

from jarvis.adapters.stt.speech_analyzer import SpeechAnalyzerStreamError
from jarvis.config import JarvisConfig
from jarvis.core.action_broker import ActionBroker, ActionRequest
from jarvis.core.capability_broker import Capability, CapabilityBroker
from jarvis.models.conversation import Role, RouteDecision, RouteTarget
from jarvis.models.events import EventType
from jarvis.models.memory import Memory, MemoryCategory, MemorySource
from jarvis.models.state import JarvisState
from jarvis.runtime import JarvisRuntime, VoiceCaptureError


class FakeActivationAdapter:
    def __init__(self, responses):
        self._responses = list(responses)

    async def listen(self) -> bool:
        return self._responses.pop(0)


class FakeSTTSession:
    def __init__(self, events):
        self._events = list(events)
        self.stopped = False

    async def iter_events(self):
        for event in self._events:
            yield event

    async def stop(self) -> None:
        self.stopped = True


class DelayedSTTSession(FakeSTTSession):
    def __init__(self, events, delay_s: float):
        super().__init__(events)
        self._delay_s = delay_s

    async def iter_events(self):
        for index, event in enumerate(self._events):
            if index > 0:
                await asyncio.sleep(self._delay_s)
            yield event


class FailingSTTSession:
    def __init__(self, exc: Exception):
        self.exc = exc
        self.stopped = False

    async def iter_events(self):
        raise self.exc
        yield  # pragma: no cover

    async def stop(self) -> None:
        self.stopped = True


class FakeSTTAdapter:
    def __init__(self, events):
        self.events = events

    async def start_live_session(self):
        session_events = self.events
        self.events = []  # O watcher não ver eventos
        return FakeSTTSession(session_events)


class DelayedSTTAdapter(FakeSTTAdapter):
    def __init__(self, events, delay_s: float):
        super().__init__(events)
        self.delay_s = delay_s

    async def start_live_session(self):
        session_events = self.events
        self.events = []
        return DelayedSTTSession(session_events, self.delay_s)


class FailingSTTAdapter:
    def __init__(self, exc: Exception):
        self.exc = exc

    async def start_live_session(self):
        return FailingSTTSession(self.exc)


class FakePlaybackBackend:
    def __init__(self):
        self.played = []
        self.stop_calls = 0

    async def play(self, audio_bytes: bytes, sample_rate_hz: int) -> None:
        self.played.append((audio_bytes, sample_rate_hz))

    async def stop(self) -> None:
        self.stop_calls += 1

    async def shutdown(self) -> None:
        return None


class SlowHotPathAdapter:
    def __init__(self):
        self.cancelled = False

    async def chat_stream(self, messages, tools, max_kv_size, tool_invoker=None):
        del messages, tools, max_kv_size, tool_invoker
        yield "Primeira frase completa com detalhes suficientes para ser despachada agora. "
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        yield "Segunda frase completa com detalhes suficientes para ser despachada depois."

    async def cancel_current_response(self) -> bool:
        self.cancelled = True
        return True


class ToolCallingHotPathAdapter:
    async def chat_stream(self, messages, tools, max_kv_size, tool_invoker=None):
        del messages, tools, max_kv_size
        assert tool_invoker is not None
        result = await tool_invoker("system.get_time", {})
        yield "Agora sao %s. Posso seguir." % result["time"]

    async def cancel_current_response(self) -> bool:
        return False


class CapturingHotPathAdapter:
    def __init__(self):
        self.messages = None
        self.tools = None

    async def chat_stream(self, messages, tools, max_kv_size, tool_invoker=None):
        del max_kv_size, tool_invoker
        self.messages = list(messages)
        self.tools = list(tools)
        yield "Resposta curta."

    async def cancel_current_response(self) -> bool:
        return False


class FakeMemorySystem:
    def __init__(self, memories=None):
        self.memories = list(memories or [])
        self.calls = []
        self.persisted = []

    async def recall(
        self, query: str, route_target: RouteTarget, top_k: int | None = None
    ):
        self.calls.append((query, route_target, top_k))
        return list(self.memories)

    async def maybe_persist_turn(self, user_text: str, assistant_text: str):
        self.persisted.append((user_text, assistant_text))
        return None


@pytest.mark.asyncio
class TestRuntime:
    @pytest_asyncio.fixture(autouse=True)
    async def _runtime(self):
        self.runtime = JarvisRuntime.from_config(
            JarvisConfig(allowed_file_roots=["/tmp"]),
            enable_native_backends=False,
        )
        yield
        await self.runtime.shutdown()

    async def test_direct_tool_response_for_time(self):
        response = await self.runtime.respond_text("Que horas sao agora?")
        assert response.route.target == RouteTarget.DIRECT_TOOL
        assert "Agora sao" in response.full_text

    async def test_deliberative_response_uses_streaming_adapter(self):
        response = await self.runtime.respond_text(
            "Analisa esse erro e explica por que ele quebra"
        )
        assert response.route.target == RouteTarget.DELIBERATIVE
        assert len(response.sentences) >= 2
        assert "Analisei o pedido" in response.full_text

    async def test_stream_text_turn_emits_incremental_sentences(self):
        chunks = []
        async for chunk in self.runtime.stream_text_turn(
            "Analisa esse erro e explica por que ele quebra"
        ):
            chunks.append(chunk)

        assert len(chunks) >= 2
        assert chunks[0].sentence.endswith(".")
        assert chunks[0].index == 0
        assert chunks[1].index == 1

    async def test_browser_route_formats_response(self):
        with patch("webbrowser.open", return_value=True):
            response = await self.runtime.respond_text("Pesquise arquitetura hexagonal")
        assert response.route.tool_name == "browser.search"
        assert "Busca aberta" in response.full_text

    async def test_runtime_recovers_to_idle_after_error(self):
        with pytest.raises(ValueError):
            await self.runtime.respond_text("Define um timer")
        assert self.runtime.state_machine.state.value == "idle"

    async def test_capture_voice_turn_reads_stt_events(self):
        self.runtime.stt_adapter = FakeSTTAdapter(
            [
                {"type": "speech_started"},
                {"type": "partial_transcript", "text": "que horas sao"},
                {"type": "speech_ended"},
                {"type": "final_transcript", "text": "Que horas sao agora?"},
            ]
        )

        turn = await self.runtime.capture_voice_turn()

        assert turn.text == "Que horas sao agora?"
        assert self.runtime.state_machine.state == JarvisState.TRANSCRIBING

    async def test_capture_voice_turn_survives_short_silence_after_ready(self):
        delay_s = self.runtime.turn_manager_config.tick_interval_ms / 1000.0 * 1.5
        self.runtime.stt_adapter = DelayedSTTAdapter(
            [
                {"type": "ready", "locale": "pt-BR", "sample_rate": 16000},
                {"type": "speech_started"},
                {"type": "partial_transcript", "text": "olá tudo bem"},
                {"type": "speech_ended"},
                {"type": "final_transcript", "text": "Olá, tudo bem?"},
            ],
            delay_s=delay_s,
        )

        turn = await self.runtime.capture_voice_turn()

        assert turn.text == "Olá, tudo bem?"
        assert self.runtime.state_machine.state == JarvisState.TRANSCRIBING

    async def test_capture_voice_turn_reports_partial_updates(self):
        self.runtime.stt_adapter = FakeSTTAdapter(
            [
                {"type": "ready", "locale": "pt-BR", "sample_rate": 16000},
                {"type": "partial_transcript", "text": "olá"},
                {"type": "partial_transcript", "text": "olá tudo bem"},
                {"type": "final_transcript", "text": "Olá, tudo bem?"},
            ]
        )
        partials = []

        turn = await self.runtime.capture_voice_turn(
            on_partial_transcript=partials.append
        )

        assert partials == ["olá", "olá tudo bem"]
        assert turn.text == "Olá, tudo bem?"

    async def test_capture_voice_turn_surfaces_stt_error_message(self):
        self.runtime.stt_adapter = FakeSTTAdapter(
            [{"type": "error", "message": "microphone_permission_denied"}]
        )

        with pytest.raises(VoiceCaptureError, match="microphone_permission_denied"):
            await self.runtime.capture_voice_turn()

        assert self.runtime.state_machine.state == JarvisState.IDLE

    async def test_capture_voice_turn_wraps_stt_stream_failures(self):
        self.runtime.stt_adapter = FailingSTTAdapter(
            SpeechAnalyzerStreamError("SpeechAnalyzer CLI failed without stderr output")
        )

        with pytest.raises(VoiceCaptureError, match="erro no reconhecimento de voz"):
            await self.runtime.capture_voice_turn()

        assert self.runtime.state_machine.state == JarvisState.IDLE

    async def test_run_voice_foreground_handles_one_turn(self):
        self.runtime.activation_adapter = FakeActivationAdapter([True])
        self.runtime.stt_adapter = FakeSTTAdapter(
            [
                {"type": "speech_started"},
                {"type": "partial_transcript", "text": "que horas sao"},
                {"type": "speech_ended"},
                {"type": "final_transcript", "text": "Que horas sao agora?"},
            ]
        )
        self.runtime.playback_backend = FakePlaybackBackend()

        with patch("builtins.print") as print_mock:
            await self.runtime.run_voice_foreground(turn_limit=1)

        printed_lines = [call.args[0] for call in print_mock.call_args_list]
        assert any(line.startswith("voce>") for line in printed_lines)
        assert any(line.startswith("jarvis>") for line in printed_lines)
        assert self.runtime.playback_backend.played
        assert self.runtime.state_machine.state == JarvisState.IDLE

    async def test_run_voice_foreground_prints_partial_transcripts(self):
        self.runtime.activation_adapter = FakeActivationAdapter([True])
        self.runtime.stt_adapter = FakeSTTAdapter(
            [
                {"type": "ready", "locale": "pt-BR", "sample_rate": 16000},
                {"type": "partial_transcript", "text": "olá"},
                {"type": "partial_transcript", "text": "olá tudo bem"},
                {"type": "final_transcript", "text": "Olá, tudo bem?"},
            ]
        )
        self.runtime.playback_backend = FakePlaybackBackend()

        with patch("builtins.print") as print_mock:
            with patch("sys.stdout.isatty", return_value=False):
                await self.runtime.run_voice_foreground(turn_limit=1)

        printed_lines = [call.args[0] for call in print_mock.call_args_list]
        assert any(line.startswith("voce~>") for line in printed_lines)
        assert any(line.startswith("voce>") for line in printed_lines)

    async def test_run_voice_foreground_reports_capture_failures_without_traceback(
        self,
    ):
        self.runtime.activation_adapter = FakeActivationAdapter([True])
        self.runtime.stt_adapter = FakeSTTAdapter([{"type": "ready"}])

        with patch("builtins.print") as print_mock:
            await self.runtime.run_voice_foreground(turn_limit=1)

        printed_lines = [call.args[0] for call in print_mock.call_args_list]
        assert any(line.startswith("jarvis>") for line in printed_lines)
        assert any("nenhuma fala utilizavel" in line for line in printed_lines)
        assert self.runtime.state_machine.state == JarvisState.IDLE

    async def test_interrupt_current_turn_cancels_active_response(self):
        self.runtime.hot_path_adapter = SlowHotPathAdapter()

        chunks = []

        async def consume():
            async for chunk in self.runtime.stream_text_turn("Me lembra do resumo"):
                chunks.append(chunk)

        task = asyncio.create_task(consume())
        while self.runtime._active_turn is None:
            await asyncio.sleep(0.01)
        for _ in range(50):
            if chunks:
                break
            await asyncio.sleep(0.01)

        interrupted = await self.runtime.interrupt_current_turn()
        await task

        assert interrupted
        assert self.runtime.hot_path_adapter.cancelled
        assert len(chunks) == 1
        assert self.runtime.state_machine.state == JarvisState.IDLE

    async def test_hot_path_can_execute_llm_requested_tool(self):
        self.runtime.hot_path_adapter = ToolCallingHotPathAdapter()

        response = await self.runtime.respond_text("Me lembra do contexto")

        assert response.route.target == RouteTarget.HOT_PATH
        assert "Agora sao" in response.full_text
        assert self.runtime.state_machine.state == JarvisState.IDLE

    async def test_current_user_turn_is_not_duplicated_in_prompt_context(self):
        self.runtime.hot_path_adapter = CapturingHotPathAdapter()

        await self.runtime.respond_text("Me lembra do resumo")

        user_messages = [
            message.content
            for message in (self.runtime.hot_path_adapter.messages or [])
            if message.role == Role.USER
        ]
        assert user_messages == ["Me lembra do resumo"]

    async def test_runtime_only_exposes_capability_enabled_tools_to_llm(self):
        self.runtime.hot_path_adapter = CapturingHotPathAdapter()

        await self.runtime.respond_text("Me lembra do resumo")

        tool_names = [
            tool["name"] for tool in (self.runtime.hot_path_adapter.tools or [])
        ]
        assert "browser.fetch_url" in tool_names
        assert "calendar.list_events" in tool_names
        assert "files.list" in tool_names
        assert "files.read" in tool_names
        assert "shell.execute" not in tool_names
        assert "files.move" not in tool_names

    async def test_runtime_uses_hot_path_memory_policy_after_routing(self):
        self.runtime.hot_path_adapter = CapturingHotPathAdapter()
        self.runtime.memory_system = cast(
            Any,
            FakeMemorySystem(
                [
                    Memory(
                        content="Usuario prefere respostas curtas",
                        category=MemoryCategory.PROFILE,
                        source=MemorySource.EXPLICIT,
                        confidence=0.9,
                        recency_weight=1.0,
                        scope="global",
                    )
                ]
            ),
        )

        await self.runtime.respond_text("Me lembra do resumo")

        memory_system = cast(Any, self.runtime.memory_system)
        assert memory_system.calls == [
            ("Me lembra do resumo", RouteTarget.HOT_PATH, None)
        ]
        memory_messages = [
            message.content
            for message in (self.runtime.hot_path_adapter.messages or [])
            if message.role == Role.SYSTEM and "Memorias relevantes" in message.content
        ]
        assert len(memory_messages) == 1
        assert "Usuario prefere respostas curtas" in memory_messages[0]

    async def test_runtime_uses_deliberative_memory_policy_for_complex_requests(self):
        capturing_adapter = CapturingHotPathAdapter()
        self.runtime.deliberative_adapter = capturing_adapter
        self.runtime.memory_system = cast(
            Any,
            FakeMemorySystem(
                [
                    Memory(
                        content="Projeto jarvis usa sqlite vec para busca semantica",
                        category=MemoryCategory.PROCEDURAL,
                        source=MemorySource.EXPLICIT,
                        confidence=0.85,
                        recency_weight=1.0,
                        scope="project:jarvis",
                    )
                ]
            ),
        )

        await self.runtime.respond_text(
            "Analisa esse erro e explica por que ele quebra"
        )

        memory_system = cast(Any, self.runtime.memory_system)
        assert memory_system.calls == [
            (
                "Analisa esse erro e explica por que ele quebra",
                RouteTarget.DELIBERATIVE,
                None,
            )
        ]
        memory_messages = [
            message.content
            for message in (capturing_adapter.messages or [])
            if message.role == Role.SYSTEM and "Memorias relevantes" in message.content
        ]
        assert len(memory_messages) == 1
        assert "project:jarvis" in memory_messages[0]

    async def test_runtime_skips_memory_recall_for_direct_tools(self):
        self.runtime.memory_system = cast(Any, FakeMemorySystem())

        await self.runtime.respond_text("Que horas sao agora?")

        memory_system = cast(Any, self.runtime.memory_system)
        assert memory_system.calls == []

    async def test_direct_tool_confirmation_is_rendered_as_user_message(self):
        async def set_volume(level: int) -> dict:
            return {"volume": level}

        registry = self.runtime.tool_registry
        registry.register(
            "system.set_volume",
            "Ajusta volume",
            set_volume,
            input_schema={
                "type": "object",
                "properties": {"level": {"type": "integer", "minimum": 0}},
                "required": ["level"],
                "additionalProperties": False,
            },
        )
        self.runtime.action_broker = ActionBroker(
            CapabilityBroker(
                [
                    Capability(
                        "system.set_volume",
                        enabled=True,
                        requires_confirmation=True,
                        side_effects=["system_volume"],
                    )
                ]
            ),
            registry,
        )

        response = await self.runtime._execute_direct_tool(
            "ajusta o volume para 40",
            RouteDecision(
                target=RouteTarget.DIRECT_TOOL,
                tool_name="system.set_volume",
                reason="acao direta de sistema",
            ),
        )

        assert "Preciso da sua confirmacao" in response
        assert "system_volume" in response

    async def test_llm_tool_confirmation_returns_structured_payload(self):
        async def set_volume(level: int) -> dict:
            return {"volume": level}

        registry = self.runtime.tool_registry
        registry.register("system.set_volume", "Ajusta volume", set_volume)
        self.runtime.action_broker = ActionBroker(
            CapabilityBroker(
                [
                    Capability(
                        "system.set_volume",
                        enabled=True,
                        requires_confirmation=True,
                        side_effects=["system_volume"],
                    )
                ]
            ),
            registry,
        )

        payload = await self.runtime._invoke_llm_tool(
            "system.set_volume", {"level": 20}
        )

        assert payload == {
            "status": "confirmation_required",
            "tool_name": "system.set_volume",
            "scope": "global",
            "side_effects": ["system_volume"],
            "message": "tool system.set_volume requires explicit confirmation",
        }

    async def test_runtime_tool_events_include_side_effects_and_audit_flags(self):
        subscription = await self.runtime.event_bus.subscribe([EventType.TOOL_EXECUTED])

        async def set_volume(level: int) -> dict:
            return {"volume": level}

        registry = self.runtime.tool_registry
        registry.register(
            "system.set_volume",
            "Ajusta volume",
            set_volume,
            input_schema={
                "type": "object",
                "properties": {"level": {"type": "integer", "minimum": 0}},
                "required": ["level"],
                "additionalProperties": False,
            },
        )
        self.runtime.action_broker = ActionBroker(
            CapabilityBroker(
                [
                    Capability(
                        "system.set_volume",
                        enabled=True,
                        requires_confirmation=True,
                        side_effects=["system_volume"],
                    )
                ]
            ),
            registry,
        )

        result = await self.runtime._execute_action_request(
            ActionRequest(
                tool_name="system.set_volume",
                arguments={"level": 40},
                confirmed=True,
                source="direct_tool",
            ),
            acting_reason="testing",
        )

        assert result.side_effects == ["system_volume"]
        event = await subscription.get()
        assert event.payload["side_effects"] == ["system_volume"]
        assert event.payload["confirmed"]
        assert event.payload["audit_logged"]

    async def test_runtime_events_include_session_turn_and_backend_metadata(self):
        subscription = await self.runtime.event_bus.subscribe(
            [EventType.ROUTE_SELECTED, EventType.ASSISTANT_COMPLETED]
        )

        await self.runtime.respond_text("Que horas sao agora?")

        route_event = await subscription.get()
        completed_event = await subscription.get()

        assert route_event.event_type == EventType.ROUTE_SELECTED
        assert completed_event.event_type == EventType.ASSISTANT_COMPLETED
        assert "session_id" in route_event.payload
        assert (
            route_event.payload["session_id"] == completed_event.payload["session_id"]
        )
        assert "turn_id" in route_event.payload
        assert route_event.payload["turn_id"] == completed_event.payload["turn_id"]
        assert route_event.payload["mode"] == "text"
        assert route_event.payload["backend"] == "system.get_time"
        assert completed_event.payload["backend"] == "system.get_time"
        assert route_event.payload["route"] == RouteTarget.DIRECT_TOOL.value
