import asyncio
import unittest
from typing import Any, cast
from unittest.mock import patch

from jarvis.core.action_broker import ActionBroker, ActionRequest
from jarvis.core.capability_broker import Capability, CapabilityBroker
from jarvis.config import JarvisConfig
from jarvis.models.conversation import Role, RouteDecision, RouteTarget
from jarvis.models.events import EventType
from jarvis.models.memory import Memory, MemoryCategory, MemorySource
from jarvis.models.state import JarvisState
from jarvis.runtime import JarvisRuntime


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


class FakeSTTAdapter:
    def __init__(self, events):
        self.events = events

    async def start_live_session(self):
        session_events = self.events
        self.events = []  # O watcher não ver eventos
        return FakeSTTSession(session_events)


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


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = JarvisRuntime.from_config(
            JarvisConfig(allowed_file_roots=["/tmp"]),
            enable_native_backends=False,
        )

    async def test_direct_tool_response_for_time(self):
        response = await self.runtime.respond_text("Que horas sao agora?")
        self.assertEqual(response.route.target, RouteTarget.DIRECT_TOOL)
        self.assertIn("Agora sao", response.full_text)

    async def test_deliberative_response_uses_streaming_adapter(self):
        response = await self.runtime.respond_text(
            "Analisa esse erro e explica por que ele quebra"
        )
        self.assertEqual(response.route.target, RouteTarget.DELIBERATIVE)
        self.assertGreaterEqual(len(response.sentences), 2)
        self.assertIn("Analisei o pedido", response.full_text)

    async def test_stream_text_turn_emits_incremental_sentences(self):
        chunks = []
        async for chunk in self.runtime.stream_text_turn(
            "Analisa esse erro e explica por que ele quebra"
        ):
            chunks.append(chunk)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[0].sentence.endswith("."))
        self.assertEqual(chunks[0].index, 0)
        self.assertEqual(chunks[1].index, 1)

    async def test_browser_route_formats_response(self):
        with patch("webbrowser.open", return_value=True):
            response = await self.runtime.respond_text("Pesquise arquitetura hexagonal")
        self.assertEqual(response.route.tool_name, "browser.search")
        self.assertIn("Busca aberta", response.full_text)

    async def test_runtime_recovers_to_idle_after_error(self):
        with self.assertRaises(ValueError):
            await self.runtime.respond_text("Define um timer")
        self.assertEqual(self.runtime.state_machine.state.value, "idle")

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

        self.assertEqual(turn.text, "Que horas sao agora?")
        self.assertEqual(self.runtime.state_machine.state, JarvisState.TRANSCRIBING)

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
        self.assertTrue(any(line.startswith("voce>") for line in printed_lines))
        self.assertTrue(any(line.startswith("jarvis>") for line in printed_lines))
        self.assertTrue(self.runtime.playback_backend.played)
        self.assertEqual(self.runtime.state_machine.state, JarvisState.IDLE)

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

        self.assertTrue(interrupted)
        self.assertTrue(self.runtime.hot_path_adapter.cancelled)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(self.runtime.state_machine.state, JarvisState.IDLE)

    async def test_hot_path_can_execute_llm_requested_tool(self):
        self.runtime.hot_path_adapter = ToolCallingHotPathAdapter()

        response = await self.runtime.respond_text("Me lembra do contexto")

        self.assertEqual(response.route.target, RouteTarget.HOT_PATH)
        self.assertIn("Agora sao", response.full_text)
        self.assertEqual(self.runtime.state_machine.state, JarvisState.IDLE)

    async def test_current_user_turn_is_not_duplicated_in_prompt_context(self):
        self.runtime.hot_path_adapter = CapturingHotPathAdapter()

        await self.runtime.respond_text("Me lembra do resumo")

        user_messages = [
            message.content
            for message in (self.runtime.hot_path_adapter.messages or [])
            if message.role == Role.USER
        ]
        self.assertEqual(user_messages, ["Me lembra do resumo"])

    async def test_runtime_only_exposes_capability_enabled_tools_to_llm(self):
        self.runtime.hot_path_adapter = CapturingHotPathAdapter()

        await self.runtime.respond_text("Me lembra do resumo")

        tool_names = [
            tool["name"] for tool in (self.runtime.hot_path_adapter.tools or [])
        ]
        self.assertIn("browser.fetch_url", tool_names)
        self.assertIn("calendar.list_events", tool_names)
        self.assertIn("files.list", tool_names)
        self.assertIn("files.read", tool_names)
        self.assertNotIn("shell.execute", tool_names)
        self.assertNotIn("files.move", tool_names)

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
        self.assertEqual(
            memory_system.calls,
            [("Me lembra do resumo", RouteTarget.HOT_PATH, None)],
        )
        memory_messages = [
            message.content
            for message in (self.runtime.hot_path_adapter.messages or [])
            if message.role == Role.SYSTEM and "Memorias relevantes" in message.content
        ]
        self.assertEqual(len(memory_messages), 1)
        self.assertIn("Usuario prefere respostas curtas", memory_messages[0])

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
        self.assertEqual(
            memory_system.calls,
            [
                (
                    "Analisa esse erro e explica por que ele quebra",
                    RouteTarget.DELIBERATIVE,
                    None,
                )
            ],
        )
        memory_messages = [
            message.content
            for message in (capturing_adapter.messages or [])
            if message.role == Role.SYSTEM and "Memorias relevantes" in message.content
        ]
        self.assertEqual(len(memory_messages), 1)
        self.assertIn("project:jarvis", memory_messages[0])

    async def test_runtime_skips_memory_recall_for_direct_tools(self):
        self.runtime.memory_system = cast(Any, FakeMemorySystem())

        await self.runtime.respond_text("Que horas sao agora?")

        memory_system = cast(Any, self.runtime.memory_system)
        self.assertEqual(memory_system.calls, [])

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

        self.assertIn("Preciso da sua confirmacao", response)
        self.assertIn("system_volume", response)

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

        self.assertEqual(
            payload,
            {
                "status": "confirmation_required",
                "tool_name": "system.set_volume",
                "scope": "global",
                "side_effects": ["system_volume"],
                "message": "tool system.set_volume requires explicit confirmation",
            },
        )

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

        self.assertEqual(result.side_effects, ["system_volume"])
        event = await subscription.get()
        self.assertEqual(event.payload["side_effects"], ["system_volume"])
        self.assertTrue(event.payload["confirmed"])
        self.assertTrue(event.payload["audit_logged"])

    async def test_runtime_events_include_session_turn_and_backend_metadata(self):
        subscription = await self.runtime.event_bus.subscribe(
            [EventType.ROUTE_SELECTED, EventType.ASSISTANT_COMPLETED]
        )

        await self.runtime.respond_text("Que horas sao agora?")

        route_event = await subscription.get()
        completed_event = await subscription.get()

        self.assertEqual(route_event.event_type, EventType.ROUTE_SELECTED)
        self.assertEqual(completed_event.event_type, EventType.ASSISTANT_COMPLETED)
        self.assertIn("session_id", route_event.payload)
        self.assertEqual(
            route_event.payload["session_id"], completed_event.payload["session_id"]
        )
        self.assertIn("turn_id", route_event.payload)
        self.assertEqual(
            route_event.payload["turn_id"], completed_event.payload["turn_id"]
        )
        self.assertEqual(route_event.payload["mode"], "text")
        self.assertEqual(route_event.payload["backend"], "system.get_time")
        self.assertEqual(completed_event.payload["backend"], "system.get_time")
        self.assertEqual(route_event.payload["route"], RouteTarget.DIRECT_TOOL.value)


if __name__ == "__main__":
    unittest.main()
