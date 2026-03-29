import asyncio
import unittest
from unittest.mock import patch

from jarvis.config import JarvisConfig
from jarvis.models.conversation import RouteTarget
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
        self.session = FakeSTTSession(events)

    async def start_live_session(self):
        return self.session


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


if __name__ == "__main__":
    unittest.main()
