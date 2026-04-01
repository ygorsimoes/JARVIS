import array
import importlib
import time
import types
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest

from jarvis.adapters.llm.mlx_lm import MLXLMAdapter
from jarvis.adapters.tts.avspeech import AVSpeechAdapter
from jarvis.adapters.tts.fallback import FallbackTTSAdapter
from jarvis.adapters.tts.mlx_audio_kokoro import MLXAudioKokoroAdapter
from jarvis.models.conversation import Message, Role


@pytest.mark.asyncio
class TestMLXLMAdapter:
    async def test_chat_stream_uses_loaded_model_and_streams_text(self):
        adapter = MLXLMAdapter(
            model_repo="mlx-community/test-model",
            max_tokens=8,
            temperature=0.4,
            top_p=0.8,
            repetition_penalty=1.2,
        )

        class FakeTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt, tokenize):
                self.messages = messages
                self.add_generation_prompt = add_generation_prompt
                self.tokenize = tokenize
                return "PROMPT"

        fake_tokenizer = FakeTokenizer()
        sampler_calls = []
        logits_calls = []

        def fake_load(model_repo):
            return object(), fake_tokenizer

        def fake_stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens,
            max_kv_size,
            sampler=None,
            logits_processors=None,
        ):
            del model, tokenizer, prompt, max_tokens, max_kv_size
            assert sampler == "sampler"
            assert logits_processors == "logits"
            yield types.SimpleNamespace(text="Ola ")
            yield types.SimpleNamespace(
                text="mundo",
                finish_reason="stop",
                prompt_tokens=4,
                generation_tokens=2,
                generation_tps=12.5,
                peak_memory=1.5,
            )

        fake_module = types.SimpleNamespace(
            load=fake_load, stream_generate=fake_stream_generate
        )
        real_import_module = importlib.import_module

        def fake_import_module(name):
            if name == "mlx_lm":
                return fake_module
            if name == "mlx_lm.sample_utils":
                return types.SimpleNamespace(
                    make_sampler=lambda **kwargs: (
                        sampler_calls.append(kwargs) or "sampler"
                    ),
                    make_logits_processors=lambda **kwargs: (
                        logits_calls.append(kwargs) or "logits"
                    ),
                )
            return real_import_module(name)

        with patch(
            "jarvis.adapters.llm.mlx_lm.importlib.import_module",
            side_effect=fake_import_module,
        ):
            with patch(
                "jarvis.adapters.llm.mlx_lm.resolve_cached_model_reference",
                side_effect=lambda model_repo: model_repo,
            ):
                chunks = []
                async for chunk in adapter.chat_stream(
                    messages=[Message(role=Role.USER, content="Oi")],
                    tools=[],
                    max_kv_size=64,
                ):
                    chunks.append(chunk)

        assert chunks == ["Ola ", "mundo"]
        assert fake_tokenizer.messages == [{"role": "user", "content": "Oi"}]
        assert sampler_calls == [{"temp": 0.4, "top_p": 0.8}]
        assert logits_calls == [{"repetition_penalty": 1.2}]
        assert adapter.last_generation_stats["finish_reason"] == "stop"

    async def test_chat_stream_raises_clear_error_when_mlx_lm_is_missing(self):
        adapter = MLXLMAdapter(model_repo="mlx-community/test-model")
        with patch(
            "jarvis.adapters.llm.mlx_lm.resolve_cached_model_reference",
            side_effect=lambda model_repo: model_repo,
        ):
            with patch(
                "jarvis.adapters.llm.mlx_lm.importlib.import_module",
                side_effect=ImportError,
            ):
                with pytest.raises(RuntimeError):
                    async for _ in adapter.chat_stream([], [], 0):
                        pass

    async def test_cancel_current_response_stops_stream_after_current_chunk(self):
        adapter = MLXLMAdapter(model_repo="mlx-community/test-model", max_tokens=8)

        class FakeTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt, tokenize):
                del messages, add_generation_prompt, tokenize
                return "PROMPT"

        def fake_load(model_repo):
            del model_repo
            return object(), FakeTokenizer()

        def fake_stream_generate(model, tokenizer, prompt, max_tokens, max_kv_size):
            del model, tokenizer, prompt, max_tokens, max_kv_size
            yield types.SimpleNamespace(text="Primeiro ")
            time.sleep(0.05)
            yield types.SimpleNamespace(text="Segundo")

        fake_module = types.SimpleNamespace(
            load=fake_load, stream_generate=fake_stream_generate
        )

        with patch(
            "jarvis.adapters.llm.mlx_lm.resolve_cached_model_reference",
            side_effect=lambda model_repo: model_repo,
        ):
            with patch(
                "jarvis.adapters.llm.mlx_lm.importlib.import_module",
                return_value=fake_module,
            ):
                chunks = []

                async def consume():
                    async for chunk in adapter.chat_stream(
                        messages=[Message(role=Role.USER, content="Oi")],
                        tools=[],
                        max_kv_size=64,
                    ):
                        chunks.append(chunk)
                        if len(chunks) == 1:
                            await adapter.cancel_current_response()

                await consume()

        assert chunks == ["Primeiro "]

    async def test_chat_stream_strips_thinking_traces_from_visible_output(self):
        adapter = MLXLMAdapter(model_repo="mlx-community/test-model")

        class FakeTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt, tokenize):
                del messages, add_generation_prompt, tokenize
                return "PROMPT"

        def fake_load(model_repo):
            del model_repo
            return object(), FakeTokenizer()

        def fake_stream_generate(model, tokenizer, prompt, max_tokens, max_kv_size):
            del model, tokenizer, prompt, max_tokens, max_kv_size
            yield types.SimpleNamespace(text="<think>")
            yield types.SimpleNamespace(text="analise interna")
            yield types.SimpleNamespace(text="</think>Resposta limpa")

        fake_module = types.SimpleNamespace(
            load=fake_load, stream_generate=fake_stream_generate
        )

        with patch(
            "jarvis.adapters.llm.mlx_lm.resolve_cached_model_reference",
            side_effect=lambda model_repo: model_repo,
        ):
            with patch(
                "jarvis.adapters.llm.mlx_lm.importlib.import_module",
                return_value=fake_module,
            ):
                chunks = []
                async for chunk in adapter.chat_stream(
                    messages=[Message(role=Role.USER, content="Oi")],
                    tools=[],
                    max_kv_size=64,
                ):
                    chunks.append(chunk)

        assert chunks == ["Resposta limpa"]


@pytest.mark.asyncio
class TestMLXAudioKokoroAdapter:
    async def test_synthesize_stream_uses_loaded_model(self):
        adapter = MLXAudioKokoroAdapter(
            model_repo="mlx-community/Kokoro-82M-bf16",
            voice="pm_santa",
            lang_code="p",
        )

        class FakeAudio:
            def tobytes(self):
                return b"audio-bytes"

        class FakeModel:
            def generate(self, text, voice, lang_code):
                self.last_request = (text, voice, lang_code)
                yield types.SimpleNamespace(audio=FakeAudio())

        fake_model = FakeModel()
        fake_module = types.SimpleNamespace(load_model=lambda model_repo: fake_model)
        with patch.dict(
            "sys.modules",
            {
                "mlx_audio.tts.utils": fake_module,
                "misaki": ModuleType("misaki"),
                "num2words": ModuleType("num2words"),
                "spacy": ModuleType("spacy"),
                "phonemizer": ModuleType("phonemizer"),
                "espeakng_loader": ModuleType("espeakng_loader"),
                "sentencepiece": ModuleType("sentencepiece"),
            },
        ):
            chunks = []
            async for chunk in adapter.synthesize_stream("ola mundo"):
                chunks.append(chunk)

        assert chunks == [b"audio-bytes"]
        assert fake_model.last_request == ("ola mundo", "pm_santa", "p")

    async def test_synthesize_stream_converts_array_like_audio_to_bytes(self):
        adapter = MLXAudioKokoroAdapter(
            model_repo="mlx-community/Kokoro-82M-bf16",
            voice="pm_santa",
            lang_code="p",
        )

        class FakeModel:
            def generate(self, text, voice, lang_code):
                del text, voice, lang_code
                yield types.SimpleNamespace(audio=[0.0, 0.5])

        fake_model = FakeModel()
        fake_module = types.SimpleNamespace(load_model=lambda model_repo: fake_model)
        with patch.dict(
            "sys.modules",
            {
                "mlx_audio.tts.utils": fake_module,
                "misaki": ModuleType("misaki"),
                "num2words": ModuleType("num2words"),
                "spacy": ModuleType("spacy"),
                "phonemizer": ModuleType("phonemizer"),
                "espeakng_loader": ModuleType("espeakng_loader"),
                "sentencepiece": ModuleType("sentencepiece"),
            },
        ):
            chunks = []
            async for chunk in adapter.synthesize_stream("ola mundo"):
                chunks.append(chunk)

        assert chunks == [np.asarray([0.0, 0.5], dtype=np.float32).tobytes()]


@pytest.mark.asyncio
class TestFallbackTTSAdapter:
    async def test_falls_back_to_avspeech_when_primary_fails_before_audio(self):
        primary = MLXAudioKokoroAdapter(
            model_repo="mlx-community/Kokoro-82M-bf16",
            voice="pm_santa",
            lang_code="p",
        )
        fallback = AVSpeechAdapter()
        adapter = FallbackTTSAdapter(
            primary,
            fallback,
            primary_name="mlx_audio_kokoro",
            fallback_name="avspeech",
        )

        with patch.object(
            primary,
            "synthesize_stream",
            side_effect=RuntimeError("tts backend exploded"),
        ):
            with patch.object(fallback, "synthesize_stream") as fallback_stream:

                async def generate(text: str):
                    del text
                    yield b"fallback-audio"

                fallback_stream.side_effect = generate
                chunks = []
                async for chunk in adapter.synthesize_stream("ola mundo"):
                    chunks.append(chunk)

        assert chunks == [b"fallback-audio"]


class TestAVSpeechAdapter:
    @pytest.mark.asyncio
    async def test_synthesize_stream_yields_rendered_audio_bytes(self):
        adapter = AVSpeechAdapter(voice="Luciana", sample_rate_hz=24000)

        with patch.object(adapter, "_render_wav_bytes", return_value=b"pcm-bytes"):
            chunks = []
            async for chunk in adapter.synthesize_stream("ola mundo"):
                chunks.append(chunk)

        assert chunks == [b"pcm-bytes"]

    @pytest.mark.asyncio
    async def test_cancel_current_synthesis_terminates_active_process(self):
        adapter = AVSpeechAdapter()
        events = []

        class FakeProcess:
            def __init__(self):
                self.returncode = None

            def terminate(self):
                events.append("terminate")
                self.returncode = 0

            def kill(self):
                events.append("kill")
                self.returncode = -9

            async def wait(self):
                events.append("wait")
                return self.returncode

        adapter.__dict__["_active_process"] = FakeProcess()
        cancelled = await adapter.cancel_current_synthesis()

        assert cancelled
        assert events == ["terminate", "wait"]
        assert adapter._active_process is None

    def test_read_wav_as_float32_converts_pcm16_to_float_bytes(self):
        adapter = AVSpeechAdapter(sample_rate_hz=24000)
        samples = array.array("h", [0, 16384, -16384])

        with patch("wave.open") as wave_open:
            handle = wave_open.return_value.__enter__.return_value
            handle.readframes.return_value = samples.tobytes()
            handle.getnframes.return_value = 3
            handle.getsampwidth.return_value = 2
            handle.getnchannels.return_value = 1

            audio_bytes = adapter._read_wav_as_float32(Path("dummy.wav"))

        floats = array.array("f")
        floats.frombytes(audio_bytes)
        assert len(floats) == 3
        assert abs(floats[1] - 0.5) < 1e-3
        assert abs(floats[2] - (-0.5)) < 1e-3
