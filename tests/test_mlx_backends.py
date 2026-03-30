import asyncio
import time
import types
import unittest
from unittest.mock import patch

from jarvis.adapters.llm.mlx_lm import MLXLMAdapter
from jarvis.adapters.tts.mlx_audio_kokoro import MLXAudioKokoroAdapter
from jarvis.models.conversation import Message, Role


class MLXLMAdapterTests(unittest.IsolatedAsyncioTestCase):
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
            self.assertEqual(sampler, "sampler")
            self.assertEqual(logits_processors, "logits")
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
            raise ImportError(name)

        with patch(
            "jarvis.adapters.llm.mlx_lm.importlib.import_module",
            side_effect=fake_import_module,
        ):
            chunks = []
            async for chunk in adapter.chat_stream(
                messages=[Message(role=Role.USER, content="Oi")],
                tools=[],
                max_kv_size=64,
            ):
                chunks.append(chunk)

        self.assertEqual(chunks, ["Ola ", "mundo"])
        self.assertEqual(fake_tokenizer.messages, [{"role": "user", "content": "Oi"}])
        self.assertEqual(sampler_calls, [{"temp": 0.4, "top_p": 0.8}])
        self.assertEqual(logits_calls, [{"repetition_penalty": 1.2}])
        self.assertEqual(adapter.last_generation_stats["finish_reason"], "stop")

    async def test_chat_stream_raises_clear_error_when_mlx_lm_is_missing(self):
        adapter = MLXLMAdapter(model_repo="mlx-community/test-model")
        with patch(
            "jarvis.adapters.llm.mlx_lm.importlib.import_module",
            side_effect=ImportError,
        ):
            with self.assertRaises(RuntimeError):
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

        self.assertEqual(chunks, ["Primeiro "])


class MLXAudioKokoroAdapterTests(unittest.IsolatedAsyncioTestCase):
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
        with patch.dict("sys.modules", {"mlx_audio.tts.utils": fake_module}):
            chunks = []
            async for chunk in adapter.synthesize_stream("ola mundo"):
                chunks.append(chunk)

        self.assertEqual(chunks, [b"audio-bytes"])
        self.assertEqual(fake_model.last_request, ("ola mundo", "pm_santa", "p"))


if __name__ == "__main__":
    unittest.main()
