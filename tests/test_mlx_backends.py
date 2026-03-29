import types
import unittest
from unittest.mock import patch

from jarvis.adapters.llm.mlx_lm import MLXLMAdapter
from jarvis.adapters.tts.mlx_audio_kokoro import MLXAudioKokoroAdapter
from jarvis.models.conversation import Message, Role


class MLXLMAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_stream_uses_loaded_model_and_streams_text(self):
        adapter = MLXLMAdapter(model_repo="mlx-community/test-model", max_tokens=8)

        class FakeTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt, tokenize):
                self.messages = messages
                self.add_generation_prompt = add_generation_prompt
                self.tokenize = tokenize
                return "PROMPT"

        fake_tokenizer = FakeTokenizer()

        def fake_load(model_repo):
            return object(), fake_tokenizer

        def fake_stream_generate(model, tokenizer, prompt, max_tokens, max_kv_size):
            del model, tokenizer, prompt, max_tokens, max_kv_size
            yield types.SimpleNamespace(text="Ola ")
            yield types.SimpleNamespace(text="mundo")

        fake_module = types.SimpleNamespace(
            load=fake_load, stream_generate=fake_stream_generate
        )
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

        self.assertEqual(chunks, ["Ola ", "mundo"])
        self.assertEqual(fake_tokenizer.messages, [{"role": "user", "content": "Oi"}])

    async def test_chat_stream_raises_clear_error_when_mlx_lm_is_missing(self):
        adapter = MLXLMAdapter(model_repo="mlx-community/test-model")
        with patch(
            "jarvis.adapters.llm.mlx_lm.importlib.import_module",
            side_effect=ImportError,
        ):
            with self.assertRaises(RuntimeError):
                async for _ in adapter.chat_stream([], [], 0):
                    pass


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
