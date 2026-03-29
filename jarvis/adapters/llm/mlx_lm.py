from __future__ import annotations

import asyncio
import importlib
import threading
from typing import Any, AsyncIterator, Callable, List

from ...models.conversation import Message


class MLXLMAdapter:
    def __init__(self, model_repo: str, max_tokens: int = 512) -> None:
        self.model_repo = model_repo
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None
        self._load: Callable[..., Any] | None = None
        self._stream_generate: Callable[..., Any] | None = None

    async def chat_stream(
        self,
        messages: List[Message],
        tools: List[dict],
        max_kv_size: int,
        tool_invoker=None,
    ) -> AsyncIterator[str]:
        del tools, tool_invoker
        model, tokenizer, stream_generate = await asyncio.to_thread(
            self._ensure_model_loaded
        )
        prompt = tokenizer.apply_chat_template(
            [self._serialize_message(message) for message in messages],
            add_generation_prompt=True,
            tokenize=False,
        )

        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                for response in stream_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=self.max_tokens,
                    max_kv_size=max_kv_size,
                ):
                    text = getattr(response, "text", "")
                    if text:
                        asyncio.run_coroutine_threadsafe(queue.put(text), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def cancel_current_response(self) -> bool:
        return False

    async def close_session(self) -> None:
        return None

    async def shutdown(self) -> None:
        await asyncio.sleep(0)

    def _ensure_model_loaded(self) -> tuple[object, object, Callable[..., Any]]:
        if self._model is not None and self._tokenizer is not None:
            assert self._stream_generate is not None
            return self._model, self._tokenizer, self._stream_generate
        try:
            module = importlib.import_module("mlx_lm")
        except ImportError as exc:
            raise RuntimeError(
                "mlx-lm nao esta instalado. Instale o extra macos-ml para usar o backend deliberativo real."
            ) from exc
        load = getattr(module, "load")
        stream_generate = getattr(module, "stream_generate")
        self._model, self._tokenizer = load(self.model_repo)
        self._load = load
        self._stream_generate = stream_generate
        assert self._stream_generate is not None
        return self._model, self._tokenizer, self._stream_generate

    @staticmethod
    def _serialize_message(message: Message) -> dict:
        return {"role": message.role.value, "content": message.content}
