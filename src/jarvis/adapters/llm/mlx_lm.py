from __future__ import annotations

import asyncio
import importlib
import threading
from typing import Any, AsyncIterator, Callable, List

from ...models.conversation import Message


class MLXLMAdapter:
    def __init__(
        self,
        model_repo: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> None:
        self.model_repo = model_repo
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self._model = None
        self._tokenizer = None
        self._load: Callable[..., Any] | None = None
        self._stream_generate: Callable[..., Any] | None = None
        self._cancel_event = threading.Event()
        self._generation_lock = threading.Lock()
        self._generation_active = False
        self._last_generation_stats: dict[str, object] = {}

    @property
    def last_generation_stats(self) -> dict[str, object]:
        return dict(self._last_generation_stats)

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
        tokenizer_impl: Any = tokenizer
        prompt = tokenizer_impl.apply_chat_template(
            [self._serialize_message(message) for message in messages],
            add_generation_prompt=True,
            tokenize=False,
        )

        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        generation_options = await asyncio.to_thread(
            self._build_generation_options, tokenizer_impl, max_kv_size
        )
        self._cancel_event.clear()
        self._last_generation_stats = {}

        def worker() -> None:
            with self._generation_lock:
                self._generation_active = True
            try:
                for response in stream_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=self.max_tokens,
                    **generation_options,
                ):
                    if self._cancel_event.is_set():
                        break
                    text = getattr(response, "text", "")
                    if text:
                        asyncio.run_coroutine_threadsafe(queue.put(text), loop)
                    finish_reason = getattr(response, "finish_reason", None)
                    if finish_reason:
                        self._last_generation_stats = {
                            "finish_reason": finish_reason,
                            "prompt_tokens": getattr(response, "prompt_tokens", None),
                            "generation_tokens": getattr(
                                response, "generation_tokens", None
                            ),
                            "generation_tps": getattr(response, "generation_tps", None),
                            "peak_memory": getattr(response, "peak_memory", None),
                        }
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
            finally:
                with self._generation_lock:
                    self._generation_active = False
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
        with self._generation_lock:
            was_active = self._generation_active
        self._cancel_event.set()
        return was_active

    async def close_session(self) -> None:
        self._cancel_event.set()
        return None

    async def shutdown(self) -> None:
        self._cancel_event.set()
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

    def _build_generation_options(
        self, tokenizer: object, max_kv_size: int
    ) -> dict[str, object]:
        options: dict[str, object] = {"max_kv_size": max_kv_size}
        sampler = self._build_sampler(tokenizer)
        if sampler is not None:
            options["sampler"] = sampler
        logits_processors = self._build_logits_processors(tokenizer)
        if logits_processors is not None:
            options["logits_processors"] = logits_processors
        return options

    def _build_sampler(self, tokenizer: object) -> object | None:
        del tokenizer
        try:
            sample_utils = importlib.import_module("mlx_lm.sample_utils")
        except ImportError:
            return None
        make_sampler = getattr(sample_utils, "make_sampler", None)
        if make_sampler is None:
            return None
        return make_sampler(temp=self.temperature, top_p=self.top_p)

    def _build_logits_processors(self, tokenizer: object) -> object | None:
        del tokenizer
        if self.repetition_penalty <= 1.0:
            return None
        try:
            sample_utils = importlib.import_module("mlx_lm.sample_utils")
        except ImportError:
            return None
        make_logits_processors = getattr(sample_utils, "make_logits_processors", None)
        if make_logits_processors is None:
            return None
        return make_logits_processors(repetition_penalty=self.repetition_penalty)

    @staticmethod
    def _serialize_message(message: Message) -> dict:
        return {"role": message.role.value, "content": message.content}
