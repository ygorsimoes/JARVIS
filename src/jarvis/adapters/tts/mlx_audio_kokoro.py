from __future__ import annotations

import asyncio
import threading
from typing import Any, AsyncIterator, cast


class MLXAudioKokoroAdapter:
    def __init__(self, model_repo: str, voice: str, lang_code: str) -> None:
        self.model_repo = model_repo
        self.voice = voice
        self.lang_code = lang_code
        self._model: Any | None = None
        self._load_error: RuntimeError | None = None
        self._cancel_event = threading.Event()

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        self._cancel_event.clear()
        model = await asyncio.to_thread(self._ensure_model)
        assert model is not None
        model = cast(Any, model)
        for result in model.generate(text, voice=self.voice, lang_code=self.lang_code):
            if self._cancel_event.is_set():
                break
            audio = getattr(result, "audio", None)
            if audio is None:
                continue
            yield audio.tobytes()

    async def cancel_current_synthesis(self) -> bool:
        self._cancel_event.set()
        return True

    async def shutdown(self) -> None:
        await asyncio.sleep(0)

    def _ensure_model(self):
        if self._load_error is not None:
            raise self._load_error
        if self._model is not None:
            return self._model
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            self._load_error = RuntimeError(
                "mlx-audio is not installed. Install the 'macos-ml' dependency extra to enable Kokoro TTS."
            )
            raise self._load_error from exc

        for dependency in ("misaki", "num2words", "spacy"):
            try:
                __import__(dependency)
            except ImportError as exc:
                self._load_error = RuntimeError(
                    "mlx-audio Kokoro requires the Python dependency '%s'." % dependency
                )
                raise self._load_error from exc

        load_model_any = cast(Any, load_model)
        self._model = load_model_any(self.model_repo)  # pyright: ignore[reportArgumentType]
        return self._model
