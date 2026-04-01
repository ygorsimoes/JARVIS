from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator


class MLXAudioQwen3Adapter:
    def __init__(self, model_repo: str, speaker_id: str) -> None:
        self.model_repo = model_repo
        self.speaker_id = speaker_id
        self._model = None
        self._cancel_event = threading.Event()

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        self._cancel_event.clear()

        # Load the model in a thread if not loaded yet
        model = await asyncio.to_thread(self._ensure_model)

        def _generate():
            # Simulando iterator de mlx_audio generator pro Qwen3
            return model.generate(text, speaker=self.speaker_id)

        generator = await asyncio.to_thread(_generate)

        for result in generator:
            if self._cancel_event.is_set():
                break

            # Assume que retorna fragmentos de audio da mesma forma
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
        if self._model is not None:
            return self._model
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise RuntimeError(
                "mlx-audio is not installed. Install the 'macos-ml' dependency extra to enable Qwen3 TTS."
            ) from exc

        self._model = load_model(self.model_repo)
        return self._model
