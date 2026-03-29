from __future__ import annotations

import asyncio
from typing import AsyncIterator


class MLXAudioKokoroAdapter:
    def __init__(self, model_repo: str, voice: str, lang_code: str) -> None:
        self.model_repo = model_repo
        self.voice = voice
        self.lang_code = lang_code
        self._model = None

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        model = await asyncio.to_thread(self._ensure_model)
        for result in model.generate(text, voice=self.voice, lang_code=self.lang_code):
            audio = getattr(result, "audio", None)
            if audio is None:
                continue
            yield audio.tobytes()

    async def shutdown(self) -> None:
        await asyncio.sleep(0)

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise RuntimeError(
                "mlx-audio is not installed. Install the 'macos-ml' dependency extra to enable Kokoro TTS."
            ) from exc
        self._model = load_model(self.model_repo)
        return self._model
