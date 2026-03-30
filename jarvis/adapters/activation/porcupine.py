import asyncio
import os
import struct
from typing import Optional

class PorcupineActivationAdapter:
    def __init__(self, keyword: str = "jarvis") -> None:
        self.keyword = keyword
        self._porcupine = None
        self._audio_stream = None
        self._pa = None

    def _ensure_initialized(self):
        if self._porcupine is not None:
            return

        try:
            import pvporcupine
            import pyaudio
        except ImportError as exc:
            raise RuntimeError("pvporcupine e/ou pyaudio nao instalados. Instale-os com 'pip install pvporcupine pyaudio'.") from exc

        access_key = os.environ.get("PORCUPINE_ACCESS_KEY")
        if not access_key:
            raise RuntimeError("A variavel de ambiente PORCUPINE_ACCESS_KEY nao esta definida.")

        self._porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=[self.keyword]
        )

        self._pa = pyaudio.PyAudio()
        self._audio_stream = self._pa.open(
            rate=self._porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self._porcupine.frame_length,
            start=False 
        )

    async def listen(self) -> bool:
        """Bloqueia e retorna True quando a wake word eh detectada."""
        self._ensure_initialized()
        self._audio_stream.start_stream()

        try:
            while True:
                # Usa to_thread para evitar bloquear o event loop enquanto leimos o audio via bloqueio do OS
                pcm = await asyncio.to_thread(self._audio_stream.read, self._porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * self._porcupine.frame_length, pcm)

                result = self._porcupine.process(pcm)
                if result >= 0:
                    return True
        finally:
            self._audio_stream.stop_stream()

    async def shutdown(self) -> None:
        if self._audio_stream is not None:
            self._audio_stream.close()
            self._audio_stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        if self._porcupine is not None:
            self._porcupine.delete()
            self._porcupine = None
