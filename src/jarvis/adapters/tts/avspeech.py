from __future__ import annotations

import array
import asyncio
import tempfile
import wave
from pathlib import Path


class AVSpeechAdapter:
    def __init__(
        self,
        voice: str | None = "Luciana",
        sample_rate_hz: int = 24000,
        rate: int = 175,
    ) -> None:
        self.voice = voice
        self.sample_rate_hz = sample_rate_hz
        self.rate = rate
        self._active_process: asyncio.subprocess.Process | None = None

    async def synthesize_stream(self, text: str):
        audio_bytes = await self._render_wav_bytes(text)
        if audio_bytes:
            yield audio_bytes

    async def cancel_current_synthesis(self) -> bool:
        process = self._active_process
        if process is None or process.returncode is not None:
            return False
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        finally:
            if self._active_process is process:
                self._active_process = None
        return True

    async def shutdown(self) -> None:
        await self.cancel_current_synthesis()

    async def _render_wav_bytes(self, text: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="jarvis-avspeech-") as temp_dir:
            aiff_path = Path(temp_dir) / "utterance.aiff"
            wav_path = Path(temp_dir) / "utterance.wav"
            await self._run_command(*self._say_command(text, aiff_path))
            await self._run_command(*self._afconvert_command(aiff_path, wav_path))
            return await asyncio.to_thread(self._read_wav_as_float32, wav_path)

    def _say_command(self, text: str, output_path: Path) -> tuple[str, ...]:
        command = ["say"]
        if self.voice:
            command.extend(["-v", self.voice])
        command.extend(["-r", str(self.rate), "-o", str(output_path), text])
        return tuple(command)

    def _afconvert_command(
        self, input_path: Path, output_path: Path
    ) -> tuple[str, ...]:
        return (
            "afconvert",
            "-f",
            "WAVE",
            "-d",
            "LEI16@%d" % self.sample_rate_hz,
            str(input_path),
            str(output_path),
        )

    async def _run_command(self, *command: str) -> None:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self._active_process = process
        try:
            _, stderr = await process.communicate()
        except asyncio.CancelledError:
            await self.cancel_current_synthesis()
            raise
        finally:
            if self._active_process is process:
                self._active_process = None
        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="ignore").strip()
            if not message:
                message = "%s failed with exit code %s" % (
                    command[0],
                    process.returncode,
                )
            raise RuntimeError(message)

    def _read_wav_as_float32(self, wav_path: Path) -> bytes:
        with wave.open(str(wav_path), "rb") as handle:
            frames = handle.readframes(handle.getnframes())
            sample_width = handle.getsampwidth()
            channel_count = handle.getnchannels()
        if sample_width != 2:
            raise RuntimeError(
                "AVSpeech fallback expected 16-bit WAV output, got %d bytes"
                % sample_width
            )

        samples = array.array("h")
        samples.frombytes(frames)
        if channel_count > 1:
            mono_samples = array.array("h")
            for index in range(0, len(samples), channel_count):
                chunk = samples[index : index + channel_count]
                mono_samples.append(int(sum(chunk) / len(chunk)))
            samples = mono_samples

        normalized = array.array("f", (sample / 32768.0 for sample in samples))
        return normalized.tobytes()
