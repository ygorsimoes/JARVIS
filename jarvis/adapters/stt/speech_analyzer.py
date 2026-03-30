from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Deque, Dict, Optional


class SpeechAnalyzerStreamError(RuntimeError):
    pass


@dataclass
class SpeechAnalyzerSession:
    process: asyncio.subprocess.Process
    locale: str
    backend_name: str = "speech_analyzer"
    stderr_lines: Deque[str] = field(default_factory=lambda: deque(maxlen=20))
    stderr_task: Optional[asyncio.Task] = None

    async def iter_events(self) -> AsyncIterator[Dict[str, Any]]:
        sequence = 0
        assert self.process.stdout is not None
        async for raw_line in self.process.stdout:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise SpeechAnalyzerStreamError(
                    "SpeechAnalyzer CLI emitted an invalid event payload"
                )
            yield self._normalize_event(event, sequence=sequence)
            sequence += 1

        return_code = await self.process.wait()
        if return_code not in {0, None}:
            raise SpeechAnalyzerStreamError(self.stderr_summary())

    async def stop(self) -> None:
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()

        if self.stderr_task is not None:
            self.stderr_task.cancel()
            try:
                await self.stderr_task
            except asyncio.CancelledError:
                pass

    def stderr_summary(self) -> str:
        if not self.stderr_lines:
            return "SpeechAnalyzer CLI failed without stderr output"
        return " ".join(self.stderr_lines)

    def _normalize_event(self, event: Dict[str, Any], sequence: int) -> Dict[str, Any]:
        normalized = dict(event)
        normalized.setdefault("locale", self.locale)
        normalized.setdefault("stt_backend", self.backend_name)
        normalized["sequence"] = sequence
        normalized["received_at_monotonic"] = time.monotonic()
        if normalized.get("type") == "speech_detector_result":
            normalized["speech_detected"] = bool(normalized.get("speech_detected"))
        return normalized


class SpeechAnalyzerSTTAdapter:
    def __init__(self, binary_path: str, locale: str = "pt-BR") -> None:
        self.binary_path = binary_path
        self.locale = locale

    async def start_live_session(self) -> SpeechAnalyzerSession:
        process = await asyncio.create_subprocess_exec(
            self.binary_path,
            "--live",
            "--locale",
            self.locale,
            "--format",
            "ndjson",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        session = SpeechAnalyzerSession(process=process, locale=self.locale)
        session.stderr_task = asyncio.create_task(self._drain_stderr(session))
        return session

    async def iter_events(self) -> AsyncIterator[Dict[str, Any]]:
        session = await self.start_live_session()
        try:
            async for event in session.iter_events():
                yield event
        finally:
            await session.stop()

    async def transcribe_stream(self) -> AsyncIterator[str]:
        async for event in self.iter_events():
            if event.get("type") == "final_transcript":
                yield event.get("text", "")

    async def _drain_stderr(self, session: SpeechAnalyzerSession) -> None:
        if session.process.stderr is None:
            return
        async for raw_line in session.process.stderr:
            line = raw_line.decode("utf-8").strip()
            if line:
                session.stderr_lines.append(line)

    async def shutdown(self) -> None:
        await asyncio.sleep(0)
