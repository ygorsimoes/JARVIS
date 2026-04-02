from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional


@dataclass(frozen=True)
class SentenceStreamerConfig:
    min_dispatch_tokens: int = 8
    min_soft_boundary_chars: int = 40
    clause_fallback_chars: int = 72
    max_buffer_chars: int = 160
    hard_split_min_tokens: int = 18
    max_pending_segments: int = 2
    backpressure_poll_interval_s: float = 0.01


class SentenceStreamer:
    HARD_ENDS = (".", "!", "?")
    SOFT_ENDS = (":", "\n")
    COMMON_ABBREVIATIONS = (
        "sr.",
        "sra.",
        "dr.",
        "dra.",
        "prof.",
        "etc.",
        "vs.",
        "ex.",
    )
    CLAUSE_BREAKS = (", ", "; ", "\n")

    def __init__(self, config: Optional[SentenceStreamerConfig] = None) -> None:
        self.config = config or SentenceStreamerConfig()
        self._buffer = ""

    def push_text(self, text: str) -> List[str]:
        self._buffer += text
        ready: List[str] = []
        while True:
            boundary_index = self._find_dispatch_boundary()
            if boundary_index is None:
                break
            sentence = self._buffer[:boundary_index].strip()
            self._buffer = self._buffer[boundary_index:].lstrip()
            if sentence:
                ready.append(sentence)
        return ready

    def flush(self) -> List[str]:
        trailing = self._buffer.strip()
        self._buffer = ""
        return [trailing] if trailing else []

    async def pump(
        self, token_stream: AsyncIterator[str], sink_queue: asyncio.Queue
    ) -> None:
        try:
            async for chunk in token_stream:
                for sentence in self.push_text(chunk):
                    while sink_queue.qsize() > self.config.max_pending_segments:
                        await asyncio.sleep(self.config.backpressure_poll_interval_s)
                    await sink_queue.put(sentence)
            for sentence in self.flush():
                await sink_queue.put(sentence)
        except Exception as exc:
            await sink_queue.put(exc)
        finally:
            await sink_queue.put(None)

    async def collect(self, token_stream: AsyncIterator[str]) -> List[str]:
        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(self.pump(token_stream, queue))
        sentences: List[str] = []
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            sentences.append(item)
        await task
        return sentences

    def _find_dispatch_boundary(self) -> Optional[int]:
        stripped_buffer = self._buffer.rstrip()
        if not stripped_buffer:
            return None

        for index, character in enumerate(self._buffer):
            if character not in self.HARD_ENDS + self.SOFT_ENDS:
                continue
            candidate = self._buffer[: index + 1].strip()
            if not candidate:
                continue
            token_count = self._token_count(candidate)
            long_enough = token_count >= self.config.min_dispatch_tokens
            if (
                character in self.HARD_ENDS
                and long_enough
                and self._is_terminal_boundary(index)
            ):
                return index + 1
            if (
                character in self.SOFT_ENDS
                and long_enough
                and len(candidate) >= self.config.min_soft_boundary_chars
            ):
                return index + 1

        if len(stripped_buffer) >= self.config.max_buffer_chars:
            return self._find_clause_fallback_boundary()
        return None

    def _is_terminal_boundary(self, index: int) -> bool:
        if index < 0 or index >= len(self._buffer):
            return False
        character = self._buffer[index]
        if character != ".":
            return True
        if 0 < index < len(self._buffer) - 1:
            previous = self._buffer[index - 1]
            following = self._buffer[index + 1]
            if previous.isdigit() and following.isdigit():
                return False
        lowered_prefix = self._buffer[: index + 1].lower().rstrip()
        return not any(
            lowered_prefix.endswith(item) for item in self.COMMON_ABBREVIATIONS
        )

    def _find_clause_fallback_boundary(self) -> Optional[int]:
        upper_bound = min(len(self._buffer), self.config.max_buffer_chars)
        for separator in self.CLAUSE_BREAKS:
            index = self._buffer.rfind(
                separator,
                self.config.clause_fallback_chars,
                upper_bound,
            )
            if index == -1:
                continue
            boundary = index + len(separator.rstrip())
            candidate = self._buffer[:boundary].strip()
            if self._token_count(candidate) >= self.config.min_dispatch_tokens:
                return boundary

        if self._token_count(self._buffer) < self.config.hard_split_min_tokens:
            return None

        boundary = self._buffer.rfind(
            " ",
            self.config.clause_fallback_chars,
            upper_bound,
        )
        if boundary == -1:
            return None
        candidate = self._buffer[:boundary].strip()
        if self._token_count(candidate) < self.config.min_dispatch_tokens:
            return None
        return boundary

    @staticmethod
    def _token_count(text: str) -> int:
        return len(re.findall(r"\S+", text))
