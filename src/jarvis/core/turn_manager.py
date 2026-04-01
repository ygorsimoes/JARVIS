from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TurnManagerConfig:
    silence_timeout_ms: int = 800
    partial_commit_min_chars: int = 16
    partial_stability_ms: int = 250
    tick_interval_ms: int = 100
    max_turn_duration_s: float = 30.0

    @property
    def silence_timeout_s(self) -> float:
        return self.silence_timeout_ms / 1000.0

    @property
    def partial_stability_s(self) -> float:
        return self.partial_stability_ms / 1000.0


@dataclass
class CompletedTurn:
    text: str
    reason: str
    used_partial: bool
    started_at_monotonic: float
    ended_at_monotonic: float
    metadata: dict = field(default_factory=dict)


class TurnManager:
    INCOMPLETE_SUFFIXES = (
        " e",
        " ou",
        " mas",
        " porque",
        " quando",
        " enquanto",
        " se",
        " que",
        ",",
        ":",
        "...",
    )

    def __init__(self, config: Optional[TurnManagerConfig] = None) -> None:
        self.config = config or TurnManagerConfig()
        self.reset()

    def reset(self) -> None:
        self.speech_active = False
        self.partial_text = ""
        self.final_text = ""
        self.started_at_monotonic: Optional[float] = None
        self.silence_started_at_monotonic: Optional[float] = None
        self.last_event_at_monotonic: Optional[float] = None
        self.last_partial_at_monotonic: Optional[float] = None

    def consume_event(
        self, event: dict, now: Optional[float] = None
    ) -> Optional[CompletedTurn]:
        now = now if now is not None else time.monotonic()
        self.last_event_at_monotonic = now
        event_type = event.get("type")

        if event_type == "speech_started":
            return self.consume_vad_signal(True, now=now, reason="speech_started")

        if event_type == "speech_ended":
            return self.consume_vad_signal(False, now=now, reason="speech_ended")

        if event_type == "partial_transcript":
            return self.consume_partial_transcript(event.get("text", ""), now=now)

        if event_type == "final_transcript":
            return self.consume_final_transcript(event.get("text", ""), now=now)

        if event_type == "speech_detector_result":
            return self.consume_vad_signal(
                bool(event.get("speech_detected")),
                now=now,
                reason="speech_detector_result",
            )

        return self._maybe_complete_for_max_duration(now)

    def consume_vad_signal(
        self,
        speech_detected: bool,
        now: Optional[float] = None,
        reason: str = "vad_signal",
    ) -> Optional[CompletedTurn]:
        now = now if now is not None else time.monotonic()
        self.last_event_at_monotonic = now
        if speech_detected:
            self._on_speech_started(now)
            return None

        self._on_speech_ended(now)
        if self.final_text:
            return self._complete("%s_with_final" % reason, now, used_partial=False)
        return None

    def consume_partial_transcript(
        self, text: object, now: Optional[float] = None
    ) -> Optional[CompletedTurn]:
        now = now if now is not None else time.monotonic()
        self.last_event_at_monotonic = now
        if self.started_at_monotonic is None:
            self.started_at_monotonic = now
        normalized = self._normalized_text(text)
        if normalized:
            self.partial_text = normalized
            self.last_partial_at_monotonic = now
            if not self.speech_active:
                self.silence_started_at_monotonic = now
        return self._maybe_complete_for_max_duration(now)

    def consume_final_transcript(
        self, text: object, now: Optional[float] = None
    ) -> Optional[CompletedTurn]:
        now = now if now is not None else time.monotonic()
        self.last_event_at_monotonic = now
        if self.started_at_monotonic is None:
            self.started_at_monotonic = now
        normalized = self._normalized_text(text)
        if normalized:
            self.final_text = normalized
        if not self.speech_active and self.final_text:
            return self._complete(
                "final_transcript_after_silence", now, used_partial=False
            )
        return self._maybe_complete_for_max_duration(now)

    def tick(self, now: Optional[float] = None) -> Optional[CompletedTurn]:
        now = now if now is not None else time.monotonic()

        completed = self._maybe_complete_for_max_duration(now)
        if completed is not None:
            return completed

        if self.speech_active or self.silence_started_at_monotonic is None:
            return None

        silence_elapsed = now - self.silence_started_at_monotonic
        if silence_elapsed < self.config.silence_timeout_s:
            return None

        if self.final_text:
            return self._complete("trailing_silence_final", now, used_partial=False)

        if self._partial_is_committable(now):
            return self._complete("trailing_silence_partial", now, used_partial=True)

        return None

    def finalize(
        self, reason: str = "stream_ended", now: Optional[float] = None
    ) -> Optional[CompletedTurn]:
        now = now if now is not None else time.monotonic()
        if self.final_text:
            return self._complete(reason, now, used_partial=False)
        if self._partial_is_committable(now, require_stability=False):
            return self._complete(reason, now, used_partial=True)
        return None

    def _on_speech_started(self, now: float) -> None:
        self.speech_active = True
        self.silence_started_at_monotonic = None
        if self.started_at_monotonic is None:
            self.started_at_monotonic = now

    def _on_speech_ended(self, now: float) -> None:
        if self.started_at_monotonic is None:
            self.started_at_monotonic = now
        self.speech_active = False
        if self.silence_started_at_monotonic is None:
            self.silence_started_at_monotonic = now

    def _maybe_complete_for_max_duration(self, now: float) -> Optional[CompletedTurn]:
        if self.started_at_monotonic is None:
            return None
        if now - self.started_at_monotonic < self.config.max_turn_duration_s:
            return None
        if self.final_text:
            return self._complete("max_turn_duration_final", now, used_partial=False)
        if self._partial_is_committable(now):
            return self._complete("max_turn_duration_partial", now, used_partial=True)
        return None

    def _partial_is_committable(
        self, now: Optional[float] = None, require_stability: bool = True
    ) -> bool:
        stripped = self.partial_text.strip()
        if len(stripped) < self.config.partial_commit_min_chars:
            return False
        if require_stability and not self._partial_is_stable(now):
            return False
        if self._looks_incomplete(stripped):
            return False
        if stripped.endswith((".", "!", "?")):
            return True
        return len(stripped.split()) >= 5

    def _partial_is_stable(self, now: Optional[float]) -> bool:
        if self.last_partial_at_monotonic is None:
            return False
        if now is None:
            now = time.monotonic()
        return now - self.last_partial_at_monotonic >= self.config.partial_stability_s

    def _looks_incomplete(self, text: str) -> bool:
        lowered = text.rstrip().lower()
        return any(lowered.endswith(suffix) for suffix in self.INCOMPLETE_SUFFIXES)

    def _complete(self, reason: str, now: float, used_partial: bool) -> CompletedTurn:
        started_at = self.started_at_monotonic or now
        text = self.partial_text if used_partial else self.final_text
        completed = CompletedTurn(
            text=text.strip(),
            reason=reason,
            used_partial=used_partial,
            started_at_monotonic=started_at,
            ended_at_monotonic=now,
            metadata={
                "speech_active": self.speech_active,
                "had_final": bool(self.final_text),
                "had_partial": bool(self.partial_text),
                "silence_duration_ms": self._silence_duration_ms(now),
                "turn_duration_ms": int((now - started_at) * 1000),
            },
        )
        self.reset()
        return completed

    def _silence_duration_ms(self, now: float) -> int:
        if self.silence_started_at_monotonic is None:
            return 0
        return int(max(0.0, now - self.silence_started_at_monotonic) * 1000)

    @staticmethod
    def _normalized_text(value: object) -> str:
        return str(value or "").strip()
