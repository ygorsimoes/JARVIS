from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class EventType(str, Enum):
    ACTIVATION_TRIGGERED = "activation_triggered"
    STATE_CHANGED = "state_changed"
    INTERRUPTION_REQUESTED = "interruption_requested"
    STT_READY = "stt_ready"
    VAD_ACTIVITY = "vad_activity"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    PARTIAL_TRANSCRIPT = "partial_transcript"
    FINAL_TRANSCRIPT = "final_transcript"
    TURN_READY = "turn_ready"
    USER_TURN = "user_turn"
    ROUTE_SELECTED = "route_selected"
    ASSISTANT_SENTENCE = "assistant_sentence"
    ASSISTANT_COMPLETED = "assistant_completed"
    TTS_SENTENCE_QUEUED = "tts_sentence_queued"
    TTS_STARTED = "tts_started"
    TTS_COMPLETED = "tts_completed"
    PLAYBACK_STARTED = "playback_started"
    PLAYBACK_COMPLETED = "playback_completed"
    TOOL_EXECUTED = "tool_executed"
    ERROR = "error"


@dataclass
class Event:
    event_type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
