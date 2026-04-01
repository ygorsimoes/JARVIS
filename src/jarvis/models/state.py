from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class JarvisState(str, Enum):
    IDLE = "idle"
    ARMED = "armed"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    ACTING = "acting"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass
class StateTransition:
    previous_state: JarvisState
    new_state: JarvisState
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
