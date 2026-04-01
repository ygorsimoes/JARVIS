from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class MemoryCategory(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    PROFILE = "profile"
    PROCEDURAL = "procedural"


class MemorySource(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    SYSTEM = "system"


@dataclass
class Memory:
    content: str
    category: MemoryCategory
    source: MemorySource
    confidence: float
    recency_weight: float
    scope: str
    created_at: datetime = field(default_factory=utc_now)
    last_accessed: datetime = field(default_factory=utc_now)
