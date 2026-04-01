from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class RouteTarget(str, Enum):
    DIRECT_TOOL = "direct_tool"
    HOT_PATH = "hot_path"
    DELIBERATIVE = "deliberative"


@dataclass
class Message:
    role: Role
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class Turn:
    role: Role
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class RouteDecision:
    target: RouteTarget
    reason: str
    tool_name: Optional[str] = None
    confidence: float = 1.0
