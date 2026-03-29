from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .capability_broker import CapabilityBroker, CapabilityDeniedError, ConfirmationRequiredError
from ..models.actions import ToolAction
from ..tools import ToolRegistry


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass
class ActionRequest:
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    scope: str = "global"
    confirmed: bool = False

    @classmethod
    def from_action(
        cls,
        action: ToolAction,
        scope: str = "global",
        confirmed: bool = False,
    ) -> "ActionRequest":
        return cls(
            tool_name=action.tool_name,
            arguments=action.to_arguments(),
            scope=scope,
            confirmed=confirmed,
        )


@dataclass
class ActionResult:
    tool_name: str
    output: Any
    executed_at: datetime = field(default_factory=utc_now)


class ActionBroker:
    def __init__(self, capability_broker: CapabilityBroker, tool_registry: ToolRegistry) -> None:
        self._capability_broker = capability_broker
        self._tool_registry = tool_registry

    async def execute(self, request: ActionRequest) -> ActionResult:
        self._capability_broker.authorize(
            request.tool_name,
            scope=request.scope,
            confirmed=request.confirmed,
        )
        output = await self._tool_registry.execute(request.tool_name, **request.arguments)
        return ActionResult(tool_name=request.tool_name, output=output)


__all__ = [
    "ActionBroker",
    "ActionRequest",
    "ActionResult",
    "CapabilityDeniedError",
    "ConfirmationRequiredError",
]
