from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

from ..models.actions import ToolAction
from ..tools import ToolRegistry
from .capability_broker import (
    CapabilityBroker,
    CapabilityDeniedError,
    ConfirmationRequiredError,
)


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass
class ActionRequest:
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    scope: str = "global"
    confirmed: bool = False
    source: str = "runtime"

    @classmethod
    def from_action(
        cls,
        action: ToolAction,
        scope: str = "global",
        confirmed: bool = False,
        source: str = "runtime",
    ) -> "ActionRequest":
        return cls(
            tool_name=action.tool_name,
            arguments=action.to_arguments(),
            scope=scope,
            confirmed=confirmed,
            source=source,
        )


@dataclass
class ActionResult:
    tool_name: str
    output: Any
    scope: str = "global"
    confirmed: bool = False
    side_effects: list[str] = field(default_factory=list)
    audit_logged: bool = False
    executed_at: datetime = field(default_factory=utc_now)


@dataclass
class ActionAuditEntry:
    tool_name: str
    scope: str
    source: str
    status: str
    confirmed: bool
    side_effects: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)


class ActionBroker:
    def __init__(
        self, capability_broker: CapabilityBroker, tool_registry: ToolRegistry
    ) -> None:
        self._capability_broker = capability_broker
        self._tool_registry = tool_registry
        self._audit_log: list[ActionAuditEntry] = []

    def describe_available_tools(self, scope: str = "global") -> list[dict[str, Any]]:
        descriptions: list[dict[str, Any]] = []
        for capability in self._capability_broker.list_enabled(scope=scope):
            try:
                tool = self._tool_registry.get(capability.tool_name)
            except KeyError:
                continue
            descriptions.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "scopes": list(capability.scopes),
                    "risk_level": capability.risk_level.value,
                    "requires_confirmation": capability.requires_confirmation
                    or capability.risk_level.value == "destructive",
                    "side_effects": list(capability.side_effects),
                }
            )
        return descriptions

    async def execute(self, request: ActionRequest) -> ActionResult:
        capability = self._capability_broker.authorize(
            request.tool_name,
            scope=request.scope,
            confirmed=request.confirmed,
        )
        try:
            output = await self._tool_registry.execute(
                request.tool_name, **request.arguments
            )
        except Exception:
            self._record(
                request,
                status="execution_failed",
                side_effects=capability.side_effects,
                audit_log=capability.audit_log,
            )
            raise

        self._record(
            request,
            status="executed",
            side_effects=capability.side_effects,
            audit_log=capability.audit_log,
        )
        return ActionResult(
            tool_name=request.tool_name,
            output=output,
            scope=request.scope,
            confirmed=request.confirmed,
            side_effects=list(capability.side_effects),
            audit_logged=capability.audit_log,
        )

    @property
    def audit_log(self) -> list[ActionAuditEntry]:
        return list(self._audit_log)

    def _record(
        self,
        request: ActionRequest,
        *,
        status: str,
        side_effects: list[str],
        audit_log: bool,
    ) -> None:
        if not audit_log:
            return
        self._audit_log.append(
            ActionAuditEntry(
                tool_name=request.tool_name,
                scope=request.scope,
                source=request.source,
                status=status,
                confirmed=request.confirmed,
                side_effects=list(side_effects),
            )
        )


__all__ = [
    "ActionBroker",
    "ActionRequest",
    "ActionResult",
    "ActionAuditEntry",
    "CapabilityDeniedError",
    "ConfirmationRequiredError",
]
