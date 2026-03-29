from __future__ import annotations

from typing import Any, Dict, Literal, Union

from pydantic import BaseModel, Field


class ToolAction(BaseModel):
    tool_name: str

    def to_arguments(self) -> Dict[str, Any]:
        payload = self.model_dump()
        payload.pop("tool_name", None)
        return payload


class GetTimeAction(ToolAction):
    tool_name: Literal["system.get_time"] = "system.get_time"


class OpenAppAction(ToolAction):
    tool_name: Literal["system.open_app"] = "system.open_app"
    app_name: str = Field(min_length=1)


class BrowserSearchAction(ToolAction):
    tool_name: Literal["browser.search"] = "browser.search"
    query: str = Field(min_length=1)


class StartTimerAction(ToolAction):
    tool_name: Literal["timer.start"] = "timer.start"
    duration_seconds: int = Field(gt=0)
    label: str = Field(min_length=1)


DirectToolAction = Union[GetTimeAction, OpenAppAction, BrowserSearchAction, StartTimerAction]
