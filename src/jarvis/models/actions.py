from __future__ import annotations

from typing import Any, ClassVar, Dict, Literal, Union

from pydantic import BaseModel, Field


class ToolAction(BaseModel):
    def to_arguments(self) -> Dict[str, Any]:
        return self.model_dump()


class GetTimeAction(ToolAction):
    tool_name: ClassVar[Literal["system.get_time"]] = "system.get_time"


class OpenAppAction(ToolAction):
    tool_name: ClassVar[Literal["system.open_app"]] = "system.open_app"
    app_name: str = Field(min_length=1)


class SetVolumeAction(ToolAction):
    tool_name: ClassVar[Literal["system.set_volume"]] = "system.set_volume"
    level: int = Field(ge=0, le=100)


class BrowserSearchAction(ToolAction):
    tool_name: ClassVar[Literal["browser.search"]] = "browser.search"
    query: str = Field(min_length=1)


class BrowserFetchURLAction(ToolAction):
    tool_name: ClassVar[Literal["browser.fetch_url"]] = "browser.fetch_url"
    url: str = Field(min_length=1)


class StartTimerAction(ToolAction):
    tool_name: ClassVar[Literal["timer.start"]] = "timer.start"
    duration_seconds: int = Field(gt=0)
    label: str = Field(min_length=1)


class ListTimersAction(ToolAction):
    tool_name: ClassVar[Literal["timer.list"]] = "timer.list"


class CancelTimerAction(ToolAction):
    tool_name: ClassVar[Literal["timer.cancel"]] = "timer.cancel"
    timer_id: str = Field(min_length=1)


class ListCalendarEventsAction(ToolAction):
    tool_name: ClassVar[Literal["calendar.list_events"]] = "calendar.list_events"
    days: int = Field(default=1, ge=1)


class ListFilesAction(ToolAction):
    tool_name: ClassVar[Literal["files.list"]] = "files.list"
    path: str = Field(min_length=1)


class ReadFileAction(ToolAction):
    tool_name: ClassVar[Literal["files.read"]] = "files.read"
    path: str = Field(min_length=1)


class MoveFileAction(ToolAction):
    tool_name: ClassVar[Literal["files.move"]] = "files.move"
    source_path: str = Field(min_length=1)
    destination_path: str = Field(min_length=1)


class ExecuteShellAction(ToolAction):
    tool_name: ClassVar[Literal["shell.execute"]] = "shell.execute"
    command: str = Field(min_length=1)


ToolActionType = Union[
    GetTimeAction,
    OpenAppAction,
    SetVolumeAction,
    BrowserSearchAction,
    BrowserFetchURLAction,
    StartTimerAction,
    ListTimersAction,
    CancelTimerAction,
    ListCalendarEventsAction,
    ListFilesAction,
    ReadFileAction,
    MoveFileAction,
    ExecuteShellAction,
]


DirectToolAction = Union[
    GetTimeAction,
    OpenAppAction,
    BrowserSearchAction,
    StartTimerAction,
    ListCalendarEventsAction,
    BrowserFetchURLAction,
    SetVolumeAction,
    ListTimersAction,
    CancelTimerAction,
]
