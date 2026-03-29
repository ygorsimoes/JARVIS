from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from ..config import JarvisConfig
from .browser import BrowserTool
from .files import FilesTool
from .system import SystemTool
from .timer import TimerTool

ToolHandler = Callable[..., Awaitable[Any]]


class ToolValidationError(ValueError):
    pass


@dataclass
class RegisteredTool:
    name: str
    description: str
    handler: ToolHandler
    input_schema: Optional[dict] = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        description: str,
        handler: ToolHandler,
        input_schema: Optional[dict] = None,
    ) -> None:
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            handler=handler,
            input_schema=input_schema,
        )

    def get(self, name: str) -> RegisteredTool:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError("tool %s is not registered" % name)
        return tool

    async def execute(self, name: str, **kwargs: Any) -> Any:
        tool = self.get(name)
        self._validate_input(tool, kwargs)
        return await tool.handler(**kwargs)

    def describe(self) -> list:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    def _validate_input(self, tool: RegisteredTool, payload: Dict[str, Any]) -> None:
        schema = tool.input_schema or {"type": "object", "properties": {}}
        if schema.get("type") != "object":
            raise ToolValidationError("tool %s must use an object schema" % tool.name)

        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional_properties = schema.get("additionalProperties", False)

        for field_name in required:
            if field_name not in payload:
                raise ToolValidationError("tool %s requires field %s" % (tool.name, field_name))

        for field_name, value in payload.items():
            field_schema = properties.get(field_name)
            if field_schema is None:
                if additional_properties:
                    continue
                raise ToolValidationError("tool %s does not accept field %s" % (tool.name, field_name))
            self._validate_field(tool.name, field_name, value, field_schema)

    def _validate_field(self, tool_name: str, field_name: str, value: Any, schema: Dict[str, Any]) -> None:
        field_type = schema.get("type")
        if field_type == "string" and not isinstance(value, str):
            raise ToolValidationError("tool %s field %s must be a string" % (tool_name, field_name))
        if field_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ToolValidationError("tool %s field %s must be an integer" % (tool_name, field_name))
            minimum = schema.get("minimum")
            if minimum is not None and value < minimum:
                raise ToolValidationError(
                    "tool %s field %s must be >= %s" % (tool_name, field_name, minimum)
                )
        if field_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ToolValidationError("tool %s field %s must be a number" % (tool_name, field_name))
        if field_type == "boolean" and not isinstance(value, bool):
            raise ToolValidationError("tool %s field %s must be a boolean" % (tool_name, field_name))


def build_default_registry(config: JarvisConfig) -> ToolRegistry:
    registry = ToolRegistry()

    timer_tool = TimerTool()
    system_tool = SystemTool(config.system_allowed_apps)
    browser_tool = BrowserTool()
    files_tool = FilesTool(config.allowed_file_roots)

    registry.register(
        "system.get_time",
        "Retorna o horario local atual.",
        system_tool.get_time,
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )
    registry.register(
        "system.open_app",
        "Abre um aplicativo instalado no macOS.",
        system_tool.open_app,
        input_schema={
            "type": "object",
            "properties": {"app_name": {"type": "string"}},
            "required": ["app_name"],
            "additionalProperties": False,
        },
    )
    registry.register(
        "timer.start",
        "Cria um timer local.",
        timer_tool.start,
        input_schema={
            "type": "object",
            "properties": {
                "duration_seconds": {"type": "integer", "minimum": 1},
                "label": {"type": "string"},
            },
            "required": ["duration_seconds"],
            "additionalProperties": False,
        },
    )
    registry.register(
        "timer.list",
        "Lista timers ativos.",
        timer_tool.list,
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )
    registry.register(
        "timer.cancel",
        "Cancela um timer ativo.",
        timer_tool.cancel,
        input_schema={
            "type": "object",
            "properties": {"timer_id": {"type": "string"}},
            "required": ["timer_id"],
            "additionalProperties": False,
        },
    )
    registry.register(
        "browser.search",
        "Abre uma busca na web.",
        browser_tool.search,
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    registry.register(
        "files.list",
        "Lista conteudo de um diretorio permitido.",
        files_tool.list,
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )
    registry.register(
        "files.read",
        "Le um arquivo de texto permitido.",
        files_tool.read,
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )
    return registry
