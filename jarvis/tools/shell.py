from __future__ import annotations

import asyncio
from typing import Any, Dict

from .security import parse_safe_shell_command


class ShellTool:
    def __init__(self, allowed_commands: list[str] | None = None):
        self.allowed_commands = allowed_commands or [
            "ls",
            "echo",
            "pwd",
            "date",
            "whoami",
            "uname",
            "uptime",
        ]
        self._allowed_commands = set(self.allowed_commands)

    async def execute(self, command: str) -> Dict[str, Any]:
        """
        Executes a shell command on macOS if it is in the allowlist.
        """
        try:
            args = parse_safe_shell_command(command)
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}

        base_cmd = args[0]
        if base_cmd not in self._allowed_commands:
            return {
                "status": "error",
                "error": f"Comando '{base_cmd}' bloqueado pela politica de seguranca. Permitidos: {', '.join(self.allowed_commands)}",
            }

        try:
            process = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            return {
                "status": "success" if process.returncode == 0 else "error",
                "exit_code": process.returncode,
                "stdout": stdout.decode("utf-8").strip(),
                "stderr": stderr.decode("utf-8").strip(),
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
