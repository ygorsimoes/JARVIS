import asyncio
import shlex
from typing import Dict, List, Any


class ShellTool:
    def __init__(self, allowed_commands: List[str] = None):
        self.allowed_commands = allowed_commands or ["ls", "echo", "pwd", "date", "whoami", "uname", "uptime"]

    async def execute(self, command: str) -> Dict[str, Any]:
        """
        Executes a shell command on macOS if it is in the allowlist.
        """
        # Parse logic to verify base command
        args = shlex.split(command)
        if not args:
            return {"status": "error", "error": "Comando vazio"}

        base_cmd = args[0]
        if base_cmd not in self.allowed_commands:
            return {
                "status": "error", 
                "error": f"Comando '{base_cmd}' bloqueado pela politica de seguranca. Permitidos: {', '.join(self.allowed_commands)}"
            }

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            return {
                "status": "success" if process.returncode == 0 else "error",
                "exit_code": process.returncode,
                "stdout": stdout.decode("utf-8").strip(),
                "stderr": stderr.decode("utf-8").strip()
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
