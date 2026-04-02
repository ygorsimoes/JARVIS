from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from pathlib import Path
from typing import (
    AsyncIterator,
    Awaitable,
    Callable,
    Deque,
    List,
    Optional,
)
from urllib.parse import urlparse

from ...models.conversation import Message, Role
from ...observability import get_logger

logger = get_logger(__name__)


class FoundationModelsBridgeAdapter:
    def __init__(
        self,
        base_url: str,
        instructions: str,
        timeout: float = 60.0,
        session_id: Optional[str] = None,
        bridge_binary_path: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.instructions = instructions
        self.timeout = timeout
        self.session_id = session_id
        self.bridge_binary_path = bridge_binary_path
        self._process: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._stderr_lines: Deque[str] = deque(maxlen=50)
        self._startup_lock = asyncio.Lock()
        self._owns_process = False
        self._session_tools_signature: Optional[str] = None

    async def reset_session(self) -> None:
        await self.close_session()
        self.session_id = None

    async def close_session(self) -> None:
        if not self.session_id:
            return

        request = urllib.request.Request(
            "%s/sessions/%s" % (self.base_url, self.session_id),
            method="DELETE",
        )
        try:
            response = await asyncio.to_thread(self._open_request, request)
            await asyncio.to_thread(response.read)
            await asyncio.to_thread(response.close)
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        finally:
            self.session_id = None
            self._session_tools_signature = None

    async def cancel_current_response(self) -> bool:
        if not self.session_id:
            return False

        request = urllib.request.Request(
            "%s/sessions/%s/cancel" % (self.base_url, self.session_id),
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = await asyncio.to_thread(self._open_request, request)
            body = await asyncio.to_thread(response.read)
            await asyncio.to_thread(response.close)
        except (urllib.error.URLError, OSError, TimeoutError):
            return False
        payload = json.loads(body.decode("utf-8"))
        return bool(payload.get("cancelled"))

    async def shutdown(self) -> None:
        await self.close_session()
        if (
            self._process is not None
            and self._owns_process
            and self._process.returncode is None
        ):
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except ProcessLookupError:
                pass
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
        self._process = None
        self._stderr_task = None
        self._owns_process = False

    async def prewarm(self, tools: List[dict] | None = None) -> None:
        await self._ensure_service_running()
        await self._ensure_session(tools or [])

    async def healthcheck(self) -> bool:
        health = await self._service_health()
        if health is not None and health["available"]:
            return True
        if not self.bridge_binary_path:
            return False
        try:
            await self._ensure_service_running()
        except RuntimeError:
            return False
        health = await self._service_health()
        return bool(health and health["available"])

    async def _service_health(self) -> dict[str, object] | None:
        request = urllib.request.Request("%s/health" % self.base_url, method="GET")
        try:
            response = await asyncio.to_thread(self._open_request, request)
            status_code = getattr(response, "status", 200)
            body = await asyncio.to_thread(response.read)
            await asyncio.to_thread(response.close)
        except urllib.error.HTTPError as exc:
            status_code = exc.code
            body = exc.read()
            exc.close()
        except (urllib.error.URLError, TimeoutError):
            return None

        payload: dict[str, object] = {}
        if body:
            try:
                decoded = json.loads(body.decode("utf-8"))
                if isinstance(decoded, dict):
                    payload = decoded
            except json.JSONDecodeError:
                payload = {}

        return {
            "status_code": status_code,
            "payload": payload,
            "available": payload.get("status") == "ok"
            and payload.get("availability") == "available",
        }

    async def chat_stream(
        self,
        messages: List[Message],
        tools: List[dict],
        max_kv_size: int,
        tool_invoker: Callable[[str, dict], Awaitable[object]] | None = None,
    ) -> AsyncIterator[str]:
        del max_kv_size
        await self._ensure_service_running()
        session_created = await self._ensure_session(tools)
        prompt = self._bridge_prompt(messages, include_history=session_created)
        if not prompt:
            raise RuntimeError("foundation models adapter requires a user prompt")
        response_messages = [self._serialize_message(messages[-1])] if messages else []
        payload = {
            "prompt": prompt,
            "messages": response_messages,
            "stream": True,
        }

        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                request = urllib.request.Request(
                    "%s/sessions/%s/responses" % (self.base_url, self.session_id),
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    for raw_line in response:
                        line = raw_line.decode("utf-8").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        event = json.loads(line[5:].strip())
                        event_type = event.get("type")
                        if event_type in {"response_chunk", "text", "fm.text"}:
                            text = event.get("text", "")
                            if text:
                                asyncio.run_coroutine_threadsafe(queue.put(text), loop)
                        elif event_type in {"tool_call", "fm.tool_call"}:
                            if tool_invoker is None:
                                raise RuntimeError(
                                    "bridge requested tool execution without a tool invoker"
                                )
                            invoke = tool_invoker
                            assert invoke is not None
                            tool_name = event.get("name")
                            if not isinstance(tool_name, str) or not tool_name:
                                raise RuntimeError(
                                    "bridge emitted an invalid tool call event"
                                )
                            call_id = event.get("call_id") or event.get("callId")
                            if not isinstance(call_id, str) or not call_id:
                                raise RuntimeError(
                                    "bridge emitted a tool call without call_id"
                                )
                            args = event.get("args")
                            if args is None:
                                args = {}
                            if not isinstance(args, dict):
                                raise RuntimeError(
                                    "bridge emitted tool args that are not an object"
                                )

                            async def invoke_tool() -> object:
                                return await invoke(tool_name, args)

                            future = asyncio.run_coroutine_threadsafe(
                                invoke_tool(),
                                loop,
                            )
                            try:
                                result = future.result()
                            except Exception as exc:
                                result = {"error": str(exc)}
                            self._submit_tool_result(call_id, result)
                        elif event_type in {"tool_result", "fm.tool_result"}:
                            continue
                        elif event_type in {
                            "response_end",
                            "completed",
                            "fm.completed",
                        }:
                            break
                        elif event_type in {"error", "fm.error"}:
                            raise RuntimeError(
                                event.get("message", "Foundation Models bridge error")
                            )
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def _ensure_session(self, tools: List[dict]) -> bool:
        tools_signature = json.dumps(tools, sort_keys=True)
        if self.session_id and self._session_tools_signature == tools_signature:
            logger.info(
                "Reusing Foundation Models session",
                session_id=self.session_id,
                tools_count=len(tools),
            )
            return False
        if self.session_id and self._session_tools_signature != tools_signature:
            await self.close_session()
        payload = {
            "instructions": self.instructions,
            "tools": tools,
            "session_id": str(uuid.uuid4()),
        }
        request = urllib.request.Request(
            "%s/sessions" % self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = await asyncio.to_thread(self._open_request, request)
        body = await asyncio.to_thread(response.read)
        await asyncio.to_thread(response.close)
        parsed = json.loads(body.decode("utf-8"))
        self.session_id = parsed["session_id"]
        self._session_tools_signature = tools_signature
        logger.info(
            "Created Foundation Models session",
            session_id=self.session_id,
            tools_count=len(tools),
        )
        return True

    async def _ensure_service_running(self) -> None:
        health = await self._service_health()
        if health is not None:
            return
        if not self.bridge_binary_path:
            return

        async with self._startup_lock:
            health = await self._service_health()
            if health is not None:
                return

            binary = Path(self.bridge_binary_path)
            if not binary.exists():
                raise RuntimeError(
                    "Foundation Models bridge binary not found at %s" % binary
                )

            started = time.perf_counter()
            if self._process is None or self._process.returncode is not None:
                host, port = self._host_and_port()
                logger.info(
                    "Starting Foundation Models bridge",
                    base_url=self.base_url,
                    binary=str(binary),
                )
                self._process = await asyncio.create_subprocess_exec(
                    str(binary),
                    "--host",
                    host,
                    "--port",
                    str(port),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                self._owns_process = True
                self._stderr_task = asyncio.create_task(self._drain_stderr())

            for _ in range(40):
                health = await self._service_health()
                if health is not None:
                    payload = health.get("payload")
                    availability_payload: dict[str, object] = (
                        payload if isinstance(payload, dict) else {}
                    )
                    logger.info(
                        "Foundation Models bridge ready",
                        base_url=self.base_url,
                        status=availability_payload.get("status"),
                        availability=availability_payload.get("availability"),
                        startup_ms=int((time.perf_counter() - started) * 1000),
                    )
                    return
                await asyncio.sleep(0.25)

            raise RuntimeError(
                "Foundation Models bridge failed to start: %s" % self._stderr_summary()
            )

    async def _drain_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        async for raw_line in self._process.stderr:
            line = raw_line.decode("utf-8").strip()
            if line:
                self._stderr_lines.append(line)

    def _stderr_summary(self) -> str:
        if not self._stderr_lines:
            return "no stderr output"
        return " ".join(self._stderr_lines)

    def _host_and_port(self) -> tuple[str, int]:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"}:
            raise RuntimeError("unsupported bridge URL %s" % self.base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return host, port

    def _open_request(self, request: urllib.request.Request):
        return urllib.request.urlopen(request, timeout=self.timeout)

    def _submit_tool_result(self, call_id: str, result: object) -> None:
        if not self.session_id:
            raise RuntimeError("cannot submit tool result without an active session")
        payload = json.dumps({"result": result}).encode("utf-8")
        request = urllib.request.Request(
            "%s/sessions/%s/tool-results/%s"
            % (self.base_url, self.session_id, call_id),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = self._open_request(request)
        try:
            response.read()
        finally:
            response.close()

    @staticmethod
    def _serialize_message(message: Message) -> dict:
        return {
            "role": message.role.value,
            "content": message.content,
            "metadata": message.metadata,
        }

    @classmethod
    def _bridge_prompt(cls, messages: List[Message], *, include_history: bool) -> str:
        latest_user = cls._last_user_message(messages)
        if not latest_user:
            return ""

        context_lines: list[str] = []
        dynamic_system_messages = [
            message.content.strip()
            for message in messages
            if message.role == Role.SYSTEM and message.content.strip()
        ]
        if len(dynamic_system_messages) > 1:
            context_lines.extend(dynamic_system_messages[1:])

        if include_history:
            history_lines: list[str] = []
            for message in messages:
                if message.role == Role.SYSTEM:
                    continue
                if message.role == Role.USER and message.content == latest_user:
                    continue
                history_lines.append(
                    "%s: %s"
                    % (cls._label_for_role(message.role), message.content.strip())
                )
            if history_lines:
                context_lines.append(
                    "Historico recente:\n%s" % "\n".join(history_lines)
                )

        if not context_lines:
            return latest_user
        return "%s\n\nPedido atual:\n%s" % ("\n\n".join(context_lines), latest_user)

    @staticmethod
    def _last_user_message(messages: List[Message]) -> str:
        for message in reversed(messages):
            if message.role == Role.USER:
                return message.content
        return ""

    @staticmethod
    def _label_for_role(role: Role) -> str:
        if role == Role.USER:
            return "Usuario"
        if role == Role.ASSISTANT:
            return "JARVIS"
        if role == Role.TOOL:
            return "Tool"
        return "Sistema"
