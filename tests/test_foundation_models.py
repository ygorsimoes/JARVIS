import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jarvis.adapters.llm.foundation_models import FoundationModelsBridgeAdapter
from jarvis.models.conversation import Message, Role


class _BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    created_session_payload = None
    response_payload = None
    created_session_count = 0
    cancelled = False
    deleted = False
    submitted_tool_result_payload = None
    submitted_tool_result_path = None
    tool_result_ready = threading.Event()

    @classmethod
    def reset_state(cls):
        cls.created_session_payload = None
        cls.response_payload = None
        cls.created_session_count = 0
        cls.cancelled = False
        cls.deleted = False
        cls.submitted_tool_result_payload = None
        cls.submitted_tool_result_path = None
        cls.tool_result_ready = threading.Event()

    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return

        body = json.dumps({"status": "ok", "availability": "available"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""

        if self.path == "/sessions":
            _BridgeHandler.created_session_payload = json.loads(body.decode("utf-8"))
            _BridgeHandler.created_session_count += 1
            payload = json.dumps({"session_id": "session-123"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/sessions/session-123/cancel":
            _BridgeHandler.cancelled = True
            payload = json.dumps({"cancelled": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/sessions/session-123/responses":
            _BridgeHandler.response_payload = json.loads(body.decode("utf-8"))
            _BridgeHandler.submitted_tool_result_payload = None
            _BridgeHandler.submitted_tool_result_path = None
            _BridgeHandler.tool_result_ready.clear()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            initial_events = [
                {"type": "response_chunk", "text": "Ola "},
                {
                    "type": "tool_call",
                    "name": "system.get_time",
                    "call_id": "call-123",
                    "args": {},
                },
            ]
            for event in initial_events:
                line = "data: %s\n\n" % json.dumps(event)
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()

            if not _BridgeHandler.tool_result_ready.wait(timeout=2):
                raise RuntimeError("tool result was not submitted")

            submitted_payload = _BridgeHandler.submitted_tool_result_payload or {
                "result": None
            }
            trailing_events = [
                {
                    "type": "tool_result",
                    "name": "system.get_time",
                    "call_id": "call-123",
                    "result": submitted_payload["result"],
                },
                {"type": "response_chunk", "text": "mundo."},
                {"type": "response_end"},
            ]
            for event in trailing_events:
                line = "data: %s\n\n" % json.dumps(event)
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
            return

        if self.path == "/sessions/session-123/tool-results/call-123":
            _BridgeHandler.submitted_tool_result_payload = json.loads(
                body.decode("utf-8")
            )
            _BridgeHandler.submitted_tool_result_path = self.path
            _BridgeHandler.tool_result_ready.set()
            payload = json.dumps({"accepted": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(404)
        self.end_headers()

    def do_DELETE(self):
        if self.path == "/sessions/session-123":
            _BridgeHandler.deleted = True
            payload = json.dumps({"deleted": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


class FoundationModelsAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        _BridgeHandler.reset_state()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _BridgeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host = self.server.server_address[0]
        port = self.server.server_address[1]
        self.base_url = "http://%s:%s" % (host, port)

    async def asyncTearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()

    async def test_healthcheck(self):
        adapter = FoundationModelsBridgeAdapter(
            base_url=self.base_url,
            instructions="Teste",
        )
        self.assertTrue(await adapter.healthcheck())

    async def test_chat_stream_creates_session_and_yields_text(self):
        adapter = FoundationModelsBridgeAdapter(
            base_url=self.base_url,
            instructions="Teste",
        )
        messages = [Message(role=Role.USER, content="Ola")]

        async def tool_invoker(tool_name, args):
            self.assertEqual(tool_name, "system.get_time")
            self.assertEqual(args, {})
            return {"time": "12:34"}

        chunks = []
        async for chunk in adapter.chat_stream(
            messages=messages,
            tools=[],
            max_kv_size=0,
            tool_invoker=tool_invoker,
        ):
            chunks.append(chunk)

        self.assertEqual(chunks, ["Ola ", "mundo."])
        self.assertEqual(adapter.session_id, "session-123")
        created_payload = _BridgeHandler.created_session_payload
        response_payload = _BridgeHandler.response_payload
        self.assertIsNotNone(created_payload)
        self.assertIsNotNone(response_payload)
        assert created_payload is not None
        assert response_payload is not None
        self.assertEqual(created_payload["instructions"], "Teste")
        self.assertEqual(response_payload["prompt"], "Ola")
        self.assertEqual(
            _BridgeHandler.submitted_tool_result_path,
            "/sessions/session-123/tool-results/call-123",
        )
        self.assertEqual(
            _BridgeHandler.submitted_tool_result_payload, {"result": {"time": "12:34"}}
        )

    async def test_chat_stream_reuses_existing_session(self):
        adapter = FoundationModelsBridgeAdapter(
            base_url=self.base_url, instructions="Teste"
        )
        messages = [Message(role=Role.USER, content="Primeiro turno")]

        async def tool_invoker(tool_name, args):
            self.assertEqual(tool_name, "system.get_time")
            self.assertEqual(args, {})
            return {"time": "12:34"}

        async for _ in adapter.chat_stream(
            messages=messages,
            tools=[],
            max_kv_size=0,
            tool_invoker=tool_invoker,
        ):
            pass

        second_messages = [Message(role=Role.USER, content="Segundo turno")]
        async for _ in adapter.chat_stream(
            messages=second_messages,
            tools=[],
            max_kv_size=0,
            tool_invoker=tool_invoker,
        ):
            pass

        self.assertEqual(_BridgeHandler.created_session_count, 1)
        self.assertEqual(adapter.session_id, "session-123")

    async def test_cancel_and_close_session(self):
        adapter = FoundationModelsBridgeAdapter(
            base_url=self.base_url, instructions="Teste"
        )
        adapter.session_id = "session-123"

        cancelled = await adapter.cancel_current_response()
        await adapter.close_session()

        self.assertTrue(cancelled)
        self.assertTrue(_BridgeHandler.cancelled)
        self.assertTrue(_BridgeHandler.deleted)
        self.assertIsNone(adapter.session_id)


if __name__ == "__main__":
    unittest.main()
