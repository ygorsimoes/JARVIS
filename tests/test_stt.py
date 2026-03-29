import os
import stat
import tempfile
import textwrap
import unittest

from jarvis.adapters.stt.speech_analyzer import SpeechAnalyzerSTTAdapter, SpeechAnalyzerStreamError


class SpeechAnalyzerAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_iter_events_reads_ndjson_stream(self):
        script = self._make_script(
            """
            import json
            import sys

            events = [
                {"type": "speech_started"},
                {"type": "partial_transcript", "text": "que horas sao"},
                {"type": "speech_ended"},
                {"type": "final_transcript", "text": "Que horas sao agora?"},
            ]
            for event in events:
                sys.stdout.write(json.dumps(event) + "\\n")
                sys.stdout.flush()
            """
        )

        adapter = SpeechAnalyzerSTTAdapter(script, locale="pt-BR")
        events = []
        async for event in adapter.iter_events():
            events.append(event)

        self.assertEqual(events[-1]["type"], "final_transcript")
        self.assertEqual(events[-1]["text"], "Que horas sao agora?")

    async def test_iter_events_surfaces_stderr_on_failure(self):
        script = self._make_script(
            """
            import sys

            sys.stderr.write("model_not_ready\\n")
            sys.stderr.flush()
            raise SystemExit(2)
            """
        )

        adapter = SpeechAnalyzerSTTAdapter(script, locale="pt-BR")
        with self.assertRaises(SpeechAnalyzerStreamError) as context:
            async for _ in adapter.iter_events():
                pass

        self.assertIn("model_not_ready", str(context.exception))

    def _make_script(self, body: str) -> str:
        temp_dir = tempfile.mkdtemp(prefix="jarvis-stt-test-")
        path = os.path.join(temp_dir, "fake_cli.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("#!/usr/bin/env python3\n")
            handle.write(textwrap.dedent(body))
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        return path


if __name__ == "__main__":
    unittest.main()
