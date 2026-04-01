import os
import stat
import tempfile
import textwrap

import pytest

from jarvis.adapters.stt.speech_analyzer import (
    SpeechAnalyzerStreamError,
    SpeechAnalyzerSTTAdapter,
)


@pytest.mark.asyncio
class TestSpeechAnalyzerAdapter:
    async def test_iter_events_reads_ndjson_stream(self):
        script = self._make_script(
            """
            import json
            import sys

            events = [
                {"type": "ready", "locale": "pt-BR", "sample_rate": 16000},
                {"type": "speech_started"},
                {"type": "partial_transcript", "text": "que horas sao"},
                {"type": "speech_detector_result", "speech_detected": 1},
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

        assert events[0]["type"] == "ready"
        assert events[0]["locale"] == "pt-BR"
        assert events[0]["stt_backend"] == "speech_analyzer"
        assert events[0]["sequence"] == 0
        assert "received_at_monotonic" in events[0]
        assert events[3]["speech_detected"]
        assert events[-1]["type"] == "final_transcript"
        assert events[-1]["text"] == "Que horas sao agora?"
        assert events[-1]["sequence"] == 5

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
        with pytest.raises(SpeechAnalyzerStreamError) as context:
            async for _ in adapter.iter_events():
                pass

        assert "model_not_ready" in str(context.value)

    async def test_iter_events_rejects_non_object_payloads(self):
        script = self._make_script(
            """
            import json
            import sys

            sys.stdout.write(json.dumps(["not", "an", "object"]) + "\\n")
            sys.stdout.flush()
            """
        )

        adapter = SpeechAnalyzerSTTAdapter(script, locale="pt-BR")
        with pytest.raises(
            SpeechAnalyzerStreamError,
            match="invalid event payload",
        ):
            async for _ in adapter.iter_events():
                pass

    async def test_transcribe_stream_yields_only_final_transcripts(self):
        script = self._make_script(
            """
            import json
            import sys

            events = [
                {"type": "ready", "locale": "pt-BR", "sample_rate": 16000},
                {"type": "partial_transcript", "text": "que horas"},
                {"type": "final_transcript", "text": "Que horas sao agora?"},
                {"type": "final_transcript", "text": "Define um timer de dez minutos"},
            ]
            for event in events:
                sys.stdout.write(json.dumps(event) + "\\n")
                sys.stdout.flush()
            """
        )

        adapter = SpeechAnalyzerSTTAdapter(script, locale="pt-BR")

        transcripts = [text async for text in adapter.transcribe_stream()]

        assert transcripts == [
            "Que horas sao agora?",
            "Define um timer de dez minutos",
        ]

    async def test_iter_events_uses_default_error_summary_without_stderr(self):
        script = self._make_script(
            """
            raise SystemExit(3)
            """
        )

        adapter = SpeechAnalyzerSTTAdapter(script, locale="pt-BR")
        with pytest.raises(
            SpeechAnalyzerStreamError,
            match="failed without stderr output",
        ):
            async for _ in adapter.iter_events():
                pass

    def _make_script(self, body: str) -> str:
        temp_dir = tempfile.mkdtemp(prefix="jarvis-stt-test-")
        path = os.path.join(temp_dir, "fake_cli.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("#!/usr/bin/env python3\n")
            handle.write(textwrap.dedent(body))
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        return path
