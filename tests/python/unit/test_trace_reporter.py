import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from jarvis.bus import EventBus
from jarvis.models.events import Event, EventType
from jarvis.observability.trace import VoiceTraceReporter


@pytest.mark.asyncio
class TestVoiceTraceReporter:
    async def test_reporter_emits_terminal_lines_and_jsonl(self, tmp_path):
        lines = []
        reporter = VoiceTraceReporter(
            session_id="session-123",
            mode="compact",
            jsonl_path=str(tmp_path / "trace.jsonl"),
            line_writer=lines.append,
        )
        bus = EventBus()
        await reporter.start(bus)
        reporter.emit_session_configuration(
            {
                "activation_backend": "push_to_talk",
                "stt_backend": "speech_analyzer",
                "stt_locale": "pt-BR",
                "hot_path_backend": "foundation_models",
                "deliberative_backend": "mlx_lm",
                "deliberative_model": "mlx-community/Qwen3-8B-4bit",
                "fallback_backend": "mlx_lm_fallback",
                "fallback_model": "mlx-community/Qwen3-4B-4bit",
                "tts_backend": "mlx_audio_kokoro+avspeech",
                "tts_model": "mlx-community/Kokoro-82M-bf16",
                "tts_voice": "pm_santa",
                "playback_backend": "sounddevice",
            }
        )

        turn_id = "turn-abc-123"
        base_time = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        base_ns = 1_000_000_000
        events = [
            Event(
                event_type=EventType.ACTIVATION_TRIGGERED,
                payload={
                    "session_id": "session-123",
                    "turn_id": turn_id,
                    "backend": "push_to_talk",
                    "source": "foreground",
                },
                created_at=base_time,
                created_at_monotonic_ns=base_ns,
            ),
            Event(
                event_type=EventType.TURN_READY,
                payload={
                    "session_id": "session-123",
                    "turn_id": turn_id,
                    "backend": "speech_analyzer",
                    "reason": "final_transcript",
                    "used_partial": False,
                },
                created_at=base_time + timedelta(milliseconds=220),
                created_at_monotonic_ns=base_ns + 220_000_000,
            ),
            Event(
                event_type=EventType.ROUTE_SELECTED,
                payload={
                    "session_id": "session-123",
                    "turn_id": turn_id,
                    "backend": "mlx_lm",
                    "target": "hot_path",
                    "effective_target": "deliberative",
                    "policy_reason": "hot_path route fell back to mlx_lm backend",
                    "fallback_used": True,
                    "backend_detail": {"model": "mlx-community/Qwen3-8B-4bit"},
                },
                created_at=base_time + timedelta(milliseconds=240),
                created_at_monotonic_ns=base_ns + 240_000_000,
            ),
            Event(
                event_type=EventType.ASSISTANT_SENTENCE,
                payload={
                    "session_id": "session-123",
                    "turn_id": turn_id,
                    "backend": "mlx_lm",
                    "sentence": "Ola.",
                    "index": 0,
                },
                created_at=base_time + timedelta(milliseconds=420),
                created_at_monotonic_ns=base_ns + 420_000_000,
            ),
            Event(
                event_type=EventType.TTS_STARTED,
                payload={
                    "session_id": "session-123",
                    "turn_id": turn_id,
                    "backend": "mlx_audio_kokoro+avspeech",
                    "tts_backend": "mlx_audio_kokoro+avspeech",
                    "tts_effective_backend": "avspeech",
                    "tts_fallback_active": True,
                    "tts_last_error": "missing spacy",
                    "tts_model": "mlx-community/Kokoro-82M-bf16",
                    "tts_voice": "pm_santa",
                    "index": 0,
                },
                created_at=base_time + timedelta(milliseconds=470),
                created_at_monotonic_ns=base_ns + 470_000_000,
            ),
            Event(
                event_type=EventType.PLAYBACK_STARTED,
                payload={
                    "session_id": "session-123",
                    "turn_id": turn_id,
                    "playback_backend": "sounddevice",
                    "tts_effective_backend": "avspeech",
                    "tts_fallback_active": True,
                    "index": 0,
                },
                created_at=base_time + timedelta(milliseconds=500),
                created_at_monotonic_ns=base_ns + 500_000_000,
            ),
            Event(
                event_type=EventType.PLAYBACK_COMPLETED,
                payload={
                    "session_id": "session-123",
                    "turn_id": turn_id,
                    "playback_backend": "sounddevice",
                    "tts_effective_backend": "avspeech",
                    "tts_fallback_active": True,
                    "index": 0,
                },
                created_at=base_time + timedelta(milliseconds=760),
                created_at_monotonic_ns=base_ns + 760_000_000,
            ),
            Event(
                event_type=EventType.ASSISTANT_COMPLETED,
                payload={
                    "session_id": "session-123",
                    "turn_id": turn_id,
                    "backend": "mlx_lm",
                    "text": "Ola.",
                },
                created_at=base_time + timedelta(milliseconds=700),
                created_at_monotonic_ns=base_ns + 700_000_000,
            ),
        ]

        for event in events:
            await bus.publish(event)
        await asyncio.sleep(0.05)
        await reporter.shutdown(bus)

        assert any("session_config" in line for line in lines)
        assert any(
            "route target=hot_path effective=deliberative backend=mlx_lm model=mlx-community/Qwen3-8B-4bit fallback=yes"
            in line
            for line in lines
        )
        assert any(
            "summary total=760ms capture=220ms route=20ms first_sentence=180ms tts_start=50ms playback_start=30ms backend=mlx_lm effective=deliberative fallback=yes tts=mlx_audio_kokoro+avspeech tts_effective=avspeech tts_fallback=yes playback=sounddevice"
            in line
            for line in lines
        )

        records = [
            json.loads(line)
            for line in (tmp_path / "trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert records[0]["record_type"] == "session_config"
        route_records = [
            record for record in records if record.get("event_type") == "route_selected"
        ]
        assert route_records[0]["payload"]["fallback_used"] is True

    @pytest.mark.asyncio
    async def test_reporter_formats_conversation_line(self):
        reporter = VoiceTraceReporter(session_id="session-123", mode="compact")
        reporter.register_turn_start("turn-1", monotonic_ns=1_000_000_000)

        line = reporter.format_conversation_line("voce>", "Ola", turn_id="turn-1")

        assert "voce> Ola" in line
        assert "id=turn-1" in line
