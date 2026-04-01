from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..bus import EventBus, EventSubscription
from ..models.events import Event, EventType


@dataclass
class _TurnTraceState:
    started_ns: int
    activation_ns: int | None = None
    stt_ready_ns: int | None = None
    speech_started_ns: int | None = None
    speech_ended_ns: int | None = None
    turn_ready_ns: int | None = None
    route_selected_ns: int | None = None
    first_sentence_ns: int | None = None
    first_tts_started_ns: int | None = None
    first_playback_started_ns: int | None = None
    assistant_completed_ns: int | None = None
    last_playback_completed_ns: int | None = None
    backend: str | None = None
    effective_target: str | None = None
    fallback_used: bool = False
    tts_backend: str | None = None
    tts_effective_backend: str | None = None
    tts_fallback_active: bool = False
    tts_last_error: str | None = None
    playback_backend: str | None = None
    last_sentence_index: int | None = None
    last_playback_completed_index: int | None = None
    summary_emitted: bool = False


class VoiceTraceReporter:
    def __init__(
        self,
        *,
        session_id: str,
        mode: str = "compact",
        jsonl_path: str | None = None,
        line_writer: Callable[[str], None] | None = None,
    ) -> None:
        self.session_id = session_id
        self.mode = mode
        self.jsonl_path = jsonl_path
        self._line_writer = line_writer or self._default_line_writer
        self._subscription: EventSubscription | None = None
        self._consumer_task: asyncio.Task | None = None
        self._jsonl_handle = None
        self._session_started_ns: int | None = None
        self._turns: dict[str, _TurnTraceState] = {}

    async def start(self, event_bus: EventBus) -> None:
        if self.mode == "off" or self._consumer_task is not None:
            return
        self._subscription = await event_bus.subscribe()
        if self.jsonl_path:
            path = Path(self.jsonl_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_handle = path.open("a", encoding="utf-8")
        self._consumer_task = asyncio.create_task(
            self._consume_events(),
            name="jarvis-voice-trace-reporter",
        )

    async def shutdown(self, event_bus: EventBus | None = None) -> None:
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None

        if self._subscription is not None and event_bus is not None:
            await event_bus.unsubscribe(self._subscription)
        self._subscription = None

        if self._jsonl_handle is not None:
            self._jsonl_handle.close()
            self._jsonl_handle = None

    def register_turn_start(
        self, turn_id: str, *, monotonic_ns: int | None = None
    ) -> None:
        started_ns = monotonic_ns or self._now_ns()
        self._turns.setdefault(turn_id, _TurnTraceState(started_ns=started_ns))
        if self._session_started_ns is None:
            self._session_started_ns = started_ns

    def emit_session_configuration(self, config: dict[str, object]) -> None:
        now = datetime.now().astimezone()
        line = self._format_trace_line(
            timestamp=now,
            session_elapsed_ms=self._session_elapsed_ms(self._now_ns()),
            turn_elapsed_ms=None,
            turn_id=None,
            message=(
                "session_config activation=%s stt=%s locale=%s hot_path=%s deliberative=%s(%s) fallback=%s(%s) tts=%s(%s voice=%s) playback=%s"
                % (
                    config.get("activation_backend"),
                    config.get("stt_backend"),
                    config.get("stt_locale"),
                    config.get("hot_path_backend"),
                    config.get("deliberative_backend"),
                    config.get("deliberative_model"),
                    config.get("fallback_backend"),
                    config.get("fallback_model"),
                    config.get("tts_backend"),
                    config.get("tts_model"),
                    config.get("tts_voice"),
                    config.get("playback_backend"),
                )
            ),
        )
        self._write_line(line)
        self._write_jsonl(
            {
                "record_type": "session_config",
                "session_id": self.session_id,
                "created_at": now.isoformat(),
                "payload": config,
            }
        )

    def format_conversation_line(
        self,
        prefix: str,
        text: str,
        *,
        turn_id: str | None,
    ) -> str:
        now = datetime.now().astimezone()
        now_ns = self._now_ns()
        return self._format_trace_line(
            timestamp=now,
            session_elapsed_ms=self._session_elapsed_ms(now_ns),
            turn_elapsed_ms=self._turn_elapsed_ms(turn_id, now_ns),
            turn_id=turn_id,
            message="%s %s" % (prefix, text),
        )

    async def _consume_events(self) -> None:
        assert self._subscription is not None
        try:
            while True:
                event = await self._subscription.get()
                self._handle_event(event)
        except asyncio.CancelledError:
            raise

    def _handle_event(self, event: Event) -> None:
        payload = dict(event.payload)
        turn_id = payload.get("turn_id")
        if isinstance(turn_id, str):
            self.register_turn_start(
                turn_id, monotonic_ns=event.created_at_monotonic_ns
            )
            turn_state = self._turns[turn_id]
        else:
            turn_state = None

        if (
            event.event_type == EventType.ACTIVATION_TRIGGERED
            and turn_state is not None
        ):
            turn_state.activation_ns = event.created_at_monotonic_ns
        elif event.event_type == EventType.STT_READY and turn_state is not None:
            turn_state.stt_ready_ns = event.created_at_monotonic_ns
        elif event.event_type == EventType.SPEECH_STARTED and turn_state is not None:
            turn_state.speech_started_ns = event.created_at_monotonic_ns
        elif event.event_type == EventType.SPEECH_ENDED and turn_state is not None:
            turn_state.speech_ended_ns = event.created_at_monotonic_ns
        elif event.event_type == EventType.TURN_READY and turn_state is not None:
            turn_state.turn_ready_ns = event.created_at_monotonic_ns
        elif event.event_type == EventType.ROUTE_SELECTED and turn_state is not None:
            turn_state.route_selected_ns = event.created_at_monotonic_ns
            turn_state.backend = self._as_str(payload.get("backend"))
            turn_state.effective_target = self._as_str(payload.get("effective_target"))
            turn_state.fallback_used = bool(payload.get("fallback_used"))
        elif (
            event.event_type == EventType.ASSISTANT_SENTENCE and turn_state is not None
        ):
            if turn_state.first_sentence_ns is None:
                turn_state.first_sentence_ns = event.created_at_monotonic_ns
            index = payload.get("index")
            if isinstance(index, int):
                turn_state.last_sentence_index = index
        elif event.event_type == EventType.TTS_STARTED and turn_state is not None:
            if turn_state.first_tts_started_ns is None:
                turn_state.first_tts_started_ns = event.created_at_monotonic_ns
            turn_state.tts_backend = self._as_str(
                payload.get("tts_backend") or payload.get("backend")
            )
            turn_state.tts_effective_backend = (
                self._as_str(payload.get("tts_effective_backend"))
                or turn_state.tts_effective_backend
            )
            turn_state.tts_fallback_active = bool(payload.get("tts_fallback_active"))
            turn_state.tts_last_error = self._as_str(payload.get("tts_last_error"))
        elif event.event_type == EventType.PLAYBACK_STARTED and turn_state is not None:
            if turn_state.first_playback_started_ns is None:
                turn_state.first_playback_started_ns = event.created_at_monotonic_ns
            turn_state.playback_backend = self._as_str(payload.get("playback_backend"))
            turn_state.tts_effective_backend = (
                self._as_str(payload.get("tts_effective_backend"))
                or turn_state.tts_effective_backend
            )
            turn_state.tts_fallback_active = bool(payload.get("tts_fallback_active"))
            turn_state.tts_last_error = self._as_str(payload.get("tts_last_error"))
        elif (
            event.event_type == EventType.PLAYBACK_COMPLETED and turn_state is not None
        ):
            turn_state.last_playback_completed_ns = event.created_at_monotonic_ns
            index = payload.get("index")
            if isinstance(index, int):
                turn_state.last_playback_completed_index = index
            turn_state.tts_effective_backend = (
                self._as_str(payload.get("tts_effective_backend"))
                or turn_state.tts_effective_backend
            )
            turn_state.tts_fallback_active = bool(payload.get("tts_fallback_active"))
            turn_state.tts_last_error = self._as_str(payload.get("tts_last_error"))
        elif (
            event.event_type == EventType.ASSISTANT_COMPLETED and turn_state is not None
        ):
            turn_state.assistant_completed_ns = event.created_at_monotonic_ns

        self._write_jsonl(self._event_record(event, turn_id))
        trace_message = self._trace_message_for_event(event)
        if trace_message is not None:
            self._write_line(
                self._format_trace_line(
                    timestamp=event.created_at.astimezone(),
                    session_elapsed_ms=self._session_elapsed_ms(
                        event.created_at_monotonic_ns
                    ),
                    turn_elapsed_ms=self._turn_elapsed_ms(
                        turn_id, event.created_at_monotonic_ns
                    ),
                    turn_id=turn_id if isinstance(turn_id, str) else None,
                    message=trace_message,
                )
            )

        if turn_state is not None:
            summary = self._summarize_turn(turn_id, turn_state)
            if summary is not None:
                self._write_line(
                    self._format_trace_line(
                        timestamp=event.created_at.astimezone(),
                        session_elapsed_ms=self._session_elapsed_ms(
                            event.created_at_monotonic_ns
                        ),
                        turn_elapsed_ms=self._turn_elapsed_ms(
                            turn_id, event.created_at_monotonic_ns
                        ),
                        turn_id=turn_id if isinstance(turn_id, str) else None,
                        message=summary,
                    )
                )

    def _trace_message_for_event(self, event: Event) -> str | None:
        payload = event.payload
        if event.event_type == EventType.ACTIVATION_TRIGGERED:
            return "activation backend=%s source=%s" % (
                payload.get("backend"),
                payload.get("source"),
            )
        if event.event_type == EventType.STT_READY:
            return "stt_ready backend=%s locale=%s sample_rate=%s channels=%s" % (
                payload.get("backend"),
                payload.get("locale"),
                payload.get("sample_rate"),
                payload.get("channel_count"),
            )
        if event.event_type == EventType.SPEECH_STARTED:
            return "speech_started"
        if event.event_type == EventType.SPEECH_ENDED:
            return "speech_ended"
        if event.event_type == EventType.FINAL_TRANSCRIPT:
            text = self._compact_text(payload.get("text"))
            return "final_transcript text=%s" % json.dumps(text, ensure_ascii=True)
        if event.event_type == EventType.TURN_READY:
            return "turn_ready reason=%s used_partial=%s" % (
                payload.get("reason"),
                payload.get("used_partial"),
            )
        if event.event_type == EventType.ROUTE_SELECTED:
            backend_detail = payload.get("backend_detail") or {}
            model = None
            if isinstance(backend_detail, dict):
                model = backend_detail.get("model")
            return (
                "route target=%s effective=%s backend=%s model=%s fallback=%s reason=%s"
                % (
                    payload.get("target"),
                    payload.get("effective_target"),
                    payload.get("backend"),
                    model or "-",
                    "yes" if payload.get("fallback_used") else "no",
                    json.dumps(
                        str(payload.get("policy_reason") or payload.get("reason") or "")
                    ),
                )
            )
        if event.event_type == EventType.TOOL_EXECUTED:
            return "tool_executed tool=%s" % payload.get("tool_name")
        if event.event_type == EventType.INTERRUPTION_REQUESTED:
            return "interrupt reason=%s" % payload.get("reason")
        if event.event_type == EventType.TTS_STARTED:
            return (
                "tts_started backend=%s effective=%s model=%s voice=%s fallback=%s index=%s"
                % (
                    payload.get("tts_backend") or payload.get("backend"),
                    payload.get("tts_effective_backend")
                    or payload.get("tts_backend")
                    or payload.get("backend"),
                    payload.get("tts_model") or "-",
                    payload.get("tts_voice") or "-",
                    "yes" if payload.get("tts_fallback_active") else "no",
                    payload.get("index"),
                )
            )
        if event.event_type == EventType.PLAYBACK_STARTED:
            return (
                "playback_started backend=%s tts_effective=%s fallback=%s index=%s"
                % (
                    payload.get("playback_backend") or "-",
                    payload.get("tts_effective_backend")
                    or payload.get("tts_backend")
                    or payload.get("backend")
                    or "-",
                    "yes" if payload.get("tts_fallback_active") else "no",
                    payload.get("index"),
                )
            )
        if event.event_type == EventType.ERROR:
            return "error message=%s" % json.dumps(
                str(payload.get("message") or ""), ensure_ascii=True
            )
        return None

    def _summarize_turn(self, turn_id: object, state: _TurnTraceState) -> str | None:
        if not isinstance(turn_id, str):
            return None
        if state.summary_emitted:
            return None
        summary_anchor_ns = state.assistant_completed_ns
        if state.first_tts_started_ns is not None:
            if (
                state.last_sentence_index is None
                or state.last_playback_completed_index != state.last_sentence_index
                or state.last_playback_completed_ns is None
            ):
                return None
            summary_anchor_ns = state.last_playback_completed_ns
        if summary_anchor_ns is None:
            return None
        total_ms = self._delta_ms(state.activation_ns, summary_anchor_ns)
        capture_ms = self._delta_ms(state.activation_ns, state.turn_ready_ns)
        route_ms = self._delta_ms(state.turn_ready_ns, state.route_selected_ns)
        first_sentence_ms = self._delta_ms(
            state.route_selected_ns, state.first_sentence_ns
        )
        tts_ms = self._delta_ms(state.first_sentence_ns, state.first_tts_started_ns)
        playback_ms = self._delta_ms(
            state.first_tts_started_ns, state.first_playback_started_ns
        )
        state.summary_emitted = True
        return (
            "summary total=%s capture=%s route=%s first_sentence=%s tts_start=%s playback_start=%s backend=%s effective=%s fallback=%s tts=%s tts_effective=%s tts_fallback=%s playback=%s"
            % (
                self._format_metric(total_ms),
                self._format_metric(capture_ms),
                self._format_metric(route_ms),
                self._format_metric(first_sentence_ms),
                self._format_metric(tts_ms),
                self._format_metric(playback_ms),
                state.backend or "-",
                state.effective_target or "-",
                "yes" if state.fallback_used else "no",
                state.tts_backend or "-",
                state.tts_effective_backend or state.tts_backend or "-",
                "yes" if state.tts_fallback_active else "no",
                state.playback_backend or "-",
            )
        )

    def _event_record(self, event: Event, turn_id: object) -> dict[str, object]:
        now_ns = event.created_at_monotonic_ns
        return {
            "record_type": "event",
            "session_id": self.session_id,
            "turn_id": turn_id if isinstance(turn_id, str) else None,
            "event_type": event.event_type.value,
            "created_at": event.created_at.isoformat(),
            "created_at_monotonic_ns": now_ns,
            "session_elapsed_ms": self._session_elapsed_ms(now_ns),
            "turn_elapsed_ms": self._turn_elapsed_ms(
                turn_id if isinstance(turn_id, str) else None,
                now_ns,
            ),
            "payload": event.payload,
        }

    def _format_trace_line(
        self,
        *,
        timestamp: datetime,
        session_elapsed_ms: int | None,
        turn_elapsed_ms: int | None,
        turn_id: str | None,
        message: str,
    ) -> str:
        prefix = timestamp.strftime("%H:%M:%S.%f")[:-3]
        parts = [prefix]
        if session_elapsed_ms is not None:
            parts.append("session=+%04dms" % session_elapsed_ms)
        if turn_elapsed_ms is not None:
            parts.append("turn=+%04dms" % turn_elapsed_ms)
        if turn_id is not None:
            parts.append("id=%s" % turn_id[:8])
        return "[%s] %s" % (" ".join(parts), message)

    def _session_elapsed_ms(self, now_ns: int) -> int | None:
        if self._session_started_ns is None:
            self._session_started_ns = now_ns
            return 0
        return int((now_ns - self._session_started_ns) / 1_000_000)

    def _turn_elapsed_ms(self, turn_id: str | None, now_ns: int) -> int | None:
        if turn_id is None:
            return None
        state = self._turns.get(turn_id)
        if state is None:
            return None
        return int((now_ns - state.started_ns) / 1_000_000)

    @staticmethod
    def _delta_ms(start_ns: int | None, end_ns: int | None) -> int | None:
        if start_ns is None or end_ns is None:
            return None
        return int((end_ns - start_ns) / 1_000_000)

    @staticmethod
    def _format_metric(value_ms: int | None) -> str:
        if value_ms is None:
            return "-"
        return "%dms" % value_ms

    @staticmethod
    def _compact_text(value: object) -> str:
        text = str(value or "").strip()
        if len(text) <= 120:
            return text
        return "%s..." % text[:117]

    @staticmethod
    def _as_str(value: object) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _default_line_writer(line: str) -> None:
        print(line, file=sys.stderr, flush=True)

    @staticmethod
    def _now_ns() -> int:
        return time.perf_counter_ns()

    def _write_line(self, line: str) -> None:
        if self.mode == "off":
            return
        self._line_writer(line)

    def _write_jsonl(self, record: dict[str, object]) -> None:
        if self._jsonl_handle is None:
            return
        self._jsonl_handle.write(
            json.dumps(record, ensure_ascii=True, default=str) + "\n"
        )
        self._jsonl_handle.flush()
