from jarvis.core.turn_manager import TurnManager, TurnManagerConfig


class TestTurnManager:
    def test_completes_final_transcript_after_speech_end(self):
        manager = TurnManager(TurnManagerConfig(silence_timeout_ms=800))

        assert manager.consume_event({"type": "speech_started"}, now=0.0) is None
        assert (
            manager.consume_event(
                {"type": "partial_transcript", "text": "que horas sao"}, now=0.1
            )
            is None
        )
        assert manager.consume_event({"type": "speech_ended"}, now=0.2) is None
        completed = manager.consume_event(
            {"type": "final_transcript", "text": "Que horas sao agora?"},
            now=0.25,
        )

        assert completed is not None
        assert completed is not None
        assert completed.text == "Que horas sao agora?"
        assert not completed.used_partial

    def test_commits_partial_after_trailing_silence(self):
        manager = TurnManager(
            TurnManagerConfig(silence_timeout_ms=800, partial_commit_min_chars=10)
        )

        manager.consume_event({"type": "speech_started"}, now=0.0)
        manager.consume_event(
            {"type": "partial_transcript", "text": "Isso fecha uma frase."},
            now=0.1,
        )
        manager.consume_event({"type": "speech_ended"}, now=0.2)
        completed = manager.tick(now=1.1)

        assert completed is not None
        assert completed is not None
        assert completed.text == "Isso fecha uma frase."
        assert completed.used_partial

    def test_does_not_commit_incomplete_partial_clause(self):
        manager = TurnManager(
            TurnManagerConfig(silence_timeout_ms=800, partial_commit_min_chars=10)
        )

        manager.consume_vad_signal(True, now=0.0)
        manager.consume_partial_transcript("Eu estava pensando em", now=0.1)
        manager.consume_vad_signal(False, now=0.2)

        completed = manager.tick(now=1.2)
        assert completed is None

    def test_requires_partial_to_be_stable_before_committing_after_silence(self):
        manager = TurnManager(
            TurnManagerConfig(
                silence_timeout_ms=100,
                partial_commit_min_chars=10,
                partial_stability_ms=250,
            )
        )

        manager.consume_vad_signal(True, now=0.0)
        manager.consume_partial_transcript("Isso fecha uma frase.", now=0.05)
        manager.consume_vad_signal(False, now=0.06)

        assert manager.tick(now=0.18) is None

        completed = manager.tick(now=0.31)
        assert completed is not None
        assert completed is not None
        assert completed.text == "Isso fecha uma frase."
        assert completed.metadata["silence_duration_ms"] == 250
