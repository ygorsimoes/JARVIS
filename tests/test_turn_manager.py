import unittest

from jarvis.core.turn_manager import TurnManager, TurnManagerConfig


class TurnManagerTests(unittest.TestCase):
    def test_completes_final_transcript_after_speech_end(self):
        manager = TurnManager(TurnManagerConfig(silence_timeout_ms=800))

        self.assertIsNone(manager.consume_event({"type": "speech_started"}, now=0.0))
        self.assertIsNone(
            manager.consume_event({"type": "partial_transcript", "text": "que horas sao"}, now=0.1)
        )
        self.assertIsNone(manager.consume_event({"type": "speech_ended"}, now=0.2))
        completed = manager.consume_event(
            {"type": "final_transcript", "text": "Que horas sao agora?"},
            now=0.25,
        )

        self.assertIsNotNone(completed)
        assert completed is not None
        self.assertEqual(completed.text, "Que horas sao agora?")
        self.assertFalse(completed.used_partial)

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

        self.assertIsNotNone(completed)
        assert completed is not None
        self.assertEqual(completed.text, "Isso fecha uma frase.")
        self.assertTrue(completed.used_partial)

    def test_does_not_commit_incomplete_partial_clause(self):
        manager = TurnManager(
            TurnManagerConfig(silence_timeout_ms=800, partial_commit_min_chars=10)
        )

        manager.consume_vad_signal(True, now=0.0)
        manager.consume_partial_transcript("Eu estava pensando em", now=0.1)
        manager.consume_vad_signal(False, now=0.2)

        completed = manager.tick(now=1.2)
        self.assertIsNone(completed)


if __name__ == "__main__":
    unittest.main()
