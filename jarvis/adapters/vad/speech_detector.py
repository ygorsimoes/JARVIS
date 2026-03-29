from __future__ import annotations

from typing import Dict


class SpeechDetectorAdapter:
    @staticmethod
    def event_has_speech(event: Dict[str, object]) -> bool:
        if event.get("type") == "speech_started":
            return True
        if event.get("type") == "speech_detector_result":
            return bool(event.get("speech_detected"))
        return False

    @staticmethod
    def event_is_silence(event: Dict[str, object]) -> bool:
        if event.get("type") == "speech_ended":
            return True
        if event.get("type") == "speech_detector_result":
            return not bool(event.get("speech_detected"))
        return False
