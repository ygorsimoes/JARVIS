from __future__ import annotations

from typing import Dict, Optional


class SpeechDetectorAdapter:
    @classmethod
    def classify_event(cls, event: Dict[str, object]) -> Optional[dict]:
        event_type = event.get("type")
        if event_type == "speech_started":
            return {
                "speech_detected": True,
                "source": "speech_started",
                "confidence": None,
            }
        if event_type == "speech_ended":
            return {
                "speech_detected": False,
                "source": "speech_ended",
                "confidence": None,
            }
        if event_type == "speech_detector_result":
            confidence = event.get("confidence")
            return {
                "speech_detected": bool(event.get("speech_detected")),
                "source": "speech_detector_result",
                "confidence": confidence,
            }
        return None

    @classmethod
    def event_has_speech(cls, event: Dict[str, object]) -> bool:
        classified = cls.classify_event(event)
        return bool(classified and classified["speech_detected"])

    @classmethod
    def event_is_silence(cls, event: Dict[str, object]) -> bool:
        classified = cls.classify_event(event)
        return bool(classified and not classified["speech_detected"])

    @classmethod
    def signal_reason(cls, event: Dict[str, object]) -> Optional[str]:
        classified = cls.classify_event(event)
        if classified is None:
            return None
        return str(classified["source"])
