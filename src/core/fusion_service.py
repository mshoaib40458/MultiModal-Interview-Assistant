"""
Late Fusion Service
Combines SER and FER for enhanced emotion recognition
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import Config

logger = Config.get_logger(__name__)


class LateFusionService:
    """Late fusion of Speech and Facial Emotion Recognition"""

    # Emotion mapping for consistency
    EMOTIONS = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]

    # Fusion weights (can be tuned)
    SER_WEIGHT = 0.6  # Speech often more reliable
    FER_WEIGHT = 0.4

    def __init__(self):
        logger.info("Late Fusion Service initialized")

    def normalize_emotion(self, emotion: str) -> str:
        """Normalize emotion labels"""
        emotion_lower = emotion.lower()

        # Handle variations
        if "ang" in emotion_lower:
            return "angry"
        elif "disg" in emotion_lower:
            return "disgust"
        elif "fear" in emotion_lower or "afraid" in emotion_lower:
            return "fear"
        elif "happ" in emotion_lower or "joy" in emotion_lower:
            return "happy"
        elif "sad" in emotion_lower:
            return "sad"
        elif "surpr" in emotion_lower:
            return "surprise"
        else:
            return "neutral"

    def fuse_emotions(
        self, ser_result: Optional[Dict], fer_result: Optional[Dict]
    ) -> Dict:
        """Fuse SER and FER results using late fusion"""

        # Handle missing data
        if not ser_result and not fer_result:
            return {
                "fused_emotion": "neutral",
                "confidence": 0.0,
                "ser_emotion": None,
                "fer_emotion": None,
                "fusion_method": "none",
            }

        if not ser_result:
            assert fer_result is not None
            return {
                "fused_emotion": self.normalize_emotion(
                    fer_result["dominant_emotion"]
                ),
                "confidence": fer_result["confidence"],
                "ser_emotion": None,
                "fer_emotion": fer_result["dominant_emotion"],
                "fusion_method": "fer_only",
            }

        if not fer_result:
            assert ser_result is not None
            return {
                "fused_emotion": self.normalize_emotion(
                    ser_result["dominant_emotion"]
                ),
                "confidence": ser_result["confidence"],
                "ser_emotion": ser_result["dominant_emotion"],
                "fer_emotion": None,
                "fusion_method": "ser_only",
            }

        # Both available - perform fusion
        ser_emotion = self.normalize_emotion(ser_result["dominant_emotion"])
        fer_emotion = self.normalize_emotion(fer_result["dominant_emotion"])

        # If emotions agree, high confidence
        if ser_emotion == fer_emotion:
            fused_confidence = (
                ser_result["confidence"] * self.SER_WEIGHT
                + fer_result["confidence"] * self.FER_WEIGHT
            )

            return {
                "fused_emotion": ser_emotion,
                "confidence": min(
                    fused_confidence * 1.2, 1.0
                ),  # Boost for agreement
                "ser_emotion": ser_emotion,
                "fer_emotion": fer_emotion,
                "fusion_method": "agreement",
                "agreement": True,
            }

        # Emotions disagree - weighted fusion
        # Use confidence-weighted voting
        ser_score = ser_result["confidence"] * self.SER_WEIGHT
        fer_score = fer_result["confidence"] * self.FER_WEIGHT

        if ser_score > fer_score:
            fused_emotion = ser_emotion
            fused_confidence = ser_score
        else:
            fused_emotion = fer_emotion
            fused_confidence = fer_score

        return {
            "fused_emotion": fused_emotion,
            "confidence": fused_confidence,
            "ser_emotion": ser_emotion,
            "fer_emotion": fer_emotion,
            "fusion_method": "weighted",
            "agreement": False,
            "ser_confidence": ser_result["confidence"],
            "fer_confidence": fer_result["confidence"],
        }

    def fuse_multiple(
        self, ser_results: List[Dict], fer_results: List[Dict]
    ) -> List[Dict]:
        """Fuse multiple SER and FER results"""

        # Match by filename or index
        fused_results = []

        # Create lookup for FER results
        fer_lookup = {
            r.get("filename", f"frame_{i}"): r
            for i, r in enumerate(fer_results)
        }

        for ser_result in ser_results:
            filename = ser_result.get("filename", "")
            fer_result = fer_lookup.get(filename)

            fused = self.fuse_emotions(ser_result, fer_result)
            fused["filename"] = filename
            fused_results.append(fused)

        return fused_results

    def get_fusion_summary(self, fused_results: List[Dict]) -> Dict:
        """Get summary of fused emotion results"""

        if not fused_results:
            return {
                "dominant_emotion": "neutral",
                "average_confidence": 0.0,
                "agreement_rate": 0.0,
                "emotion_distribution": {},
            }

        # Count emotions
        emotion_counts: Dict[str, int] = {}
        total_confidence = 0.0
        agreement_count = 0

        for result in fused_results:
            emotion = result["fused_emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_confidence += result["confidence"]

            if result.get("agreement", False):
                agreement_count += 1

        # Calculate stats
        total_samples = len(fused_results)
        emotion_distribution = {
            emotion: (count / total_samples) * 100
            for emotion, count in emotion_counts.items()
        }

        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]

        return {
            "dominant_emotion": dominant_emotion,
            "average_confidence": total_confidence / total_samples,
            "agreement_rate": (agreement_count / total_samples) * 100,
            "emotion_distribution": emotion_distribution,
            "total_samples": total_samples,
            "fusion_timeline": [
                {
                    "filename": r.get("filename"),
                    "emotion": r["fused_emotion"],
                    "confidence": r["confidence"],
                    "agreement": r.get("agreement", False),
                }
                for r in fused_results
            ],
        }


# Singleton instance
_fusion_service: Optional[LateFusionService] = None


def get_fusion_service() -> LateFusionService:
    """Get or create fusion service instance"""
    global _fusion_service
    if _fusion_service is None:
        _fusion_service = LateFusionService()
    return _fusion_service
