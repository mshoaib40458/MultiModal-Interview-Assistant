"""
Facial Emotion Recognition (FER) Service
Uses DeepFace for real-time emotion detection
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import Config

logger = Config.get_logger(__name__)


class FERService:
    """Facial Emotion Recognition Service"""

    EMOTIONS = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]

    def __init__(self):
        self.model_name = "Emotion"
        self.detector_backend = "opencv"
        logger.info("FER Service initialized")

    def analyze_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Analyze single frame for emotions"""
        try:
            # DeepFace analyze
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True,
            )

            if isinstance(result, list):
                result = result[0]

            emotion_scores = result.get("emotion", {})
            dominant_emotion = result.get("dominant_emotion", "neutral")

            logger.debug(f"Detected emotion: {dominant_emotion}")

            return {
                "dominant_emotion": dominant_emotion,
                "emotion_scores": emotion_scores,
                "confidence": emotion_scores.get(dominant_emotion, 0.0)
                / 100.0,
                "region": result.get("region", {}),
            }

        except Exception as e:
            logger.warning(f"FER analysis failed: {str(e)}")
            return None

    async def analyze_video_async(
        self, video_path: str, sample_rate: int = 30
    ) -> List[Dict]:
        """Analyze video file for emotions asynchronously"""
        import asyncio

        return await asyncio.to_thread(
            self.analyze_video, video_path, sample_rate
        )

    def analyze_video(
        self, video_path: str, sample_rate: int = 30
    ) -> List[Dict]:
        """Analyze video file for emotions at intervals"""

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        # Get frame properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Fallback

        frame_interval = max(
            1, int(fps // sample_rate)
        )  # Sample every N frames

        emotions = []
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    result = self.analyze_frame(frame)
                    if result:
                        result["timestamp"] = frame_count / fps
                        result["frame_number"] = frame_count
                        emotions.append(result)

                frame_count += 1

            logger.info(
                f"Analyzed {len(emotions)} frames from video ({video_path})"
            )
            return emotions

        finally:
            cap.release()

    def get_emotion_summary(self, emotions: List[Dict]) -> Dict:
        """Get summary statistics of emotions"""
        if not emotions:
            return {
                "dominant_overall": "neutral",
                "emotion_distribution": {},
                "average_confidence": 0.0,
                "total_frames": 0,
            }

        # Count emotions
        emotion_counts = {}
        total_confidence = 0.0

        for emotion_data in emotions:
            emotion = emotion_data["dominant_emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_confidence += emotion_data["confidence"]

        # Calculate percentages
        total_frames = len(emotions)
        emotion_distribution = {
            emotion: (count / total_frames) * 100
            for emotion, count in emotion_counts.items()
        }

        # Get dominant overall emotion
        dominant_overall = max(emotion_counts.items(), key=lambda x: x[1])[0]

        return {
            "dominant_overall": dominant_overall,
            "emotion_distribution": emotion_distribution,
            "average_confidence": total_confidence / total_frames,
            "total_frames": total_frames,
            "emotion_timeline": [
                {
                    "timestamp": e["timestamp"],
                    "emotion": e["dominant_emotion"],
                    "confidence": e["confidence"],
                }
                for e in emotions
            ],
        }


# Singleton instance
_fer_service: Optional[FERService] = None


def get_fer_service() -> FERService:
    """Get or create FER service instance"""
    global _fer_service
    if _fer_service is None:
        _fer_service = FERService()
    return _fer_service
