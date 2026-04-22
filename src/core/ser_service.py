"""
Speech Emotion Recognition (SER) Service - Production Version
Uses HuggingFace API for emotion detection from audio
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import Config

logger = Config.get_logger(__name__)


class SERService:
    """Speech Emotion Recognition Service"""

    API_URL = "https://router.huggingface.co/models/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

    EMOTION_MAPPING = {
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "sad": "sad",
        "surprise": "surprise",
        "neutral": "neutral",
    }

    def __init__(self):
        if not Config.HF_API_KEY:
            raise ValueError("HF_API_KEY not configured")

        self.headers = {"Authorization": f"Bearer {Config.HF_API_KEY}"}
        logger.info("SER Service initialized")

    async def analyze_audio_async(self, audio_path: str) -> Optional[Dict]:
        """Analyze audio file for emotion asynchronously"""
        import httpx

        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None

        try:
            with open(audio_path, "rb") as f:
                data = f.read()

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.API_URL,
                    headers=self.headers,
                    content=data,
                    timeout=30.0,
                )

            if response.status_code == 200:
                results = response.json()

                if isinstance(results, list) and results:
                    dominant = max(results, key=lambda x: x.get("score", 0))

                    emotion_data = {
                        "filename": os.path.basename(audio_path),
                        "dominant_emotion": dominant.get("label", "neutral"),
                        "confidence": dominant.get("score", 0.0),
                        "all_predictions": results,
                    }

                    logger.info(
                        f"SER Async Analysis: {emotion_data['dominant_emotion']} ({emotion_data['confidence']:.2f})"
                    )
                    return emotion_data
                else:
                    logger.warning(
                        f"Unexpected API response format: {results}"
                    )
                    return None

            elif response.status_code == 503:
                logger.warning(
                    "Model is loading, please retry in a few seconds"
                )
                return None
            else:
                logger.error(
                    f"API Error {response.status_code}: {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"SER async analysis failed: {str(e)}")
            return None

    async def analyze_multiple_files_async(
        self, audio_files: List[str]
    ) -> List[Dict]:
        """Analyze multiple audio files concurrently"""
        import asyncio

        tasks = [self.analyze_audio_async(f) for f in audio_files]
        results = await asyncio.gather(*tasks)

        # Filter out None results
        valid_results = [r for r in results if r is not None]

        logger.info(
            f"Async: Analyzed {len(valid_results)}/{len(audio_files)} audio files"
        )
        return valid_results

    def analyze_audio(self, audio_path: str) -> Optional[Dict]:
        """Analyze audio file for emotion (Synchronous wrapper)"""
        # Kept for backward compatibility
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we are already in an async loop, we should probably be using analyze_audio_async
                # But as a fallback we can't easily wait here.
                # For now, keep the requests implementation for the sync method.
                with open(audio_path, "rb") as f:
                    data = f.read()
                response = requests.post(
                    self.API_URL, headers=self.headers, data=data, timeout=30
                )
                if response.status_code == 200:
                    results = response.json()
                    if isinstance(results, list) and results:
                        dominant = max(
                            results, key=lambda x: x.get("score", 0)
                        )
                        return {
                            "filename": os.path.basename(audio_path),
                            "dominant_emotion": dominant.get(
                                "label", "neutral"
                            ),
                            "confidence": dominant.get("score", 0.0),
                            "all_predictions": results,
                        }
                return None
            else:
                return asyncio.run(self.analyze_audio_async(audio_path))
        except Exception:
            # Fallback to requests if asyncio fails
            import requests

            try:
                with open(audio_path, "rb") as f:
                    data = f.read()
                response = requests.post(
                    self.API_URL, headers=self.headers, data=data, timeout=30
                )
                if response.status_code == 200:
                    results = response.json()
                    if isinstance(results, list) and results:
                        dominant = max(
                            results, key=lambda x: x.get("score", 0)
                        )
                        return {
                            "filename": os.path.basename(audio_path),
                            "dominant_emotion": dominant.get(
                                "label", "neutral"
                            ),
                            "confidence": dominant.get("score", 0.0),
                            "all_predictions": results,
                        }
                return None
            except Exception:
                return None

    def analyze_multiple_files(self, audio_files: List[str]) -> List[Dict]:
        """Analyze multiple audio files (Synchronous wrapper)"""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Fallback to sequential sync loop if already running
                results = []
                for f in audio_files:
                    res = self.analyze_audio(f)
                    if res:
                        results.append(res)
                return results
            return asyncio.run(self.analyze_multiple_files_async(audio_files))
        except Exception:
            results = []
            for f in audio_files:
                res = self.analyze_audio(f)
                if res:
                    results.append(res)
            return results

    def get_emotion_summary(self, emotion_results: List[Dict]) -> Dict:
        """Get summary of emotions across multiple files"""

        if not emotion_results:
            return {
                "dominant_overall": "neutral",
                "emotion_distribution": {},
                "average_confidence": 0.0,
                "total_samples": 0,
            }

        # Count emotions
        emotion_counts: Dict[str, int] = {}
        total_confidence = 0.0

        for result in emotion_results:
            emotion = result["dominant_emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_confidence += result["confidence"]

        # Calculate distribution
        total_samples = len(emotion_results)
        emotion_distribution = {
            emotion: (count / total_samples) * 100
            for emotion, count in emotion_counts.items()
        }

        # Get dominant overall
        dominant_overall = max(emotion_counts.items(), key=lambda x: x[1])[0]

        return {
            "dominant_overall": dominant_overall,
            "emotion_distribution": emotion_distribution,
            "average_confidence": total_confidence / total_samples,
            "total_samples": total_samples,
            "emotion_timeline": [
                {
                    "filename": r["filename"],
                    "emotion": r["dominant_emotion"],
                    "confidence": r["confidence"],
                }
                for r in emotion_results
            ],
        }


# Singleton instance
_ser_service: Optional[SERService] = None


def get_ser_service() -> SERService:
    """Get or create SER service instance"""
    global _ser_service
    if _ser_service is None:
        _ser_service = SERService()
    return _ser_service
