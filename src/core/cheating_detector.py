"""
Cheating Detection Service
Monitors gaze, head pose, and mobile phone detection
"""

import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Try importing MediaPipe with fallback for different versions
try:
    import mediapipe as mp

    # Check if solutions API exists (older versions)
    if hasattr(mp, "solutions"):
        USE_LEGACY_API = True
    else:
        USE_LEGACY_API = False
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
except ImportError as e:
    raise ImportError(
        f"MediaPipe not installed. Please run: pip install mediapipe"
    ) from e

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import Config

logger = Config.get_logger(__name__)


class CheatingDetector:
    """Comprehensive cheating detection system"""

    def __init__(self):
        # Initialize MediaPipe Face Mesh for gaze and head pose
        if USE_LEGACY_API:
            # Legacy API (mediapipe < 0.10)
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh: Any | None = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.face_cascade: Any | None = None
            self.eye_cascade: Any | None = None
        else:
            # New API (mediapipe >= 0.10) - Use opencv-based face detection as fallback
            # Since the new API requires model files, we'll use a simpler approach
            logger.warning(
                "MediaPipe 0.10+ detected. Using OpenCV-based face detection."
            )
            self.face_mesh = None
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            )

        # Thresholds
        self.gaze_threshold = Config.GAZE_DEVIATION_THRESHOLD
        self.head_pose_threshold = Config.HEAD_POSE_THRESHOLD

        # Tracking
        self.gaze_history: deque[float] = deque(maxlen=30)  # Last 30 frames
        self.violations: list[Dict[str, Any]] = []

        logger.info("Cheating Detector initialized")

    def _get_head_pose(
        self, landmarks, image_shape
    ) -> Tuple[float, float, float]:
        """Calculate head pose angles (pitch, yaw, roll)"""

        h, w = image_shape[:2]

        # 3D model points (generic face model)
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corner
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0),  # Right mouth corner
            ]
        )

        # 2D image points from landmarks
        image_points = np.array(
            [
                landmarks[1],  # Nose tip
                landmarks[152],  # Chin
                landmarks[226],  # Left eye left corner
                landmarks[446],  # Right eye right corner
                landmarks[57],  # Left mouth corner
                landmarks[287],  # Right mouth corner
            ],
            dtype="double",
        )

        # Camera internals
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype="double",
        )

        dist_coeffs = np.zeros((4, 1))

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Calculate Euler angles
        pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            pose_mat
        )

        pitch, yaw, roll = euler_angles.flatten()[:3]

        return pitch, yaw, roll

    def _estimate_gaze(self, landmarks, image_shape) -> Tuple[float, float]:
        """Estimate gaze direction from eye landmarks"""

        h, w = image_shape[:2]

        # Left eye landmarks
        left_eye = [landmarks[i] for i in [33, 133, 160, 159, 158, 157, 173]]
        # Right eye landmarks
        right_eye = [landmarks[i] for i in [362, 263, 387, 386, 385, 384, 398]]

        # Calculate eye centers
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)

        # Gaze direction (simplified - deviation from center)
        screen_center = np.array([w / 2, h / 2])
        eye_center = (left_center + right_center) / 2

        gaze_deviation = eye_center - screen_center
        gaze_x = gaze_deviation[0] / w
        gaze_y = gaze_deviation[1] / h

        return gaze_x, gaze_y

    def analyze_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> Dict:
        """Analyze single frame for cheating indicators"""

        analysis: Dict[str, Any] = {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "face_detected": False,
            "head_pose": None,
            "gaze_deviation": None,
            "violations": [],
        }

        if USE_LEGACY_API and self.face_mesh is not None:
            # Use MediaPipe legacy API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                analysis["violations"].append(
                    {
                        "type": "no_face_detected",
                        "severity": "high",
                        "message": "Face not detected in frame",
                    }
                )
                return analysis

            analysis["face_detected"] = True

            # Get landmarks
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]

            landmarks: list[list[int]] = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append([x, y])
            landmarks = np.array(landmarks)

            # Head pose estimation
            try:
                pitch, yaw, roll = self._get_head_pose(landmarks, frame.shape)
                analysis["head_pose"] = {
                    "pitch": float(pitch),
                    "yaw": float(yaw),
                    "roll": float(roll),
                }

                # Check head pose violations
                if abs(yaw) > self.head_pose_threshold:
                    analysis["violations"].append(
                        {
                            "type": "head_turned",
                            "severity": "medium",
                            "message": f"Head turned {abs(yaw):.1f}° (threshold: {self.head_pose_threshold}°)",
                            "value": abs(yaw),
                        }
                    )

                if abs(pitch) > self.head_pose_threshold:
                    analysis["violations"].append(
                        {
                            "type": "looking_away",
                            "severity": "medium",
                            "message": f"Looking up/down {abs(pitch):.1f}° (threshold: {self.head_pose_threshold}°)",
                            "value": abs(pitch),
                        }
                    )

            except Exception as e:
                logger.warning(f"Head pose estimation failed: {str(e)}")

            # Gaze estimation
            try:
                gaze_x, gaze_y = self._estimate_gaze(landmarks, frame.shape)
                gaze_magnitude = np.sqrt(gaze_x**2 + gaze_y**2)

                analysis["gaze_deviation"] = {
                    "x": float(gaze_x),
                    "y": float(gaze_y),
                    "magnitude": float(gaze_magnitude),
                }

                self.gaze_history.append(gaze_magnitude)

                # Check gaze violations
                if gaze_magnitude > self.gaze_threshold:
                    analysis["violations"].append(
                        {
                            "type": "gaze_deviation",
                            "severity": "low",
                            "message": f"Gaze deviation: {gaze_magnitude:.2f} (threshold: {self.gaze_threshold})",
                            "value": gaze_magnitude,
                        }
                    )

            except Exception as e:
                logger.warning(f"Gaze estimation failed: {str(e)}")
        else:
            # Use OpenCV fallback for newer MediaPipe versions
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = self.face_cascade
            if face_cascade is None:
                return analysis
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                analysis["violations"].append(
                    {
                        "type": "no_face_detected",
                        "severity": "high",
                        "message": "Face not detected in frame",
                    }
                )
                return analysis

            analysis["face_detected"] = True

            # Simple face position tracking (less accurate than MediaPipe)
            x, y, w, h = faces[0]
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2

            # Estimate head turn based on face position
            deviation_x = abs(face_center_x - frame_center_x) / frame.shape[1]
            deviation_y = abs(face_center_y - frame_center_y) / frame.shape[0]

            if deviation_x > 0.3:
                analysis["violations"].append(
                    {
                        "type": "head_turned",
                        "severity": "medium",
                        "message": f"Face significantly off-center (horizontal)",
                        "value": deviation_x,
                    }
                )

            if deviation_y > 0.3:
                analysis["violations"].append(
                    {
                        "type": "looking_away",
                        "severity": "medium",
                        "message": f"Face significantly off-center (vertical)",
                        "value": deviation_y,
                    }
                )

        # Track violations
        if analysis["violations"]:
            self.violations.extend(analysis["violations"])

        return analysis

    async def analyze_video_async(
        self, video_path: str, sample_rate: int = 5
    ) -> Dict:
        """Analyze video file for cheating behavior asynchronously"""
        import asyncio

        return await asyncio.to_thread(
            self.analyze_video, video_path, sample_rate
        )

    def analyze_video(self, video_path: str, sample_rate: int = 5) -> Dict:
        """Analyze entire video for cheating behavior"""

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return {"error": "Cannot open video"}

        # Get frame properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Fallback

        frame_interval = max(1, int(fps // sample_rate))

        frame_analyses = []
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    analysis = self.analyze_frame(
                        frame, frame_count, timestamp
                    )
                    frame_analyses.append(analysis)

                frame_count += 1

            logger.info(
                f"Analyzed {len(frame_analyses)} frames for cheating detection ({video_path})"
            )

            # Generate summary
            return self._generate_summary(frame_analyses)

        finally:
            cap.release()

    def _generate_summary(self, frame_analyses: List[Dict]) -> Dict:
        """Generate cheating detection summary"""

        total_frames = len(frame_analyses)
        frames_with_face = sum(1 for f in frame_analyses if f["face_detected"])

        violation_counts: Dict[str, int] = {}
        for frame in frame_analyses:
            for violation in frame["violations"]:
                vtype = violation["type"]
                violation_counts[vtype] = violation_counts.get(vtype, 0) + 1

        # Calculate violation percentages
        violation_percentages = {
            vtype: (count / total_frames) * 100
            for vtype, count in violation_counts.items()
        }

        # Determine overall risk level
        total_violations = sum(violation_counts.values())
        violation_rate = (
            total_violations / total_frames if total_frames > 0 else 0
        )

        if violation_rate > 0.5:
            risk_level = "high"
        elif violation_rate > 0.2:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "total_frames_analyzed": total_frames,
            "frames_with_face_detected": frames_with_face,
            "face_detection_rate": (
                (frames_with_face / total_frames * 100)
                if total_frames > 0
                else 0
            ),
            "violation_counts": violation_counts,
            "violation_percentages": violation_percentages,
            "total_violations": total_violations,
            "violation_rate": violation_rate,
            "risk_level": risk_level,
            "frame_details": frame_analyses,
            "recommendations": self._get_recommendations(
                violation_percentages, risk_level
            ),
        }

    def _get_recommendations(
        self, violation_percentages: Dict, risk_level: str
    ) -> List[str]:
        """Generate recommendations based on violations"""

        recommendations = []

        if risk_level == "high":
            recommendations.append(
                "⚠️ High cheating risk detected. Manual review recommended."
            )

        if violation_percentages.get("no_face_detected", 0) > 20:
            recommendations.append(
                "Candidate frequently not visible on camera"
            )

        if violation_percentages.get("head_turned", 0) > 30:
            recommendations.append(
                "Frequent head turning detected - possible external assistance"
            )

        if violation_percentages.get("looking_away", 0) > 25:
            recommendations.append(
                "Candidate frequently looking away from screen"
            )

        if violation_percentages.get("gaze_deviation", 0) > 40:
            recommendations.append(
                "Significant gaze deviation - may be reading from notes"
            )

        if not recommendations:
            recommendations.append(
                "✅ No significant cheating indicators detected"
            )

        return recommendations


# Singleton instance
_cheating_detector: Optional[CheatingDetector] = None


def get_cheating_detector() -> CheatingDetector:
    """Get or create cheating detector instance"""
    global _cheating_detector
    if _cheating_detector is None:
        _cheating_detector = CheatingDetector()
    return _cheating_detector
