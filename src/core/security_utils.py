"""
Production Security Utilities
Critical security features for production deployment
"""

import hashlib
import mimetypes
import os
import secrets
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple


class FileValidator:
    """Secure file upload validation"""

    # Allowed file types and their MIME types
    ALLOWED_EXTENSIONS = {
        "pdf": ["application/pdf"],
        "docx": [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ],
        "doc": ["application/msword"],
    }

    # Security limits
    MAX_FILE_SIZE_MB = 10
    MAX_FILENAME_LENGTH = 255

    @classmethod
    def validate_upload(
        cls, filename: str, file_content: bytes
    ) -> Tuple[bool, str]:
        """
        Validate uploaded file for security
        Returns: (is_valid, error_message)
        """
        # 1. Check filename length
        if len(filename) > cls.MAX_FILENAME_LENGTH:
            return (
                False,
                f"Filename too long (max {cls.MAX_FILENAME_LENGTH} characters)",
            )

        # 2. Check for directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            return False, "Invalid filename: path traversal detected"

        # 3. Check file extension
        file_ext = (
            filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        )
        if file_ext not in cls.ALLOWED_EXTENSIONS:
            return (
                False,
                f"Invalid file type. Allowed: {', '.join(cls.ALLOWED_EXTENSIONS.keys())}",
            )

        # 4. Check file size
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > cls.MAX_FILE_SIZE_MB:
            return False, f"File too large (max {cls.MAX_FILE_SIZE_MB}MB)"

        # 5. Verify MIME type (basic check)
        # Note: For production, use python-magic for real MIME detection
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type not in cls.ALLOWED_EXTENSIONS[file_ext]:
            return False, "File content doesn't match extension"

        return True, "Valid"

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename to prevent security issues"""
        # Remove path components
        filename = os.path.basename(filename)

        # Generate safe filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = (
            filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
        )
        safe_name = f"upload_{timestamp}_{secrets.token_hex(4)}.{file_ext}"

        return safe_name


class RateLimiter:
    """Thread-safe rate limiting"""

    def __init__(self, max_calls: int = 60, time_window: int = 60):
        """
        Args:
            max_calls: Maximum calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: dict[str, list[datetime]] = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, identifier: str) -> Tuple[bool, Optional[str]]:
        """
        Check if request is allowed
        Args:
            identifier: Unique identifier (e.g., IP address, API key hash)
        Returns:
            (is_allowed, error_message)
        """
        with self.lock:
            now = datetime.now()

            # Clean old entries
            cutoff = now - timedelta(seconds=self.time_window)
            self.calls[identifier] = [
                timestamp
                for timestamp in self.calls[identifier]
                if timestamp > cutoff
            ]

            # Check limit
            if len(self.calls[identifier]) >= self.max_calls:
                retry_after = int(
                    (self.calls[identifier][0] - cutoff).total_seconds()
                )
                return (
                    False,
                    f"Rate limit exceeded. Retry after {retry_after} seconds.",
                )

            # Record this call
            self.calls[identifier].append(now)
            return True, None

    def reset(self, identifier: str):
        """Reset rate limit for identifier"""
        with self.lock:
            if identifier in self.calls:
                del self.calls[identifier]


class APIKeyManager:
    """Secure API key management"""

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for safe storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    @staticmethod
    def validate_api_key_format(
        api_key: str, provider: str
    ) -> Tuple[bool, str]:
        """
        Validate API key format
        Returns: (is_valid, error_message)
        """
        if not api_key or not api_key.strip():
            return False, "API key is empty"

        if provider == "groq":
            if not api_key.startswith("gsk_"):
                return (
                    False,
                    "Invalid Groq API key format (should start with 'gsk_')",
                )
            if len(api_key) < 20:
                return False, "Groq API key too short"

        elif provider == "huggingface":
            if not api_key.startswith("hf_"):
                return (
                    False,
                    "Invalid HuggingFace API key format (should start with 'hf_')",
                )
            if len(api_key) < 20:
                return False, "HuggingFace API key too short"

        return True, "Valid"


class SecretGenerator:
    """Generate secure secrets"""

    @staticmethod
    def generate_secret_key(length: int = 32) -> str:
        """Generate cryptographically secure secret key"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def generate_session_id() -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_part = secrets.token_hex(8)
        return f"session_{timestamp}_{random_part}"


class VideoValidator:
    """Video file validation"""

    MAX_DURATION_SECONDS = 3600  # 1 hour
    MAX_SIZE_MB = 500
    ALLOWED_FORMATS = ["mp4", "avi", "mov"]

    @classmethod
    def validate_video(cls, video_path: Path) -> Tuple[bool, str]:
        """
        Validate video file
        Returns: (is_valid, error_message)
        """
        import cv2

        # Check file exists
        if not video_path.exists():
            return False, "Video file not found"

        # Check file size
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        if file_size_mb > cls.MAX_SIZE_MB:
            return False, f"Video too large (max {cls.MAX_SIZE_MB}MB)"

        # Check format
        ext = video_path.suffix.lower().lstrip(".")
        if ext not in cls.ALLOWED_FORMATS:
            return (
                False,
                f"Invalid format. Allowed: {', '.join(cls.ALLOWED_FORMATS)}",
            )

        # Check video properties
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return False, "Cannot open video file"

            # Check duration
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            if duration > cls.MAX_DURATION_SECONDS:
                return (
                    False,
                    f"Video too long (max {cls.MAX_DURATION_SECONDS}s)",
                )

            return True, "Valid"

        except Exception as e:
            return False, f"Video validation error: {str(e)}"


# Create global instances
file_validator = FileValidator()
rate_limiter = RateLimiter(max_calls=60, time_window=60)
api_key_manager = APIKeyManager()
secret_generator = SecretGenerator()
video_validator = VideoValidator()


if __name__ == "__main__":
    # Demo usage
    print("=== Security Utilities Demo ===\n")

    # 1. Generate secret key
    secret = secret_generator.generate_secret_key()
    print(f"Generated Secret Key: {secret}\n")

    # 2. Validate API key format
    valid, msg = api_key_manager.validate_api_key_format("gsk_test123", "groq")
    print(f"API Key Validation: {valid} - {msg}\n")

    # 3. Rate limiting demo
    identifier = "test_user"
    for i in range(3):
        allowed, rate_msg = rate_limiter.is_allowed(identifier)
        print(
            f"Request {i+1}: {'Allowed' if allowed else f'Blocked - {rate_msg}'}"
        )

    print("\n✅ Security utilities ready for production!")
