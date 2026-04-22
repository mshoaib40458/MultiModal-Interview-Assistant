"""
Centralized configuration management for AI Interview
Production-ready configuration with comprehensive validation
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from the project root, not the caller's working directory
# Priority: .env.local (local dev) > .env (template)
load_dotenv(BASE_DIR / ".env.local", override=True)
load_dotenv(BASE_DIR / ".env")


class Config:
    """Application configuration"""

    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    HF_API_KEY: str = os.getenv("HF_API_KEY", "")

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Interview Settings
    MAX_QUESTIONS: int = int(os.getenv("MAX_QUESTIONS", "10"))
    RECORDING_DURATION: int = int(os.getenv("RECORDING_DURATION", "60"))
    VIDEO_FPS: int = int(os.getenv("VIDEO_FPS", "30"))
    AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))

    # Directories
    DOMAINS_DIR: Path = BASE_DIR / os.getenv("DOMAINS_DIR", "Domains")
    OUTPUT_DIR: Path = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs")
    LOGS_DIR: Path = BASE_DIR / os.getenv("LOGS_DIR", "logs")

    # Security
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY", "dev-secret-key-change-in-production"
    )
    ALLOWED_HOSTS: list = os.getenv(
        "ALLOWED_HOSTS", "localhost,127.0.0.1"
    ).split(",")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    MAX_VIDEO_DURATION: int = int(os.getenv("MAX_VIDEO_DURATION", "3600"))

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR}/interview_data.db"
    )

    # Rate Limiting
    MAX_API_CALLS_PER_MINUTE: int = int(
        os.getenv("MAX_API_CALLS_PER_MINUTE", "60")
    )
    MAX_UPLOADS_PER_HOUR: int = int(os.getenv("MAX_UPLOADS_PER_HOUR", "10"))

    # Model Settings
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.7"))
    GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "1024"))

    # Cheating Detection Thresholds
    GAZE_DEVIATION_THRESHOLD: float = float(
        os.getenv("GAZE_DEVIATION_THRESHOLD", "0.3")
    )
    HEAD_POSE_THRESHOLD: float = float(
        os.getenv("HEAD_POSE_THRESHOLD", "30.0")
    )
    MOBILE_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("MOBILE_CONFIDENCE_THRESHOLD", "0.5")
    )

    # Monitoring (optional)
    SENTRY_DSN: str = os.getenv("SENTRY_DSN", "")

    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration"""
        errors = []
        warnings = []

        # API Key Validation
        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is not set")
        elif not cls.GROQ_API_KEY.startswith("gsk_"):
            warnings.append(
                "GROQ_API_KEY format looks incorrect (should start with 'gsk_')"
            )

        if not cls.HF_API_KEY:
            errors.append("HF_API_KEY is not set")
        elif not cls.HF_API_KEY.startswith("hf_"):
            warnings.append(
                "HF_API_KEY format looks incorrect (should start with 'hf_')"
            )

        # Production Security Checks
        if cls.ENVIRONMENT == "production":
            if cls.DEBUG:
                errors.append(
                    "DEBUG is enabled in production - this MUST be False!"
                )

            weak_keys = [
                "dev-secret-key-change-in-production",
                "your-secret-key-here-change-in-production",
                "CHANGE_THIS_IN_PRODUCTION_GENERATE_STRONG_KEY",
            ]
            if cls.SECRET_KEY in weak_keys:
                errors.append(
                    "SECRET_KEY is using default value - CRITICAL SECURITY RISK!"
                )

            if len(cls.SECRET_KEY) < 32:
                warnings.append(
                    "SECRET_KEY is too short for production (should be 32+ characters)"
                )

            if cls.LOG_LEVEL == "DEBUG":
                warnings.append(
                    "LOG_LEVEL=DEBUG in production may expose sensitive information"
                )

            # Check for SQLite in production
            if cls.DATABASE_URL.startswith("sqlite"):
                warnings.append(
                    "Using SQLite in production - PostgreSQL is recommended"
                )

        # Create required directories
        for dir_path in [cls.OUTPUT_DIR, cls.LOGS_DIR, cls.DOMAINS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Display errors and warnings
        if errors:
            error_msg = "Configuration errors:\n" + "\n".join(
                f"  ❌ {e}" for e in errors
            )
            print(f"\n{'='*60}")
            print("⚠️  CONFIGURATION ERRORS DETECTED")
            print(f"{'='*60}")
            print(error_msg)
            print(f"{'='*60}\n")

            if cls.ENVIRONMENT == "production":
                raise ValueError(error_msg)
            else:
                print(
                    "⚠️  Running in development mode with configuration issues"
                )
                print(
                    "Please copy .env.example to .env.local and fill in your values\n"
                )

        if warnings:
            print(f"\n{'='*60}")
            print("⚠️  CONFIGURATION WARNINGS")
            print(f"{'='*60}")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
            print(f"{'='*60}\n")

        return len(errors) == 0

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get configured logger (legacy support)"""
        # Use new logging system if available
        try:
            from src.core.logging_config import get_production_logger

            return get_production_logger(name)
        except ImportError:
            # Fallback to basic logging
            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, cls.LOG_LEVEL))

            if not logger.handlers:
                # Console handler
                console_handler = logging.StreamHandler()
                console_handler.setLevel(getattr(logging, cls.LOG_LEVEL))

                # File handler
                log_file = cls.LOGS_DIR / f"{name}.log"
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)

                # Formatter
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                console_handler.setFormatter(formatter)
                file_handler.setFormatter(formatter)

                logger.addHandler(console_handler)
                logger.addHandler(file_handler)

            return logger

    @classmethod
    def initialize_sentry(cls):
        """Initialize Sentry error tracking (optional)"""
        if cls.SENTRY_DSN and cls.ENVIRONMENT == "production":
            try:
                import sentry_sdk

                sentry_sdk.init(
                    dsn=cls.SENTRY_DSN,
                    environment=cls.ENVIRONMENT,
                    traces_sample_rate=0.1,
                )
                print("✅ Sentry error tracking initialized")
            except ImportError:
                print("⚠️  Sentry SDK not installed, error tracking disabled")


# Validate configuration on import
try:
    Config.validate()

    # Initialize optional services
    Config.initialize_sentry()

except ValueError:
    if Config.ENVIRONMENT == "production":
        raise
    # In development, just warn
    pass
