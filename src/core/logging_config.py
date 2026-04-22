"""
Enhanced structured logging with context and correlation
Production-ready logging setup with multiple handlers
"""

import json
import logging
import sys
import threading
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

# Thread-local storage for request context
_context = threading.local()


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context if available
        if hasattr(_context, "session_id"):
            log_data["session_id"] = _context.session_id
        if hasattr(_context, "user_id"):
            log_data["user_id"] = _context.user_id
        if hasattr(_context, "request_id"):
            log_data["request_id"] = _context.request_id

        # Add exception info if present
        if record.exc_info:
            exc_type = record.exc_info[0]
            log_data["exception"] = {
                "type": exc_type.__name__ if exc_type else "Exception",
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        return json.dumps(log_data)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console output for development"""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


class ProductionLogger:
    """Production-ready logger setup"""

    _initialized = False
    _loggers: Dict[str, logging.Logger] = {}

    @classmethod
    def setup(
        cls,
        name: str,
        log_dir: Path,
        level: str = "INFO",
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = False,
    ) -> logging.Logger:
        """
        Setup production logger

        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Log level
            enable_console: Enable console output
            enable_file: Enable file output
            enable_json: Enable JSON structured logging
        """
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        logger.handlers = []  # Clear existing handlers

        # Console handler (colored for development)
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))

            if sys.stdout.isatty():  # Terminal supports colors
                console_formatter: logging.Formatter = ColoredConsoleFormatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            else:
                console_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )

            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # File handler (rotating)
        if enable_file:
            log_dir.mkdir(parents=True, exist_ok=True)

            # Main log file (rotates daily)
            log_file = log_dir / f"{name}.log"
            file_handler = TimedRotatingFileHandler(
                log_file,
                when="midnight",
                interval=1,
                backupCount=30,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)

            if enable_json:
                file_formatter: logging.Formatter = StructuredFormatter()
            else:
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )

            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            # Error log file (only errors and above)
            error_log_file = log_dir / f"{name}_errors.log"
            error_handler = RotatingFileHandler(
                error_log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_formatter)
            logger.addHandler(error_handler)

        cls._loggers[name] = logger
        return logger

    @classmethod
    def set_context(cls, **kwargs):
        """Set logging context for current thread"""
        for key, value in kwargs.items():
            setattr(_context, key, value)

    @classmethod
    def clear_context(cls):
        """Clear logging context"""
        for attr in ["session_id", "user_id", "request_id"]:
            if hasattr(_context, attr):
                delattr(_context, attr)


class AuditLogger:
    """Separate audit trail logger for compliance"""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Audit log (never rotates, permanent record)
        audit_file = log_dir / "audit.log"
        handler = logging.FileHandler(audit_file, encoding="utf-8")
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)

    def log_event(
        self,
        event_type: str,
        action: str,
        status: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log audit event"""
        event = {
            "event_type": event_type,
            "action": action,
            "status": status,
            "session_id": session_id,
            "user_id": user_id,
            "resource": resource,
            "metadata": metadata or {},
        }

        self.logger.info(json.dumps(event))


# Singleton instances
_production_logger = None
_audit_logger = None


def get_production_logger(name: str, **kwargs) -> logging.Logger:
    """Get or create production logger"""
    from config import Config

    return ProductionLogger.setup(
        name=name,
        log_dir=Config.LOGS_DIR,
        level=Config.LOG_LEVEL,
        enable_json=Config.ENVIRONMENT == "production",
        **kwargs,
    )


def get_audit_logger() -> AuditLogger:
    """Get audit logger singleton"""
    global _audit_logger
    if _audit_logger is None:
        from config import Config

        _audit_logger = AuditLogger(Config.LOGS_DIR)
    return _audit_logger


# Performance logging decorator
def log_performance(func):
    """Decorator to log function execution time"""
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_production_logger(func.__module__)
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise

    return wrapper


# Async performance logging
def log_async_performance(func):
    """Decorator to log async function execution time"""
    import time
    from functools import wraps

    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = get_production_logger(func.__module__)
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(
                f"{func.__name__} (async) completed in {elapsed:.3f}s"
            )
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"{func.__name__} (async) failed after {elapsed:.3f}s: {e}"
            )
            raise

    return wrapper
