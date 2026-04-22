"""
Health check and monitoring endpoints
Production monitoring and observability
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import psutil

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import Config

logger = Config.get_logger(__name__)


class HealthCheck:
    """System health monitoring"""

    def __init__(self):
        self.start_time = datetime.utcnow()
        self.total_requests = 0
        self.failed_requests = 0
        self.api_call_history = []

    def get_uptime(self) -> timedelta:
        """Get system uptime"""
        return datetime.utcnow() - self.start_time

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_root = Path.cwd().anchor or "/"
            disk = psutil.disk_usage(disk_root)

            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": psutil.cpu_count(),
                },
                "memory": {
                    "total_mb": memory.total / (1024 * 1024),
                    "available_mb": memory.available / (1024 * 1024),
                    "used_mb": memory.used / (1024 * 1024),
                    "percent": memory.percent,
                },
                "disk": {
                    "total_gb": disk.total / (1024 * 1024 * 1024),
                    "used_gb": disk.used / (1024 * 1024 * 1024),
                    "free_gb": disk.free / (1024 * 1024 * 1024),
                    "percent": disk.percent,
                },
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}

    def check_dependencies(self) -> Dict[str, Dict[str, Any]]:
        """Check all service dependencies"""
        checks = {}

        # Database configuration
        database_url = Config.DATABASE_URL
        if database_url.startswith("sqlite:///"):
            database_path = Path(database_url.replace("sqlite:///", "", 1))
            checks["database"] = {
                "status": "configured",
                "url": "sqlite",
                "path": str(database_path),
                "healthy": database_path.parent.exists(),
            }
        elif database_url:
            checks["database"] = {
                "status": "configured",
                "url": (
                    database_url.split("@")[-1]
                    if "@" in database_url
                    else database_url.split("://", 1)[0]
                ),
                "healthy": True,
            }
        else:
            checks["database"] = {
                "status": "not_configured",
                "healthy": False,
            }

        # API Keys
        checks["api_keys"] = {
            "groq": {
                "configured": bool(Config.GROQ_API_KEY),
                "format_valid": (
                    Config.GROQ_API_KEY.startswith("gsk_")
                    if Config.GROQ_API_KEY
                    else False
                ),
            },
            "huggingface": {
                "configured": bool(Config.HF_API_KEY),
                "format_valid": (
                    Config.HF_API_KEY.startswith("hf_")
                    if Config.HF_API_KEY
                    else False
                ),
            },
        }

        # Directories
        checks["directories"] = {
            "domains": Config.DOMAINS_DIR.exists(),
            "output": Config.OUTPUT_DIR.exists(),
            "logs": Config.LOGS_DIR.exists(),
        }

        # Models/Libraries
        checks["models"] = {}

        # Check spaCy
        try:
            import spacy

            try:
                _nlp = spacy.load("en_core_web_sm")
                checks["models"]["spacy"] = {
                    "status": "loaded",
                    "healthy": True,
                }
            except Exception:
                checks["models"]["spacy"] = {
                    "status": "model_missing",
                    "healthy": False,
                }
        except Exception as e:
            checks["models"]["spacy"] = {
                "status": "error",
                "error": str(e),
                "healthy": False,
            }

        # Check DeepFace
        try:
            from deepface import DeepFace

            checks["models"]["deepface"] = {
                "status": "available",
                "healthy": True,
            }
        except Exception as e:
            checks["models"]["deepface"] = {
                "status": "error",
                "error": str(e),
                "healthy": False,
            }

        # Check MediaPipe
        try:
            import mediapipe as mp

            checks["models"]["mediapipe"] = {
                "status": "available",
                "healthy": True,
            }
        except Exception as e:
            checks["models"]["mediapipe"] = {
                "status": "error",
                "error": str(e),
                "healthy": False,
            }

        # Check OpenCV
        try:
            import cv2

            checks["models"]["opencv"] = {
                "status": "available",
                "version": cv2.__version__,
                "healthy": True,
            }
        except Exception as e:
            checks["models"]["opencv"] = {
                "status": "error",
                "error": str(e),
                "healthy": False,
            }

        return checks

    def get_full_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        uptime = self.get_uptime()
        metrics = self.get_system_metrics()
        dependencies = self.check_dependencies()

        # Determine overall health
        all_healthy = all(
            dep.get("healthy", True)
            for dep in dependencies.values()
            if isinstance(dep, dict) and "healthy" in dep
        )

        # Check models
        models_healthy = all(
            model.get("healthy", True)
            for model in dependencies.get("models", {}).values()
        )

        overall_status = (
            "healthy" if (all_healthy and models_healthy) else "degraded"
        )

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": uptime.total_seconds(),
            "uptime_formatted": str(uptime),
            "environment": Config.ENVIRONMENT,
            "debug_mode": Config.DEBUG,
            "version": "2.0",
            "system_metrics": metrics,
            "dependencies": dependencies,
            "statistics": {
                "total_requests": self.total_requests,
                "failed_requests": self.failed_requests,
                "success_rate": (self.total_requests - self.failed_requests)
                / max(self.total_requests, 1)
                * 100,
            },
        }

    def record_request(self, success: bool = True):
        """Record API request for statistics"""
        self.total_requests += 1
        if not success:
            self.failed_requests += 1

    def export_metrics(self, format: str = "json") -> str:
        """Export health metrics in various formats"""
        status = self.get_full_status()

        if format == "json":
            return json.dumps(status, indent=2)
        elif format == "prometheus":
            # Prometheus format
            metrics = []
            metrics.append(
                f"# HELP interview_uptime_seconds System uptime in seconds"
            )
            metrics.append(f"# TYPE interview_uptime_seconds gauge")
            metrics.append(
                f'interview_uptime_seconds {status["uptime_seconds"]}'
            )

            metrics.append(
                f"# HELP interview_requests_total Total number of requests"
            )
            metrics.append(f"# TYPE interview_requests_total counter")
            metrics.append(f"interview_requests_total {self.total_requests}")

            metrics.append(f"# HELP interview_requests_failed Failed requests")
            metrics.append(f"# TYPE interview_requests_failed counter")
            metrics.append(f"interview_requests_failed {self.failed_requests}")

            if status["system_metrics"]:
                cpu = status["system_metrics"]["cpu"]["usage_percent"]
                metrics.append(
                    f"# HELP interview_cpu_usage CPU usage percentage"
                )
                metrics.append(f"# TYPE interview_cpu_usage gauge")
                metrics.append(f"interview_cpu_usage {cpu}")

                mem = status["system_metrics"]["memory"]["percent"]
                metrics.append(
                    f"# HELP interview_memory_usage Memory usage percentage"
                )
                metrics.append(f"# TYPE interview_memory_usage gauge")
                metrics.append(f"interview_memory_usage {mem}")

            return "\n".join(metrics)
        else:
            return str(status)


# Singleton instance
_health_check = None


def get_health_check() -> HealthCheck:
    """Get health check singleton"""
    global _health_check
    if _health_check is None:
        _health_check = HealthCheck()
    return _health_check


def run_health_check() -> Dict[str, Any]:
    """Run complete health check and return results"""
    health = get_health_check()
    return health.get_full_status()


if __name__ == "__main__":
    # CLI health check
    health = get_health_check()
    status = health.get_full_status()

    print("\n" + "=" * 60)
    print("AI Interview System Health Check")
    print("=" * 60)
    print(f"Status: {status['status'].upper()}")
    print(f"Uptime: {status['uptime_formatted']}")
    print(f"Environment: {status['environment']}")
    print(f"Debug Mode: {status['debug_mode']}")
    print("\nSystem Metrics:")
    if status["system_metrics"]:
        print(
            f"  CPU Usage: {status['system_metrics']['cpu']['usage_percent']:.1f}%"
        )
        print(
            f"  Memory: {status['system_metrics']['memory']['percent']:.1f}%"
        )
        print(f"  Disk: {status['system_metrics']['disk']['percent']:.1f}%")

    print("\nDependencies:")
    for name, check in status["dependencies"].items():
        if isinstance(check, dict) and "status" in check:
            print(f"  {name}: {check['status']}")
        else:
            print(f"  {name}: {check}")

    print("=" * 60)

    # Exit with appropriate code
    exit(0 if status["status"] == "healthy" else 1)
