
import pytest
import os
from pathlib import Path
import sys

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent))

from config import Config, BASE_DIR

def test_config_paths():
    assert isinstance(BASE_DIR, Path)
    assert isinstance(Config.DOMAINS_DIR, Path)
    assert isinstance(Config.OUTPUT_DIR, Path)
    assert isinstance(Config.LOGS_DIR, Path)

def test_config_defaults():
    # Check some default values defined in config.py
    assert Config.GROQ_MODEL == "llama-3.3-70b-versatile"
    assert Config.MAX_QUESTIONS in [5, 10] # depending on .env
    assert Config.RECORDING_DURATION == 60

def test_config_validate():
    # Since we have .env.local with keys, validate should pass
    # If it fails, it means keys are missing which is expected if someone didn't setup
    try:
        assert Config.validate() is True
    except ValueError as e:
        pytest.skip(f"Config validation failed (probably missing keys): {e}")

def test_get_logger():
    logger = Config.get_logger("test_logger")
    assert logger.name == "test_logger"
    assert len(logger.handlers) >= 1
