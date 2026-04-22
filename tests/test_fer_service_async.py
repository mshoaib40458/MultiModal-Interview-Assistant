
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
from pathlib import Path

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent))

from src.core.fer_service import FERService

@pytest.fixture
def fer_service():
    return FERService()

@pytest.mark.asyncio
async def test_analyze_video_async(fer_service):
    # Mock synchronous analyze_video
    with patch.object(FERService, 'analyze_video', return_value=[{"dominant_emotion": "happy"}]) as mock_sync:
        result = await fer_service.analyze_video_async("test_video.mp4")
        assert result == [{"dominant_emotion": "happy"}]
        mock_sync.assert_called_once_with("test_video.mp4", 30)
