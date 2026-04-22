
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
from pathlib import Path
import os

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent))

from src.core.ser_service import SERService

@pytest.fixture
def ser_service():
    with patch('config.Config.HF_API_KEY', 'fake-key'):
        return SERService()

@pytest.mark.asyncio
async def test_analyze_audio_async(ser_service):
    # Mock httpx AsyncClient
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"label": "happy", "score": 0.9}]
    
    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        # Create a dummy file
        with open("test_audio.wav", "wb") as f:
            f.write(b"dummy data")
            
        try:
            result = await ser_service.analyze_audio_async("test_audio.wav")
            assert result['dominant_emotion'] == "happy"
            assert result['confidence'] == 0.9
        finally:
            if os.path.exists("test_audio.wav"):
                os.remove("test_audio.wav")

@pytest.mark.asyncio
async def test_analyze_multiple_files_async(ser_service):
    # Mock analyze_audio_async
    ser_service.analyze_audio_async = AsyncMock(side_effect=[
        {"dominant_emotion": "happy", "score": 0.9},
        {"dominant_emotion": "sad", "score": 0.8}
    ])
    
    results = await ser_service.analyze_multiple_files_async(["f1.wav", "f2.wav"])
    assert len(results) == 2
    assert results[0]['dominant_emotion'] == "happy"
    assert results[1]['dominant_emotion'] == "sad"
