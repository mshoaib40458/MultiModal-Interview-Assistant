
import pytest
import sys
from pathlib import Path

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent))

from src.core.fusion_service import LateFusionService

@pytest.fixture
def fusion_service():
    return LateFusionService()

def test_normalize_emotion(fusion_service):
    assert fusion_service.normalize_emotion("Happy") == "happy"
    assert fusion_service.normalize_emotion("JOYFUL") == "happy"
    assert fusion_service.normalize_emotion("Angry") == "angry"
    assert fusion_service.normalize_emotion("Sadness") == "sad"
    assert fusion_service.normalize_emotion("Unknown") == "neutral"

def test_fuse_emotions_none(fusion_service):
    result = fusion_service.fuse_emotions(None, None)
    assert result['fused_emotion'] == 'neutral'
    assert result['fusion_method'] == 'none'

def test_fuse_emotions_ser_only(fusion_service):
    ser_res = {'dominant_emotion': 'happy', 'confidence': 0.8}
    result = fusion_service.fuse_emotions(ser_res, None)
    assert result['fused_emotion'] == 'happy'
    assert result['fusion_method'] == 'ser_only'
    assert result['confidence'] == 0.8

def test_fuse_emotions_fer_only(fusion_service):
    fer_res = {'dominant_emotion': 'sad', 'confidence': 0.7}
    result = fusion_service.fuse_emotions(None, fer_res)
    assert result['fused_emotion'] == 'sad'
    assert result['fusion_method'] == 'fer_only'
    assert result['confidence'] == 0.7

def test_fuse_emotions_agreement(fusion_service):
    ser_res = {'dominant_emotion': 'happy', 'confidence': 0.8}
    fer_res = {'dominant_emotion': 'happy', 'confidence': 0.6}
    result = fusion_service.fuse_emotions(ser_res, fer_res)
    assert result['fused_emotion'] == 'happy'
    assert result['fusion_method'] == 'agreement'
    assert result['agreement'] is True
    # Confidence should be boosted: (0.8*0.6 + 0.6*0.4) * 1.2 = 0.72 * 1.2 = 0.864
    assert result['confidence'] == pytest.approx(0.864)

def test_fuse_emotions_disagreement_ser_wins(fusion_service):
    ser_res = {'dominant_emotion': 'happy', 'confidence': 0.9}
    fer_res = {'dominant_emotion': 'sad', 'confidence': 0.4}
    result = fusion_service.fuse_emotions(ser_res, fer_res)
    assert result['fused_emotion'] == 'happy'
    assert result['fusion_method'] == 'weighted'
    assert result['agreement'] is False
    # ser_score = 0.9 * 0.6 = 0.54
    # fer_score = 0.4 * 0.4 = 0.16
    assert result['confidence'] == pytest.approx(0.54)

def test_get_fusion_summary_empty(fusion_service):
    summary = fusion_service.get_fusion_summary([])
    assert summary['dominant_emotion'] == 'neutral'
    assert summary['average_confidence'] == 0.0

def test_get_fusion_summary_populated(fusion_service):
    fused_results = [
        {'fused_emotion': 'happy', 'confidence': 0.9, 'agreement': True},
        {'fused_emotion': 'happy', 'confidence': 0.8, 'agreement': True},
        {'fused_emotion': 'neutral', 'confidence': 0.5, 'agreement': False}
    ]
    summary = fusion_service.get_fusion_summary(fused_results)
    assert summary['dominant_emotion'] == 'happy'
    assert summary['total_samples'] == 3
    assert summary['average_confidence'] == pytest.approx(0.7333, abs=1e-3)
    assert summary['agreement_rate'] == pytest.approx(66.6666, abs=1e-3)
