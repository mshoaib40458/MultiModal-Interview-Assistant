
import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent))

from src.core.llm_service import LLMService, QuestionResponse, EvaluationScore

@pytest.fixture
def mock_config():
    with patch('config.Config.GROQ_API_KEY', 'fake-key'):
        yield

@pytest.fixture
def llm_service(mock_config):
    with patch('src.core.llm_service.Groq') as mock_groq:
        service = LLMService()
        service.client = MagicMock()
        return service

def test_generate_question(llm_service):
    # Mock the Groq client response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"question": "How are you?", "category": "General", "difficulty": "easy"}'
    
    llm_service.client.chat.completions.create.return_value = mock_response
    
    question = llm_service.generate_question(
        subdomain="AI",
        domain_details={"topics": "ML"},
        asked_questions=[]
    )
    
    assert "How are you?" in question
    assert llm_service.client.chat.completions.create.called

def test_generate_question_follow_up(llm_service):
    # Mock the Groq client response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "What about Docker?"
    
    llm_service.client.chat.completions.create.return_value = mock_response
    
    question = llm_service.generate_question(
        subdomain="AI",
        domain_details={"topics": "ML"},
        context="I use Python.",
        follow_up=True
    )
    
    assert "What about Docker?" in question

def test_evaluate_answer(llm_service):
    # Mock the Groq client response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"accuracy": 8, "clarity": 9, "preciseness": 7, "feedback": "Good job", "overall_score": 8.0}'
    
    llm_service.client.chat.completions.create.return_value = mock_response
    
    evaluation = llm_service.evaluate_answer("What is AI?", "Artificial Intelligence")
    
    assert evaluation.accuracy == 8
    assert evaluation.overall_score == 8.0
    assert evaluation.feedback == "Good job"

def test_should_follow_up(llm_service):
    # Short answer - no follow up
    assert llm_service.should_follow_up("Yes.") is False
    
    # Long answer with experience keywords - should follow up
    detailed_answer = "In my previous project, I had a great experience implementing a machine learning solution. We worked on deep learning models and built a full pipeline."
    # Check word count > 50 and keywords >= 2
    # experience, project, worked, built, deep learning are keywords
    assert llm_service.should_follow_up(detailed_answer + " " + "word " * 60) is True
