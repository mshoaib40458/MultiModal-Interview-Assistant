"""
LLM Service using Groq API for question generation and evaluation
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import Groq
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import Config

logger = Config.get_logger(__name__)


class QuestionResponse(BaseModel):
    """Structured question response"""

    question: str = Field(..., description="The interview question")
    category: str = Field(..., description="Question category")
    difficulty: str = Field(
        default="medium", description="Question difficulty"
    )


class EvaluationScore(BaseModel):
    """Structured evaluation response"""

    accuracy: int = Field(..., ge=1, le=10, description="Accuracy score 1-10")
    clarity: int = Field(..., ge=1, le=10, description="Clarity score 1-10")
    preciseness: int = Field(
        ..., ge=1, le=10, description="Preciseness score 1-10"
    )
    feedback: str = Field(..., description="Detailed feedback")
    overall_score: float = Field(..., ge=0, le=10, description="Overall score")


class LLMService:
    """Production-ready LLM service with Groq"""

    def __init__(self):
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not configured")

        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model = Config.GROQ_MODEL
        self.temperature = Config.GROQ_TEMPERATURE
        self.max_tokens = Config.GROQ_MAX_TOKENS

        logger.info(f"LLM Service initialized with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _call_groq(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Make API call to Groq with retry logic"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                response_format=(
                    {"type": "json_object"} if json_mode else {"type": "text"}
                ),
            )

            content = response.choices[0].message.content
            logger.debug(f"Groq API response: {content[:100]}...")
            return content

        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise

    def generate_question(
        self,
        subdomain: str,
        domain_details: Dict[str, Any],
        context: Optional[str] = None,
        follow_up: bool = False,
        question_type: Optional[str] = None,
        asked_questions: Optional[List[str]] = None,
        topics: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate interview question using Groq"""

        asked_questions = asked_questions or []

        # Build prompt
        if follow_up and context:
            prompt = f"""You are an expert technical interviewer for {subdomain}.

Based on the candidate's previous response: "{context}"

Generate ONE insightful follow-up question that:
1. Dives deeper into their answer
2. Tests their practical understanding
3. Is specific and technical
4. Ends with a question mark

Previous questions asked (DO NOT repeat):
{chr(10).join(f"- {q}" for q in asked_questions[-3:])}

Return ONLY the question, nothing else."""
        else:
            topic_guidance = ""
            if question_type and topics and question_type in topics:
                topic_guidance = f"\nFocus on: {topics[question_type]}"

            prompt = f"""{domain_details.get('llm_guidance', '')}

You are conducting a technical interview for: {subdomain}

Generate ONE {question_type or 'technical'} question that:
1. Tests deep understanding of {subdomain}
2. Is clear and specific
3. Encourages detailed answers
4. Is appropriate for the role
5. Ends with a question mark
{topic_guidance}

Previous questions asked (DO NOT repeat):
{chr(10).join(f"- {q}" for q in asked_questions[-5:])}

Return ONLY the question, nothing else."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert technical interviewer. Generate clear, specific interview questions.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._call_groq(messages, temperature=0.8)

            # Clean up response
            question = response.strip()

            # Ensure it's a question
            if not question.endswith("?"):
                question += "?"

            # Remove any numbering or prefixes
            question = question.lstrip("0123456789.-) ")

            logger.info(f"Generated question: {question[:50]}...")
            return question

        except Exception as e:
            logger.error(f"Failed to generate question: {str(e)}")
            # Fallback question
            return f"Can you describe your experience with {subdomain}?"

    def evaluate_answer(self, question: str, answer: str) -> EvaluationScore:
        """Evaluate candidate's answer using Groq with structured output"""

        prompt = f"""Evaluate this interview response:

Question: {question}
Answer: {answer}

Provide a structured evaluation with:
1. Accuracy (1-10): How correct and relevant is the answer?
2. Clarity (1-10): How clear and well-articulated is the response?
3. Preciseness (1-10): How concise and to-the-point is the answer?
4. Feedback: 2-3 sentences of constructive feedback
5. Overall Score (0-10): Weighted average

Return as JSON with keys: accuracy, clarity, preciseness, feedback, overall_score

Be fair but critical. Reward depth and practical knowledge."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert technical interviewer evaluating candidate responses. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._call_groq(
                messages, temperature=0.3, json_mode=True
            )

            # Parse JSON response
            data = json.loads(response)

            # Calculate overall score if not provided
            if "overall_score" not in data:
                data["overall_score"] = round(
                    (data["accuracy"] + data["clarity"] + data["preciseness"])
                    / 3,
                    1,
                )

            evaluation = EvaluationScore(**data)
            logger.info(
                f"Evaluation complete: Overall {evaluation.overall_score}/10"
            )
            return evaluation

        except Exception as e:
            logger.error(f"Failed to evaluate answer: {str(e)}")
            # Return default evaluation
            return EvaluationScore(
                accuracy=5,
                clarity=5,
                preciseness=5,
                feedback="Unable to evaluate response due to technical error.",
                overall_score=5.0,
            )

    def should_follow_up(self, response: str) -> bool:
        """Determine if response warrants a follow-up question"""
        keywords = [
            "experience",
            "approach",
            "method",
            "challenge",
            "solution",
            "details",
            "job",
            "internship",
            "project",
            "problem",
            "implementation",
            "design",
            "optimization",
            "tool",
            "process",
            "framework",
            "worked",
            "developed",
            "built",
            "created",
        ]

        response_lower = response.lower()
        keyword_count = sum(1 for kw in keywords if kw in response_lower)

        # Follow up if response is detailed (>50 words) and mentions experience
        word_count = len(response.split())
        should_follow = keyword_count >= 2 and word_count > 50

        logger.debug(
            f"Follow-up decision: {should_follow} (keywords: {keyword_count}, words: {word_count})"
        )
        return should_follow


# Singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
