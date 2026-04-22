"""
Resume Parser Service
Extracts information from PDF/DOCX resumes
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import docx
import PyPDF2
import spacy
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import Config

logger = Config.get_logger(__name__)


class ResumeData(BaseModel):
    """Structured resume data"""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = []
    experience: List[Dict[str, str]] = []
    education: List[Dict[str, str]] = []
    projects: List[Dict[str, str]] = []
    summary: Optional[str] = None


class ResumeParser:
    """Resume parsing and information extraction"""

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            logger.warning(
                "Spacy model not found. Run: python -m spacy download en_core_web_sm"
            )
            self.nlp = None

        # Regex patterns
        self.email_pattern = (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )
        self.phone_pattern = (
            r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        )

        # Common skill keywords
        self.skill_keywords = {
            "python",
            "java",
            "javascript",
            "c++",
            "c#",
            "ruby",
            "php",
            "swift",
            "kotlin",
            "go",
            "rust",
            "typescript",
            "sql",
            "nosql",
            "mongodb",
            "postgresql",
            "mysql",
            "redis",
            "docker",
            "kubernetes",
            "aws",
            "azure",
            "gcp",
            "tensorflow",
            "pytorch",
            "keras",
            "scikit-learn",
            "pandas",
            "numpy",
            "react",
            "angular",
            "vue",
            "node.js",
            "django",
            "flask",
            "fastapi",
            "spring",
            "git",
            "ci/cd",
            "jenkins",
            "linux",
            "machine learning",
            "deep learning",
            "nlp",
            "computer vision",
            "data science",
            "data analysis",
            "api",
            "rest",
            "graphql",
            "agile",
            "scrum",
            "devops",
            "microservices",
            "blockchain",
            "ai",
            "ml",
        }

        logger.info("Resume Parser initialized")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

            logger.info(f"Extracted {len(text)} characters from PDF")
            return text

        except Exception as e:
            logger.error(f"Failed to extract PDF text: {str(e)}")
            return ""

    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])

            logger.info(f"Extracted {len(text)} characters from DOCX")
            return text

        except Exception as e:
            logger.error(f"Failed to extract DOCX text: {str(e)}")
            return ""

    def extract_contact_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract contact information"""

        # Email
        email_match = re.search(self.email_pattern, text)
        email = email_match.group(0) if email_match else None

        # Phone
        phone_match = re.search(self.phone_pattern, text)
        phone = phone_match.group(0) if phone_match else None

        # Name (first line or first PERSON entity)
        name = None
        if self.nlp:
            doc = self.nlp(text[:500])  # Check first 500 chars
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text
                    break

        if not name:
            # Fallback: first non-empty line
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if lines:
                name = lines[0]

        return {"name": name, "email": email, "phone": phone}

    def extract_skills(self, text: str) -> List[str]:
        """Extract technical skills"""

        text_lower = text.lower()
        found_skills = []

        for skill in self.skill_keywords:
            if skill in text_lower:
                found_skills.append(skill.title())

        # Remove duplicates and sort
        found_skills = sorted(list(set(found_skills)))

        logger.info(f"Extracted {len(found_skills)} skills")
        return found_skills

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract resume sections"""

        sections: Dict[str, str] = {}
        current_section = "summary"
        current_content: List[str] = []

        # Common section headers
        section_keywords = {
            "experience": [
                "experience",
                "work history",
                "employment",
                "professional experience",
            ],
            "education": ["education", "academic", "qualification"],
            "projects": ["projects", "portfolio", "work samples"],
            "skills": ["skills", "technical skills", "competencies"],
            "summary": ["summary", "objective", "profile", "about"],
        }

        lines = text.split("\n")

        for line in lines:
            line_lower = line.lower().strip()

            # Check if line is a section header
            section_found = False
            for section_name, keywords in section_keywords.items():
                if any(keyword in line_lower for keyword in keywords):
                    # Save previous section
                    if current_content:
                        sections[current_section] = "\n".join(current_content)

                    current_section = section_name
                    current_content = []
                    section_found = True
                    break

            if not section_found and line.strip():
                current_content.append(line.strip())

        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content)

        return sections

    def parse_experience(self, experience_text: str) -> List[Dict[str, str]]:
        """Parse work experience section"""

        experiences = []

        # Split by common delimiters
        entries = re.split(r"\n\s*\n", experience_text)

        for entry in entries:
            if len(entry.strip()) < 20:  # Skip very short entries
                continue

            exp_data = {
                "description": entry.strip(),
                "title": "",
                "company": "",
                "duration": "",
            }

            # Try to extract title (first line often)
            lines = [
                line.strip() for line in entry.split("\n") if line.strip()
            ]
            if lines:
                exp_data["title"] = lines[0]

            # Look for date patterns
            date_pattern = r"(\d{4}|\w+\s+\d{4})\s*[-–]\s*(\d{4}|\w+\s+\d{4}|present|current)"
            date_match = re.search(date_pattern, entry, re.IGNORECASE)
            if date_match:
                exp_data["duration"] = date_match.group(0)

            experiences.append(exp_data)

        return experiences[:5]  # Limit to 5 most recent

    def parse_resume(self, file_path: str | Path) -> ResumeData:
        """Parse resume file and extract structured data"""

        resume_path = Path(file_path)

        if not resume_path.exists():
            logger.error(f"Resume file not found: {resume_path}")
            return ResumeData()

        # Extract text based on file type
        if resume_path.suffix.lower() == ".pdf":
            text = self.extract_text_from_pdf(str(resume_path))
        elif resume_path.suffix.lower() in [".docx", ".doc"]:
            text = self.extract_text_from_docx(str(resume_path))
        else:
            logger.error(f"Unsupported file format: {resume_path.suffix}")
            return ResumeData()

        if not text:
            return ResumeData()

        # Extract information
        contact_info = self.extract_contact_info(text)
        skills = self.extract_skills(text)
        sections = self.extract_sections(text)

        # Parse experience
        experience = []
        if "experience" in sections:
            experience = self.parse_experience(sections["experience"])

        # Build resume data
        resume_data = ResumeData(
            name=contact_info.get("name"),
            email=contact_info.get("email"),
            phone=contact_info.get("phone"),
            skills=skills,
            experience=experience,
            summary=sections.get("summary", "")[:500],  # Limit summary
        )

        logger.info(f"Successfully parsed resume: {resume_data.name}")
        return resume_data

    def generate_context_for_llm(self, resume_data: ResumeData) -> str:
        """Generate context string for LLM question personalization"""

        context_parts = []

        if resume_data.name:
            context_parts.append(f"Candidate: {resume_data.name}")

        if resume_data.skills:
            skills_str = ", ".join(resume_data.skills[:10])
            context_parts.append(f"Technical Skills: {skills_str}")

        if resume_data.experience:
            exp_count = len(resume_data.experience)
            context_parts.append(f"Work Experience: {exp_count} position(s)")

            # Add first experience
            if resume_data.experience[0].get("title"):
                context_parts.append(
                    f"Recent Role: {resume_data.experience[0]['title']}"
                )

        if resume_data.summary:
            context_parts.append(f"Summary: {resume_data.summary[:200]}")

        return "\n".join(context_parts)


# Singleton instance
_resume_parser: Optional[ResumeParser] = None


def get_resume_parser() -> ResumeParser:
    """Get or create resume parser instance"""
    global _resume_parser
    if _resume_parser is None:
        _resume_parser = ResumeParser()
    return _resume_parser
