# AI Multi-Modal Interview Assistant

An AI-powered interview platform that combines large language models, speech emotion recognition, facial emotion recognition, resume parsing, and cheating-detection heuristics to support structured technical interviews and generate post-interview reports.

## Overview

The application simulates a technical interview workflow by combining multiple analysis layers:

- Question generation tailored to a chosen domain
- Resume parsing for context-aware follow-up questions
- Audio transcription and speech emotion analysis
- Facial emotion analysis from video input
- Behavioral signals for cheating detection
- Scoring, feedback generation, and report creation

The project exposes two entry points:

- `main_production.py` for a command-line interview flow
- `streamlit_app.py` for an interactive web interface

## Key Features

- Domain-specific interview question generation
- Resume-aware questioning and candidate context extraction
- Speech emotion recognition for tone and confidence signals
- Facial emotion recognition for visual engagement analysis
- Cheating detection using gaze and head-pose heuristics
- Automated scoring with structured feedback
- PDF report generation with charts and summary insights
- Logging and health-check support for operational visibility

## Technology Stack

- Python 3.10+
- Streamlit
- OpenCV
- SpeechRecognition
- Groq API
- Hugging Face Inference API
- DeepFace
- MediaPipe
- spaCy
- ReportLab
- pandas
- NumPy
- Matplotlib

## Project Structure

```text
AI-Multi-Modal-Interview-Assistant/
├── config.py
├── main_production.py
├── streamlit_app.py
├── src/
│   └── core/
│       ├── cheating_detector.py
│       ├── fer_service.py
│       ├── fusion_service.py
│       ├── health_check.py
│       ├── llm_service.py
│       ├── logging_config.py
│       ├── report_generator.py
│       ├── resume_parser.py
│       ├── security_utils.py
│       └── ser_service.py
├── Domains/
├── tests/
├── requirements.txt
└── requirements-prod.txt
```

## Prerequisites

- Python 3.10 or newer
- Microphone and camera access for full multimodal functionality
- Valid API keys for the external services used by the application

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the application dependencies.
4. Install the spaCy language model used by the resume parser.

```bash
git clone <repo-url>
cd AI-Multi-Modal-Interview-Assistant
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-prod.txt
python -m spacy download en_core_web_sm
```

## Configuration

The application reads environment variables from `.env.local` and `.env`.

Recommended variables:

- `GROQ_API_KEY`
- `HF_API_KEY`
- `SECRET_KEY`

Recommended production settings:

- `ENVIRONMENT=production`
- `DEBUG=False`
- A strong, unique `SECRET_KEY`
- A production-ready database URL if you connect one
- Appropriate host and access restrictions for your deployment

## Running the Application

### Web Interface

```bash
streamlit run streamlit_app.py
```

### Command-Line Interview Flow

```bash
python main_production.py
```

## Testing

Run the test suite with:

```bash
pytest -q
```

## Health Monitoring

A health-check module is available at `src/core/health_check.py` for runtime validation and system monitoring.

## Troubleshooting

### Missing API Keys

If the application fails to start or cannot call external services, verify that `GROQ_API_KEY` and `HF_API_KEY` are defined in your environment.

### Camera or Microphone Access

If video or audio capture fails, confirm that the device is available and that no other application is using it.

### spaCy Model Missing

If resume parsing fails with a model error, install the English spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Security Notes

- Do not commit API keys or secrets.
- Review uploaded resumes before processing them in production.
- Use strong credentials and environment isolation in deployed environments.
- Limit camera and microphone access to trusted contexts.

## Support

If you extend the project, keep the README, configuration, and runtime behavior aligned so the documentation stays accurate and maintainable.
