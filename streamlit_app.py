import json
import os

# --- Import Internal Modules ---
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import speech_recognition as sr
import streamlit as st

sys.path.append(str(Path(__file__).parent))
from config import Config
from src.core.cheating_detector import get_cheating_detector
from src.core.fer_service import get_fer_service
from src.core.fusion_service import get_fusion_service
from src.core.llm_service import get_llm_service
from src.core.report_generator import get_report_generator
from src.core.resume_parser import get_resume_parser
from src.core.security_utils import file_validator, rate_limiter
from src.core.ser_service import get_ser_service

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Multimodal Interview Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Production Security Check
if Config.ENVIRONMENT == "production":
    if Config.DEBUG:
        st.error(
            "⚠️ CRITICAL WARNING: DEBUG mode is enabled in PRODUCTION environment!"
        )
        st.info("Please set DEBUG=False in .env.local")
        st.stop()

# Initialize Session State
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "resume_data" not in st.session_state:
    st.session_state.resume_data = None
if "domain_config" not in st.session_state:
    st.session_state.domain_config = {}
if "audio_files" not in st.session_state:
    st.session_state.audio_files = []
if "video_recording_active" not in st.session_state:
    st.session_state.video_recording_active = False
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if "video_output_path" not in st.session_state:
    session_dir = Config.OUTPUT_DIR / f"session_{st.session_state.session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    st.session_state.video_output_path = str(
        session_dir / "interview_video.mp4"
    )
    st.session_state.session_dir = session_dir
if "video_thread" not in st.session_state:
    st.session_state.video_thread = None
if "stop_video_event" not in st.session_state:
    st.session_state.stop_video_event = threading.Event()
if "camera_index" not in st.session_state:
    st.session_state.camera_index = 0
if "answer_timestamps" not in st.session_state:
    st.session_state.answer_timestamps = []
if "video_start_time" not in st.session_state:
    st.session_state.video_start_time = None
if "audio_captured" not in st.session_state:
    st.session_state.audio_captured = False
if "temp_transcription" not in st.session_state:
    st.session_state.temp_transcription = ""
if "current_question" not in st.session_state:
    st.session_state.current_question = None

# Rate Limiting Check
allowed, limit_msg = rate_limiter.is_allowed(st.session_state.session_id)
if not allowed:
    st.error(f"⚠️ {limit_msg}")
    try:
        st.stop()
    except Exception:
        pass


# --- Helper Functions ---
def load_domains():
    domains = []
    # Create domains dir if not exists
    if not Config.DOMAINS_DIR.exists():
        Config.DOMAINS_DIR.mkdir(parents=True, exist_ok=True)
        # Create a default domain
        with open(Config.DOMAINS_DIR / "general.txt", "w") as f:
            f.write("[details]\ndescription=General Interview\n")

    for file in Config.DOMAINS_DIR.glob("*.txt"):
        domain_name = file.stem.replace("_", " ").title()
        domains.append(domain_name)
    return sorted(domains)


def load_domain_config_file(domain_name):
    domain_file = f"{domain_name.replace(' ', '_').lower()}.txt"
    domain_path = Config.DOMAINS_DIR / domain_file
    config = {}
    current_section = None

    if domain_path.exists():
        with open(domain_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1].lower()
                    config[current_section] = {}
                elif "=" in line and current_section:
                    key, value = line.split("=", 1)
                    config[current_section][key.strip()] = value.strip().strip(
                        '"'
                    )
    return config


def get_available_cameras():
    """Detect available camera indices"""
    available = []
    # Check first 5 indices
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
        except Exception:
            pass
    return available if available else [0]  # Default to 0 if none found


# Shared container for video frames to allow preview while recording
class VideoState:
    last_frame = None
    lock = threading.Lock()
    is_running = False


if "video_state" not in st.session_state:
    st.session_state.video_state = VideoState()


def record_video_background(output_path, stop_event, camera_index=0):
    """Records video in a background thread and updates shared state for preview"""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ Error: Could not open camera with index {camera_index}")
        return

    # Use higher resolution if possible
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception:
        pass

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0

    try:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    except Exception as e:
        print(f"Failed to create video writer: {e}")
        return

    st.session_state.video_start_time = time.time()
    st.session_state.video_state.is_running = True

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            # Write to file
            try:
                out.write(frame)
            except Exception:
                pass

            # Update shared state for preview (thread-safe)
            if not stop_event.is_set():
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with st.session_state.video_state.lock:
                        st.session_state.video_state.last_frame = frame_rgb
                except Exception:
                    pass
        else:
            break
        time.sleep(1 / fps)  # Control frame rate

    st.session_state.video_state.is_running = False
    cap.release()
    out.release()


@st.cache_resource
def get_services():
    return {
        "llm": get_llm_service(),
        "ser": get_ser_service(),
        "fer": get_fer_service(),
        "fusion": get_fusion_service(),
        "cheating": get_cheating_detector(),
        "parser": get_resume_parser(),
        "report": get_report_generator(),
    }


try:
    services = get_services()
except Exception as exc:
    st.error(f"Failed to initialize application services: {exc}")
    st.stop()

# --- Sidebar Logic ---
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/000000/artificial-intelligence.png",
        width=80,
    )
    st.title("Interview Setup")

    # Environment Status
    if Config.ENVIRONMENT == "production":
        st.caption("🔒 Production Mode")
    else:
        st.caption("🛠️ Development Mode")

    if not st.session_state.interview_started:
        candidate_name = st.text_input("Candidate Name", "Candidate")
        selected_domain = st.selectbox("Select Domain", load_domains())
        num_questions = st.number_input(
            "Number of Questions", min_value=1, max_value=10, value=3
        )
        uploaded_resume = st.file_uploader(
            "Upload Resume (PDF/DOCX)", type=["pdf", "docx", "doc"]
        )

        with st.expander("⚙️ Camera Settings"):
            # Auto-detect cameras
            available_cams = get_available_cameras()
            camera_index = st.selectbox(
                "Select Camera", available_cams, index=0
            )
            st.session_state.camera_index = camera_index

            if st.button("📷 Test Camera"):
                cap = cv2.VideoCapture(camera_index)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="Camera Test Successful!")
                else:
                    st.error("Failed to Capture. Try another index.")
                cap.release()

        if st.button("🚀 Start Interview", type="primary"):
            st.session_state.interview_started = True
            st.session_state.candidate_name = candidate_name
            st.session_state.domain = selected_domain
            st.session_state.num_questions = num_questions
            st.session_state.domain_config = load_domain_config_file(
                selected_domain
            )

            # Process Resume
            if uploaded_resume:
                # Security Validation
                is_valid, error_msg = file_validator.validate_upload(
                    uploaded_resume.name, uploaded_resume.getvalue()
                )

                if not is_valid:
                    st.error(f"❌ Resume Rejected: {error_msg}")
                    st.stop()

                # Sanitize filename
                safe_filename = file_validator.sanitize_filename(
                    uploaded_resume.name
                )
                resume_path = st.session_state.session_dir / safe_filename

                with open(resume_path, "wb") as f:
                    f.write(uploaded_resume.getbuffer())

                with st.spinner("Analyzing Resume..."):
                    st.session_state.resume_data = services[
                        "parser"
                    ].parse_resume(str(resume_path))
                    st.success("✅ Resume Analyzed Successfully!")

            # Start Video Recording Thread
            st.session_state.stop_video_event.clear()
            # Initialize shared state
            if "video_state" in st.session_state:
                st.session_state.video_state.last_frame = None

            st.session_state.video_thread = threading.Thread(
                target=record_video_background,
                args=(
                    st.session_state.video_output_path,
                    st.session_state.stop_video_event,
                    st.session_state.camera_index,
                ),
            )
            st.session_state.video_thread.start()
            st.rerun()

    else:
        st.info(f"👤 Candidate: {st.session_state.candidate_name}")
        st.info(f"📚 Domain: {st.session_state.domain}")
        st.warning("🔴 Recording in Progress")
        if st.button("Abort Interview", type="secondary"):
            st.session_state.stop_video_event.set()
            if st.session_state.video_thread:
                st.session_state.video_thread.join()
            st.session_state.clear()
            st.rerun()

# --- Main Content Area ---
if not st.session_state.interview_started:
    st.markdown("""
    # 🤖 MultiModal AI Interview
    ### AI-Powered Multimodal Interview Assessment System
    
    **Features:**
    - 🧠 **AI Question Generation** tailored to your domain
    - 🎤 **Speech Emotion Recognition** to analyze confidence
    - 📹 **Facial Expression Analysis** for engagement
    - 📄 **Resume Integration** for personalized questions
    - 📊 **Detailed Feedback Report** generated instantly
    
    **Instructions:**
    1. Fill in your details in the sidebar.
    2. Upload your resume (optional but recommended).
    3. Click **Start Interview**.
    4. Answer questions clearly into your microphone.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.info("Ensure your camera and microphone are connected.")
    with col2:
        st.info("Find a quiet, well-lit environment.")

elif st.session_state.current_question_index < st.session_state.num_questions:
    # --- Interview In Progress ---

    # Generate Question Logic (Only if we don't have one yet)
    if st.session_state.current_question is None:
        with st.spinner("🧠 AI is generating the next question..."):

            # Prepare context
            context = None
            if st.session_state.qa_history:
                context = st.session_state.qa_history[-1]["answer"]

            resume_context = ""
            if st.session_state.resume_data:
                resume_context = services["parser"].generate_context_for_llm(
                    st.session_state.resume_data
                )
                if not context:
                    context = resume_context

            # Generate
            question = services["llm"].generate_question(
                subdomain=st.session_state.domain,
                domain_details=st.session_state.domain_config.get(
                    "details", {}
                ),
                context=context,
                follow_up=(st.session_state.current_question_index > 0),
                asked_questions=[
                    q["question"] for q in st.session_state.qa_history
                ],
            )

            # Store temporary question into session state
            st.session_state.current_question = question

    # Display Question
    st.progress(
        (st.session_state.current_question_index + 1)
        / st.session_state.num_questions
    )
    st.markdown(
        f"### Question {st.session_state.current_question_index + 1} / {st.session_state.num_questions}"
    )
    st.markdown(f"## ❓ {st.session_state.current_question}")

    st.divider()

    # Recording Area
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### 📹 Your Camera")

        # Live Camera Preview using Shared State
        camera_placeholder = st.empty()

        if (
            hasattr(st.session_state, "video_state")
            and st.session_state.video_state.last_frame is not None
        ):
            # Display the frame from the background thread
            with st.session_state.video_state.lock:
                if st.session_state.video_state.last_frame is not None:
                    camera_placeholder.image(
                        st.session_state.video_state.last_frame,
                        caption="🔴 Live Recording",
                        use_container_width=True,
                    )
        else:
            # Fallback if no frame yet
            st.warning("📷 connecting to camera...")

            # Auto-refresh to check for new frames
            time.sleep(0.5)
            st.rerun()

        if st.button("🔄 Refresh Preview"):
            st.rerun()

    with col2:
        st.markdown("#### 🎙️ Your Answer")

        # Audio Recording Logic
        if st.button("🎤 Start Recording Answer"):
            audio_file = (
                st.session_state.session_dir
                / f"answer_{st.session_state.current_question_index + 1}.wav"
            )

            # Calculate start time relative to video
            ans_start = time.time() - (
                st.session_state.video_start_time or time.time()
            )

            with st.spinner("Listening... Speak now!"):
                recognizer = sr.Recognizer()
                recognizer.pause_threshold = 2.0
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    try:
                        audio = recognizer.listen(
                            source,
                            timeout=8,
                            phrase_time_limit=Config.RECORDING_DURATION,
                        )
                    except sr.WaitTimeoutError:
                        audio = None

            # Calculate end time
            ans_end = time.time() - (
                st.session_state.video_start_time or time.time()
            )
            st.session_state.answer_timestamps.append((ans_start, ans_end))

            if audio:
                # Save Audio to temp for preview
                with open(audio_file, "wb") as f:
                    f.write(audio.get_wav_data())

                # Transcribe
                try:
                    text = recognizer.recognize_google(audio)
                    st.session_state.temp_transcription = text
                except Exception:
                    st.session_state.temp_transcription = (
                        "[Unintelligible / No Speech Detected]"
                    )

                st.session_state.audio_captured = True
                st.rerun()
            else:
                st.error("No speech detected. Please try again.")

    # Show review area if audio was captured
    if st.session_state.get("audio_captured", False):
        st.info("📝 **Transcription Preview:**")
        text = st.session_state.get("temp_transcription", "")
        edited_text = st.text_area(
            "You can refine your answer here before submitting:", value=text
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Confirm & Next Question", type="primary"):
                # Use current audio file
                audio_file = (
                    st.session_state.session_dir
                    / f"answer_{st.session_state.current_question_index + 1}.wav"
                )
                st.session_state.audio_files.append(str(audio_file))

                # Evaluate
                with st.spinner("Evaluating your answer..."):
                    eval_score = services["llm"].evaluate_answer(
                        st.session_state.current_question, edited_text
                    )

                st.session_state.qa_history.append(
                    {
                        "question": st.session_state.current_question,
                        "answer": edited_text,
                        "audio_file": str(audio_file),
                        "evaluation": eval_score.dict(),
                    }
                )

                # Cleanup and move forward
                st.session_state.audio_captured = False
                st.session_state.current_question_index += 1
                st.session_state.current_question = None  # Reset query
                st.rerun()

        with c2:
            if st.button("🔄 Retake Recording", type="secondary"):
                st.session_state.audio_captured = False
                st.rerun()

else:
    # --- Interview Complete / Report Generation ---
    st.balloons()
    st.title("🎉 Interview Complete!")

    # Stop Video
    if not st.session_state.stop_video_event.is_set():
        st.session_state.stop_video_event.set()
        if st.session_state.video_thread:
            st.session_state.video_thread.join()

    with st.status(
        "Generating Comprehensive Report...", expanded=True
    ) as status:
        import asyncio

        # Helper to run async analysis
        async def run_analysis():
            tasks = [
                services["ser"].analyze_multiple_files_async(
                    st.session_state.audio_files
                ),
                services["fer"].analyze_video_async(
                    st.session_state.video_output_path, sample_rate=5
                ),
                services["cheating"].analyze_video_async(
                    st.session_state.video_output_path, sample_rate=3
                ),
            ]
            return await asyncio.gather(*tasks)

        st.write("🔄 Analyzing Multimodal Data in Parallel...")
        try:
            import nest_asyncio

            nest_asyncio.apply()
            results = asyncio.run(run_analysis())

            ser_results = results[0]
            fer_results = results[1]
            cheating_results = results[2]

            ser_map = {res["filename"]: res for res in ser_results}
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            ser_results, fer_results, cheating_results = [], [], {}
            ser_map = {}

        # 3. Fusion
        st.write("🔗 Fusing Multimodal Data...")
        fused_results = []
        for i, qa in enumerate(st.session_state.qa_history):
            filename = Path(qa["audio_file"]).name
            ser_res = ser_map.get(filename)

            # Find relevant FER frames based on timestamps
            fer_res = None
            if i < len(st.session_state.answer_timestamps):
                start_t, end_t = st.session_state.answer_timestamps[i]
                frames = [
                    f
                    for f in fer_results
                    if start_t <= f.get("timestamp", -1) <= end_t
                ]
                if frames:
                    emotions = [f["dominant_emotion"] for f in frames]
                    fer_res = {
                        "dominant_emotion": max(
                            set(emotions), key=emotions.count
                        ),
                        "confidence": 0.8,
                    }  # simplified

            fused = services["fusion"].fuse_emotions(ser_res, fer_res)
            fused["filename"] = filename
            fused_results.append(fused)
            qa["emotion_analysis"] = fused

        fusion_summary = services["fusion"].get_fusion_summary(fused_results)

        # Final Report
        overall_score = (
            sum(
                [
                    q["evaluation"]["overall_score"]
                    for q in st.session_state.qa_history
                ]
            )
            / len(st.session_state.qa_history)
            if st.session_state.qa_history
            else 0
        )

        report_data = {
            "candidate_name": st.session_state.candidate_name,
            "domain": st.session_state.domain,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "total_questions": st.session_state.num_questions,
            "overall_score": overall_score,
            "dominant_emotion": fusion_summary.get(
                "dominant_emotion", "neutral"
            ),
            "cheating_risk": cheating_results.get("risk_level", "unknown"),
            "emotion_data": fusion_summary,
            "evaluation_scores": [
                q["evaluation"] for q in st.session_state.qa_history
            ],
            "qa_pairs": st.session_state.qa_history,
            "recommendations": [],
        }

        # Generate PDF
        pdf_path = services["report"].generate_report(report_data)
        status.update(
            label="Report Generated Successfully!",
            state="complete",
            expanded=False,
        )

    # Display Results
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Score", f"{overall_score:.1f}/10")
    col2.metric(
        "Dominant Emotion",
        fusion_summary.get("dominant_emotion", "N/A").title(),
    )
    col3.metric(
        "Cheating Risk", cheating_results.get("risk_level", "N/A").title()
    )

    st.subheader("📝 Question Breakdown")
    for idx, qa in enumerate(st.session_state.qa_history):
        with st.expander(f"Q{idx+1}: {qa['question']}"):
            st.markdown(f"**Answer:** {qa['answer']}")
            st.markdown(f"**Feedback:** {qa['evaluation']['feedback']}")
            st.caption(
                f"Emotion detected: {qa.get('emotion_analysis', {}).get('fused_emotion', 'N/A')}"
            )

    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📄 Download Full PDF Report",
            data=pdf_file,
            file_name="Interview_Report.pdf",
            mime="application/pdf",
        )

    if st.button("Start New Interview"):
        st.session_state.clear()
        st.rerun()
