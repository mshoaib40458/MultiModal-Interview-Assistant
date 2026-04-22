"""
Production Interview Orchestrator
Main system coordinating all interview components
"""

import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import speech_recognition as sr

sys.path.append(str(Path(__file__).resolve().parent))
from config import Config
from src.core.cheating_detector import get_cheating_detector
from src.core.fer_service import get_fer_service
from src.core.fusion_service import get_fusion_service
from src.core.llm_service import get_llm_service
from src.core.report_generator import get_report_generator
from src.core.resume_parser import get_resume_parser
from src.core.ser_service import get_ser_service

logger = Config.get_logger(__name__)


class InterviewOrchestrator:
    """Main interview orchestration system"""

    def __init__(self):
        self.llm = get_llm_service()
        self.ser = get_ser_service()
        self.fer = get_fer_service()
        self.fusion = get_fusion_service()
        self.cheating = get_cheating_detector()
        self.resume_parser = get_resume_parser()
        self.report_gen = get_report_generator()

        self.session_data = {
            "start_time": None,
            "end_time": None,
            "candidate_name": None,
            "domain": None,
            "qa_pairs": [],
            "audio_files": [],
            "video_path": None,
            "resume_data": None,
            "answer_timestamps": [],  # Store (start, end) for each answer
        }

        self.stop_video_event = threading.Event()
        self.video_thread = None
        self.video_start_time = None

        logger.info("Interview Orchestrator initialized")

    def load_domain_config(self, domain_file: str) -> Dict:
        """Load domain configuration"""
        domain_path = Config.DOMAINS_DIR / domain_file

        if not domain_path.exists():
            logger.error(f"Domain file not found: {domain_path}")
            return {}

        config: Dict[str, Dict[str, str]] = {}
        current_section = None

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

        logger.info(f"Loaded domain config: {domain_file}")
        return config

    def get_available_domains(self) -> List[str]:
        """Get list of available interview domains"""
        domains = []

        for file in Config.DOMAINS_DIR.glob("*.txt"):
            domain_name = file.stem.replace("_", " ").title()
            domains.append(domain_name)

        return sorted(domains)

    def record_audio(self, filename: str, duration: int = 60) -> Optional[str]:
        """Record audio response"""
        recognizer = sr.Recognizer()

        try:
            with sr.Microphone() as source:
                logger.info("🎤 Recording... Speak your answer")
                print("\n🎤 Recording started. Speak your answer...")

                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=1)

                # Record audio
                audio = recognizer.listen(
                    source, timeout=duration, phrase_time_limit=duration
                )

                # Save audio file
                with open(filename, "wb") as f:
                    f.write(audio.get_wav_data())

                logger.info(f"Audio saved: {filename}")
                return filename

        except Exception as e:
            logger.error(f"Audio recording failed: {str(e)}")
            return None

    def transcribe_audio(self, audio_file: str) -> Optional[str]:
        """Transcribe audio to text"""
        recognizer = sr.Recognizer()

        try:
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)

                # Try Google Speech Recognition
                transcription = recognizer.recognize_google(audio)
                logger.info(f"Transcription: {transcription[:50]}...")
                return transcription

        except sr.UnknownValueError:
            logger.warning("Speech was unintelligible")
            return None
        except sr.RequestError as e:
            logger.error(f"Transcription service error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return None

    def record_video_thread(self, output_path: str):
        """Record video in a separate thread"""
        logger.info(f"Starting video recording to {output_path}")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logger.error("Cannot open camera")
            return

        # Get camera properties
        fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.video_start_time = time.time()  # accurate start time

        try:
            while not self.stop_video_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                # Small sleep to yield CPU
                time.sleep(1 / fps)

        except Exception as e:
            logger.error(f"Video recording error: {e}")
        finally:
            cap.release()
            out.release()
            logger.info("Video recording stopped")

    def start_video_recording(self, output_path: str):
        """Start the video recording thread"""
        self.stop_video_event.clear()
        self.video_thread = threading.Thread(
            target=self.record_video_thread, args=(output_path,)
        )
        self.video_thread.daemon = True
        self.video_thread.start()

    def stop_video_recording(self):
        """Stop the video recording thread"""
        if self.video_thread and self.video_thread.is_alive():
            logger.info("Stopping video recording...")
            self.stop_video_event.set()
            self.video_thread.join()

    def conduct_interview(
        self,
        domain: str,
        num_questions: int = 5,
        resume_path: Optional[str] = None,
        candidate_name: Optional[str] = None,
    ) -> Dict:
        """Conduct complete interview"""

        logger.info(f"Starting interview: {domain}")
        print("\n" + "=" * 60)
        print("🤖 AI Interview - AI Multimodal Interview Assistant")
        print("=" * 60 + "\n")

        # Initialize session
        self.session_data["start_time"] = datetime.now()
        # start_timestamp removed, using video_start_time
        self.session_data["domain"] = domain
        self.session_data["candidate_name"] = candidate_name or "Candidate"

        # Parse resume if provided
        resume_context = ""
        if resume_path:
            logger.info(f"Parsing resume: {resume_path}")
            print("📄 Parsing resume...")
            resume_data = self.resume_parser.parse_resume(resume_path)
            self.session_data["resume_data"] = resume_data
            resume_context = self.resume_parser.generate_context_for_llm(
                resume_data
            )
            print(f"✅ Resume parsed: {resume_data.name or 'Unknown'}")
            print(f"   Skills detected: {len(resume_data.skills)}")

        # Load domain configuration
        domain_file = f"{domain.replace(' ', '_').lower()}.txt"
        domain_config = self.load_domain_config(domain_file)

        if not domain_config:
            logger.error("Failed to load domain configuration")
            return {"error": "Domain configuration not found"}

        # Setup output directory
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Config.OUTPUT_DIR / f"session_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        # Video recording path
        video_path = session_dir / "interview_video.mp4"
        self.session_data["video_path"] = str(video_path)

        print("\n📹 Interview will be recorded")
        print(f"📁 Session directory: {session_dir}")
        print("\nStarting interview in 3 seconds...")
        time.sleep(3)

        # Start Video Recording
        self.start_video_recording(str(video_path))

        # Wait for video to initialize
        timeout_start = time.time()
        while (
            self.video_start_time is None and time.time() - timeout_start < 5
        ):
            time.sleep(0.1)

        if self.video_start_time is None:
            logger.warning(
                "Video recording did not initialize properly. Timestamps may be inaccurate."
            )
            self.video_start_time = time.time()

        # Conduct Q&A
        asked_questions: List[str] = []

        try:
            for i in range(num_questions):
                print("\n" + "─" * 60)
                print(f"Question {i+1}/{num_questions}")
                print("─" * 60 + "\n")

                # Generate question
                context = (
                    self.session_data["qa_pairs"][-1]["answer"]
                    if self.session_data["qa_pairs"]
                    else resume_context
                )
                follow_up = i > 0 and len(context) > 100

                question = self.llm.generate_question(
                    subdomain=domain,
                    domain_details=domain_config.get("details", {}),
                    context=context if follow_up else None,
                    follow_up=follow_up,
                    asked_questions=asked_questions,
                )

                asked_questions.append(question)

                print(f"❓ {question}\n")

                # Record answer
                audio_file = session_dir / f"answer_{i+1}.wav"
                print("🎤 Recording your answer (speak clearly)...")

                # Track start time relative to video start
                ans_start = time.time() - self.video_start_time

                recorded = self.record_audio(
                    str(audio_file), duration=Config.RECORDING_DURATION
                )

                # Track end time relative to video start
                ans_end = time.time() - self.video_start_time
                self.session_data["answer_timestamps"].append(
                    (ans_start, ans_end)
                )

                if not recorded:
                    print("⚠️  Recording failed. Skipping...")
                    continue

                self.session_data["audio_files"].append(str(audio_file))

                # Transcribe
                print("📝 Transcribing...")
                transcription = self.transcribe_audio(str(audio_file))

                if not transcription:
                    print("⚠️  Could not understand speech. Please try again.")
                    transcription = "[Transcription failed]"
                else:
                    print(f"✅ Transcribed: {transcription[:100]}...")

                # Evaluate answer
                print("🔍 Evaluating answer...")
                evaluation = self.llm.evaluate_answer(question, transcription)

                print(
                    f"📊 Scores: Accuracy: {evaluation.accuracy}/10, "
                    f"Clarity: {evaluation.clarity}/10, "
                    f"Preciseness: {evaluation.preciseness}/10"
                )

                # Store Q&A pair
                self.session_data["qa_pairs"].append(
                    {
                        "question": question,
                        "answer": transcription,
                        "audio_file": str(audio_file),
                        "evaluation": {
                            "accuracy": evaluation.accuracy,
                            "clarity": evaluation.clarity,
                            "preciseness": evaluation.preciseness,
                            "feedback": evaluation.feedback,
                            "overall_score": evaluation.overall_score,
                        },
                    }
                )

                time.sleep(1)
        finally:
            # Enure video stops even if error occurs
            self.stop_video_recording()

        # Post-interview analysis
        print("\n" + "=" * 60)
        print("📊 Analyzing interview data (Parallel Async)...")
        print("=" * 60 + "\n")

        self.session_data["end_time"] = datetime.now()

        # Define async analysis runner
        import asyncio

        async def run_parallel_analysis():
            tasks = []

            # 1. SER Task
            print("🎭 Analysis: Audio emotions...")
            tasks.append(
                self.ser.analyze_multiple_files_async(
                    self.session_data["audio_files"]
                )
            )

            if video_path.exists():
                # 2. FER Task
                print("🎥 Analysis: Facial emotions...")
                tasks.append(
                    self.fer.analyze_video_async(
                        str(video_path), sample_rate=5
                    )
                )

                # 3. Cheating Task
                print("🔍 Analysis: Cheating indicators...")
                tasks.append(
                    self.cheating.analyze_video_async(
                        str(video_path), sample_rate=3
                    )
                )

            return await asyncio.gather(*tasks)

        # Run all analysis tasks in parallel
        analysis_results = asyncio.run(run_parallel_analysis())

        # Unpack results
        ser_results = analysis_results[0]
        ser_map = {res["filename"]: res for res in ser_results}

        fer_results = []
        cheating_results = {}

        if video_path.exists():
            fer_results = analysis_results[1]
            cheating_results = analysis_results[2]

        # 3. Smart Fusion (Syncing Audio & Video)
        print("🔗 Fusing multimodal data...")
        fused_results = []

        for i, qa in enumerate(self.session_data["qa_pairs"]):
            audio_path = Path(qa["audio_file"])
            filename = audio_path.name

            # Get SER result
            ser_res = ser_map.get(filename)

            # Calculate FER for this specific answer duration
            if i < len(self.session_data["answer_timestamps"]):
                start_t, end_t = self.session_data["answer_timestamps"][i]

                # Filter frames within this answer's time window
                answer_frames = [
                    f
                    for f in fer_results
                    if start_t <= f.get("timestamp", -1) <= end_t
                ]

                # Aggregate FER emotions for this answer
                if answer_frames:
                    # Voting for dominant emotion
                    emotions = [f["dominant_emotion"] for f in answer_frames]
                    confidences = [f["confidence"] for f in answer_frames]

                    if emotions:
                        dominant_fer = max(set(emotions), key=emotions.count)
                        avg_conf = sum(confidences) / len(confidences)

                        fer_res = {
                            "dominant_emotion": dominant_fer,
                            "confidence": avg_conf,
                        }
                    else:
                        fer_res = None
                else:
                    fer_res = None
            else:
                fer_res = None

            # Fuse them
            fused = self.fusion.fuse_emotions(ser_res, fer_res)
            fused["filename"] = filename
            fused_results.append(fused)

            # Attach fusion data back to Q&A pair for granular reporting
            qa["emotion_analysis"] = fused

        # Creates summaries
        fusion_summary = self.fusion.get_fusion_summary(fused_results)

        # Calculate overall score
        evaluation_scores = [
            qa["evaluation"] for qa in self.session_data["qa_pairs"]
        ]
        overall_score = (
            sum(e["overall_score"] for e in evaluation_scores)
            / len(evaluation_scores)
            if evaluation_scores
            else 0
        )

        # Generate report
        print("📄 Generating comprehensive report...")

        report_data = {
            "candidate_name": self.session_data["candidate_name"],
            "domain": domain,
            "date": self.session_data["start_time"].strftime("%Y-%m-%d %H:%M"),
            "duration_minutes": (
                self.session_data["end_time"] - self.session_data["start_time"]
            ).seconds
            // 60,
            "total_questions": num_questions,
            "overall_score": overall_score,
            "dominant_emotion": fusion_summary.get(
                "dominant_emotion", "neutral"
            ),
            "cheating_risk": cheating_results.get("risk_level", "unknown"),
            "emotion_data": fusion_summary,
            "evaluation_scores": evaluation_scores,
            "qa_pairs": self.session_data["qa_pairs"],
            "recommendations": self._generate_recommendations(
                overall_score, fusion_summary, cheating_results
            ),
        }

        report_path = self.report_gen.generate_report(report_data)

        # Save session data
        session_json = session_dir / "session_data.json"
        with open(session_json, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        print("\n" + "=" * 60)
        print("✅ Interview Complete!")
        print("=" * 60)
        print(f"\n📊 Overall Score: {overall_score:.1f}/10")
        print(
            f"🎭 Dominant Emotion: {fusion_summary.get('dominant_emotion', 'N/A')}"
        )
        print(f"🔍 Cheating Risk: {cheating_results.get('risk_level', 'N/A')}")
        print(f"\n📄 Report saved: {report_path}")
        print(f"📁 Session data: {session_dir}")

        return report_data

    def _generate_recommendations(
        self,
        overall_score: float,
        emotion_summary: Dict,
        cheating_results: Dict,
    ) -> List[str]:
        """Generate hiring recommendations"""

        recommendations = []

        # Performance-based
        if overall_score >= 8:
            recommendations.append(
                "✅ Strong candidate - Highly recommended for next round"
            )
        elif overall_score >= 6:
            recommendations.append(
                "👍 Good candidate - Recommended with minor reservations"
            )
        elif overall_score >= 4:
            recommendations.append(
                "⚠️  Average performance - Consider for junior roles"
            )
        else:
            recommendations.append("❌ Below expectations - Not recommended")

        # Emotion-based
        dominant_emotion = emotion_summary.get("dominant_emotion", "neutral")
        if dominant_emotion == "happy":
            recommendations.append("Positive and enthusiastic demeanor")
        elif dominant_emotion in ["sad", "fear"]:
            recommendations.append(
                "May lack confidence - consider stress management"
            )
        elif dominant_emotion == "angry":
            recommendations.append("⚠️  Showed signs of frustration")

        # Cheating-based
        risk_level = cheating_results.get("risk_level", "unknown")
        if risk_level == "high":
            recommendations.append(
                "🚨 High cheating risk detected - Manual review required"
            )
        elif risk_level == "medium":
            recommendations.append("⚠️  Some suspicious behavior detected")
        elif risk_level == "low":
            recommendations.append("✅ No significant integrity concerns")

        # Specific improvements
        if cheating_results.get("recommendations"):
            recommendations.extend(cheating_results["recommendations"])

        return recommendations


def main():
    """Main entry point"""
    try:
        orchestrator = InterviewOrchestrator()

        # Get available domains
        domains = orchestrator.get_available_domains()

        print("\n🤖 AI Interview - AI Multimodal Interview Assistant")
        print("=" * 60)
        print("\nAvailable Domains:")
        for i, domain in enumerate(domains, 1):
            print(f"{i}. {domain}")

        # Select domain
        try:
            choice = int(input("\nSelect domain (number): ")) - 1
            if 0 <= choice < len(domains):
                selected_domain = domains[choice]
            else:
                print("Invalid selection. Exiting.")
                return
        except ValueError:
            print("Invalid input. Exiting.")
            return

        # Get candidate info
        candidate_name = input("Candidate name: ").strip() or "Candidate"

        # Resume (optional)
        resume_path = input(
            "Resume path (optional, press Enter to skip): "
        ).strip()
        if resume_path and not Path(resume_path).exists():
            print("⚠️  Resume file not found. Continuing without resume.")
            resume_path = None

        # Number of questions
        try:
            default_qs = Config.MAX_QUESTIONS
            num_in = input(
                f"Number of questions (default: {default_qs}): "
            ).strip()
            num_questions = int(num_in) if num_in else default_qs
        except ValueError:
            num_questions = 5

        # Conduct interview
        orchestrator.conduct_interview(
            domain=selected_domain,
            num_questions=num_questions,
            resume_path=resume_path,
            candidate_name=candidate_name,
        )

        print("\n✅ Interview session completed successfully!")

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}", exc_info=True)
        print(f"\n❌ Configuration error: {str(e)}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Interview interrupted by user")
    except Exception as e:
        logger.error(f"Interview failed: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
