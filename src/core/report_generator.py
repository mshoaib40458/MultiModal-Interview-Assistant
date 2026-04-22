"""
Production Interview Report Generator
Creates comprehensive PDF reports with visualizations
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

matplotlib.use("Agg")  # Non-interactive backend
import sys

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import Config

logger = Config.get_logger(__name__)


class ReportGenerator:
    """Generate comprehensive interview reports"""

    def __init__(self):
        self.output_dir = Config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Report Generator initialized")

    def create_emotion_chart(
        self,
        emotion_data: Dict,
        output_path: str,
        title: str = "Emotion Distribution",
    ):
        """Create emotion distribution pie chart"""

        distribution = emotion_data.get("emotion_distribution", {})

        if not distribution:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        emotions = list(distribution.keys())
        percentages = list(distribution.values())

        colors_map = {
            "happy": "#4CAF50",
            "neutral": "#9E9E9E",
            "sad": "#2196F3",
            "angry": "#F44336",
            "surprise": "#FF9800",
            "fear": "#9C27B0",
            "disgust": "#795548",
        }

        chart_colors = [colors_map.get(e, "#607D8B") for e in emotions]

        ax.pie(
            percentages,
            labels=emotions,
            autopct="%1.1f%%",
            colors=chart_colors,
            startangle=90,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def create_score_chart(self, scores: List[Dict], output_path: str):
        """Create bar chart for evaluation scores"""

        if not scores:
            return None

        categories = ["Accuracy", "Clarity", "Preciseness"]
        avg_scores: Dict[str, List[float]] = {cat: [] for cat in categories}

        for score in scores:
            avg_scores["Accuracy"].append(score.get("accuracy", 0))
            avg_scores["Clarity"].append(score.get("clarity", 0))
            avg_scores["Preciseness"].append(score.get("preciseness", 0))

        # Calculate averages
        avg_values = [
            sum(avg_scores["Accuracy"]) / len(avg_scores["Accuracy"]),
            sum(avg_scores["Clarity"]) / len(avg_scores["Clarity"]),
            sum(avg_scores["Preciseness"]) / len(avg_scores["Preciseness"]),
        ]

        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.bar(
            categories, avg_values, color=["#2196F3", "#4CAF50", "#FF9800"]
        )
        ax.set_ylim(0, 10)
        ax.set_ylabel("Score (out of 10)", fontsize=12)
        ax.set_title(
            "Average Evaluation Scores", fontsize=14, fontweight="bold"
        )
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def generate_report(
        self, interview_data: Dict, output_filename: Optional[str] = None
    ) -> str:
        """Generate comprehensive PDF report"""

        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"interview_report_{timestamp}.pdf"

        output_path = self.output_dir / output_filename

        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        # Container for PDF elements
        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#1976D2"),
            spaceAfter=30,
            alignment=TA_CENTER,
        )

        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=16,
            textColor=colors.HexColor("#424242"),
            spaceAfter=12,
            spaceBefore=12,
        )

        # Title
        story.append(Paragraph("🤖 AI Interview", title_style))
        story.append(
            Paragraph(
                "AI Multimodal Interview Assessment Report", styles["Heading3"]
            )
        )
        story.append(Spacer(1, 0.3 * inch))

        # Candidate Information
        story.append(Paragraph("Candidate Information", heading_style))

        candidate_data = [
            ["Name:", interview_data.get("candidate_name", "N/A")],
            ["Domain:", interview_data.get("domain", "N/A")],
            [
                "Date:",
                interview_data.get(
                    "date", datetime.now().strftime("%Y-%m-%d %H:%M")
                ),
            ],
            [
                "Duration:",
                f"{interview_data.get('duration_minutes', 0)} minutes",
            ],
            ["Questions:", str(interview_data.get("total_questions", 0))],
        ]

        candidate_table = Table(candidate_data, colWidths=[2 * inch, 4 * inch])
        candidate_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.HexColor("#E3F2FD"),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )

        story.append(candidate_table)
        story.append(Spacer(1, 0.3 * inch))

        # Overall Performance Summary
        story.append(Paragraph("Overall Performance Summary", heading_style))

        overall_score = interview_data.get("overall_score", 0)
        performance_level = (
            "Excellent"
            if overall_score >= 8
            else (
                "Good"
                if overall_score >= 6
                else "Average" if overall_score >= 4 else "Needs Improvement"
            )
        )

        summary_data = [
            ["Overall Score:", f"{overall_score:.1f}/10"],
            ["Performance Level:", performance_level],
            [
                "Dominant Emotion:",
                interview_data.get("dominant_emotion", "N/A"),
            ],
            ["Cheating Risk:", interview_data.get("cheating_risk", "N/A")],
        ]

        summary_table = Table(summary_data, colWidths=[2 * inch, 4 * inch])
        summary_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.HexColor("#E8F5E9"),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )

        story.append(summary_table)
        story.append(Spacer(1, 0.4 * inch))

        # Create charts
        charts_created = []

        # Emotion chart
        if "emotion_data" in interview_data:
            emotion_chart_path = self.output_dir / "emotion_distribution.png"
            self.create_emotion_chart(
                interview_data["emotion_data"],
                str(emotion_chart_path),
                "Emotional State Distribution",
            )
            if emotion_chart_path.exists():
                charts_created.append(("emotion", emotion_chart_path))

        # Score chart
        if "evaluation_scores" in interview_data:
            score_chart_path = self.output_dir / "evaluation_scores.png"
            self.create_score_chart(
                interview_data["evaluation_scores"], str(score_chart_path)
            )
            if score_chart_path.exists():
                charts_created.append(("scores", score_chart_path))

        # Add charts to report
        for chart_type, chart_path in charts_created:
            story.append(
                Image(str(chart_path), width=5 * inch, height=3 * inch)
            )
            story.append(Spacer(1, 0.2 * inch))

        # Page break before detailed sections
        story.append(PageBreak())

        # Detailed Q&A Section
        story.append(Paragraph("Question & Answer Details", heading_style))

        qa_pairs = interview_data.get("qa_pairs", [])
        for i, qa in enumerate(qa_pairs, 1):
            story.append(
                Paragraph(
                    f"<b>Q{i}:</b> {qa.get('question', 'N/A')}",
                    styles["Normal"],
                )
            )
            story.append(Spacer(1, 0.1 * inch))
            story.append(
                Paragraph(
                    f"<b>A{i}:</b> {qa.get('answer', 'N/A')}", styles["Normal"]
                )
            )
            story.append(Spacer(1, 0.1 * inch))

            if "evaluation" in qa:
                eval_data = qa["evaluation"]
                story.append(
                    Paragraph(
                        f"<b>Scores:</b> Accuracy: {eval_data.get('accuracy', 0)}/10, "
                        f"Clarity: {eval_data.get('clarity', 0)}/10, "
                        f"Preciseness: {eval_data.get('preciseness', 0)}/10",
                        styles["Normal"],
                    )
                )
                story.append(
                    Paragraph(
                        f"<b>Feedback:</b> {eval_data.get('feedback', 'N/A')}",
                        styles["Normal"],
                    )
                )

            story.append(Spacer(1, 0.2 * inch))

        # Recommendations
        story.append(PageBreak())
        story.append(Paragraph("Recommendations", heading_style))

        recommendations = interview_data.get("recommendations", [])
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))

        # Build PDF
        doc.build(story)

        logger.info(f"Report generated: {output_path}")
        return str(output_path)


# Singleton instance
_report_generator: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """Get or create report generator instance"""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator
