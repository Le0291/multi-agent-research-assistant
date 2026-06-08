"""
report_exporter.py — Export the final report to Markdown and PDF.

PDF generation uses ReportLab (Windows-compatible) as primary,
with a fallback to a simple text-based PDF if ReportLab fails.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.config import config

logger = logging.getLogger(__name__)


def save_markdown(content: str, topic: str) -> str:
    """
    Save the report as a .md file and return the file path.
    """
    slug = topic[:40].replace(" ", "_").replace("/", "-")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{slug}_{timestamp}.md"
    filepath = config.reports_dir / filename
    filepath.write_text(content, encoding="utf-8")
    logger.info("Markdown report saved: %s", filepath)
    return str(filepath)


def save_pdf(markdown_content: str, topic: str) -> str:
    """
    Convert Markdown to PDF using ReportLab.

    Falls back to a plain-text PDF on any error.
    """
    slug = topic[:40].replace(" ", "_").replace("/", "-")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{slug}_{timestamp}.pdf"
    filepath = config.reports_dir / filename

    try:
        _save_pdf_reportlab(markdown_content, filepath)
    except Exception as exc:
        logger.warning("ReportLab PDF failed: %s — using plain text PDF.", exc)
        try:
            _save_pdf_plain(markdown_content, filepath)
        except Exception as exc2:
            logger.error("Plain text PDF also failed: %s", exc2)
            return ""

    logger.info("PDF report saved: %s", filepath)
    return str(filepath)


def _save_pdf_reportlab(content: str, filepath: Path) -> None:
    """Generate a styled PDF with ReportLab."""
    from reportlab.lib.pagesizes import A4  # noqa: PLC0415
    from reportlab.lib.styles import getSampleStyleSheet  # noqa: PLC0415
    from reportlab.lib.units import cm  # noqa: PLC0415
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable  # noqa: PLC0415
    from reportlab.lib import colors  # noqa: PLC0415

    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue

        if line.startswith("# "):
            # H1 — report title
            story.append(Paragraph(line[2:], styles["Title"]))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        elif line.startswith("## "):
            story.append(Spacer(1, 10))
            story.append(Paragraph(line[3:], styles["Heading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:], styles["Heading3"]))
        elif line.startswith("- ") or line.startswith("* "):
            story.append(Paragraph(f"• {line[2:]}", styles["Normal"]))
        else:
            # Strip Markdown bold/italic for cleaner PDF
            clean = line.replace("**", "").replace("*", "").replace("`", "")
            story.append(Paragraph(clean, styles["Normal"]))

    doc.build(story)


def _save_pdf_plain(content: str, filepath: Path) -> None:
    """Ultra-simple plain-text PDF using ReportLab's canvas."""
    from reportlab.pdfgen import canvas  # noqa: PLC0415
    from reportlab.lib.pagesizes import A4  # noqa: PLC0415

    c = canvas.Canvas(str(filepath), pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica", 10)

    for line in content.split("\n"):
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        c.drawString(50, y, line[:100])
        y -= 14

    c.save()
