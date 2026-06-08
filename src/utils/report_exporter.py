"""
report_exporter.py — Export the final report to Markdown and PDF.

PDF generation uses ReportLab with:
  - markdown2 for parsing Markdown structure
  - Embedded images (from generated_images/) inline in the PDF
  - Styled headings, body text, bullet points, and figure captions
"""

from __future__ import annotations

import base64
import logging
import re
from datetime import datetime
from pathlib import Path

from src.config import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Markdown → HTML (for Streamlit UI rendering)                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def markdown_to_html(content: str, image_paths: list[str] | None = None) -> str:
    """
    Convert Markdown report to HTML using markdown2.

    Images referenced as ![Figure N](generated_images/file.png) are replaced
    with base64-encoded <img> tags so they render inline in the browser.

    Args:
        content:     Markdown text of the report.
        image_paths: List of absolute paths to generated images (optional).

    Returns:
        HTML string ready for st.markdown(..., unsafe_allow_html=True).
    """
    try:
        import markdown2  # noqa: PLC0415
    except ImportError:
        logger.warning("markdown2 not installed — returning raw markdown.")
        return content

    # Build a lookup: filename → absolute path
    path_map: dict[str, str] = {}
    if image_paths:
        for p in image_paths:
            path_map[Path(p).name] = p

    # Replace image markdown with base64 <img> tags before parsing
    def _embed_image(match: re.Match) -> str:
        alt  = match.group(1)
        ref  = match.group(2)           # e.g. "generated_images/dalle_1_xxx.png"
        name = Path(ref).name
        abs_path = path_map.get(name) or (
            str(config.images_dir / name)
            if (config.images_dir / name).exists() else None
        )
        if abs_path and Path(abs_path).exists():
            img_bytes = Path(abs_path).read_bytes()
            b64       = base64.b64encode(img_bytes).decode()
            return (
                f'<figure style="text-align:center;margin:24px 0;">'
                f'<img src="data:image/png;base64,{b64}" '
                f'alt="{alt}" style="max-width:100%;border-radius:8px;'
                f'box-shadow:0 4px 20px rgba(0,0,0,0.4);"/>'
                f'<figcaption style="color:#bfc7d2;font-size:0.85rem;'
                f'margin-top:8px;">{alt}</figcaption></figure>'
            )
        # Image not found — show a placeholder
        return f'<p style="color:#ffba4b;">⚠ Image not found: {ref}</p>'

    content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', _embed_image, content)

    # Convert remaining Markdown to HTML
    html = markdown2.markdown(
        content,
        extras=[
            "fenced-code-blocks",
            "tables",
            "header-ids",
            "strike",
            "footnotes",
        ],
    )
    return html


# ═══════════════════════════════════════════════════════════════════════════ #
#  Markdown file saver                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def save_markdown(content: str, topic: str) -> str:
    """Save the report as a .md file and return the file path."""
    slug      = topic[:40].replace(" ", "_").replace("/", "-")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename  = f"{slug}_{timestamp}.md"
    filepath  = config.reports_dir / filename
    filepath.write_text(content, encoding="utf-8")
    logger.info("Markdown report saved: %s", filepath)
    return str(filepath)


# ═══════════════════════════════════════════════════════════════════════════ #
#  PDF saver                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

def save_pdf(
    markdown_content: str,
    topic: str,
    image_paths: list[str] | None = None,
) -> str:
    """
    Convert Markdown + images to a styled PDF using ReportLab.

    Images referenced as ![...](generated_images/xxx.png) are embedded
    inline at the exact position they appear in the report.

    Args:
        markdown_content: The full Markdown report text.
        topic:            Research topic (used for filename).
        image_paths:      Optional list of absolute paths to generated images.

    Returns:
        Absolute path to the saved PDF, or "" on failure.
    """
    slug      = topic[:40].replace(" ", "_").replace("/", "-")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename  = f"{slug}_{timestamp}.pdf"
    filepath  = config.reports_dir / filename

    try:
        _save_pdf_reportlab(markdown_content, filepath, image_paths or [])
    except Exception as exc:
        logger.error("PDF generation failed: %s", exc)
        return ""

    logger.info("PDF report saved: %s", filepath)
    return str(filepath)


# ═══════════════════════════════════════════════════════════════════════════ #
#  ReportLab PDF builder                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

def _save_pdf_reportlab(
    content: str,
    filepath: Path,
    image_paths: list[str],
) -> None:
    """Build a styled PDF with embedded images using ReportLab."""
    from reportlab.lib.pagesizes   import A4                          # noqa: PLC0415
    from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle  # noqa: PLC0415
    from reportlab.lib.units       import cm                          # noqa: PLC0415
    from reportlab.lib             import colors                      # noqa: PLC0415
    from reportlab.platypus        import (                           # noqa: PLC0415
        SimpleDocTemplate, Paragraph, Spacer,
        HRFlowable, Image as RLImage, KeepTogether,
    )
    from reportlab.lib.enums       import TA_CENTER                   # noqa: PLC0415

    PAGE_W = A4[0] - 5 * cm   # usable width after margins

    # ── Build path lookup ─────────────────────────────────────────────────────
    path_map: dict[str, str] = {}
    for p in image_paths:
        path_map[Path(p).name] = p

    # ── Style definitions ─────────────────────────────────────────────────────
    styles  = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        spaceAfter=12,
        textColor=colors.HexColor("#1a1c1e"),
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=18,
        spaceAfter=6,
        textColor=colors.HexColor("#006397"),
    )
    h3_style = ParagraphStyle(
        "H3",
        parent=styles["Heading3"],
        fontSize=11,
        leading=14,
        spaceBefore=10,
        spaceAfter=4,
        textColor=colors.HexColor("#00639b"),
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=16,
        spaceAfter=3,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=body_style,
        fontSize=9,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )

    # ── Doc template ──────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    story = []
    fig_counter = [0]   # mutable counter for figure captions

    def _embed_image_block(ref: str, alt: str) -> None:
        """Resolve image path and append an Image + caption to story."""
        name     = Path(ref).name
        abs_path = path_map.get(name) or (
            str(config.images_dir / name)
            if (config.images_dir / name).exists() else None
        )
        if abs_path and Path(abs_path).exists():
            fig_counter[0] += 1
            try:
                img = RLImage(abs_path, width=PAGE_W, height=PAGE_W * 0.6)
                img.hAlign = "CENTER"
                caption_text = alt or f"Figure {fig_counter[0]}"
                story.append(Spacer(1, 10))
                story.append(
                    KeepTogether([
                        img,
                        Spacer(1, 4),
                        Paragraph(caption_text, caption_style),
                    ])
                )
                story.append(Spacer(1, 10))
            except Exception as exc:
                logger.warning("Could not embed image %s: %s", abs_path, exc)
        else:
            logger.warning("Image not found for PDF: %s", ref)

    # ── Line-by-line Markdown parser ──────────────────────────────────────────
    IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

    def _clean(text: str) -> str:
        """Strip Markdown bold/italic/code for safe ReportLab Paragraph text."""
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)   # bold → <b>
        text = re.sub(r'\*(.+?)\*',     r'<i>\1</i>', text)    # italic → <i>
        text = re.sub(r'`(.+?)`',       r'<font face="Courier">\1</font>', text)
        # Strip citation refs [N] — keep them as superscript
        text = re.sub(r'\[(\d+)\]', r'<super>[\1]</super>', text)
        return text

    for raw_line in content.split("\n"):
        line = raw_line.rstrip()

        # ── Image reference ───────────────────────────────────────────────────
        m = IMAGE_RE.match(line.strip())
        if m:
            _embed_image_block(ref=m.group(2), alt=m.group(1))
            continue

        # ── Headings ──────────────────────────────────────────────────────────
        if line.startswith("# "):
            story.append(Spacer(1, 6))
            story.append(Paragraph(_clean(line[2:]), title_style))
            story.append(HRFlowable(width="100%", thickness=1,
                                    color=colors.HexColor("#cdd5dd")))
            story.append(Spacer(1, 6))
        elif line.startswith("## "):
            story.append(Paragraph(_clean(line[3:]), h2_style))
        elif line.startswith("### "):
            story.append(Paragraph(_clean(line[4:]), h3_style))

        # ── Bullet points ─────────────────────────────────────────────────────
        elif line.startswith("- ") or line.startswith("* "):
            story.append(Paragraph(f"• {_clean(line[2:])}", bullet_style))

        # ── Horizontal rule ───────────────────────────────────────────────────
        elif line.strip() in ("---", "***", "___"):
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#cdd5dd")))
            story.append(Spacer(1, 4))

        # ── Empty line ────────────────────────────────────────────────────────
        elif not line.strip():
            story.append(Spacer(1, 6))

        # ── Normal paragraph ──────────────────────────────────────────────────
        else:
            story.append(Paragraph(_clean(line), body_style))

    doc.build(story)
