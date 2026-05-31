"""
citations.py — Citation formatting utilities.

Builds numbered reference lists in Markdown and Chicago/APA-style formats
from SourceRecord objects.
"""

from __future__ import annotations

from datetime import datetime
from src.state import SourceRecord


def build_references_section(sources: list[SourceRecord]) -> str:
    """
    Build a Markdown ## References section with numbered entries.

    Format: [N] Title. Source Type. URL. Retrieved: date.
    """
    lines = ["## References\n"]
    for i, src in enumerate(sources, 1):
        date_str = src.retrieval_date[:10] if src.retrieval_date else datetime.utcnow().strftime("%Y-%m-%d")
        lines.append(
            f"[{i}] **{src.title}**. "
            f"*{src.source_type.replace('_', ' ').title()}*. "
            f"{src.url}. "
            f"Retrieved: {date_str}."
        )
    return "\n".join(lines)


def inline_cite(source_index: int) -> str:
    """Return an inline citation marker like [3]."""
    return f"[{source_index}]"
