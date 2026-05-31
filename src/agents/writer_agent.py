"""
writer_agent.py — Academic report writer.

Produces a polished Markdown report following the structured outline from the
analyzer agent.  Every factual claim must be followed by a citation reference
number [N] pointing to the references section.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import invoke_claude
from src.state import ResearchState
from src.utils.citations import build_references_section

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert academic writer.  Write a comprehensive research
report in Markdown format based on the provided outline, sources, entities, and analysis.

Rules:
- Use numbered citations [1], [2], … after every factual claim.
- Structure the report exactly following the provided outline sections.
- Include the Named Entities and Source Classification sections.
- Use clear headings (##, ###).
- Write in formal academic English.
- Aim for 1 500–2 500 words (excluding references).
- Do NOT invent facts not supported by the provided sources.
- Leave a placeholder like ![Figure 1](FIGURE_1) where figures should be embedded.
"""


def _format_sources_for_prompt(state: ResearchState) -> str:
    """Compress sources into a numbered reference block for the writer prompt."""
    lines = []
    for i, src in enumerate(state.classified_sources[:15], 1):
        lines.append(
            f"[{i}] {src.title}\n"
            f"    URL: {src.url}\n"
            f"    Type: {src.source_type} | Domain: {src.domain}\n"
            f"    Snippet: {src.snippet[:400]}"
        )
    return "\n\n".join(lines)


def _format_entities_section(state: ResearchState) -> str:
    """Build a compact entity table for the prompt."""
    rows = []
    for e in state.entities[:25]:
        rows.append(f"| {e.text} | {e.category} | {e.count} |")
    header = "| Entity | Category | Occurrences |\n|--------|----------|-------------|"
    return header + "\n" + "\n".join(rows) if rows else "No entities extracted."


def _format_classification_summary(state: ResearchState) -> str:
    """Build a Markdown table summarising source types."""
    from collections import Counter  # noqa: PLC0415
    type_counts: Counter = Counter(s.source_type for s in state.classified_sources)
    rows = "\n".join(f"| {t} | {c} |" for t, c in type_counts.most_common())
    return "| Source Type | Count |\n|-------------|-------|\n" + rows


def _embed_illustrations(draft: str, illustrations: list[str]) -> str:
    """Replace FIGURE_N placeholders with actual image links."""
    for i, path in enumerate(illustrations, 1):
        # Use just the filename for portability
        filename = Path(path).name
        placeholder = f"FIGURE_{i}"
        img_md = f"generated_images/{filename}"
        draft = draft.replace(placeholder, img_md)
    return draft


def writer_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: write the full Markdown report draft.

    Input state fields used : topic, outline, classified_sources, entities,
                              themes, evidence_summary, contradictions,
                              illustrations, critic_feedback, revision_count
    Output state fields set  : draft, references, status
    """
    logger.info(
        "Writer Agent: writing draft (revision %d).", state.revision_count
    )

    # Include critic feedback in the prompt on revisions
    revision_note = ""
    if state.revision_count > 0 and state.critic_feedback:
        revision_note = (
            f"\n\n## REVISION INSTRUCTIONS (iteration {state.revision_count})\n"
            f"The previous draft was reviewed and returned for revision.  "
            f"Please address ALL of the following feedback:\n\n{state.critic_feedback}\n"
        )

    source_block = _format_sources_for_prompt(state)
    entity_table = _format_entities_section(state)
    classification_table = _format_classification_summary(state)
    outline_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state.outline))
    themes_text = "\n".join(f"- {t}" for t in state.themes)
    contradictions_text = "\n".join(f"- {c}" for c in state.contradictions) or "None identified."

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"# Research Topic\n{state.topic}\n\n"
                f"# Report Outline\n{outline_text}\n\n"
                f"# Key Themes\n{themes_text}\n\n"
                f"# Source Contradictions\n{contradictions_text}\n\n"
                f"# Entity Table (for Named Entities section)\n{entity_table}\n\n"
                f"# Source Classification Summary\n{classification_table}\n\n"
                f"# Sources (cite these as [1], [2], …)\n{source_block}\n"
                f"{revision_note}"
                "\n\nWrite the complete Markdown report now."
            )
        ),
    ]

    try:
        draft = invoke_claude(
            messages,
            cost_metrics=state.cost_metrics,
            temperature=0.4,
            max_tokens=6000,
        )
    except Exception as exc:
        logger.error("Writer Agent: Claude call failed: %s", exc)
        draft = f"# {state.topic}\n\n*Report generation failed: {exc}*\n"
        state.errors.append(f"Writer failed: {exc}")

    # ── Embed illustration links ──────────────────────────────────────────────
    if state.illustrations:
        draft = _embed_illustrations(draft, state.illustrations)

    # ── Build reference list ──────────────────────────────────────────────────
    references = build_references_section(state.classified_sources)

    # Append references to draft if not already present
    if "## References" not in draft and "# References" not in draft:
        draft += "\n\n" + references

    logger.info("Writer Agent: draft length = %d chars.", len(draft))
    return {
        "draft": draft,
        "references": [src.url for src in state.classified_sources],
        "status": "reviewing",
    }
