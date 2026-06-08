"""
analyzer_agent.py — Theme synthesis and report structure agent.

Reads classified sources + NER output and produces:
  - 4–6 thematic areas with supporting evidence
  - Cross-source contradictions
  - A structured report outline for the writer
  - Image prompts for the illustration agent
  - Optional: transformer architecture + MoE analysis sections
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import invoke_claude
from src.state import ResearchState
from src.tools.vector_store import index_sources

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research synthesis analyst.  Given source summaries and
named entities, produce a structured JSON analysis.

Return ONLY a valid JSON object with these keys:
  themes:         list of 4-6 theme strings
  contradictions: list of strings describing source conflicts
  evidence_summary: object mapping each theme to a 2-sentence evidence summary
  outline:        list of report section titles (Introduction, …, Conclusion)
  image_prompts:  list of 2-3 concise prompts for academic-style diagrams/charts
  transformer_config: object with keys: architecture_notes, attention_mechanism, decoder_only_notes
  moe_analysis:   object with keys: dense_vs_moe, switch_transformer, mixtral, deepseek_moe, tradeoffs
"""


def _build_source_summary(state: ResearchState) -> str:
    """Condense classified sources into a prompt-sized text block."""
    lines = []
    for i, src in enumerate(state.classified_sources[:12], 1):
        lines.append(
            f"[{i}] {src.title} ({src.source_type}/{src.domain})\n"
            f"    URL: {src.url}\n"
            f"    Snippet: {src.snippet[:300]}"
        )
    return "\n\n".join(lines)


def _build_entity_summary(state: ResearchState) -> str:
    """Summarise top entities for the analyst."""
    top = state.entities[:20]
    return ", ".join(f"{e.text} ({e.category}, ×{e.count})" for e in top)


def analyzer_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: synthesise sources and NER into structured analysis.

    Input state fields used : classified_sources, entities, topic, cost_metrics
    Output state fields set  : themes, contradictions, evidence_summary, outline,
                               image_prompts, transformer_config, moe_analysis, status
    """
    logger.info("Analyzer Agent: synthesising %d sources.", len(state.classified_sources))

    # ── Index sources in ChromaDB for semantic retrieval ──────────────────────
    try:
        index_sources(state.classified_sources, state.topic)
    except Exception as exc:
        logger.warning("Vector store indexing failed (non-fatal): %s", exc)

    source_summary = _build_source_summary(state)
    entity_summary = _build_entity_summary(state)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Research topic: {state.topic}\n\n"
                f"Sub-questions:\n" + "\n".join(f"- {q}" for q in state.sub_questions) + "\n\n"
                f"Sources:\n{source_summary}\n\n"
                f"Key entities: {entity_summary}\n\n"
                "Produce the JSON analysis now."
            )
        ),
    ]

    try:
        raw = invoke_claude(messages, cost_metrics=state.cost_metrics, temperature=0.3)

        # Extract JSON from response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in analyzer response")
        analysis = json.loads(match.group())
    except Exception as exc:
        logger.error("Analyzer JSON parse failed: %s", exc)
        # Provide sensible fallbacks so the pipeline continues
        analysis = {
            "themes": [
                f"Overview of {state.topic}",
                "Key Components and Architecture",
                "Applications and Use Cases",
                "Challenges and Limitations",
                "Future Directions",
            ],
            "contradictions": ["Insufficient sources to identify contradictions."],
            "evidence_summary": {},
            "outline": [
                "Executive Summary",
                "Introduction",
                f"Overview of {state.topic}",
                "Technical Architecture",
                "Applications",
                "Challenges",
                "Future Directions",
                "Conclusion",
                "References",
            ],
            "image_prompts": [
                f"Architecture diagram of {state.topic}",
                "Comparison chart of key approaches",
            ],
            "transformer_config": {},
            "moe_analysis": {},
        }

    logger.info("Analyzer: %d themes, %d outline sections.", len(analysis.get("themes", [])), len(analysis.get("outline", [])))
    return {
        "themes": analysis.get("themes", []),
        "contradictions": analysis.get("contradictions", []),
        "evidence_summary": analysis.get("evidence_summary", {}),
        "outline": analysis.get("outline", []),
        "image_prompts": analysis.get("image_prompts", []),
        "transformer_config": analysis.get("transformer_config", {}),
        "moe_analysis": analysis.get("moe_analysis", {}),
        "status": "illustrating",
    }
