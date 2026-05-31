"""
classification_agent.py — Source quality and type classifier.

For each raw source we determine:
  - source_type  : academic paper | documentation | blog | news | tutorial |
                   dataset | opinion | unknown
  - domain       : technical | business | healthcare | education | AI | policy | general
  - relevance    : high | medium | low

Low-quality (low relevance) sources are logged but removed from the active set
so later agents do not waste tokens on irrelevant material.

Classification uses Claude for accuracy.  A keyword-based fallback is used if
the LLM call fails.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import invoke_claude
from src.state import ResearchState, SourceRecord

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a source quality classifier.  Given a URL, title, and
text snippet, return ONLY a JSON object with these keys:
  source_type: one of [academic_paper, documentation, blog, news, tutorial, dataset, opinion, unknown]
  domain: one of [technical, business, healthcare, education, AI, policy, general]
  relevance: one of [high, medium, low]
No extra text, only JSON."""

# ── Keyword heuristics for fallback classification ────────────────────────────
_TYPE_KEYWORDS = {
    "academic_paper": ["arxiv", "doi.org", "springer", "ieee", "acm.org", "scholar", "pubmed", "researchgate"],
    "documentation": ["docs.", "documentation", "readthedocs", "developer.", "api."],
    "news": ["bbc", "reuters", "techcrunch", "wired", "forbes", "cnn", "nyt"],
    "tutorial": ["tutorial", "how-to", "howto", "guide", "step-by-step"],
    "dataset": ["kaggle", "huggingface.co/datasets", "data.gov", "zenodo"],
    "blog": ["medium.com", "substack", "blogspot", "wordpress", "towardsdatascience"],
}

_DOMAIN_KEYWORDS = {
    "AI": ["ai", "machine learning", "deep learning", "neural", "llm", "gpt", "transformer", "claude"],
    "technical": ["software", "code", "algorithm", "engineering", "architecture"],
    "healthcare": ["health", "medical", "clinical", "hospital", "patient"],
    "business": ["business", "market", "economy", "finance", "startup"],
    "education": ["education", "university", "school", "learning", "course"],
    "policy": ["policy", "regulation", "law", "government", "ethics"],
}


def _keyword_classify(src: SourceRecord) -> dict[str, str]:
    """Fast heuristic fallback when the LLM call fails."""
    url_lower = src.url.lower()
    text_lower = (src.title + " " + src.snippet).lower()

    # source_type
    source_type = "unknown"
    for stype, keywords in _TYPE_KEYWORDS.items():
        if any(kw in url_lower for kw in keywords):
            source_type = stype
            break

    # domain
    domain = "general"
    for dom, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            domain = dom
            break

    # relevance — use the numeric score already on the source
    if src.relevance_score >= 7:
        relevance = "high"
    elif src.relevance_score >= 4:
        relevance = "medium"
    else:
        relevance = "low"

    return {"source_type": source_type, "domain": domain, "relevance": relevance}


def _classify_source(src: SourceRecord, cost_metrics: Any) -> dict[str, str]:
    """
    Classify a single source using Claude.

    Falls back to keyword heuristics on any error.
    """
    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"URL: {src.url}\n"
                    f"Title: {src.title}\n"
                    f"Snippet: {src.snippet[:600]}"
                )
            ),
        ]
        raw = invoke_claude(messages, cost_metrics=cost_metrics, temperature=0.0)

        # Robustly extract the JSON object from Claude's response
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as exc:
        logger.warning("Claude classification failed for %s: %s — using fallback.", src.url, exc)

    return _keyword_classify(src)


def classification_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: classify and filter sources.

    Input state fields used : raw_sources, cost_metrics
    Output state fields set  : classified_sources, classification_log, status
    """
    logger.info("Classification Agent: classifying %d sources.", len(state.raw_sources))

    classified: list[SourceRecord] = []
    log: list[str] = []

    for src in state.raw_sources:
        labels = _classify_source(src, state.cost_metrics)

        # Apply labels back to the source record
        src.source_type = labels.get("source_type", "unknown")
        src.domain = labels.get("domain", "general")
        src.relevance_tier = labels.get("relevance", "medium")

        if src.relevance_tier == "low":
            # Log and discard low-quality sources
            reason = (
                f"Discarded [{src.source_type}/{src.domain}] "
                f"score={src.relevance_score:.1f} — {src.url}"
            )
            log.append(reason)
            logger.info("  %s", reason)
        else:
            classified.append(src)
            logger.info(
                "  Kept [%s/%s/%s] score=%.1f — %s",
                src.source_type, src.domain, src.relevance_tier,
                src.relevance_score, src.url,
            )

    logger.info(
        "Classification: %d kept, %d discarded.",
        len(classified), len(state.raw_sources) - len(classified),
    )
    return {
        "classified_sources": classified,
        "classification_log": log,
        "status": "extracting_entities",
    }
