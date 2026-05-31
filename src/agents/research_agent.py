"""
research_agent.py — ReAct-style web research agent.

ReAct pattern (Reason + Act):
  THOUGHT: What do I need to find?  Which query should I run?
  ACTION:  Call web_search(query) or scrape_page(url).
  OBSERVATION: Parse the result.
  THOUGHT: Is the source good enough?  Do I need more sources?
  ... repeat until ≥ MIN_SOURCES high-quality sources are collected.

This loop is implemented explicitly here (not via LangChain's AgentExecutor)
for full control and transparency.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import config
from src.llm import invoke_claude
from src.state import ResearchState, SourceRecord
from src.tools.mcp_tools import call_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a thorough academic research agent.
Given a research topic and sub-questions, rate each source snippet 1-10 for
relevance.  Return ONLY the integer score, nothing else."""


def _score_source(topic: str, snippet: str, cost_metrics: Any) -> float:
    """Ask Claude to score a source's relevance (1–10) to the topic."""
    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Topic: {topic}\n\n"
                    f"Source snippet:\n{snippet[:800]}\n\n"
                    "Rate relevance 1-10 (integer only):"
                )
            ),
        ]
        raw = invoke_claude(messages, cost_metrics=cost_metrics, temperature=0.0)
        # Extract first number from response
        import re
        nums = re.findall(r"\d+", raw)
        return float(nums[0]) if nums else 5.0
    except Exception:
        return 5.0  # Default mid-range score on error


def research_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: gather ≥ MIN_SOURCES relevant web sources.

    ReAct loop:
      1. Generate search queries from sub_questions.
      2. Search → collect raw results.
      3. Scrape full content for top candidates.
      4. Score relevance with Claude.
      5. Repeat with refined queries if needed.

    Input state fields used : topic, sub_questions
    Output state fields set  : raw_sources, status
    """
    topic = state.topic
    sub_questions = state.sub_questions or [topic]
    collected: list[SourceRecord] = []
    seen_urls: set[str] = set()

    logger.info("Research Agent: starting ReAct loop for '%s'", topic)

    # ── THOUGHT: build search queries ─────────────────────────────────────────
    # Primary queries: one per sub-question + the raw topic
    queries = [topic] + [q for q in sub_questions[:5]]

    for query in queries:
        if len(collected) >= config.max_sources:
            break  # We have enough — stop searching

        # ── ACTION: web search ────────────────────────────────────────────────
        logger.info("ReAct ACTION: web_search(%r)", query)
        try:
            results: list[SourceRecord] = call_tool("web_search", query=query, max_results=8)
        except Exception as exc:
            logger.warning("Search failed for query %r: %s", query, exc)
            state.errors.append(f"Search failed for '{query}': {exc}")
            continue

        # ── OBSERVATION: process each result ──────────────────────────────────
        for src in results:
            if src.url in seen_urls or not src.url:
                continue  # Skip duplicates
            seen_urls.add(src.url)

            # ── ACTION: scrape full content ───────────────────────────────────
            logger.info("ReAct ACTION: scrape_page(%s)", src.url)
            try:
                full_text = call_tool("scrape_page", url=src.url)
                if full_text:
                    src.full_content = full_text
            except Exception as exc:
                logger.warning("Scrape failed for %s: %s", src.url, exc)
                # Non-fatal: we keep the snippet even without full content

            # ── THOUGHT: score relevance ──────────────────────────────────────
            score_text = src.full_content or src.snippet
            src.relevance_score = _score_source(topic, score_text, state.cost_metrics)

            # Only keep sources with relevance ≥ 4
            if src.relevance_score >= 4.0:
                collected.append(src)
                logger.info("  Kept (score=%.1f): %s", src.relevance_score, src.url)
            else:
                logger.info("  Discarded (score=%.1f): %s", src.relevance_score, src.url)

    # ── Final check ───────────────────────────────────────────────────────────
    if len(collected) < config.min_sources:
        logger.warning(
            "Only %d sources collected (minimum %d). "
            "Consider adding more search API keys.",
            len(collected), config.min_sources,
        )
        state.errors.append(
            f"Warning: collected only {len(collected)} sources "
            f"(minimum target: {config.min_sources})"
        )

    # Sort by relevance descending
    collected.sort(key=lambda s: s.relevance_score, reverse=True)

    logger.info("Research Agent: collected %d sources.", len(collected))
    return {
        "raw_sources": collected,
        "status": "classifying",
    }
