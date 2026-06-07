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


def _score_source(topic: str, content: str, cost_metrics: Any) -> float:
    """
    Score a source's relevance (1–10) to the topic using keyword heuristics.

    PERFORMANCE NOTE
    ----------------
    Previously this made ONE Claude API call per source.  With ~8 results per
    query and several queries, that meant 20-40 sequential API calls just for
    scoring — slow (30s+) and expensive, and a frequent cause of the pipeline
    appearing to "stall" on memory-limited hosts like Railway.

    We now use a fast, FREE keyword-overlap heuristic (score_relevance) that
    runs instantly with no API call.  Tavily already returns results ranked by
    relevance, so this heuristic is more than sufficient for filtering, and the
    downstream Classification Agent still uses Claude for quality labelling.

    IMPORTANT: This function is called ONLY when scraping actually succeeded
    (non-empty content).  Pages that returned HTTP 403 / failed to scrape are
    discarded before reaching this function.  We deliberately do NOT fall back
    to scoring the Tavily snippet alone — Tavily pre-selects snippets for
    relevance, so every snippet would score 10/10 even for blocked pages.
    """
    score = call_tool("score_relevance", topic=topic, text=content)
    # Apply a minimum floor only for pages with substantial scraped content.
    # Very short content (< 200 chars) is likely navigation-only or bot-check
    # pages — let those score naturally so they can be filtered out.
    if len(content) > 200 and score < 4.0:
        score = 4.0
    return float(score)


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

    # ── Preserve uploaded-file sources ────────────────────────────────────────
    # The Full Pipeline injects any user-uploaded document as a SourceRecord
    # with url="file://..." before the graph starts.  Without this guard the
    # research agent would overwrite raw_sources and silently drop the file.
    file_sources = [s for s in state.raw_sources if s.url.startswith("file://")]
    for fs in file_sources:
        seen_urls.add(fs.url)   # prevent the loop from re-processing file URLs
    if file_sources:
        logger.info(
            "Research Agent: preserving %d uploaded-file source(s).", len(file_sources)
        )

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
            # Early-exit: stop scraping the moment we have enough sources.
            # This prevents wasted scrape calls (and time) after the target
            # is met within a single query's result batch.
            if len(collected) >= config.max_sources:
                break
            if src.url in seen_urls or not src.url:
                continue  # Skip duplicates
            seen_urls.add(src.url)

            # ── ACTION: scrape full content ───────────────────────────────────
            logger.info("ReAct ACTION: scrape_page(%s)", src.url)
            scrape_ok = False
            try:
                full_text = call_tool("scrape_page", url=src.url)
                if full_text:
                    # Cap content to 8 000 chars — NER/analysis only uses the
                    # first 5 000 anyway, and smaller payloads cut session-state
                    # memory roughly 4× on Railway (prevents OOM crash on 2nd run).
                    src.full_content = full_text[:8_000]
                    scrape_ok = True
            except Exception as exc:
                logger.warning("Scrape failed for %s: %s", src.url, exc)

            # ── THOUGHT: discard inaccessible pages (e.g. HTTP 403) ──────────
            # Never score on the Tavily snippet alone — Tavily pre-selects
            # snippets for topic relevance so even a fully blocked page's snippet
            # scores 10/10.  If the real page is inaccessible, downstream agents
            # (NER, writer, …) cannot use its content anyway.
            if not scrape_ok:
                logger.info("  Discarded (inaccessible / no content): %s", src.url)
                continue

            # ── THOUGHT: score relevance on actual scraped content ────────────
            src.relevance_score = _score_source(topic, src.full_content, state.cost_metrics)

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

    # Merge uploaded-file sources back in (at the front — scored 9.0 by default)
    all_sources = file_sources + collected
    all_sources.sort(key=lambda s: s.relevance_score, reverse=True)

    logger.info(
        "Research Agent: collected %d web sources + %d file source(s) = %d total.",
        len(collected), len(file_sources), len(all_sources),
    )
    return {
        "raw_sources": all_sources,
        "status": "classifying",
    }
