"""
search_tool.py — MCP-style wrapper around Tavily / Brave search APIs.

Returns a list of SourceRecord objects so callers never touch raw HTTP dicts.
Falls back to a simple DuckDuckGo scrape if neither key is configured.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import config
from src.state import SourceRecord

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Tool descriptor (MCP-style metadata)                                        #
# --------------------------------------------------------------------------- #
TOOL_SPEC = {
    "name": "web_search",
    "description": "Search the web for information about a query and return ranked results.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "max_results": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    },
}


# --------------------------------------------------------------------------- #
#  Tavily search                                                                #
# --------------------------------------------------------------------------- #
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _search_tavily(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Call Tavily search API and return raw result dicts."""
    from tavily import TavilyClient  # lazy import — not always installed
    client = TavilyClient(api_key=config.tavily_api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        include_raw_content=True,   # We want full page text when available
        search_depth="advanced",
    )
    return response.get("results", [])


# --------------------------------------------------------------------------- #
#  Brave search (REST API)                                                      #
# --------------------------------------------------------------------------- #
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _search_brave(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Call Brave Search API and return normalised result dicts."""
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": config.brave_api_key,
    }
    params = {"q": query, "count": max_results, "search_lang": "en"}
    resp = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers=headers,
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("web", {}).get("results", []):
        results.append({
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "content": item.get("description", ""),
            "raw_content": item.get("description", ""),
        })
    return results


# --------------------------------------------------------------------------- #
#  DuckDuckGo fallback (no API key needed)                                     #
# --------------------------------------------------------------------------- #
def _search_duckduckgo_fallback(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """
    Minimal fallback: fetches DuckDuckGo HTML and parses result links.
    Only used when no API key is configured — quality is lower.
    """
    from urllib.parse import quote_plus

    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (research-bot/1.0)"}
    try:
        resp = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("DuckDuckGo fallback failed: %s", exc)
        return []

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for a in soup.select("a.result__a")[:max_results]:
        title = a.get_text(strip=True)
        href = a.get("href", "")
        results.append({"url": href, "title": title, "content": title, "raw_content": ""})
    return results


# --------------------------------------------------------------------------- #
#  Public interface                                                             #
# --------------------------------------------------------------------------- #
def web_search(query: str, max_results: int = 10) -> list[SourceRecord]:
    """
    MCP-style tool: search the web and return SourceRecord objects.

    Priority: Tavily → Brave → DuckDuckGo fallback.
    """
    raw: list[dict[str, Any]] = []

    if config.tavily_api_key:
        logger.info("Searching via Tavily: %s", query)
        try:
            raw = _search_tavily(query, max_results)
        except Exception as exc:
            logger.warning("Tavily search failed: %s", exc)

    if not raw and config.brave_api_key:
        logger.info("Searching via Brave: %s", query)
        try:
            raw = _search_brave(query, max_results)
        except Exception as exc:
            logger.warning("Brave search failed: %s", exc)

    if not raw:
        logger.warning("Falling back to DuckDuckGo for: %s", query)
        raw = _search_duckduckgo_fallback(query, max_results)

    records: list[SourceRecord] = []
    for i, item in enumerate(raw):
        records.append(
            SourceRecord(
                url=item.get("url", ""),
                title=item.get("title", "Untitled"),
                snippet=item.get("content", "")[:500],  # cap snippet at 500 chars
                full_content=item.get("raw_content", item.get("content", "")),
                relevance_score=float(10 - i),  # Initial rank; refined later by research agent
            )
        )
    return records
