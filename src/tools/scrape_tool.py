"""
scrape_tool.py — MCP-style tool for fetching and cleaning web page content.

Uses httpx + BeautifulSoup.  Returns clean plain text stripped of nav/ads.
Playwright-based scraping is in browser_tool.py for JS-heavy pages.
"""

from __future__ import annotations

import logging
import re

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

TOOL_SPEC = {
    "name": "scrape_page",
    "description": "Fetch and clean the text content of a web page URL.",
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to scrape."},
        },
        "required": ["url"],
    },
}

# Tags that typically contain navigation/ads — we discard them
_NOISE_TAGS = {"nav", "header", "footer", "aside", "script", "style", "noscript", "form"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36 research-bot/1.0"
    )
}


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
def _fetch_html(url: str, timeout: int = 15) -> str:
    """Download raw HTML for a URL.  Raises on HTTP errors."""
    resp = httpx.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def _clean_html(html: str) -> str:
    """
    Extract readable text from raw HTML.

    Steps:
      1. Parse with BeautifulSoup (lxml or html.parser).
      2. Remove noise tags.
      3. Extract text from <article>, <main>, or <body>.
      4. Collapse whitespace.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements in-place
    for tag in soup(list(_NOISE_TAGS)):
        tag.decompose()

    # Prefer semantic containers; fall back to the whole body
    container = (
        soup.find("article")
        or soup.find("main")
        or soup.find(id=re.compile(r"content|main|article", re.I))
        or soup.body
        or soup
    )

    text = container.get_text(separator="\n", strip=True)

    # Collapse runs of blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def scrape_page(url: str, max_chars: int = 8_000) -> str:
    """
    MCP-style tool: fetch a URL and return clean text (≤ max_chars).

    Returns an empty string and logs a warning on failure — callers must
    handle the possibility of empty content gracefully.
    """
    if not url or not url.startswith(("http://", "https://")):
        logger.warning("Invalid URL skipped: %r", url)
        return ""

    try:
        html = _fetch_html(url)
        text = _clean_html(html)
        return text[:max_chars]  # Limit to avoid blowing context windows
    except httpx.HTTPStatusError as exc:
        logger.warning("HTTP %s when scraping %s", exc.response.status_code, url)
    except httpx.RequestError as exc:
        logger.warning("Network error scraping %s: %s", url, exc)
    except Exception as exc:
        logger.warning("Unexpected scrape error for %s: %s", url, exc)

    return ""
