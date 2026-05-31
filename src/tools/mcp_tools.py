"""
mcp_tools.py — MCP-style tool registry.

The Model Context Protocol (MCP) defines a standard way for language models to
discover and call external tools.  We implement the same JSON-schema descriptor
pattern here so that Claude can (in principle) call these tools via its
tool_use API.

Each tool has:
  - A TOOL_SPEC dict (name, description, input_schema) — the MCP descriptor.
  - A Python callable that executes the tool.

The registry maps tool names to callables for the orchestrator.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from src.tools.search_tool import web_search, TOOL_SPEC as SEARCH_SPEC
from src.tools.scrape_tool import scrape_page, TOOL_SPEC as SCRAPE_SPEC
from src.tools.browser_tool import browser_navigate, TOOL_SPEC as BROWSER_SPEC

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Additional lightweight tools                                                #
# --------------------------------------------------------------------------- #

SUMMARISE_SPEC = {
    "name": "summarise_text",
    "description": "Return a ≤200-word plain-text summary of the provided text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "max_words": {"type": "integer", "default": 200},
        },
        "required": ["text"],
    },
}


def summarise_text(text: str, max_words: int = 200) -> str:
    """
    Very simple extractive summariser (no LLM call — avoids extra cost).

    Splits text into sentences and returns the first N words.
    The research agent uses this to trim long scraped pages before passing
    them to Claude, keeping prompt size (and cost) down.
    """
    words = text.split()
    return " ".join(words[:max_words])


RELEVANCE_SPEC = {
    "name": "score_relevance",
    "description": "Heuristically score how relevant a text snippet is to a topic (0–10).",
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["topic", "text"],
    },
}


def score_relevance(topic: str, text: str) -> float:
    """
    Quick keyword-overlap relevance score (0–10).

    A cheap heuristic so the research agent can pre-filter sources before
    spending Claude tokens on deeper analysis.
    """
    topic_words = set(topic.lower().split())
    text_lower = text.lower()
    # Count how many topic keywords appear in the text
    hits = sum(1 for w in topic_words if w in text_lower)
    # Normalise to 0–10
    return round(min(10.0, hits / max(len(topic_words), 1) * 10), 1)


# --------------------------------------------------------------------------- #
#  Tool registry                                                               #
# --------------------------------------------------------------------------- #

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "web_search": {
        "spec": SEARCH_SPEC,
        "fn": web_search,
    },
    "scrape_page": {
        "spec": SCRAPE_SPEC,
        "fn": scrape_page,
    },
    "browser_navigate": {
        "spec": BROWSER_SPEC,
        "fn": browser_navigate,
    },
    "summarise_text": {
        "spec": SUMMARISE_SPEC,
        "fn": summarise_text,
    },
    "score_relevance": {
        "spec": RELEVANCE_SPEC,
        "fn": score_relevance,
    },
}


def list_tools() -> list[dict[str, Any]]:
    """Return all MCP-style tool specs (for display / passing to Claude)."""
    return [entry["spec"] for entry in TOOL_REGISTRY.values()]


def call_tool(name: str, **kwargs: Any) -> Any:
    """
    Execute a registered tool by name.

    This is the MCP dispatch layer: the orchestrator (or Claude via tool_use)
    calls this function with the tool name and arguments.
    """
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name!r}.  Available: {list(TOOL_REGISTRY)}")

    fn: Callable = TOOL_REGISTRY[name]["fn"]
    logger.info("MCP tool call: %s(%s)", name, json.dumps(kwargs, default=str)[:120])
    return fn(**kwargs)
