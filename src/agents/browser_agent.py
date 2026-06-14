"""
browser_agent.py — Large Action Model (LAM) browser automation agent.

This agent uses real browser automation (Playwright) to:
  1. Visit the top 3 classified sources for richer structured metadata.
  2. Capture screenshots as evidence of browser activity.
  3. Extract headings and Open Graph data that HTTP-only scraping misses.

LAM Context:
  A Large Action Model differs from a standard LLM in that it doesn't just
  generate text — it performs actions in the world (navigate, click, extract).
  This agent is a simplified LAM: Claude decides WHICH pages to visit;
  Playwright executes the browser actions.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.state import ResearchState
from src.tools.browser_tool import browser_navigate

logger = logging.getLogger(__name__)

def browser_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: use Playwright to enrich the top N classified sources.

    Input state fields used : classified_sources
    Output state fields set  : browser_results, status
    """
    # Read at call time so Railway env-var changes take effect without restart.
    browser_visit_count = int(os.environ.get("BROWSER_VISIT_COUNT", "3"))

    try:
        if browser_visit_count == 0:
            logger.info("Browser Agent: disabled via BROWSER_VISIT_COUNT=0.")
            return {"browser_results": [], "status": "analyzing"}

        sources_to_visit = state.classified_sources[:browser_visit_count]

        if not sources_to_visit:
            logger.info("Browser Agent: no sources to visit.")
            return {"browser_results": [], "status": "analyzing"}

        logger.info("Browser Agent: visiting %d source(s) via Playwright (BROWSER_VISIT_COUNT=%d).", len(sources_to_visit), browser_visit_count)
        results: list[dict[str, Any]] = []

        for src in sources_to_visit:
            logger.info("  Browser navigating to: %s", src.url)

            # Call the Playwright tool — always returns a dict (never raises)
            result = browser_navigate(src.url, take_screenshot=True)

            # Merge browser-extracted metadata back into the source record
            if "body_text" in result and result["body_text"]:
                if len(result["body_text"]) > len(src.full_content or ""):
                    src.full_content = result["body_text"]

            result["source_title"] = src.title
            results.append(result)

            if "error" in result:
                # Non-fatal — record and continue
                state.errors.append(
                    f"Browser agent error for {src.url}: {result['error']}"
                )

        logger.info("Browser Agent: completed %d visit(s).", len(results))
        return {
            "browser_results": results,
            "status": "analyzing",
        }

    except Exception as exc:
        # Absolute last resort — should never reach here, but if it does we
        # record the error and let the pipeline continue rather than crashing.
        logger.exception("Browser agent unexpected failure: %s", exc)
        state.errors.append(f"Browser agent failed entirely: {exc}")
        return {"browser_results": [], "status": "analyzing"}
