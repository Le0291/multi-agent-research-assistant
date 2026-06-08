"""
llm.py — Reusable Anthropic Claude client.

Every agent imports get_claude() to get a configured ChatAnthropic instance.
Token counts from each call are forwarded to the shared CostMetrics object
so we can display live cost estimates in the UI.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage

from src.config import config
from src.state import CostMetrics

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)  # Build the client once; reuse across all agents
def get_claude(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatAnthropic:
    """
    Return a cached ChatAnthropic instance using settings from config.

    We use langchain-anthropic so that LangChain's standard runnable
    interface (invoke / stream / batch) works transparently with our graph.
    """
    if not config.anthropic_api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is missing.  "
            "Set it in your .env file or as a system environment variable."
        )

    return ChatAnthropic(
        model=config.anthropic_model,
        api_key=config.anthropic_api_key,
        temperature=temperature if temperature is not None else config.anthropic_temperature,
        max_tokens=max_tokens if max_tokens is not None else config.anthropic_max_tokens,
        # Hard 90-second per-request timeout. Without this the client can hang
        # INDEFINITELY on a network stall or an unreachable API endpoint, which
        # freezes the whole pipeline at "Running Orchestrator…" with no error.
        # With a timeout the call fails fast and the agent's fallback kicks in.
        default_request_timeout=90,
        # Retry up to 2 times on transient Anthropic errors (timeouts, 5xx, 429)
        max_retries=2,
    )


def invoke_claude(
    messages: list[BaseMessage],
    cost_metrics: Optional[CostMetrics] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Invoke Claude and return the response text.

    Also records approximate token usage into cost_metrics if provided.
    """
    llm = get_claude(temperature=temperature, max_tokens=max_tokens)

    try:
        response = llm.invoke(messages)
    except Exception as exc:
        logger.error("Claude invocation failed: %s", exc)
        raise

    content: str = response.content if hasattr(response, "content") else str(response)

    # --- Token tracking (approximate) ----------------------------------------
    # langchain-anthropic exposes usage_metadata on the AIMessage
    if cost_metrics is not None and hasattr(response, "usage_metadata"):
        usage = response.usage_metadata or {}
        cost_metrics.add(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

    return content
