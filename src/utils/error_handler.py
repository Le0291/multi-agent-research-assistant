"""
error_handler.py — Centralised error handling and fallback utilities.

Every agent uses these helpers so the app never crashes silently.
"""

from __future__ import annotations

import logging
import traceback
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def safe_node(fallback_status: str = "error") -> Callable[[F], F]:
    """
    Decorator for LangGraph node functions.

    Catches any unhandled exception, logs a full traceback, and returns a
    partial state update with the error recorded — so the graph never crashes.

    Usage:
        @safe_node(fallback_status="analyzing")
        def my_agent_node(state): ...
    """
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(state: Any) -> dict[str, Any]:
            try:
                return fn(state)
            except Exception as exc:
                tb = traceback.format_exc()
                error_msg = f"{fn.__name__} failed: {exc}"
                logger.error("%s\n%s", error_msg, tb)

                # Append to state.errors (non-fatal — pipeline continues)
                errors = list(getattr(state, "errors", []))
                errors.append(error_msg)
                return {"errors": errors, "status": fallback_status}
        return wrapper  # type: ignore[return-value]
    return decorator


def check_api_keys() -> list[str]:
    """
    Validate that required environment variables are set.

    Returns a list of warning strings (empty = all good).
    """
    from src.config import config  # noqa: PLC0415

    warnings: list[str] = []
    if not config.anthropic_api_key:
        warnings.append(
            "ANTHROPIC_API_KEY is not set. The pipeline cannot run without it."
        )
    if not config.tavily_api_key and not config.brave_api_key:
        warnings.append(
            "No search API key found (TAVILY_API_KEY or BRAVE_API_KEY). "
            "Search will use a limited fallback."
        )
    return warnings
