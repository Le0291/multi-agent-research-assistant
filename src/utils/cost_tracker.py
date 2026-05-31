"""
cost_tracker.py — Token and cost tracking display helpers.

The CostMetrics object lives on ResearchState and is updated by invoke_claude()
after every LLM call.  This module formats the data for display in Streamlit.
"""

from __future__ import annotations

from src.state import CostMetrics


def format_cost_summary(metrics: CostMetrics) -> str:
    """Return a human-readable cost summary string."""
    return (
        f"Input tokens : {metrics.input_tokens:,}\n"
        f"Output tokens: {metrics.output_tokens:,}\n"
        f"Total tokens : {metrics.input_tokens + metrics.output_tokens:,}\n"
        f"Est. cost    : ${metrics.estimated_cost_usd:.4f} USD"
    )


def format_cost_table(metrics: CostMetrics) -> dict[str, str]:
    """Return a dict suitable for st.table() display."""
    return {
        "Metric": ["Input tokens", "Output tokens", "Total tokens", "Estimated cost"],
        "Value": [
            f"{metrics.input_tokens:,}",
            f"{metrics.output_tokens:,}",
            f"{metrics.input_tokens + metrics.output_tokens:,}",
            f"${metrics.estimated_cost_usd:.4f} USD",
        ],
    }
