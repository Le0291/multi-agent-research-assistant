"""
classification_eval.py — Evaluate source classification accuracy.

Runs the classification agent on a small labelled test set and reports
precision / recall for each source_type category.

Usage:
    python -m src.evaluation.classification_eval
"""

from __future__ import annotations

from collections import defaultdict
from src.state import SourceRecord, CostMetrics
from src.agents.classification_agent import _classify_source

# ── Small hand-labelled test set ─────────────────────────────────────────────
TEST_CASES: list[tuple[SourceRecord, str, str]] = [
    (
        SourceRecord(
            url="https://arxiv.org/abs/1706.03762",
            title="Attention Is All You Need",
            snippet="We propose the Transformer, a model architecture relying entirely on attention mechanisms.",
            full_content="", relevance_score=9.0,
        ),
        "academic_paper", "AI",
    ),
    (
        SourceRecord(
            url="https://docs.python.org/3/library/asyncio.html",
            title="asyncio — Asynchronous I/O",
            snippet="This module provides infrastructure for writing single-threaded concurrent code.",
            full_content="", relevance_score=6.0,
        ),
        "documentation", "technical",
    ),
    (
        SourceRecord(
            url="https://techcrunch.com/2024/01/01/openai-raises-2b",
            title="OpenAI raises $2B in new funding",
            snippet="OpenAI, the maker of ChatGPT, announced a new funding round.",
            full_content="", relevance_score=5.0,
        ),
        "news", "business",
    ),
    (
        SourceRecord(
            url="https://towardsdatascience.com/transformers-explained",
            title="Transformers Explained: A Step-by-Step Tutorial",
            snippet="In this tutorial, we walk through the transformer architecture step by step.",
            full_content="", relevance_score=7.0,
        ),
        "tutorial", "AI",
    ),
    (
        SourceRecord(
            url="https://huggingface.co/datasets/squad",
            title="SQuAD: Stanford Question Answering Dataset",
            snippet="SQuAD is a reading comprehension dataset consisting of questions on Wikipedia articles.",
            full_content="", relevance_score=7.0,
        ),
        "dataset", "AI",
    ),
]


def run_eval():
    """Evaluate classification on the labelled test set."""
    metrics = CostMetrics()
    correct_type = 0
    correct_domain = 0

    print(f"{'URL':<50} {'Expected':<18} {'Predicted':<18} {'Match'}")
    print("-" * 100)

    for src, expected_type, expected_domain in TEST_CASES:
        result = _classify_source(src, metrics)
        pred_type = result.get("source_type", "unknown")
        pred_domain = result.get("domain", "general")

        type_match = pred_type == expected_type
        domain_match = pred_domain == expected_domain
        correct_type += type_match
        correct_domain += domain_match

        print(
            f"{src.url[:48]:<50} "
            f"{expected_type:<18} {pred_type:<18} "
            f"{'✓' if type_match else '✗'}"
        )

    n = len(TEST_CASES)
    print(f"\nSource-type accuracy : {correct_type}/{n} = {correct_type/n:.0%}")
    print(f"Domain accuracy      : {correct_domain}/{n} = {correct_domain/n:.0%}")
    print(f"Tokens used          : {metrics.input_tokens + metrics.output_tokens:,}")


if __name__ == "__main__":
    run_eval()
