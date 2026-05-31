"""
benchmark.py — Run the pipeline on 5 test topics and record metrics.

Usage:
    python -m src.evaluation.benchmark

Output: benchmark_results.csv in the project root.
"""

from __future__ import annotations

import csv
import dataclasses
import time
from datetime import datetime
from pathlib import Path

TOPICS = [
    "Transformer architecture in large language models",
    "Mixture of Experts in neural networks",
    "Retrieval-Augmented Generation (RAG) systems",
    "AI safety and alignment techniques",
    "Multimodal AI models and their applications",
]

OUTPUT_FILE = Path(__file__).resolve().parent.parent.parent / "benchmark_results.csv"

FIELDNAMES = [
    "topic", "runtime_seconds", "num_raw_sources", "num_classified_sources",
    "num_entities", "report_length_chars", "critic_score", "revision_count",
    "input_tokens", "output_tokens", "estimated_cost_usd",
    "source_types_summary", "timestamp",
]


def run_benchmark():
    """Run the full pipeline on each topic and write results to CSV."""
    from src.graph import run_pipeline  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415

    rows = []

    for topic in TOPICS:
        print(f"\n{'='*60}\nBenchmarking: {topic}\n{'='*60}")
        start = time.time()

        try:
            state = run_pipeline(topic)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append({
                "topic": topic,
                "runtime_seconds": round(time.time() - start, 1),
                "num_raw_sources": 0,
                "num_classified_sources": 0,
                "num_entities": 0,
                "report_length_chars": 0,
                "critic_score": 0,
                "revision_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0,
                "source_types_summary": f"FAILED: {exc}",
                "timestamp": datetime.utcnow().isoformat(),
            })
            continue

        elapsed = round(time.time() - start, 1)
        type_counts = Counter(s.source_type for s in state.classified_sources)
        type_summary = "; ".join(f"{t}={c}" for t, c in type_counts.most_common())

        row = {
            "topic": topic,
            "runtime_seconds": elapsed,
            "num_raw_sources": len(state.raw_sources),
            "num_classified_sources": len(state.classified_sources),
            "num_entities": len(state.entities),
            "report_length_chars": len(state.final_report),
            "critic_score": state.critic_score,
            "revision_count": state.revision_count,
            "input_tokens": state.cost_metrics.input_tokens,
            "output_tokens": state.cost_metrics.output_tokens,
            "estimated_cost_usd": round(state.cost_metrics.estimated_cost_usd, 4),
            "source_types_summary": type_summary,
            "timestamp": datetime.utcnow().isoformat(),
        }
        rows.append(row)
        print(f"  Done in {elapsed}s | score={state.critic_score}/10 | sources={len(state.classified_sources)}")

    # Write CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_benchmark()
