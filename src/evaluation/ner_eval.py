"""
ner_eval.py — Evaluate NER extraction against a gold-standard annotation set.

Usage:
    python -m src.evaluation.ner_eval
"""

from __future__ import annotations

from src.state import SourceRecord, ResearchState
from src.agents.ner_agent import ner_agent_node

# ── Gold-standard test sentences with expected entities ──────────────────────
GOLD = [
    {
        "text": "Geoffrey Hinton and Yann LeCun received the Turing Award for their work on deep learning at Google and Facebook.",
        "expected_entities": {"Geoffrey Hinton", "Yann LeCun", "Google", "Facebook", "Turing Award"},
    },
    {
        "text": "OpenAI released GPT-4 in March 2023, followed by Anthropic's Claude 2 in July 2023.",
        "expected_entities": {"OpenAI", "GPT-4", "Anthropic", "Claude 2", "March 2023", "July 2023"},
    },
    {
        "text": "The transformer architecture, introduced in the paper 'Attention Is All You Need' by Google Brain, powers models like BERT and T5.",
        "expected_entities": {"Google Brain", "BERT", "T5"},
    },
]


def _entity_f1(predicted: set[str], expected: set[str]) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for a single example."""
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)


def run_eval():
    """Evaluate NER on the gold-standard examples."""
    total_p, total_r, total_f1 = 0.0, 0.0, 0.0

    for i, example in enumerate(GOLD, 1):
        # Build a minimal state with one source per test case
        src = SourceRecord(
            url=f"https://test.example.com/{i}",
            title="Test",
            snippet=example["text"],
            full_content=example["text"],
            relevance_score=8.0,
        )
        state = ResearchState(topic="NER evaluation", classified_sources=[src])

        result = ner_agent_node(state)
        state.entities = result.get("entities", [])

        predicted = {e.text for e in state.entities}
        expected = example["expected_entities"]
        p, r, f1 = _entity_f1(predicted, expected)

        print(f"\nExample {i}: {example['text'][:70]}…")
        print(f"  Expected : {expected}")
        print(f"  Predicted: {predicted}")
        print(f"  P={p:.2f}  R={r:.2f}  F1={f1:.2f}")

        total_p += p
        total_r += r
        total_f1 += f1

    n = len(GOLD)
    print(f"\n{'='*50}")
    print(f"Avg Precision : {total_p/n:.3f}")
    print(f"Avg Recall    : {total_r/n:.3f}")
    print(f"Avg F1        : {total_f1/n:.3f}")


if __name__ == "__main__":
    run_eval()
