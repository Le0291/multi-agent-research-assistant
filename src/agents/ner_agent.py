"""
ner_agent.py — Named Entity Recognition agent.

Uses SpaCy as the primary NER engine (fast, no API cost).
Optionally augments with Claude structured extraction for domain-specific
entities (AI architectures, dataset names, etc.) that SpaCy may miss.

Produces:
  - entities        : list[EntityRecord] with frequency counts
  - entity_relationships : co-occurrence pairs (entities in same sentence)
  - entity_source_map : {entity_text → [url, url, …]}
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from typing import Any

from src.state import EntityRecord, ResearchState

logger = logging.getLogger(__name__)

# SpaCy label → our higher-level category.
# CARDINAL, ORDINAL, PERCENT, MONEY, QUANTITY, TIME are intentionally EXCLUDED —
# they almost always produce noisy entities like "128", "first", "two", "10%",
# "billions of dollars", "$5", which add zero research value.
_LABEL_MAP: dict[str, str] = {
    "PERSON":     "person",
    "ORG":        "organization",
    "GPE":        "location",
    "LOC":        "location",
    "DATE":       "date",
    "PRODUCT":    "technology",
    "WORK_OF_ART":"concept",
    "LAW":        "concept",
    "EVENT":      "concept",
    "NORP":       "concept",       # Nationality/religion/political group → concept
    "FACILITY":   "location",
    "LANGUAGE":   "concept",
}

# Technology / AI terms that SpaCy's en_core_web_sm frequently mis-classifies
# as ORG (because they often appear in organization-like context).
# We remap these to "technology" to improve category quality.
_TECH_REMAPS: set[str] = {
    "AI", "ML", "NLP", "LLM", "LLMs", "GPT", "BERT", "CNN", "RNN", "LSTM",
    "API", "GPU", "CPU", "CUDA", "TPU", "MoE", "MoEs", "RAG", "RL",
    "LoRA", "RLHF", "SFT", "DPO", "Transformer", "Transformers",
    "PyTorch", "TensorFlow", "JAX", "NumPy", "SciPy",
    "Hugging Face", "HuggingFace", "OpenAI", "Anthropic",
    "LangChain", "LangGraph", "ChromaDB", "SpaCy", "Streamlit",
    "GitHub", "ChatGPT", "GPT-4", "GPT-3", "Claude", "Gemini",
    "Llama", "Mistral", "Falcon", "Mixtral", "Tavily", "Chroma",
}

# Case-insensitive lookup set for fast tech-term checking
_TECH_REMAPS_UPPER: set[str] = {t.upper() for t in _TECH_REMAPS}

# Entities to always skip: stop words + ordinal/number words that SpaCy
# occasionally tags as ORDINAL or DATE entities.
_STOP_ENTITIES: set[str] = {
    # Articles / prepositions / conjunctions
    "the", "a", "an", "of", "for", "in", "on", "at", "to", "and",
    "or", "but", "not", "with", "by", "from", "is", "are", "was", "be",
    # Ordinal / number words
    "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth", "last", "next", "recent",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    # Common noise determiners / quantifiers
    "many", "several", "few", "more", "most", "other", "some", "any",
    "all", "each", "both", "much", "such", "same", "also", "about",
}


def _load_spacy():
    """Load the SpaCy model, auto-downloading if necessary."""
    import spacy  # noqa: PLC0415

    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        logger.info("Downloading SpaCy model %s …", model_name)
        import subprocess, sys  # noqa: PLC0415
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True, capture_output=True,
        )
        return spacy.load(model_name)


def _extract_spacy(text: str, url: str, nlp) -> list[tuple[str, str, str]]:
    """
    Run SpaCy NER on text and return (surface, spacy_label, category) tuples.

    Quality filters applied in order:
    1. Label allow-list  — only labels in _LABEL_MAP are kept (drops CARDINAL,
       ORDINAL, PERCENT, MONEY, QUANTITY, TIME which produce pure noise).
    2. Minimum length    — must be ≥ 2 chars.
    3. Stop-word filter  — common noise words and ordinal number words removed.
    4. Numeric filter    — purely numeric strings like "128" or "2024" discarded.
    5. Tech-term remap   — known AI/ML abbreviations re-labelled as "technology"
       even when SpaCy incorrectly classifies them as ORG.
    """
    doc = nlp(text[:5000])  # Cap input to keep processing fast
    results = []
    for ent in doc.ents:
        surface = ent.text.strip()
        label   = ent.label_

        # ── Filter 1: only allowed SpaCy labels ──────────────────────────────
        if label not in _LABEL_MAP:
            continue  # silently drops CARDINAL, ORDINAL, PERCENT, MONEY …

        # ── Filter 2: minimum meaningful length ──────────────────────────────
        if len(surface) < 2:
            continue

        # ── Filter 3: stop-word / ordinal-word filter ─────────────────────────
        if surface.lower() in _STOP_ENTITIES:
            continue

        # ── Filter 4: pure numeric strings ───────────────────────────────────
        # Matches "128", "2,048", "3.14", "99.9%" etc.
        if re.match(r'^\d[\d,.\s%]*$', surface):
            continue

        # ── Filter 5: tech-term category remapping ────────────────────────────
        if surface in _TECH_REMAPS or surface.upper() in _TECH_REMAPS_UPPER:
            category = "technology"
        else:
            category = _LABEL_MAP[label]

        results.append((surface, label, category))
    return results


def _cooccurrences(text: str, entities: list[str], nlp) -> list[tuple[str, str]]:
    """
    Find entity pairs that appear in the same sentence (co-occurrence).

    Co-occurrence is a lightweight proxy for semantic relationships.
    """
    doc = nlp(text[:5000])
    pairs = []
    ent_set = set(entities)
    for sent in doc.sents:
        sent_ents = [
            ent.text.strip() for ent in sent.ents
            if ent.text.strip() in ent_set
        ]
        # Generate all pairs within the sentence
        for i, a in enumerate(sent_ents):
            for b in sent_ents[i + 1:]:
                if a != b:
                    pairs.append((a, b))
    return pairs


def ner_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: extract named entities from classified sources.

    Input state fields used : classified_sources
    Output state fields set  : entities, entity_relationships, entity_source_map, status
    """
    logger.info("NER Agent: processing %d sources.", len(state.classified_sources))

    try:
        nlp = _load_spacy()
    except Exception as exc:
        logger.error("SpaCy load failed: %s — NER skipped.", exc)
        state.errors.append(f"NER skipped: SpaCy unavailable ({exc})")
        return {"status": "browsing"}

    # Counters and maps
    entity_counter: Counter = Counter()          # entity_text → total occurrences
    entity_labels: dict[str, tuple[str, str]] = {}  # entity_text → (spacy_label, category)
    entity_source_map: dict[str, list[str]] = defaultdict(list)
    all_cooc: list[tuple[str, str]] = []

    for src in state.classified_sources:
        text = src.full_content or src.snippet
        if not text:
            continue

        # ── SpaCy extraction ──────────────────────────────────────────────────
        ents = _extract_spacy(text, src.url, nlp)
        ent_names = []
        for surface, label, category in ents:
            entity_counter[surface] += 1
            entity_labels[surface] = (label, category)
            entity_source_map[surface].append(src.url)
            ent_names.append(surface)

        # ── Co-occurrence within this source ──────────────────────────────────
        cooc = _cooccurrences(text, ent_names, nlp)
        all_cooc.extend(cooc)

    # ── Build EntityRecord list ───────────────────────────────────────────────
    entity_records: list[EntityRecord] = []
    for surface, count in entity_counter.most_common(100):  # Top 100 entities
        label, category = entity_labels.get(surface, ("MISC", "concept"))
        entity_records.append(
            EntityRecord(
                text=surface,
                label=label,
                category=category,
                count=count,
                sources=list(set(entity_source_map[surface])),  # Deduplicate URLs
            )
        )

    # Deduplicate co-occurrence pairs (keep unique unordered pairs)
    seen_pairs: set[frozenset] = set()
    unique_cooc: list[tuple[str, str]] = []
    for a, b in all_cooc:
        key = frozenset({a, b})
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_cooc.append((a, b))

    logger.info(
        "NER Agent: %d unique entities, %d relationships.",
        len(entity_records), len(unique_cooc),
    )
    return {
        "entities": entity_records,
        "entity_relationships": unique_cooc[:200],  # Cap to avoid state bloat
        "entity_source_map": dict(entity_source_map),
        "status": "browsing",
    }
