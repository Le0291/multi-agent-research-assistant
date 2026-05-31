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

# SpaCy label → our higher-level category
_LABEL_MAP = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "DATE": "date",
    "TIME": "date",
    "PRODUCT": "technology",
    "WORK_OF_ART": "concept",
    "LAW": "concept",
    "EVENT": "concept",
    "NORP": "organization",
    "FACILITY": "location",
    "LANGUAGE": "concept",
    "QUANTITY": "concept",
    "CARDINAL": "concept",
    "PERCENT": "concept",
    "MONEY": "concept",
    "ORDINAL": "concept",
}

# Entities to always skip (noise words)
_STOP_ENTITIES = {
    "the", "a", "an", "of", "for", "in", "on", "at", "to", "and",
    "or", "but", "not", "with", "by", "from", "is", "are", "was",
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

    Also extracts co-occurring entity pairs from the same sentence.
    """
    doc = nlp(text[:5000])  # Cap input to keep processing fast
    results = []
    for ent in doc.ents:
        surface = ent.text.strip()
        if len(surface) < 2 or surface.lower() in _STOP_ENTITIES:
            continue
        category = _LABEL_MAP.get(ent.label_, "concept")
        results.append((surface, ent.label_, category))
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
