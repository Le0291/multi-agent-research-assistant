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
    # Additional common AI/research tech terms
    "NER", "ReAct", "MCP", "Ollama", "RAG", "FAISS", "Pinecone",
    "LangSmith", "LangServe", "Mermaid", "Playwright", "BeautifulSoup",
    "Trafilatura", "Pydantic", "FastAPI", "Docker", "Railway",
    "Markdown", "PDF", "JSON", "YAML", "REST", "GraphQL",
    "Claude API", "Brave Search", "Tavily API",
    # Compound AI/ML terms that SpaCy often mis-tags as PERSON or ORG
    "Language Model", "Large Language Model", "Large Action Model",
    "Diffusion Models", "Diffusion Model", "Foundation Model",
    "Vector Store", "Vector Database", "Embedding Model",
    "ReAct Agents", "ReAct Agent", "Autonomous Agent",
    "Interface Streamlit", "Gradio", "Python", "JavaScript",
    "Format Markdown", "Markdown PDF", "Markdown / PDF", "Markdown/PDF",
    "Programming Language", "Web Scraping",
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
    # Generic action/document words that SpaCy tags as ORG/PERSON
    "drafts", "reviews", "review", "draft", "overview", "description",
    "descriptions", "working", "receives", "deliverables", "deliverable",
    "format", "interface", "programming", "language", "system",
    "project", "course", "size", "build", "output", "input",
    "intelligence",  # too generic — "Artificial Intelligence" is caught by phrase
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


def _clean_text_for_ner(text: str) -> str:
    """
    Pre-process scraped text before feeding to SpaCy.

    Removes:
    - Full URLs (http/https lines)
    - ASCII table borders (+----+, |---|)
    - Lines with excessive special characters (table rows, garbage)
    - Duplicate whitespace / control characters
    """
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip full URL-only lines
        if re.match(r'^https?://\S+$', stripped):
            continue
        # Skip ASCII table border rows: lines made mostly of +, -, |, =, v
        if re.match(r'^[\s+\-|=v><]+$', stripped):
            continue
        # Skip lines where >40% of chars are non-alphanumeric (garbage)
        alnum = sum(c.isalnum() or c.isspace() for c in stripped)
        if len(stripped) > 10 and alnum / len(stripped) < 0.60:
            continue
        # Remove inline URLs but keep surrounding text
        stripped = re.sub(r'https?://\S+', '', stripped).strip()
        if stripped:
            clean_lines.append(stripped)
    return ' '.join(clean_lines)


def _extract_spacy(text: str, url: str, nlp) -> list[tuple[str, str, str]]:
    """
    Run SpaCy NER on text and return (surface, spacy_label, category) tuples.

    Quality filters applied in order:
    1. Text cleaning     — remove URLs, ASCII tables, garbage lines.
    2. Label allow-list  — only labels in _LABEL_MAP are kept (drops CARDINAL,
       ORDINAL, PERCENT, MONEY, QUANTITY, TIME which produce pure noise).
    3. URL entity filter — entities containing 'http' or '://' are dropped.
    4. Length bounds     — must be 2–60 chars.
    5. Word count cap    — max 5 words (real named entities are short).
    6. Stop-word filter  — common noise words removed.
    7. Numeric filter    — purely numeric strings discarded.
    8. Garbage filter    — entities with too many special chars dropped.
    9. Tech-term remap   — known AI/ML terms re-labelled as "technology".
    """
    clean = _clean_text_for_ner(text[:8000])
    doc = nlp(clean[:5000])  # SpaCy input cap
    results = []
    for ent in doc.ents:
        surface = ent.text.strip()
        label   = ent.label_

        # ── Filter 1: only allowed SpaCy labels ──────────────────────────────
        if label not in _LABEL_MAP:
            continue

        # ── Filter 2: URL in entity text ─────────────────────────────────────
        if 'http' in surface or '://' in surface or surface.startswith('www.'):
            continue

        # ── Filter 3: length bounds (2–60 chars) ─────────────────────────────
        if len(surface) < 2 or len(surface) > 60:
            continue

        # ── Filter 4: word count cap (max 4 words) ───────────────────────────
        if len(surface.split()) > 4:
            continue

        # ── Filter 4b: strip trailing noise tokens from PERSON entities ───────
        # SpaCy often appends adjacent words to person names:
        # "Abdulkarim Albanna Multi-Agent" → "Abdulkarim Albanna"
        if label == "PERSON":
            surface = re.sub(
                r'\s+(Multi[-\s]?Agent|Research|Assistant|System|Project|'
                r'Agent|Model|Framework|Platform|Tool|API|Lab|Labs|Inc|Corp)s?$',
                '', surface, flags=re.IGNORECASE,
            ).strip()
            if not surface:
                continue

        # ── Filter 5: stop-word filter ────────────────────────────────────────
        if surface.lower() in _STOP_ENTITIES:
            continue

        # ── Filter 5b: any word in entity is a hard-stop word ─────────────────
        # Catches "Deliverables Working", "Format Markdown" etc.
        _HARD_STOPS = {
            "working", "deliverables", "deliverable", "receives",
            "format", "overview", "build", "interface",
        }
        words_lower = {w.lower() for w in surface.split()}
        if words_lower & _HARD_STOPS:
            continue

        # ── Filter 6: pure numeric strings ───────────────────────────────────
        if re.match(r'^\d[\d,.\s%\-/]*$', surface):
            continue

        # ── Filter 7: garbage character filter ───────────────────────────────
        # Drop entities where >25% of chars are special (e.g., "+--+", "v v v")
        special = sum(not (c.isalnum() or c.isspace() or c in ".-,'&/()") for c in surface)
        if len(surface) > 4 and special / len(surface) > 0.25:
            continue

        # ── Filter 7b: bullet / symbol prefix ────────────────────────────────
        # Drop entities starting with bullet chars, symbols, or digits+dash
        if re.match(r'^[•·▪▸►▹\-–—*#@$%^&]', surface):
            continue

        # ── Filter 7c: numeric suffix garbage ────────────────────────────────
        # Drop entities ending in loose numbers like "Size 2-", "Description 5.1"
        if re.search(r'\s\d[\d.\-]*\s*$', surface):
            continue

        # ── Filter 7d: single-word generic terms via stop-entity check ────────
        if surface.title().lower() in _STOP_ENTITIES or surface.lower() in _STOP_ENTITIES:
            continue

        # ── Filter 7e: mixed brand+generic concatenation (SpaCy list bleed) ───
        # "Gradio Programming Language Python" — contains generic filler words
        # If entity has ≥3 words AND ≥1 word is a generic connector, drop it.
        _GENERIC_CONNECTORS = {"programming", "language", "based", "using",
                               "via", "with", "through", "for", "and", "or"}
        words = surface.lower().split()
        if len(words) >= 3 and len(set(words) & _GENERIC_CONNECTORS) >= 1:
            continue

        # ── Filter 8: tech-term category remapping ────────────────────────────
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

    # ── Normalise + deduplicate similar entities ─────────────────────────────
    # "the Model Context Protocol" → "Model Context Protocol"
    # "langchain" + "LangChain" → merged under the most-seen capitalisation
    normalised_counter: Counter = Counter()
    normalised_labels:  dict[str, tuple[str, str]] = {}
    normalised_sources: dict[str, list[str]] = defaultdict(list)
    canonical_map: dict[str, str] = {}  # normalised_key → preferred surface

    for surface, count in entity_counter.items():
        # Strip common leading articles
        core = re.sub(r'^(the|a|an)\s+', '', surface, flags=re.IGNORECASE).strip()
        key  = core.lower()

        if key not in canonical_map:
            canonical_map[key] = core  # first seen wins as display name
        else:
            # Prefer the capitalisation with the higher count
            existing = canonical_map[key]
            if entity_counter[surface] > entity_counter.get(existing, 0):
                canonical_map[key] = core

        preferred = canonical_map[key]
        normalised_counter[preferred] += count
        if key not in normalised_labels:
            normalised_labels[key] = entity_labels.get(surface, ("MISC", "concept"))
        normalised_sources[preferred].extend(entity_source_map.get(surface, []))

    # ── Build EntityRecord list ───────────────────────────────────────────────
    entity_records: list[EntityRecord] = []
    for surface, count in normalised_counter.most_common(80):  # Top 80 after dedup
        key = re.sub(r'^(the|a|an)\s+', '', surface, flags=re.IGNORECASE).strip().lower()
        label, category = normalised_labels.get(key, ("MISC", "concept"))
        entity_records.append(
            EntityRecord(
                text=surface,
                label=label,
                category=category,
                count=count,
                sources=list(set(normalised_sources[surface])),  # Deduplicate URLs
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
