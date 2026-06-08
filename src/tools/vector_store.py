"""
vector_store.py — ChromaDB-backed semantic search over collected sources.

After the research agent gathers sources, we embed them with a local
sentence-transformers model and store them in ChromaDB.  The analyzer agent
can then retrieve the most relevant chunks for each theme without re-scanning
all sources linearly.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from src.config import config
from src.state import SourceRecord

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "research_sources"
_EMBED_MODEL = "all-MiniLM-L6-v2"  # Small, fast, good quality


@lru_cache(maxsize=1)
def _get_collection():
    """
    Lazily initialise the ChromaDB client and collection (cached per process).

    Caching matters for RAG: without it, every index/query call would rebuild
    the client AND reload the local sentence-transformers embedding model
    (seconds each).  Retrieving evidence for several sub-questions would then
    reload the model several times.  With the cache the model loads exactly once.
    A persistent local directory keeps the store alive between Streamlit reruns.
    """
    import chromadb  # noqa: PLC0415  (lazy — not always installed)
    from chromadb.utils import embedding_functions  # noqa: PLC0415

    client = chromadb.PersistentClient(path=str(config.chroma_dir))

    # Sentence-transformers embedding function (runs locally — no API cost)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=_EMBED_MODEL
    )

    # get_or_create so we can call this multiple times safely
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def index_sources(sources: list[SourceRecord], topic: str) -> None:
    """
    Embed and store all sources in ChromaDB.

    Each source is split into: title + snippet (short doc) and full_content
    (long doc).  Both are indexed so retrieval can find relevant passages.
    """
    try:
        collection = _get_collection()

        # Clear existing entries for this topic to avoid stale data
        try:
            existing_ids = collection.get(where={"topic": topic})["ids"]
            if existing_ids:
                collection.delete(ids=existing_ids)
        except Exception:
            pass  # Collection might be empty — that's fine

        documents, metadatas, ids = [], [], []

        for i, src in enumerate(sources):
            # Short document: title + snippet (fast retrieval)
            short_text = f"{src.title}\n{src.snippet}"
            documents.append(short_text)
            metadatas.append({
                "topic": topic,
                "url": src.url,
                "source_type": src.source_type,
                "relevance_score": src.relevance_score,
            })
            ids.append(f"src_{i}_short")

            # Long document: full content (richer context)
            if src.full_content:
                documents.append(src.full_content[:2000])  # Cap to keep index fast
                metadatas.append({
                    "topic": topic,
                    "url": src.url,
                    "source_type": src.source_type,
                    "relevance_score": src.relevance_score,
                })
                ids.append(f"src_{i}_long")

        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info("Indexed %d document chunks in ChromaDB.", len(documents))

    except Exception as exc:
        logger.warning("ChromaDB indexing failed (non-fatal): %s", exc)


def query_store(query: str, n_results: int = 5, topic: Optional[str] = None) -> list[str]:
    """
    Return the top-N most semantically similar text chunks for a query.

    Used by the analyzer agent to pull relevant evidence per theme without
    re-reading all sources.
    """
    try:
        collection = _get_collection()
        where = {"topic": topic} if topic else None
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        # results["documents"] is a list-of-lists (one per query)
        return results["documents"][0] if results["documents"] else []
    except Exception as exc:
        logger.warning("ChromaDB query failed (non-fatal): %s", exc)
        return []
