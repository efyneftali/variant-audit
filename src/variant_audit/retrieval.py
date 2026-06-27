"""Semantic search over Qdrant — the "R" in RAG.

Given a query, embed it and return the most similar guideline chunks. Used by
the state machine's "retrieve criteria" node and measured directly by the eval
harness (recall@k).

TODO(day-2): implement semantic_search.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import settings
from . import embeddings
from qdrant_client import QdrantClient

@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


def semantic_search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Return the top_k most similar chunks to `query`.

    Steps: embeddings.embed_query(query) -> Qdrant query_points(limit=top_k) ->
    map each hit to a RetrievedChunk(text, source, score).
    """
    
    client = QdrantClient(url=settings.qdrant_url)
    query_vector = embeddings.embed_query(query)
    top_k = top_k or settings.top_k
    results = client.query_points(collection_name=settings.collection, query=query_vector, limit=top_k)
    retrieved_chunks = []
    for p in results.points:
        chunk = RetrievedChunk(text=p.payload["text"], source=p.payload["source"], score=p.score)
        retrieved_chunks.append(chunk)
    return retrieved_chunks