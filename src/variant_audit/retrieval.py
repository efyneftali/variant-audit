"""Semantic search over Qdrant — the "R" in RAG.

Given a query, embed it and return the most similar guideline chunks. Used by
the state machine's "retrieve criteria" node and measured directly by the eval
harness (recall@k).

TODO(day-2): implement semantic_search.
"""

from dataclasses import dataclass

from .config import settings


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
    raise NotImplementedError("TODO(day-2): embed query, query Qdrant, map hits")
