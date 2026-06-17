"""Local embeddings via sentence-transformers — free, runs on CPU.

The embedding model turns text into vectors so Qdrant can do similarity search.
Swapping models (EMBEDDING_MODEL env var) and re-running evals is one of your
first real experiments: e.g. all-MiniLM-L6-v2 (fast) vs bge-small-en (better).

TODO(day-2): implement all three. The model loads lazily and is cached.
"""

from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def _model():
    """Load and cache the SentenceTransformer named by settings.embedding_model."""
    raise NotImplementedError("TODO(day-2): return SentenceTransformer(settings.embedding_model)")


def embedding_dim() -> int:
    """Vector dimension of the current model (Qdrant needs this to size the collection)."""
    raise NotImplementedError("TODO(day-2): return the model's sentence-embedding dimension")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings. Normalize embeddings (cosine distance assumes unit vectors)."""
    raise NotImplementedError("TODO(day-2): encode texts, normalized, return as lists of floats")


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    raise NotImplementedError("TODO(day-2): return embed_texts([query])[0]")
