"""Local embeddings via sentence-transformers — free, runs on CPU.

The embedding model turns text into vectors so Qdrant can do similarity search.
Swapping models (EMBEDDING_MODEL env var) and re-running evals is one of your
first real experiments: e.g. all-MiniLM-L6-v2 (fast) vs bge-small-en (better).
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer named by settings.embedding_model."""
    return SentenceTransformer(settings.embedding_model)


def embedding_dim() -> int:
    """Vector dimension of the current model (Qdrant needs this to size the collection)."""
    return _model().get_sentence_embedding_dimension()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings. Normalize embeddings (cosine distance assumes unit vectors)."""
    return _model().encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
