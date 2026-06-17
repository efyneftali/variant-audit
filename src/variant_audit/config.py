"""Central configuration, read from environment (.env supported).

This one is provided working — it's plumbing, not the learning. Everything else
in the package imports `settings` from here. Read it so you know what knobs exist.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # --- LLM provider switch (local-first) ---
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))

    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    ollama_judge_model: str = field(default_factory=lambda: os.getenv("OLLAMA_JUDGE_MODEL", "llama3.1:8b"))

    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    anthropic_judge_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_JUDGE_MODEL", "claude-haiku-4-5-20251001")
    )

    # --- vector store / embeddings ---
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "variant_audit"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # --- chunking / retrieval knobs (experiment with these; measure via evals) ---
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5

    # --- bounded loop budgets for the state machine ---
    max_query_rewrites: int = 1
    max_generation_retries: int = 1


settings = Settings()
