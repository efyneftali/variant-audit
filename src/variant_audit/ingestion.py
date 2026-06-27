"""Ingestion: chunk the ACMG/ClinGen guideline corpus, embed it, load into Qdrant.

This builds the RAG knowledge base (the "rulebook"). Chunking is the highest-
leverage knob in RAG — chunk size/overlap directly affect retrieval quality, so
this is something you'll tune and measure with the eval harness.

TODO(day-2): implement chunk_text + ingest_directory.
"""

from __future__ import annotations

from .config import settings


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split a document into overlapping chunks (start with a simple word-window splitter).

    Defaults come from settings. Return a list of non-empty chunk strings.
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    words = text.split()
    stride = chunk_size - overlap

    chunks = []
    for start in range(0, len(words), stride):
        if start + chunk_size >= len(words):
            start = max(0, len(words) - chunk_size)
        chunk = words[start:start + chunk_size]
        chunk_str = " ".join(chunk)
        if chunk_str.strip() != "" and chunk_str not in chunks:
            chunks.append(chunk_str)
        if start + chunk_size >= len(words):
            break
    return chunks


def ingest_directory(corpus_dir: str = "data/corpus") -> int:
    """Chunk + embed + upsert every .md/.txt file under corpus_dir into Qdrant.

    Steps: ensure the collection exists (size = embeddings.embedding_dim(), cosine);
    for each file -> chunk_text -> embed_texts -> upsert points with payload
    {text, source, chunk_index}. Return the number of chunks ingested.

    Use a deterministic point id (hash of source+index+text) so re-ingesting
    updates instead of duplicating.
    """
    raise NotImplementedError("TODO(day-2): build points and upsert to Qdrant")
