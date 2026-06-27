"""Minimal RAG glue: retrieve chunks, stuff them into a prompt, call the LLM.

Day-6 will absorb this into the graph as a node (retrieve -> generate, with
grading/retry loops around it). Until then it's a single small function so
it's easy to call from scratch.py / the eval harness.
"""

from __future__ import annotations

from . import llm
from .retrieval import semantic_search

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. Cite the source "
    "file for each fact you use. If the context does not contain the answer, "
    "say so instead of guessing."
)


def answer_question(question: str, top_k: int | None = None) -> tuple[str, list[str]]:
    """Retrieve relevant chunks for `question` and answer it grounded in them.

    Returns (answer, sources) where sources is the deduplicated list of
    source filenames the retrieved chunks came from.
    """
    chunks = semantic_search(question, top_k=top_k)
    context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    answer = llm.complete(prompt, system=SYSTEM_PROMPT, purpose="generate")
    sources = sorted({c.source for c in chunks})
    return answer, sources
