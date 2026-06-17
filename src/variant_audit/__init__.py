"""Variant Audit — clinical variant-classification assistant + eval harness.

Package layout (fill these in as you build; see ../EvalForge_4week_plan.md):
    config.py       settings from env
    llm.py          provider-agnostic LLM call (Ollama | Anthropic)  ← the key abstraction
    embeddings.py   local sentence-transformers embeddings
    ingestion.py    chunk + embed ACMG guidelines -> Qdrant
    retrieval.py    semantic search (the "R" in RAG)
    mcp_tools/      live-evidence tools: ClinVar, gnomAD, UCSC, AlphaMissense, Ensembl
    graph.py        the LangGraph state machine (the agent)
    telemetry.py    OpenTelemetry traces + metrics
    api.py          FastAPI /ask
"""

__version__ = "0.0.1"
