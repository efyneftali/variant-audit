"""MCP tools — live evidence sources the agent queries during classification.

Each module wraps one external data source as a callable tool the LangGraph
agent (and, later, any MCP client) can invoke. RAG gives the agent the *rules*;
these give it the *facts* about a specific variant.

  clinvar.py       known clinical assertions (also your eval ground-truth source)
  gnomad.py        population allele frequency (ACMG PM2/BA1/BS1)
  ucsc.py          conservation, genes, regulatory context (UCSC REST API)
  alphamissense.py DeepMind pathogenicity score (missense only) — an input, not truth
  ensembl.py       variant consequence / VEP annotation

Build clinvar.py first (day-4: one tool end to end), then add the rest on day-7.
Each tool should: validate input, call the source, return a typed/structured
result, and fail gracefully on an unknown variant.
"""
