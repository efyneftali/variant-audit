"""Linear variant classification — no loops, no state machine.

One function: gather ClinVar evidence, retrieve relevant ACMG criteria via RAG,
ask the LLM to classify. Returns a structured result.

This gets absorbed into graph.py on Day 6 as individual nodes (gather_evidence,
retrieve_criteria, classify). Until then it's a single flat function that's easy
to call from scratch.py or the /classify API endpoint.
"""

from . import llm
from .mcp_tools.clinvar import get_clinvar_record
from .retrieval import semantic_search

SYSTEM_PROMPT = (
    "You are a clinical variant classification assistant. "
    "Use only the evidence and ACMG criteria provided — do not draw on outside knowledge. "
    "Classify the variant as exactly one of: Pathogenic, Likely Pathogenic, "
    "Uncertain Significance, Likely Benign, or Benign. "
    "Then list the specific ACMG criteria codes you relied on (e.g. PVS1, PS1, PM2). "
    "Format your response as:\n"
    "Classification: <tier>\n"
    "Criteria used: <comma-separated list, or 'none identified' if none apply>"
)


def classify_variant(variant: str) -> dict:
    """Classify a variant using ClinVar evidence + ACMG criteria RAG.

    Args:
        variant: an rsID (e.g. "rs28897696").
    Returns:
        {
          "variant": str,
          "classification": str,       # one of the 5 ACMG tiers
          "evidence_used": list[dict], # ClinVar matches that informed the call
          "criteria_cited": list[str], # ACMG guideline source files retrieved
        }
    """
    # --- Step 1: gather evidence from ClinVar ---
    clinvar_record = get_clinvar_record(variant)
    matches = clinvar_record.get("matches", [])

    # --- Step 2: build a search query from the evidence ---
    # If ClinVar found something, encode the variant name + significances into
    # the query so RAG surfaces the most relevant ACMG criteria sections.
    # If not found, fall back to just the raw variant string.
    if matches:
        significances = ", ".join(m["clinical_significance"] for m in matches)
        hgvs_names = ", ".join(m["hgvs"] for m in matches)
        search_query = f"{hgvs_names} {significances}"
    else:
        search_query = variant

    # --- Step 3: retrieve relevant ACMG criteria from the knowledge base ---
    chunks = semantic_search(search_query)
    criteria_text = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
    criteria_cited = sorted({c.source for c in chunks})

    # --- Step 4: format the evidence block ---
    if matches:
        evidence_lines = []
        for m in matches:
            evidence_lines.append(
                f"HGVS: {m['hgvs']}\n"
                f"ClinVar significance: {m['clinical_significance']}\n"
                f"Review status: {m['review_status']}"
            )
        evidence_text = "\n\n".join(evidence_lines)
    else:
        evidence_text = f"No ClinVar record found for {variant}."

    # --- Step 5: call the LLM ---
    prompt = (
        f"Variant: {variant}\n\n"
        f"== ClinVar Evidence ==\n{evidence_text}\n\n"
        f"== Relevant ACMG Criteria ==\n{criteria_text}\n\n"
        f"Classify this variant."
    )
    answer = llm.complete(prompt, system=SYSTEM_PROMPT, purpose="generate")

    return {
        "variant": variant,
        "classification": answer,
        "evidence_used": matches,
        "criteria_cited": criteria_cited,
    }
