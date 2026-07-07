"""The agent — a corrective-classification LangGraph state machine.

Flow (bounded loops guarantee termination):

    gather_evidence -> retrieve_criteria -> grade_evidence --(sufficient)--> classify
            ^                                      |                              |
            |                                      v                              v
            +----(insufficient, bounded)----------+                     check_grounded
                                                                          |        |
                                                              (grounded)  |        | (ungrounded,
                                                                  -> END  |        |  bounded retry)
                                                                          +--------+--> classify

  gather_evidence   : call MCP tools (clinvar, gnomad, ucsc, alphamissense, ensembl)
  retrieve_criteria : RAG over ACMG guidelines (retrieval.semantic_search)
  grade_evidence    : is there enough evidence to classify? (llm.complete, purpose="grade")
  classify          : combine criteria -> Pathogenic / VUS / Benign + cited criteria
  check_grounded    : are the cited criteria actually supported? (purpose="groundedness")

TODO(day-8..9): add the grading node + conditional edges + bounded loops.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .classify import SYSTEM_PROMPT
from .config import settings
from . import llm
from .mcp_tools.alphamissense import get_alphamissense_score
from .mcp_tools.clinvar import get_clinvar_record
from .mcp_tools.ensembl import get_gene_consequence
from .mcp_tools.gnomad import get_allele_frequency
from .mcp_tools.ucsc import get_genomic_context
from .retrieval import semantic_search


class GraphState(TypedDict):
    variant: str
    evidence: dict            # accumulated MCP tool results
    criteria: list            # retrieved ACMG criteria chunks
    classification: str       # final call + justification
    rewrites: int             # evidence-gathering loop counter (bounded)
    gen_retries: int          # regeneration counter (bounded)
    grounded: bool
    sufficient: bool          # grade_evidence's verdict: enough evidence to classify?


# --- nodes (each takes GraphState, returns a partial state dict) ---

def gather_evidence(state: GraphState) -> dict:
    """Call the MCP tools for this variant and accumulate structured evidence.

    Each tool degrades to a {"found": False} result for an unknown/malformed
    variant rather than raising, so one missing source never blocks the others.

    Also advances the `rewrites` counter on a retry entry (i.e. when
    grade_evidence has looped back here), which is what makes
    route_after_grading's budget check a real, bounded limit rather than dead
    code -- `state["evidence"]` is only empty on the very first pass, since
    every prior gather_evidence call always populates all five keys.
    """
    variant = state["variant"]
    is_retry = bool(state.get("evidence"))
    ensembl_record = get_gene_consequence(variant)
    return {
        "evidence": {
            "clinvar": get_clinvar_record(variant),
            "gnomad": get_allele_frequency(variant),
            "ensembl": ensembl_record,
            "ucsc": get_genomic_context(variant, consequence=ensembl_record),
            "alphamissense": get_alphamissense_score(variant, consequence=ensembl_record),
        },
        "rewrites": state.get("rewrites", 0) + 1 if is_retry else state.get("rewrites", 0),
    }


def retrieve_criteria(state: GraphState) -> dict:
    """RAG: retrieve the ACMG criteria relevant to this variant/evidence."""
    matches = state["evidence"].get("clinvar", {}).get("matches", [])

    if matches:
        significances = ", ".join(m["clinical_significance"] for m in matches)
        hgvs_names = ", ".join(m["hgvs"] for m in matches)
        search_query = f"{hgvs_names} {significances}"
    else:
        search_query = state["variant"]

    chunks = semantic_search(search_query)
    return {"criteria": chunks}


GRADE_SYSTEM_PROMPT = (
    "You are grading whether gathered variant evidence is sufficient to reach an "
    "ACMG classification (Pathogenic/Likely Pathogenic/Uncertain Significance/"
    "Likely Benign/Benign) — you are not classifying the variant yourself. "
    "Missing sources are normal and often still enough (e.g. a clear ClinVar "
    "expert-panel assertion, or a clear null-variant consequence, can be enough "
    "on its own). Only call it insufficient if essentially nothing was found "
    "anywhere and there's no basis for a call. "
    "Respond with exactly one word on the first line — 'sufficient' or "
    "'insufficient' — then a one-sentence reason on the second line."
)


def _summarize_evidence_for_grading(evidence: dict, variant: str) -> str:
    """Compact per-source presence/absence summary — enough for a sufficiency
    judgment, not the full narrative detail classify() needs."""
    clinvar = evidence.get("clinvar", {})
    gnomad = evidence.get("gnomad", {})
    ensembl = evidence.get("ensembl", {})
    ucsc = evidence.get("ucsc", {})
    alphamissense = evidence.get("alphamissense", {})

    clinvar_line = (
        f"{len(clinvar.get('matches', []))} match(es)" if clinvar.get("found") else "no record"
    )
    gnomad_line = (
        f"allele_freq={gnomad.get('allele_freq')}" if gnomad.get("found") else "no record (absent from gnomAD)"
    )
    ensembl_line = (
        f"{ensembl.get('most_severe_consequence')} (impact={ensembl.get('impact')})"
        if ensembl.get("found") else "no record"
    )
    ucsc_line = (
        f"phyloP={ucsc.get('phylop')}, phastCons={ucsc.get('phastcons')}"
        if ucsc.get("found") else "no conservation data"
    )
    if not alphamissense.get("applicable"):
        alphamissense_line = "not applicable (non-missense)"
    elif alphamissense.get("found"):
        alphamissense_line = f"score={alphamissense.get('am_pathogenicity')} ({alphamissense.get('am_class')})"
    else:
        alphamissense_line = "missense, but not in the precomputed table"

    return (
        f"Variant: {variant}\n"
        f"ClinVar: {clinvar_line}\n"
        f"gnomAD frequency: {gnomad_line}\n"
        f"Ensembl consequence: {ensembl_line}\n"
        f"Conservation (UCSC): {ucsc_line}\n"
        f"AlphaMissense: {alphamissense_line}"
    )


def _parse_sufficiency(verdict: str) -> bool:
    first_line = verdict.strip().splitlines()[0].strip().lower() if verdict.strip() else ""
    if "insufficient" in first_line:
        return False
    return first_line.startswith("sufficient")


def grade_evidence(state: GraphState) -> dict:
    """Decide whether the gathered evidence is sufficient to classify."""
    summary = _summarize_evidence_for_grading(state["evidence"], state["variant"])
    prompt = f"{summary}\n\nIs this evidence sufficient to reach an ACMG classification?"
    verdict = llm.complete(prompt, system=GRADE_SYSTEM_PROMPT, purpose="grade", max_tokens=64)
    return {"sufficient": _parse_sufficiency(verdict)}


def classify(state: GraphState) -> dict:
    """Combine ACMG criteria into a classification with cited criteria."""
    variant = state["variant"]
    matches = state["evidence"].get("clinvar", {}).get("matches", [])
    chunks = state["criteria"]

    criteria_text = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)

    if matches:
        evidence_lines = [
            f"HGVS: {m['hgvs']}\n"
            f"ClinVar significance: {m['clinical_significance']}\n"
            f"Review status: {m['review_status']}"
            for m in matches
        ]
        evidence_text = "\n\n".join(evidence_lines)
    else:
        evidence_text = f"No ClinVar record found for {variant}."

    prompt = (
        f"Variant: {variant}\n\n"
        f"== ClinVar Evidence ==\n{evidence_text}\n\n"
        f"== Relevant ACMG Criteria ==\n{criteria_text}\n\n"
        f"Classify this variant."
    )
    answer = llm.complete(prompt, system=SYSTEM_PROMPT, purpose="generate")
    return {"classification": answer}


def check_grounded(state: GraphState) -> dict:
    """Verify every cited criterion is supported by the evidence/criteria."""
    raise NotImplementedError("TODO(day-9): llm.complete(purpose='groundedness')")


# --- conditional edges ---

def route_after_grading(state: GraphState) -> str:
    """'classify' if sufficient OR out of budget; else 'gather_evidence' (bounded loop)."""
    if state["sufficient"] or state["rewrites"] >= settings.max_query_rewrites:
        return "classify"
    return "gather_evidence"


def route_after_groundedness(state: GraphState) -> str:
    """END if grounded OR out of retries; else 'classify' (bounded retry)."""
    raise NotImplementedError("TODO(day-9): use settings.max_generation_retries")


def build_graph():
    """Wire the nodes + edges into a compiled StateGraph.

    gather_evidence -> retrieve_criteria -> grade_evidence --(sufficient)--> classify -> END
            ^                                                    |
            +------------------(insufficient, bounded)-----------+

    check_grounded and its conditional edge are still day-9 (classify -> END directly).
    """
    graph = StateGraph(GraphState)
    graph.add_node("gather_evidence", gather_evidence)
    graph.add_node("retrieve_criteria", retrieve_criteria)
    graph.add_node("grade_evidence", grade_evidence)
    graph.add_node("classify", classify)

    graph.set_entry_point("gather_evidence")
    graph.add_edge("gather_evidence", "retrieve_criteria")
    graph.add_edge("retrieve_criteria", "grade_evidence")
    graph.add_conditional_edges(
        "grade_evidence",
        route_after_grading,
        {"gather_evidence": "gather_evidence", "classify": "classify"},
    )
    graph.add_edge("classify", END)

    return graph.compile()


def ask(variant: str) -> dict:
    """Run one variant through the graph; return the final state."""
    initial_state: GraphState = {
        "variant": variant,
        "evidence": {},
        "criteria": [],
        "classification": "",
        "rewrites": 0,
        "gen_retries": 0,
        "grounded": False,
        "sufficient": False,
    }
    return build_graph().invoke(initial_state)
