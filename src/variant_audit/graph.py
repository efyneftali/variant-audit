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
from .mcp_tools.clinvar import get_clinvar_record
from .mcp_tools.ensembl import get_gene_consequence
from .mcp_tools.gnomad import get_allele_frequency
from .retrieval import semantic_search


class GraphState(TypedDict):
    variant: str
    evidence: dict            # accumulated MCP tool results
    criteria: list            # retrieved ACMG criteria chunks
    classification: str       # final call + justification
    rewrites: int             # evidence-gathering loop counter (bounded)
    gen_retries: int          # regeneration counter (bounded)
    grounded: bool


# --- nodes (each takes GraphState, returns a partial state dict) ---

def gather_evidence(state: GraphState) -> dict:
    """Call the MCP tools for this variant and accumulate structured evidence.

    Each tool degrades to a {"found": False} result for an unknown/malformed
    variant rather than raising, so one missing source never blocks the others.
    """
    variant = state["variant"]
    return {
        "evidence": {
            "clinvar": get_clinvar_record(variant),
            "gnomad": get_allele_frequency(variant),
            "ensembl": get_gene_consequence(variant),
        }
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


def grade_evidence(state: GraphState) -> dict:
    """Decide whether the gathered evidence is sufficient to classify."""
    raise NotImplementedError("TODO(day-8): llm.complete(purpose='grade')")


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
    raise NotImplementedError("TODO(day-9): use settings.max_query_rewrites")


def route_after_groundedness(state: GraphState) -> str:
    """END if grounded OR out of retries; else 'classify' (bounded retry)."""
    raise NotImplementedError("TODO(day-9): use settings.max_generation_retries")


def build_graph():
    """Wire the nodes + edges into a compiled StateGraph.

    Linear for now (day-6): gather_evidence -> retrieve_criteria -> classify -> END.
    grade_evidence/check_grounded and their conditional edges land day 8-9.
    """
    graph = StateGraph(GraphState)
    graph.add_node("gather_evidence", gather_evidence)
    graph.add_node("retrieve_criteria", retrieve_criteria)
    graph.add_node("classify", classify)

    graph.set_entry_point("gather_evidence")
    graph.add_edge("gather_evidence", "retrieve_criteria")
    graph.add_edge("retrieve_criteria", "classify")
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
    }
    return build_graph().invoke(initial_state)
