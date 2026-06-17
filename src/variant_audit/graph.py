"""The agent — a corrective-classification LangGraph state machine.

Flow (bounded loops guarantee termination — the senior design point):

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

TODO(day-6): port a linear version into a StateGraph.
TODO(day-8..9): add the grading node + conditional edges + bounded loops.
TODO(day-10): the classify node combines ACMG criteria.
"""

from typing import TypedDict

from .config import settings


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
    """Call the MCP tools for this variant and accumulate structured evidence."""
    raise NotImplementedError("TODO(day-8): call mcp_tools and collect results")


def retrieve_criteria(state: GraphState) -> dict:
    """RAG: retrieve the ACMG criteria relevant to this variant/evidence."""
    raise NotImplementedError("TODO(day-6): semantic_search for applicable criteria")


def grade_evidence(state: GraphState) -> dict:
    """Decide whether the gathered evidence is sufficient to classify."""
    raise NotImplementedError("TODO(day-8): llm.complete(purpose='grade')")


def classify(state: GraphState) -> dict:
    """Combine ACMG criteria into a classification with cited criteria."""
    raise NotImplementedError("TODO(day-10): llm.complete(purpose='generate')")


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
    """Wire the nodes + edges into a compiled StateGraph."""
    raise NotImplementedError("TODO(day-6..9): construct, set entry point, add edges, compile")


def ask(variant: str) -> dict:
    """Run one variant through the graph; return the final state."""
    raise NotImplementedError("TODO(day-6): invoke build_graph() with an initial GraphState")
