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
    grounded_reason: str      # check_grounded's one-line critique, fed into the retry
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


def _format_clinvar_evidence(evidence: dict, variant: str) -> str:
    matches = evidence.get("clinvar", {}).get("matches", [])
    if not matches:
        return f"No ClinVar record found for {variant}."
    return "\n\n".join(
        f"HGVS: {m['hgvs']}\n"
        f"ClinVar significance: {m['clinical_significance']}\n"
        f"Review status: {m['review_status']}"
        for m in matches
    )


def _format_gnomad_evidence(evidence: dict) -> str:
    gnomad = evidence.get("gnomad", {})
    if not gnomad.get("found"):
        return "Not found in gnomAD (absent from the population database)."
    af = gnomad.get("allele_freq")
    if af is None:
        return "Present in gnomAD; allele frequency unavailable."
    return f"Allele frequency: {af}"


def _format_ensembl_evidence(evidence: dict) -> str:
    ensembl = evidence.get("ensembl", {})
    if not ensembl.get("found"):
        return "No predicted-consequence record."
    line = f"Most severe consequence: {ensembl.get('most_severe_consequence')} (impact={ensembl.get('impact')})"
    gene = ensembl.get("gene_symbol")
    return f"{line}\nGene: {gene}" if gene else line


def _format_ucsc_evidence(evidence: dict) -> str:
    ucsc = evidence.get("ucsc", {})
    if not ucsc.get("found"):
        return "No conservation data."
    return f"phyloP={ucsc.get('phylop')}, phastCons={ucsc.get('phastcons')}"


def _format_alphamissense_evidence(evidence: dict) -> str:
    am = evidence.get("alphamissense", {})
    if not am.get("applicable"):
        return "Not applicable (non-missense variant)."
    if not am.get("found"):
        return "Missense variant, but not present in the precomputed AlphaMissense table."
    return f"Score: {am.get('am_pathogenicity')} (class={am.get('am_class')})"


def _build_evidence_and_criteria_text(state: GraphState) -> tuple[str, str]:
    """Format ALL FIVE evidence sources + retrieved ACMG criteria the same way
    for both classify() and check_grounded() -- the groundedness check has to
    see exactly what classify() saw, or it's grading against the wrong context.

    ClinVar is presented as one submitter's assertion among five sources, not
    the headline: it's a legitimate evidence input (see DATASET.md), but the
    model must weigh it against frequency / consequence / conservation /
    computational signals rather than parrot it. Crucially the other four
    sources are ALWAYS included even when ClinVar is thin or absent -- otherwise
    a variant like the adversarial syn-001 (empty ClinVar, 7.3% gnomAD
    frequency) reaches classify() with a blank evidence block and the model
    hallucinates a call against nothing. The 7.3% frequency is exactly the fact
    that should fire BA1/BS1 and force a benign-or-uncertain call."""
    evidence = state["evidence"]
    variant = state["variant"]
    chunks = state["criteria"]

    criteria_text = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)

    evidence_text = (
        f"-- ClinVar (one submitter assertion, not the final answer) --\n"
        f"{_format_clinvar_evidence(evidence, variant)}\n\n"
        f"-- Population frequency (gnomAD) --\n"
        f"{_format_gnomad_evidence(evidence)}\n\n"
        f"-- Predicted consequence (Ensembl VEP) --\n"
        f"{_format_ensembl_evidence(evidence)}\n\n"
        f"-- Conservation (UCSC) --\n"
        f"{_format_ucsc_evidence(evidence)}\n\n"
        f"-- Computational pathogenicity (AlphaMissense) --\n"
        f"{_format_alphamissense_evidence(evidence)}"
    )

    return evidence_text, criteria_text


def classify(state: GraphState) -> dict:
    """Combine ACMG criteria into a classification with cited criteria.

    Also advances the `gen_retries` counter on a retry entry (i.e. when
    check_grounded has looped back here), mirroring gather_evidence's
    `rewrites` discipline -- `state["classification"]` is only empty on the
    very first pass, since every prior classify() call always sets it.

    On a retry, the prior (rejected) attempt AND check_grounded's critique are
    fed back into the prompt -- otherwise attempt two rebuilds the identical
    prompt from the identical state and regenerates the identical answer, and
    the whole correction loop is a placebo. The critique has to feed forward
    for the loop to actually correct anything.
    """
    variant = state["variant"]
    evidence_text, criteria_text = _build_evidence_and_criteria_text(state)
    is_retry = bool(state.get("classification"))

    prompt = (
        f"Variant: {variant}\n\n"
        f"== Evidence ==\n{evidence_text}\n\n"
        f"== Relevant ACMG Criteria ==\n{criteria_text}\n\n"
    )
    if is_retry:
        reason = state.get("grounded_reason") or "(no reason recorded)"
        prompt += (
            f"== Your previous attempt (rejected as ungrounded) ==\n{state['classification']}\n\n"
            f"== Why it was rejected ==\n{reason}\n\n"
            f"Revise your classification to fix this. Cite only ACMG criteria that appear in "
            f"the criteria above, and assert only facts supported by the evidence above."
        )
    else:
        prompt += "Classify this variant."
    answer = llm.complete(prompt, system=SYSTEM_PROMPT, purpose="generate")
    return {
        "classification": answer,
        "gen_retries": state.get("gen_retries", 0) + 1 if is_retry else state.get("gen_retries", 0),
    }


GROUNDED_SYSTEM_PROMPT = (
    "You are verifying whether a variant classification is grounded in the evidence "
    "and ACMG criteria it was given — you are not classifying the variant yourself. "
    "Check every specific claim in the classification (the tier called, and each ACMG "
    "criterion cited) against the evidence and ACMG criteria text provided. "
    "If the classification cites a criterion that isn't in the provided criteria text, "
    "or asserts a fact the evidence doesn't support, that's ungrounded. "
    "Respond with exactly one word on the first line — 'grounded' or 'ungrounded' — "
    "then a one-sentence reason on the second line."
)


def _parse_grounded(verdict: str) -> bool:
    first_line = verdict.strip().splitlines()[0].strip().lower() if verdict.strip() else ""
    if "ungrounded" in first_line:
        return False
    return first_line.startswith("grounded")


def _grounded_reason(verdict: str) -> str:
    """The one-line justification GROUNDED_SYSTEM_PROMPT asks for on line 2+.
    Falls back to the whole verdict if the model didn't split it onto its own
    line, so the retry always has *something* to react to."""
    lines = [line.strip() for line in verdict.strip().splitlines() if line.strip()]
    return " ".join(lines[1:]) if len(lines) > 1 else verdict.strip()


def check_grounded(state: GraphState) -> dict:
    """Verify every claim in the classification is supported by the evidence/criteria.

    Returns the verdict AND its reason -- the reason is fed forward into the
    regeneration retry (see classify), which is the whole point of running the
    check. Discarding it made the retry loop a no-op.
    """
    evidence_text, criteria_text = _build_evidence_and_criteria_text(state)
    prompt = (
        f"Variant: {state['variant']}\n\n"
        f"== Classification to verify ==\n{state['classification']}\n\n"
        f"== Evidence ==\n{evidence_text}\n\n"
        f"== Relevant ACMG Criteria ==\n{criteria_text}\n\n"
        f"Is every claim in this classification grounded in the evidence and criteria above?"
    )
    verdict = llm.complete(prompt, system=GROUNDED_SYSTEM_PROMPT, purpose="groundedness", max_tokens=64)
    return {"grounded": _parse_grounded(verdict), "grounded_reason": _grounded_reason(verdict)}


# --- conditional edges ---

def route_after_grading(state: GraphState) -> str:
    """'classify' if sufficient OR out of budget; else 'gather_evidence' (bounded loop)."""
    if state["sufficient"] or state["rewrites"] >= settings.max_query_rewrites:
        return "classify"
    return "gather_evidence"

1
def route_after_groundedness(state: GraphState) -> str:
    """END if grounded OR out of retries; else 'classify' (bounded retry)."""
    if state["grounded"] or state["gen_retries"] >= settings.max_generation_retries:
        return END
    return "classify"


def build_graph():
    """Wire the nodes + edges into a compiled StateGraph.

    gather_evidence -> retrieve_criteria -> grade_evidence --(sufficient)--> classify -> check_grounded --(grounded)--> END
            ^                                                    |                                            |
            +------------------(insufficient, bounded)-----------+                                            |
                                                                   ^                                            |
                                                                   +-------------(ungrounded, bounded)----------+
    """
    graph = StateGraph(GraphState)
    graph.add_node("gather_evidence", gather_evidence)
    graph.add_node("retrieve_criteria", retrieve_criteria)
    graph.add_node("grade_evidence", grade_evidence)
    graph.add_node("classify", classify)
    graph.add_node("check_grounded", check_grounded)

    graph.set_entry_point("gather_evidence")
    graph.add_edge("gather_evidence", "retrieve_criteria")
    graph.add_edge("retrieve_criteria", "grade_evidence")
    graph.add_conditional_edges(
        "grade_evidence",
        route_after_grading,
        {"gather_evidence": "gather_evidence", "classify": "classify"},
    )
    graph.add_edge("classify", "check_grounded")
    graph.add_conditional_edges(
        "check_grounded",
        route_after_groundedness,
        {"classify": "classify", END: END},
    )

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
        "grounded_reason": "",
        "sufficient": False,
    }
    return build_graph().invoke(initial_state)
