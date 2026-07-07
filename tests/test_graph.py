"""Tests for graph.py.

Unit tests mock the three external calls (get_clinvar_record, semantic_search,
llm.complete) at the module level, same convention as test_classify.py:
- per-node tests exercise gather_evidence/retrieve_criteria/classify in isolation
- a wiring test drives the compiled graph end to end with the fake stack
- a parity test checks graph.ask() and classify.classify_variant() agree, since
  the graph is a straight port of that function's logic

The integration test calls the real stack (live ClinVar + Qdrant + Ollama).
"""

from types import SimpleNamespace

import pytest

from src.variant_audit import classify, graph
from src.variant_audit.config import settings

GNOMAD_RESULT = {
    "variant": "rs28897696",
    "found": True,
    "allele_freq": 5.2569325798396634e-05,
    "genome": {"af": 5.2569325798396634e-05, "ac": 8, "an": 152180},
    "exome": {"af": 6.909611337782849e-05, "ac": 101, "an": 1461732},
}

ENSEMBL_RESULT = {
    "variant": "rs28897696",
    "found": True,
    "most_severe_consequence": "frameshift_variant",
    "impact": "HIGH",
    "gene_symbol": "BRCA1",
    "chrom": "17",
    "start": 43057066,
    "end": 43057065,
    "allele_string": "-/G",
}

UCSC_RESULT = {
    "variant": "rs28897696",
    "found": True,
    "phylop": 1.46348,
    "phastcons": 1,
    "chrom": "17",
    "start": 43057066,
    "end": 43057065,
}

ALPHAMISSENSE_RESULT = {
    "variant": "rs28897696",
    "found": True,
    "applicable": True,
    "am_pathogenicity": 0.6381,
    "am_class": "likely_pathogenic",
    "transcript_id": "ENST00000357654.8",
    "protein_variant": "A1708V",
}


@pytest.fixture
def fake_stack(monkeypatch):
    clinvar_result = {
        "variant": "rs28897696",
        "found": True,
        "matches": [
            {"hgvs": "BRCA1:c.123A>G", "clinical_significance": "Pathogenic", "review_status": "reviewed by expert panel"},
            {"hgvs": "BRCA1:c.123A>T", "clinical_significance": "Benign", "review_status": "criteria provided"},
        ],
    }
    monkeypatch.setattr(graph, "get_clinvar_record", lambda v: clinvar_result)
    monkeypatch.setattr(graph, "get_allele_frequency", lambda v: GNOMAD_RESULT)
    monkeypatch.setattr(graph, "get_gene_consequence", lambda v: ENSEMBL_RESULT)
    monkeypatch.setattr(graph, "get_genomic_context", lambda v, consequence=None: UCSC_RESULT)
    monkeypatch.setattr(graph, "get_alphamissense_score", lambda v, consequence=None: ALPHAMISSENSE_RESULT)

    chunks = [
        SimpleNamespace(text="PVS1 applies to null variants.", source="acmg_criteria.md", score=0.9),
        SimpleNamespace(text="BA1 applies when frequency > 5%.", source="acmg_criteria.md", score=0.6),
    ]
    monkeypatch.setattr(graph, "semantic_search", lambda q: chunks)

    llm_calls = {}

    def fake_complete(prompt, system="", *, purpose="generate", max_tokens=1024):
        llm_calls["prompt"] = prompt
        llm_calls["system"] = system
        llm_calls["purpose"] = purpose
        if purpose == "grade":
            # happy path: evidence is graded sufficient immediately, one pass, no retry loop
            return "sufficient\nEnough evidence to classify."
        return "Classification: Uncertain Significance\nCriteria used: none identified"

    monkeypatch.setattr(graph.llm, "complete", fake_complete)
    return clinvar_result, chunks, llm_calls


class TestGatherEvidence:
    def test_accumulates_all_five_sources_under_their_keys(self, fake_stack):
        clinvar_result, _, _ = fake_stack
        result = graph.gather_evidence({"variant": "rs28897696"})
        assert result == {
            "evidence": {
                "clinvar": clinvar_result,
                "gnomad": GNOMAD_RESULT,
                "ensembl": ENSEMBL_RESULT,
                "ucsc": UCSC_RESULT,
                "alphamissense": ALPHAMISSENSE_RESULT,
            },
            "rewrites": 0,
        }

    def test_passes_variant_through_to_clinvar_lookup(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(graph, "get_clinvar_record", lambda v: captured.update({"variant": v}) or {"found": False, "matches": []})
        monkeypatch.setattr(graph, "get_allele_frequency", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_gene_consequence", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_genomic_context", lambda v, consequence=None: {"found": False})
        monkeypatch.setattr(graph, "get_alphamissense_score", lambda v, consequence=None: {"found": False, "applicable": False})
        graph.gather_evidence({"variant": "rs999"})
        assert captured["variant"] == "rs999"

    def test_passes_variant_through_to_gnomad_lookup(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(graph, "get_clinvar_record", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_allele_frequency", lambda v: captured.update({"variant": v}) or {"found": False})
        monkeypatch.setattr(graph, "get_gene_consequence", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_genomic_context", lambda v, consequence=None: {"found": False})
        monkeypatch.setattr(graph, "get_alphamissense_score", lambda v, consequence=None: {"found": False, "applicable": False})
        graph.gather_evidence({"variant": "rs999"})
        assert captured["variant"] == "rs999"

    def test_passes_variant_through_to_ensembl_lookup(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(graph, "get_clinvar_record", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_allele_frequency", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_gene_consequence", lambda v: captured.update({"variant": v}) or {"found": False})
        monkeypatch.setattr(graph, "get_genomic_context", lambda v, consequence=None: {"found": False})
        monkeypatch.setattr(graph, "get_alphamissense_score", lambda v, consequence=None: {"found": False, "applicable": False})
        graph.gather_evidence({"variant": "rs999"})
        assert captured["variant"] == "rs999"

    def test_reuses_ensembl_result_for_ucsc_instead_of_refetching(self, monkeypatch):
        # gather_evidence must not trigger a second, slow VEP round-trip for UCSC
        ensembl_result = {"found": True, "chrom": "17", "start": 1, "end": 1}
        captured = {}
        monkeypatch.setattr(graph, "get_clinvar_record", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_allele_frequency", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_gene_consequence", lambda v: ensembl_result)
        monkeypatch.setattr(
            graph,
            "get_genomic_context",
            lambda v, consequence=None: captured.update({"consequence": consequence}) or {"found": False},
        )
        monkeypatch.setattr(graph, "get_alphamissense_score", lambda v, consequence=None: {"found": False, "applicable": False})

        graph.gather_evidence({"variant": "rs999"})

        assert captured["consequence"] is ensembl_result

    def test_reuses_ensembl_result_for_alphamissense_instead_of_refetching(self, monkeypatch):
        # same efficiency win as UCSC -- avoid a second, slow VEP round-trip
        ensembl_result = {"found": True, "most_severe_consequence": "missense_variant", "chrom": "17", "start": 1, "ref": "G", "alt": "A"}
        captured = {}
        monkeypatch.setattr(graph, "get_clinvar_record", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_allele_frequency", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_gene_consequence", lambda v: ensembl_result)
        monkeypatch.setattr(graph, "get_genomic_context", lambda v, consequence=None: {"found": False})
        monkeypatch.setattr(
            graph,
            "get_alphamissense_score",
            lambda v, consequence=None: captured.update({"consequence": consequence}) or {"found": False, "applicable": False},
        )

        graph.gather_evidence({"variant": "rs999"})

        assert captured["consequence"] is ensembl_result

    def test_degrades_gracefully_when_a_source_has_no_record(self, monkeypatch):
        monkeypatch.setattr(graph, "get_clinvar_record", lambda v: {"variant": v, "found": False})
        monkeypatch.setattr(graph, "get_allele_frequency", lambda v: {"variant": v, "found": False})
        monkeypatch.setattr(graph, "get_gene_consequence", lambda v: {"variant": v, "found": False})
        monkeypatch.setattr(graph, "get_genomic_context", lambda v, consequence=None: {"variant": v, "found": False})
        monkeypatch.setattr(graph, "get_alphamissense_score", lambda v, consequence=None: {"variant": v, "found": False, "applicable": False})

        result = graph.gather_evidence({"variant": "rs00000000000"})

        assert result["evidence"]["clinvar"]["found"] is False
        assert result["evidence"]["gnomad"]["found"] is False
        assert result["evidence"]["ensembl"]["found"] is False
        assert result["evidence"]["ucsc"]["found"] is False
        assert result["evidence"]["alphamissense"]["found"] is False


class TestRetrieveCriteria:
    def test_search_query_is_built_from_clinvar_not_raw_rsid(self, monkeypatch):
        state = {
            "variant": "rs123",
            "evidence": {"clinvar": {"found": True, "matches": [
                {"hgvs": "GENE:c.1A>G", "clinical_significance": "Pathogenic", "review_status": "x"},
            ]}},
        }
        captured = {}
        monkeypatch.setattr(graph, "semantic_search", lambda q: captured.update({"query": q}) or [])

        graph.retrieve_criteria(state)

        assert "rs123" not in captured["query"]
        assert "GENE:c.1A>G" in captured["query"] or "Pathogenic" in captured["query"]

    def test_not_found_falls_back_to_variant_string_for_search(self, monkeypatch):
        state = {"variant": "rs999", "evidence": {"clinvar": {"found": False, "matches": []}}}
        captured = {}
        monkeypatch.setattr(graph, "semantic_search", lambda q: captured.update({"query": q}) or [])

        graph.retrieve_criteria(state)

        assert captured["query"] == "rs999"

    def test_returns_criteria_key_with_search_results(self, fake_stack):
        _, chunks, _ = fake_stack
        state = {"variant": "rs28897696", "evidence": {"clinvar": {"found": True, "matches": []}}}
        result = graph.retrieve_criteria(state)
        assert result == {"criteria": chunks}


class TestClassify:
    def test_returns_classification_key_with_llm_output(self, fake_stack):
        _, chunks, _ = fake_stack
        state = {"variant": "rs28897696", "evidence": {"clinvar": {"found": True, "matches": []}}, "criteria": chunks}
        result = graph.classify(state)
        assert result == {"classification": "Classification: Uncertain Significance\nCriteria used: none identified"}

    def test_prompt_includes_clinvar_hgvs_and_significance(self, fake_stack):
        clinvar_result, chunks, llm_calls = fake_stack
        state = {"variant": "rs28897696", "evidence": {"clinvar": clinvar_result}, "criteria": chunks}
        graph.classify(state)
        assert "BRCA1:c.123A>G" in llm_calls["prompt"]
        assert "Pathogenic" in llm_calls["prompt"]

    def test_prompt_includes_retrieved_criteria_text(self, fake_stack):
        clinvar_result, chunks, llm_calls = fake_stack
        state = {"variant": "rs28897696", "evidence": {"clinvar": clinvar_result}, "criteria": chunks}
        graph.classify(state)
        assert "PVS1 applies to null variants." in llm_calls["prompt"]

    def test_not_found_uses_no_record_message(self, fake_stack):
        _, chunks, llm_calls = fake_stack
        state = {"variant": "rs999", "evidence": {"clinvar": {"found": False, "matches": []}}, "criteria": chunks}
        graph.classify(state)
        assert "No ClinVar record found for rs999" in llm_calls["prompt"]

    def test_uses_generate_purpose(self, fake_stack):
        _, chunks, llm_calls = fake_stack
        state = {"variant": "rs28897696", "evidence": {"clinvar": {"found": True, "matches": []}}, "criteria": chunks}
        graph.classify(state)
        assert llm_calls["purpose"] == "generate"

    def test_system_prompt_enforces_five_tier_vocabulary(self, fake_stack):
        _, chunks, llm_calls = fake_stack
        state = {"variant": "rs28897696", "evidence": {"clinvar": {"found": True, "matches": []}}, "criteria": chunks}
        graph.classify(state)
        system = llm_calls["system"]
        assert "Pathogenic" in system
        assert "Uncertain Significance" in system
        assert "Benign" in system


class TestGradeEvidence:
    def _grade_state(self, evidence):
        return {"variant": "rs28897696", "evidence": evidence}

    def test_sufficient_verdict_returns_true(self, monkeypatch):
        monkeypatch.setattr(graph.llm, "complete", lambda *a, **kw: "sufficient\nClinVar plus consequence data is enough.")
        result = graph.grade_evidence(self._grade_state({"clinvar": {"found": True, "matches": []}}))
        assert result == {"sufficient": True}

    def test_insufficient_verdict_returns_false(self, monkeypatch):
        monkeypatch.setattr(graph.llm, "complete", lambda *a, **kw: "insufficient\nNothing was found anywhere.")
        result = graph.grade_evidence(self._grade_state({}))
        assert result == {"sufficient": False}

    def test_verdict_parsing_is_case_insensitive_and_tolerates_trailing_text(self, monkeypatch):
        monkeypatch.setattr(graph.llm, "complete", lambda *a, **kw: "SUFFICIENT - yes, clearly enough evidence.")
        result = graph.grade_evidence(self._grade_state({}))
        assert result == {"sufficient": True}

    def test_ambiguous_verdict_defaults_to_insufficient(self, monkeypatch):
        # conservative default: if we can't tell, don't let the graph proceed as if it were sufficient
        monkeypatch.setattr(graph.llm, "complete", lambda *a, **kw: "unclear, hard to say")
        result = graph.grade_evidence(self._grade_state({}))
        assert result == {"sufficient": False}

    def test_uses_grade_purpose_and_small_max_tokens(self, monkeypatch):
        calls = {}

        def fake_complete(prompt, system="", *, purpose="generate", max_tokens=1024):
            calls["purpose"] = purpose
            calls["max_tokens"] = max_tokens
            return "sufficient\nfine"

        monkeypatch.setattr(graph.llm, "complete", fake_complete)
        graph.grade_evidence(self._grade_state({}))

        assert calls["purpose"] == "grade"
        assert calls["max_tokens"] <= 128

    def test_prompt_summarizes_all_five_sources_when_present(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(graph.llm, "complete", lambda prompt, **kw: calls.update({"prompt": prompt}) or "sufficient\nfine")

        evidence = {
            "clinvar": {"found": True, "matches": [{"hgvs": "x", "clinical_significance": "Pathogenic", "review_status": "y"}]},
            "gnomad": GNOMAD_RESULT,
            "ensembl": ENSEMBL_RESULT,
            "ucsc": UCSC_RESULT,
            "alphamissense": ALPHAMISSENSE_RESULT,
        }
        graph.grade_evidence(self._grade_state(evidence))
        prompt = calls["prompt"]

        assert "1 match(es)" in prompt
        assert "5.2569325798396634e-05" in prompt
        assert "frameshift_variant" in prompt
        assert "1.46348" in prompt
        assert "0.6381" in prompt and "likely_pathogenic" in prompt

    def test_prompt_degrades_gracefully_with_no_sources_present(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(graph.llm, "complete", lambda prompt, **kw: calls.update({"prompt": prompt}) or "insufficient\nnothing found")

        graph.grade_evidence(self._grade_state({}))
        prompt = calls["prompt"]

        assert "no record" in prompt
        assert "no conservation data" in prompt
        assert "not applicable" in prompt


class TestRouteAfterGrading:
    """Pure routing logic -- given a state, which node comes next."""

    def test_sufficient_goes_to_classify_regardless_of_budget(self):
        assert graph.route_after_grading({"sufficient": True, "rewrites": 0}) == "classify"
        assert graph.route_after_grading({"sufficient": True, "rewrites": 99}) == "classify"

    def test_insufficient_and_under_budget_loops_back_to_gather_evidence(self):
        # default settings.max_query_rewrites is 1, so rewrites=0 is still under budget
        assert graph.route_after_grading({"sufficient": False, "rewrites": 0}) == "gather_evidence"

    def test_insufficient_and_at_budget_gives_up_and_goes_to_classify(self):
        assert graph.route_after_grading({"sufficient": False, "rewrites": 1}) == "classify"

    def test_insufficient_and_over_budget_gives_up_and_goes_to_classify(self):
        assert graph.route_after_grading({"sufficient": False, "rewrites": 5}) == "classify"

    def test_respects_a_wider_or_narrower_configured_budget(self, monkeypatch):
        monkeypatch.setattr(graph, "settings", SimpleNamespace(max_query_rewrites=0))
        assert graph.route_after_grading({"sufficient": False, "rewrites": 0}) == "classify"

        monkeypatch.setattr(graph, "settings", SimpleNamespace(max_query_rewrites=3))
        assert graph.route_after_grading({"sufficient": False, "rewrites": 2}) == "gather_evidence"
        assert graph.route_after_grading({"sufficient": False, "rewrites": 3}) == "classify"


class TestGatherEvidenceRewritesCounter:
    def test_first_pass_leaves_rewrites_unchanged(self, monkeypatch):
        monkeypatch.setattr(graph, "get_clinvar_record", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_allele_frequency", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_gene_consequence", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_genomic_context", lambda v, consequence=None: {"found": False})
        monkeypatch.setattr(graph, "get_alphamissense_score", lambda v, consequence=None: {"found": False, "applicable": False})

        result = graph.gather_evidence({"variant": "rs999", "evidence": {}, "rewrites": 0})

        assert result["rewrites"] == 0

    def test_retry_pass_increments_rewrites(self, monkeypatch):
        monkeypatch.setattr(graph, "get_clinvar_record", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_allele_frequency", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_gene_consequence", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_genomic_context", lambda v, consequence=None: {"found": False})
        monkeypatch.setattr(graph, "get_alphamissense_score", lambda v, consequence=None: {"found": False, "applicable": False})

        # non-empty "evidence" is the signal that this is a retry, not the first pass
        prior_evidence = {"clinvar": {"found": False}, "gnomad": {"found": False}, "ensembl": {"found": False}, "ucsc": {"found": False}, "alphamissense": {"found": False, "applicable": False}}
        result = graph.gather_evidence({"variant": "rs999", "evidence": prior_evidence, "rewrites": 0})

        assert result["rewrites"] == 1


class TestBoundedRetryLoop:
    """End-to-end: grade_evidence -> route_after_grading -> gather_evidence, through the compiled graph."""

    def _stack(self, monkeypatch, grade_responses):
        """Mock every tool + a sequence of grade-purpose verdicts (repeats the last one)."""
        call_count = {"gather": 0}

        def fake_clinvar(v):
            call_count["gather"] += 1
            return {"found": False}

        monkeypatch.setattr(graph, "get_clinvar_record", fake_clinvar)
        monkeypatch.setattr(graph, "get_allele_frequency", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_gene_consequence", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_genomic_context", lambda v, consequence=None: {"found": False})
        monkeypatch.setattr(graph, "get_alphamissense_score", lambda v, consequence=None: {"found": False, "applicable": False})
        monkeypatch.setattr(graph, "semantic_search", lambda q: [])

        grade_calls = {"n": 0}

        def fake_complete(prompt, system="", *, purpose="generate", max_tokens=1024):
            if purpose == "grade":
                idx = min(grade_calls["n"], len(grade_responses) - 1)
                grade_calls["n"] += 1
                return grade_responses[idx]
            return "Classification: Uncertain Significance\nCriteria used: none identified"

        monkeypatch.setattr(graph.llm, "complete", fake_complete)
        return call_count

    def test_insufficient_then_sufficient_retries_exactly_once(self, monkeypatch):
        call_count = self._stack(monkeypatch, ["insufficient\nnot enough yet", "sufficient\nnow it's enough"])

        result = graph.ask("rs999")

        assert call_count["gather"] == 2  # original pass + exactly one retry
        assert result["rewrites"] == 1
        assert result["sufficient"] is True
        assert result["classification"] == "Classification: Uncertain Significance\nCriteria used: none identified"

    def test_persistently_insufficient_still_classifies_once_budget_is_exhausted(self, monkeypatch):
        # settings.max_query_rewrites defaults to 1: allowed exactly 1 retry, then must proceed anyway
        call_count = self._stack(monkeypatch, ["insufficient\nstill not enough"])

        result = graph.ask("rs999")

        assert call_count["gather"] == 2  # original pass + the one allowed retry, then gives up
        assert result["rewrites"] == 1
        assert result["sufficient"] is False
        # classify() runs regardless -- its own prompt already handles thin evidence as VUS
        assert result["classification"] == "Classification: Uncertain Significance\nCriteria used: none identified"

    def test_immediately_sufficient_never_retries(self, monkeypatch):
        call_count = self._stack(monkeypatch, ["sufficient\ngood to go"])

        result = graph.ask("rs999")

        assert call_count["gather"] == 1
        assert result["rewrites"] == 0
        assert result["sufficient"] is True


class TestBuildGraphAndAsk:
    def test_build_graph_compiles(self):
        compiled = graph.build_graph()
        assert compiled is not None

    def test_ask_runs_linear_flow_and_populates_final_state(self, fake_stack):
        clinvar_result, chunks, llm_calls = fake_stack
        result = graph.ask("rs28897696")

        assert result["variant"] == "rs28897696"
        assert result["evidence"] == {
            "clinvar": clinvar_result,
            "gnomad": GNOMAD_RESULT,
            "ensembl": ENSEMBL_RESULT,
            "ucsc": UCSC_RESULT,
            "alphamissense": ALPHAMISSENSE_RESULT,
        }
        assert result["criteria"] == chunks
        assert result["classification"] == "Classification: Uncertain Significance\nCriteria used: none identified"

    def test_ask_feeds_gather_evidence_output_into_retrieve_criteria(self, fake_stack):
        # if the query weren't built from evidence.clinvar, this hgvs wouldn't reach semantic_search's input
        clinvar_result, _, llm_calls = fake_stack
        graph.ask("rs28897696")
        assert "BRCA1:c.123A>G" in llm_calls["prompt"]


class TestParityWithClassifyVariant:
    """graph.ask() is a straight port of classify.classify_variant() — they must agree."""

    def test_same_classification_given_same_fake_stack(self, monkeypatch):
        clinvar_result = {
            "found": True,
            "matches": [
                {"hgvs": "BRCA1:c.5266dup", "clinical_significance": "Pathogenic", "review_status": "reviewed by expert panel"},
            ],
        }
        chunks = [SimpleNamespace(text="PVS1 applies to null variants.", source="acmg_criteria.md", score=0.9)]

        def fake_complete(*a, purpose="generate", **kw):
            if purpose == "grade":
                return "sufficient\nEnough evidence to classify."
            return "Classification: Pathogenic\nCriteria used: PVS1"

        for module in (classify, graph):
            monkeypatch.setattr(module, "get_clinvar_record", lambda v: clinvar_result)
            monkeypatch.setattr(module, "semantic_search", lambda q: chunks)
            monkeypatch.setattr(module.llm, "complete", fake_complete)

        # graph.gather_evidence also queries gnomAD/Ensembl/UCSC/AlphaMissense; classify_variant doesn't.
        monkeypatch.setattr(graph, "get_allele_frequency", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_gene_consequence", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_genomic_context", lambda v, consequence=None: {"found": False})
        monkeypatch.setattr(graph, "get_alphamissense_score", lambda v, consequence=None: {"found": False, "applicable": False})

        flat_result = classify.classify_variant("rs80357906")
        graph_result = graph.ask("rs80357906")

        assert graph_result["classification"] == flat_result["classification"]


@pytest.mark.integration
class TestAskIntegration:
    def test_real_variant_returns_valid_classification(self):
        result = graph.ask("rs28897696")

        assert result["variant"] == "rs28897696"
        assert result["evidence"]["clinvar"]["found"] is True
        assert len(result["criteria"]) > 0
        assert any(
            tier in result["classification"]
            for tier in ["Pathogenic", "Likely Pathogenic", "Uncertain Significance", "Likely Benign", "Benign"]
        )


@pytest.mark.integration
class TestGradeEvidenceIntegration:
    def test_real_well_populated_evidence_graded_sufficient(self):
        gathered = graph.gather_evidence({"variant": "rs28897696"})
        result = graph.grade_evidence({"variant": "rs28897696", "evidence": gathered["evidence"]})

        assert isinstance(result["sufficient"], bool)
        assert result["sufficient"] is True

    def test_real_empty_evidence_graded_insufficient(self):
        result = graph.grade_evidence({"variant": "rsFAKE", "evidence": {}})

        assert isinstance(result["sufficient"], bool)
        assert result["sufficient"] is False


@pytest.mark.integration
class TestBoundedRetryLoopIntegration:
    """The win, automated: a well-covered variant goes straight through; a
    thin-evidence variant loops back at least once, and the graph still
    terminates with a classification either way."""

    def _run_with_gather_spy(self, variant):
        call_count = {"n": 0}
        original_gather = graph.gather_evidence

        def spy(state):
            call_count["n"] += 1
            return original_gather(state)

        graph.gather_evidence = spy
        try:
            result = graph.ask(variant)
        finally:
            graph.gather_evidence = original_gather
        return result, call_count["n"]

    def test_well_covered_variant_goes_straight_through(self):
        result, gather_calls = self._run_with_gather_spy("rs28897696")

        assert gather_calls == 1
        assert result["rewrites"] == 0
        assert result["sufficient"] is True
        assert result["classification"]

    def test_thin_evidence_variant_loops_back_then_still_terminates(self):
        # a syntactically valid but nonexistent rsID: every tool comes back
        # found=False, so grading should say insufficient and trigger a retry
        result, gather_calls = self._run_with_gather_spy("rs00000000001")

        assert gather_calls >= 2  # looped back at least once
        assert gather_calls <= 1 + settings.max_query_rewrites  # never more than the budget allows
        assert result["rewrites"] == settings.max_query_rewrites  # budget fully spent
        assert result["classification"]  # still produced a call -- the graph terminated, didn't hang
