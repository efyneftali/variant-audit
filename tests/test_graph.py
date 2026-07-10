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
        if purpose == "groundedness":
            # happy path: classification passes groundedness immediately, no regeneration loop
            return "grounded\nEvery claim is supported by the evidence."
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
        assert result == {
            "classification": "Classification: Uncertain Significance\nCriteria used: none identified",
            "gen_retries": 0,
        }

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

    def test_first_pass_prompt_has_no_critique_section(self, fake_stack):
        _, chunks, llm_calls = fake_stack
        state = {"variant": "rs28897696", "evidence": {"clinvar": {"found": True, "matches": []}}, "criteria": chunks}
        graph.classify(state)
        assert "rejected" not in llm_calls["prompt"].lower()
        assert llm_calls["prompt"].rstrip().endswith("Classify this variant.")

    def test_retry_prompt_feeds_forward_prior_attempt_and_critique(self, fake_stack):
        # bug-two regression guard: attempt two must differ from attempt one, or
        # the correction loop can't correct. Both the rejected draft and the
        # groundedness critique have to reach the retry prompt.
        _, chunks, llm_calls = fake_stack
        state = {
            "variant": "rs28897696",
            "evidence": {"clinvar": {"found": True, "matches": []}},
            "criteria": chunks,
            "classification": "Classification: Pathogenic\nCriteria used: PS3",
            "grounded_reason": "PS3 was cited but no functional-assay evidence was provided.",
            "gen_retries": 0,
        }
        graph.classify(state)
        prompt = llm_calls["prompt"]
        assert "Classification: Pathogenic\nCriteria used: PS3" in prompt  # the rejected draft
        assert "PS3 was cited but no functional-assay evidence was provided." in prompt  # the critique


class TestCheckGrounded:
    def test_grounded_verdict_returns_true(self, monkeypatch):
        monkeypatch.setattr(graph.llm, "complete", lambda *a, **kw: "grounded\nEvery claim matches the evidence.")
        state = {"variant": "rs28897696", "evidence": {"clinvar": {"matches": []}}, "criteria": [], "classification": "Classification: Benign\nCriteria used: none identified"}
        result = graph.check_grounded(state)
        assert result["grounded"] is True
        assert result["grounded_reason"] == "Every claim matches the evidence."

    def test_ungrounded_verdict_returns_false(self, monkeypatch):
        # PM1 IS in the retrieved criteria, so the deterministic pre-check passes
        # and the LLM judge runs -- here it flags an unsupported fact (the kind of
        # subtle failure only the model can catch).
        monkeypatch.setattr(graph.llm, "complete", lambda *a, **kw: "ungrounded\nNo functional assay supports the pathogenic call.")
        pm1_chunk = SimpleNamespace(text="PM1: located in a mutational hotspot.", source="acmg_criteria.md", score=0.9)
        state = {"variant": "rs28897696", "evidence": {"clinvar": {"matches": []}}, "criteria": [pm1_chunk], "classification": "Classification: Pathogenic\nCriteria used: PM1"}
        result = graph.check_grounded(state)
        assert result["grounded"] is False
        # the critique is captured, not discarded -- classify() feeds it into the retry
        assert result["grounded_reason"] == "No functional assay supports the pathogenic call."

    def test_verdict_parsing_is_case_insensitive_and_tolerates_trailing_text(self, monkeypatch):
        monkeypatch.setattr(graph.llm, "complete", lambda *a, **kw: "GROUNDED - yes, fully supported.")
        state = {"variant": "rs28897696", "evidence": {}, "criteria": [], "classification": "x"}
        result = graph.check_grounded(state)
        assert result["grounded"] is True

    def test_ambiguous_verdict_defaults_to_ungrounded(self, monkeypatch):
        # conservative default: if we can't tell, don't let the graph treat it as grounded
        monkeypatch.setattr(graph.llm, "complete", lambda *a, **kw: "unclear, hard to say")
        state = {"variant": "rs28897696", "evidence": {}, "criteria": [], "classification": "x"}
        result = graph.check_grounded(state)
        assert result["grounded"] is False
        # single-line verdict: fall back to the whole thing so the retry still has something
        assert result["grounded_reason"] == "unclear, hard to say"

    def test_uses_groundedness_purpose_and_small_max_tokens(self, monkeypatch):
        calls = {}

        def fake_complete(prompt, system="", *, purpose="generate", max_tokens=1024):
            calls["purpose"] = purpose
            calls["max_tokens"] = max_tokens
            return "grounded\nfine"

        monkeypatch.setattr(graph.llm, "complete", fake_complete)
        state = {"variant": "rs28897696", "evidence": {}, "criteria": [], "classification": "x"}
        graph.check_grounded(state)

        assert calls["purpose"] == "groundedness"
        assert calls["max_tokens"] <= 128

    def test_prompt_includes_the_classification_being_verified(self, fake_stack):
        clinvar_result, chunks, llm_calls = fake_stack
        state = {
            "variant": "rs28897696",
            "evidence": {"clinvar": clinvar_result},
            "criteria": chunks,
            "classification": "Classification: Pathogenic\nCriteria used: PVS1",
        }
        graph.check_grounded(state)
        assert "Classification: Pathogenic\nCriteria used: PVS1" in llm_calls["prompt"]

    def test_prompt_includes_same_evidence_and_criteria_as_classify(self, fake_stack):
        clinvar_result, chunks, llm_calls = fake_stack
        state = {
            "variant": "rs28897696",
            "evidence": {"clinvar": clinvar_result},
            "criteria": chunks,
            "classification": "Classification: Pathogenic\nCriteria used: PVS1",
        }
        graph.check_grounded(state)
        assert "BRCA1:c.123A>G" in llm_calls["prompt"]
        assert "PVS1 applies to null variants." in llm_calls["prompt"]


class TestAcmgCodeExtraction:
    def test_extracts_all_code_families(self):
        text = "Relied on PVS1, PS3, PM2, PP3, BA1, BS1, BP4."
        assert graph._acmg_codes(text) == {"PVS1", "PS3", "PM2", "PP3", "BA1", "BS1", "BP4"}

    def test_is_case_insensitive_and_upper_cases(self):
        assert graph._acmg_codes("pvs1 and pm2") == {"PVS1", "PM2"}

    def test_strips_strength_modifier_to_base_code(self):
        # "PM2_Supporting" is the same criterion as "PM2" for grounding purposes
        assert graph._acmg_codes("Criteria used: PM2_Supporting, PS3_Moderate") == {"PM2", "PS3"}

    def test_ignores_non_acmg_tokens(self):
        assert graph._acmg_codes("BRCA1 c.123A>G rs28897696 Pathogenic") == set()


class TestPrecheckGrounded:
    """The deterministic gate that runs before the LLM judge (VA-35)."""

    def test_cited_code_absent_from_criteria_is_ungrounded_without_llm(self):
        result = graph._precheck_grounded(
            "Classification: Pathogenic\nCriteria used: PM1",
            criteria_text="[acmg] PVS1 applies to null variants.",
        )
        assert result is not None
        assert result["grounded"] is False
        assert "PM1" in result["grounded_reason"]

    def test_all_cited_codes_present_defers_to_llm(self):
        # every cited code is in the criteria -> pre-check can't rule; the LLM
        # still has to check the tier and the asserted facts
        result = graph._precheck_grounded(
            "Classification: Benign\nCriteria used: BA1",
            criteria_text="[acmg] BA1 applies when frequency > 5%.",
        )
        assert result is None

    def test_no_codes_cited_defers_to_llm(self):
        result = graph._precheck_grounded(
            "Classification: Uncertain Significance\nCriteria used: none identified",
            criteria_text="[acmg] PVS1 applies to null variants.",
        )
        assert result is None

    def test_reports_every_missing_code(self):
        result = graph._precheck_grounded(
            "Criteria used: PM1, PS4",
            criteria_text="[acmg] PVS1 applies to null variants.",
        )
        assert "PM1" in result["grounded_reason"]
        assert "PS4" in result["grounded_reason"]

    def test_check_grounded_short_circuits_and_never_calls_llm(self, monkeypatch):
        called = {"llm": False}

        def boom(*a, **kw):
            called["llm"] = True
            raise AssertionError("LLM judge must not run when the pre-check settles it")

        monkeypatch.setattr(graph.llm, "complete", boom)
        state = {
            "variant": "rs28897696",
            "evidence": {"clinvar": {"matches": []}},
            "criteria": [SimpleNamespace(text="BA1 applies when frequency > 5%.", source="acmg_criteria.md", score=0.9)],
            "classification": "Classification: Pathogenic\nCriteria used: PM6",  # not in criteria
        }
        result = graph.check_grounded(state)
        assert called["llm"] is False
        assert result["grounded"] is False
        assert "PM6" in result["grounded_reason"]


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


class TestRouteAfterGroundedness:
    """Pure routing logic -- given a state, which node comes next."""

    def test_grounded_goes_to_end_regardless_of_budget(self):
        assert graph.route_after_groundedness({"grounded": True, "gen_retries": 0}) == graph.END
        assert graph.route_after_groundedness({"grounded": True, "gen_retries": 99}) == graph.END

    def test_ungrounded_and_under_budget_loops_back_to_classify(self):
        # default settings.max_generation_retries is 1, so gen_retries=0 is still under budget
        assert graph.route_after_groundedness({"grounded": False, "gen_retries": 0}) == "classify"

    def test_ungrounded_and_at_budget_gives_up_and_ends(self):
        assert graph.route_after_groundedness({"grounded": False, "gen_retries": 1}) == graph.END

    def test_ungrounded_and_over_budget_gives_up_and_ends(self):
        assert graph.route_after_groundedness({"grounded": False, "gen_retries": 5}) == graph.END

    def test_respects_a_wider_or_narrower_configured_budget(self, monkeypatch):
        monkeypatch.setattr(graph, "settings", SimpleNamespace(max_generation_retries=0))
        assert graph.route_after_groundedness({"grounded": False, "gen_retries": 0}) == graph.END

        monkeypatch.setattr(graph, "settings", SimpleNamespace(max_generation_retries=3))
        assert graph.route_after_groundedness({"grounded": False, "gen_retries": 2}) == "classify"
        assert graph.route_after_groundedness({"grounded": False, "gen_retries": 3}) == graph.END


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


class TestClassifyGenRetriesCounter:
    def test_first_pass_leaves_gen_retries_unchanged(self, monkeypatch):
        monkeypatch.setattr(graph.llm, "complete", lambda *a, **kw: "Classification: Benign\nCriteria used: none identified")
        state = {"variant": "rs999", "evidence": {"clinvar": {"matches": []}}, "criteria": [], "gen_retries": 0}
        result = graph.classify(state)
        assert result["gen_retries"] == 0

    def test_retry_pass_increments_gen_retries(self, monkeypatch):
        monkeypatch.setattr(graph.llm, "complete", lambda *a, **kw: "Classification: Benign\nCriteria used: none identified")
        # non-empty "classification" is the signal that this is a retry, not the first pass
        state = {
            "variant": "rs999",
            "evidence": {"clinvar": {"matches": []}},
            "criteria": [],
            "classification": "Classification: Pathogenic\nCriteria used: PVS1",
            "gen_retries": 0,
        }
        result = graph.classify(state)
        assert result["gen_retries"] == 1


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


class TestBoundedRegenerationLoop:
    """End-to-end: classify -> route_after_groundedness -> classify, through the compiled graph."""

    def _stack(self, monkeypatch, grounded_responses):
        """Mock every tool + RAG (grading always sufficient, so the graph reaches
        classify on the first pass) + a sequence of groundedness-purpose verdicts
        (repeats the last one)."""
        monkeypatch.setattr(graph, "get_clinvar_record", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_allele_frequency", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_gene_consequence", lambda v: {"found": False})
        monkeypatch.setattr(graph, "get_genomic_context", lambda v, consequence=None: {"found": False})
        monkeypatch.setattr(graph, "get_alphamissense_score", lambda v, consequence=None: {"found": False, "applicable": False})
        monkeypatch.setattr(graph, "semantic_search", lambda q: [])

        classify_calls = {"n": 0}
        grounded_calls = {"n": 0}

        def fake_complete(prompt, system="", *, purpose="generate", max_tokens=1024):
            if purpose == "grade":
                return "sufficient\nfine"
            if purpose == "groundedness":
                idx = min(grounded_calls["n"], len(grounded_responses) - 1)
                grounded_calls["n"] += 1
                return grounded_responses[idx]
            classify_calls["n"] += 1
            return "Classification: Uncertain Significance\nCriteria used: none identified"

        monkeypatch.setattr(graph.llm, "complete", fake_complete)
        return classify_calls

    def test_ungrounded_then_grounded_retries_exactly_once(self, monkeypatch):
        classify_calls = self._stack(monkeypatch, ["ungrounded\nnot enough support", "grounded\nnow it's supported"])

        result = graph.ask("rs999")

        assert classify_calls["n"] == 2  # original pass + exactly one retry
        assert result["gen_retries"] == 1
        assert result["grounded"] is True
        assert result["classification"] == "Classification: Uncertain Significance\nCriteria used: none identified"

    def test_persistently_ungrounded_still_ends_once_budget_is_exhausted(self, monkeypatch):
        # settings.max_generation_retries defaults to 1: allowed exactly 1 retry, then must give up
        classify_calls = self._stack(monkeypatch, ["ungrounded\nstill not enough support"])

        result = graph.ask("rs999")

        assert classify_calls["n"] == 2  # original pass + the one allowed retry, then gives up
        assert result["gen_retries"] == 1
        assert result["grounded"] is False
        # the graph still terminates with the best-attempt classification rather than looping forever
        assert result["classification"] == "Classification: Uncertain Significance\nCriteria used: none identified"

    def test_immediately_grounded_never_retries(self, monkeypatch):
        classify_calls = self._stack(monkeypatch, ["grounded\ngood to go"])

        result = graph.ask("rs999")

        assert classify_calls["n"] == 1
        assert result["gen_retries"] == 0
        assert result["grounded"] is True


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
        # NOTE: not asserting grounded is True here -- the local judge model
        # (llama3.1:8b) is inconsistent even on well-supported classifications
        # (see TestCheckGroundedIntegration). What matters is that the graph
        # terminates with *some* verdict and never exceeds its retry budget.
        assert isinstance(result["grounded"], bool)
        assert result["gen_retries"] <= settings.max_generation_retries


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


@pytest.mark.integration
class TestCheckGroundedIntegration:
    def test_real_classification_grounded_in_real_evidence_returns_a_bool_verdict(self):
        # NOTE: the local judge model (llama3.1:8b) is inconsistent on this
        # well-supported case -- across repeated runs it flip-flops between
        # 'grounded' and 'ungrounded', sometimes with hallucinated reasoning
        # (e.g. claiming "Pathogenic" isn't a valid ACMG tier). That's a real
        # judge-calibration gap, not a bug in check_grounded's wiring, which
        # is covered deterministically by TestCheckGrounded above. Swap in a
        # stronger judge (e.g. ANTHROPIC_JUDGE_MODEL) for a reliable verdict.
        state = {
            "variant": "rs28897696",
            "evidence": {"clinvar": {"matches": [
                {"hgvs": "NM_007294.4(BRCA1):c.68_69delAG", "clinical_significance": "Pathogenic", "review_status": "reviewed by expert panel"},
            ]}},
            "criteria": [SimpleNamespace(
                text="PVS1 applies to null variants (nonsense, frameshift, ...) in a gene where loss of function is a known mechanism of disease.",
                source="acmg_criteria.md", score=0.9,
            )],
            "classification": "Classification: Pathogenic\nCriteria used: PVS1",
        }
        result = graph.check_grounded(state)

        assert isinstance(result["grounded"], bool)

    def test_fabricated_criteria_citation_is_caught_as_ungrounded(self):
        state = {
            "variant": "rs28897696",
            "evidence": {"clinvar": {"matches": []}},
            "criteria": [SimpleNamespace(
                text="PVS1 applies to null variants (nonsense, frameshift, ...) in a gene where loss of function is a known mechanism of disease.",
                source="acmg_criteria.md", score=0.9,
            )],
            "classification": (
                "Classification: Pathogenic\n"
                "Criteria used: PS3 (functional assay demonstrates a damaging effect), "
                "PP5 (reputable source recently classified as pathogenic)"
            ),
        }
        result = graph.check_grounded(state)

        assert isinstance(result["grounded"], bool)
        assert result["grounded"] is False


@pytest.mark.integration
class TestBoundedRegenerationLoopIntegration:
    """The win, automated: a well-supported classification passes groundedness
    immediately; a deliberately overreaching first draft is caught by the real
    judge, triggers exactly one reclassification, and the graph still
    terminates rather than looping forever."""

    def test_well_supported_classification_terminates_within_budget(self):
        # NOTE: not asserting grounded is True / gen_retries == 0 here -- the
        # local judge model is inconsistent even on well-supported input (see
        # TestCheckGroundedIntegration). What's under test is termination:
        # the graph never exceeds its retry budget and always ends with a
        # classification, whichever way the judge calls it.
        result = graph.ask("rs28897696")

        assert result["gen_retries"] <= settings.max_generation_retries
        assert isinstance(result["grounded"], bool)
        assert result["classification"]

    def test_overreaching_first_draft_is_caught_then_corrected_and_terminates(self, monkeypatch):
        calls = {"n": 0}
        original_classify = graph.classify

        def first_draft_overreaches(state):
            calls["n"] += 1
            if calls["n"] == 1:
                # fabricate criteria the real evidence/RAG chunks never provided
                return {
                    "classification": (
                        "Classification: Pathogenic\n"
                        "Criteria used: PS3 (functional assay demonstrates damaging effect), "
                        "PP5 (reputable source recently reclassified as pathogenic)"
                    ),
                    "gen_retries": state.get("gen_retries", 0),
                }
            return original_classify(state)

        monkeypatch.setattr(graph, "classify", first_draft_overreaches)

        result = graph.ask("rs28897696")

        assert calls["n"] >= 2  # the fabricated first draft triggered at least one reclassification
        assert calls["n"] <= 1 + settings.max_generation_retries  # never more than the budget allows
        assert result["classification"]  # still produced a call -- the graph terminated, didn't hang
