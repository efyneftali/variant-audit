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
        return "Classification: Uncertain Significance\nCriteria used: none identified"

    monkeypatch.setattr(graph.llm, "complete", fake_complete)
    return clinvar_result, chunks, llm_calls


class TestGatherEvidence:
    def test_wraps_clinvar_record_under_clinvar_key(self, fake_stack):
        clinvar_result, _, _ = fake_stack
        result = graph.gather_evidence({"variant": "rs28897696"})
        assert result == {"evidence": {"clinvar": clinvar_result}}

    def test_passes_variant_through_to_clinvar_lookup(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(graph, "get_clinvar_record", lambda v: captured.update({"variant": v}) or {"found": False, "matches": []})
        graph.gather_evidence({"variant": "rs999"})
        assert captured["variant"] == "rs999"


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


class TestBuildGraphAndAsk:
    def test_build_graph_compiles(self):
        compiled = graph.build_graph()
        assert compiled is not None

    def test_ask_runs_linear_flow_and_populates_final_state(self, fake_stack):
        clinvar_result, chunks, llm_calls = fake_stack
        result = graph.ask("rs28897696")

        assert result["variant"] == "rs28897696"
        assert result["evidence"] == {"clinvar": clinvar_result}
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

        for module in (classify, graph):
            monkeypatch.setattr(module, "get_clinvar_record", lambda v: clinvar_result)
            monkeypatch.setattr(module, "semantic_search", lambda q: chunks)
            monkeypatch.setattr(module.llm, "complete", lambda *a, **kw: "Classification: Pathogenic\nCriteria used: PVS1")

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
