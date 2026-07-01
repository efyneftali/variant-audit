"""Tests for classify.py.

Unit tests mock all three external calls (get_clinvar_record, semantic_search,
llm.complete) to verify the function's own logic: query construction, prompt
structure, return shape, and the not-found edge case.

The integration test calls the real stack (live ClinVar + Qdrant + Ollama).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.variant_audit import classify


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
    monkeypatch.setattr(classify, "get_clinvar_record", lambda v: clinvar_result)

    chunks = [
        SimpleNamespace(text="PVS1 applies to null variants.", source="acmg_criteria.md", score=0.9),
        SimpleNamespace(text="BA1 applies when frequency > 5%.", source="acmg_criteria.md", score=0.6),
    ]
    monkeypatch.setattr(classify, "semantic_search", lambda q: chunks)

    llm_calls = {}

    def fake_complete(prompt, system="", *, purpose="generate", max_tokens=1024):
        llm_calls["prompt"] = prompt
        llm_calls["system"] = system
        llm_calls["purpose"] = purpose
        return "Classification: Uncertain Significance\nCriteria used: none identified"

    monkeypatch.setattr(classify.llm, "complete", fake_complete)
    return clinvar_result, chunks, llm_calls


class TestClassifyVariant:
    def test_returns_required_keys(self, fake_stack):
        result = classify.classify_variant("rs28897696")
        assert set(result.keys()) == {"variant", "classification", "evidence_used", "criteria_cited"}

    def test_variant_key_matches_input(self, fake_stack):
        result = classify.classify_variant("rs28897696")
        assert result["variant"] == "rs28897696"

    def test_evidence_used_contains_all_clinvar_matches(self, fake_stack):
        clinvar_result, _, _ = fake_stack
        result = classify.classify_variant("rs28897696")
        assert result["evidence_used"] == clinvar_result["matches"]

    def test_criteria_cited_contains_deduplicated_sources(self, fake_stack):
        result = classify.classify_variant("rs28897696")
        assert result["criteria_cited"] == ["acmg_criteria.md"]

    def test_classification_is_llm_output(self, fake_stack):
        result = classify.classify_variant("rs28897696")
        assert "Uncertain Significance" in result["classification"]

    def test_prompt_includes_clinvar_hgvs_and_significance(self, fake_stack):
        _, _, llm_calls = fake_stack
        classify.classify_variant("rs28897696")
        assert "BRCA1:c.123A>G" in llm_calls["prompt"]
        assert "Pathogenic" in llm_calls["prompt"]

    def test_prompt_includes_retrieved_criteria_text(self, fake_stack):
        _, _, llm_calls = fake_stack
        classify.classify_variant("rs28897696")
        assert "PVS1 applies to null variants." in llm_calls["prompt"]

    def test_search_query_is_built_from_clinvar_not_raw_rsid(self, monkeypatch):
        monkeypatch.setattr(classify, "get_clinvar_record", lambda v: {
            "found": True,
            "matches": [{"hgvs": "GENE:c.1A>G", "clinical_significance": "Pathogenic", "review_status": "x"}],
        })
        captured = {}
        monkeypatch.setattr(classify, "semantic_search", lambda q: captured.update({"query": q}) or [])
        monkeypatch.setattr(classify.llm, "complete", lambda *a, **kw: "Classification: VUS\nCriteria used: none")

        classify.classify_variant("rs123")

        assert "rs123" not in captured["query"]
        assert "GENE:c.1A>G" in captured["query"] or "Pathogenic" in captured["query"]

    def test_not_found_falls_back_to_variant_string_for_search(self, monkeypatch):
        monkeypatch.setattr(classify, "get_clinvar_record", lambda v: {"found": False, "matches": []})
        captured = {}
        monkeypatch.setattr(classify, "semantic_search", lambda q: captured.update({"query": q}) or [])
        monkeypatch.setattr(classify.llm, "complete", lambda *a, **kw: "Classification: VUS\nCriteria used: none")

        classify.classify_variant("rs999")

        assert captured["query"] == "rs999"

    def test_not_found_returns_empty_evidence_used(self, monkeypatch):
        monkeypatch.setattr(classify, "get_clinvar_record", lambda v: {"found": False, "matches": []})
        monkeypatch.setattr(classify, "semantic_search", lambda q: [])
        monkeypatch.setattr(classify.llm, "complete", lambda *a, **kw: "Classification: VUS\nCriteria used: none")

        result = classify.classify_variant("rs999")
        assert result["evidence_used"] == []

    def test_uses_generate_purpose(self, fake_stack):
        _, _, llm_calls = fake_stack
        classify.classify_variant("rs28897696")
        assert llm_calls["purpose"] == "generate"

    def test_system_prompt_enforces_five_tier_vocabulary(self, fake_stack):
        _, _, llm_calls = fake_stack
        classify.classify_variant("rs28897696")
        system = llm_calls["system"]
        assert "Pathogenic" in system
        assert "Uncertain Significance" in system
        assert "Benign" in system


@pytest.mark.integration
class TestClassifyVariantIntegration:
    def test_real_variant_returns_valid_classification(self):
        result = classify.classify_variant("rs28897696")

        assert result["variant"] == "rs28897696"
        assert len(result["evidence_used"]) > 0
        assert len(result["criteria_cited"]) > 0
        assert any(
            tier in result["classification"]
            for tier in ["Pathogenic", "Likely Pathogenic", "Uncertain Significance", "Likely Benign", "Benign"]
        )
