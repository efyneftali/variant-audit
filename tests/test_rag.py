"""Tests for rag.py's answer_question.

Unit tests mock semantic_search and llm.complete to check the prompt-building
and source-deduplication logic in isolation. The integration test hits the
real Qdrant corpus + real Ollama model.
"""

import pytest

from src.variant_audit import rag
from src.variant_audit.retrieval import RetrievedChunk


@pytest.fixture
def fake_retrieval(monkeypatch):
    chunks = [
        RetrievedChunk(text="PVS1 is a very strong pathogenic criterion.", source="acmg_criteria.md", score=0.9),
        RetrievedChunk(text="VUS means insufficient evidence.", source="classification_basics.md", score=0.4),
    ]
    monkeypatch.setattr(rag, "semantic_search", lambda question, top_k=None: chunks)

    calls = {}

    def fake_complete(prompt, system="", *, purpose="generate", max_tokens=1024):
        calls["prompt"] = prompt
        calls["system"] = system
        calls["purpose"] = purpose
        return "the answer"

    monkeypatch.setattr(rag.llm, "complete", fake_complete)
    return calls


class TestAnswerQuestion:
    def test_returns_llm_answer_and_sorted_unique_sources(self, fake_retrieval):
        answer, sources = rag.answer_question("what is PVS1?")

        assert answer == "the answer"
        assert sources == ["acmg_criteria.md", "classification_basics.md"]

    def test_prompt_includes_chunk_text_and_source(self, fake_retrieval):
        rag.answer_question("what is PVS1?")

        prompt = fake_retrieval["prompt"]
        assert "PVS1 is a very strong pathogenic criterion." in prompt
        assert "[acmg_criteria.md]" in prompt
        assert "what is PVS1?" in prompt

    def test_uses_generate_purpose(self, fake_retrieval):
        rag.answer_question("what is PVS1?")
        assert fake_retrieval["purpose"] == "generate"

    def test_system_instructs_grounding_and_citation(self, fake_retrieval):
        rag.answer_question("what is PVS1?")
        system = fake_retrieval["system"].lower()
        assert "context" in system
        assert "cite" in system

    def test_no_chunks_returns_empty_sources(self, monkeypatch):
        monkeypatch.setattr(rag, "semantic_search", lambda question, top_k=None: [])
        monkeypatch.setattr(rag.llm, "complete", lambda prompt, system="", **kw: "no answer found")

        answer, sources = rag.answer_question("unrelated question")
        assert sources == []


@pytest.mark.integration
class TestAnswerQuestionIntegration:
    def test_answers_grounded_in_real_corpus(self):
        answer, sources = rag.answer_question("What is a VUS?", top_k=2)

        assert isinstance(answer, str) and len(answer) > 0
        assert len(sources) > 0
        assert all(s.endswith((".md", ".txt")) for s in sources)
