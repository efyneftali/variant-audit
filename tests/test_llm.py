"""Unit tests for llm.py routing logic.

These mock the actual network calls (Ollama/Anthropic) — they test that
complete()/_model_for() make the right decisions, not that the SDKs work.
For a real end-to-end smoke test against a live Ollama server, use scratch.py.
"""

from types import SimpleNamespace

import pytest

from src.variant_audit import llm


def _fake_settings(**overrides):
    base = dict(
        llm_provider="ollama",
        ollama_model="llama3.1:8b",
        ollama_judge_model="llama3.1:8b-judge",
        anthropic_model="claude-sonnet-4-6",
        anthropic_judge_model="claude-haiku-4-5-20251001",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestModelFor:
    def test_ollama_main_model_for_generate(self, monkeypatch):
        monkeypatch.setattr(llm, "settings", _fake_settings(llm_provider="ollama"))
        assert llm._model_for("generate") == "llama3.1:8b"

    @pytest.mark.parametrize("purpose", ["grade", "groundedness", "eval_judge"])
    def test_ollama_judge_model_for_judging_purposes(self, monkeypatch, purpose):
        monkeypatch.setattr(llm, "settings", _fake_settings(llm_provider="ollama"))
        assert llm._model_for(purpose) == "llama3.1:8b-judge"

    def test_anthropic_main_model_for_generate(self, monkeypatch):
        monkeypatch.setattr(llm, "settings", _fake_settings(llm_provider="anthropic"))
        assert llm._model_for("generate") == "claude-sonnet-4-6"

    def test_anthropic_judge_model_for_grading(self, monkeypatch):
        monkeypatch.setattr(llm, "settings", _fake_settings(llm_provider="anthropic"))
        assert llm._model_for("grade") == "claude-haiku-4-5-20251001"

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setattr(llm, "settings", _fake_settings(llm_provider="bogus"))
        with pytest.raises(ValueError):
            llm._model_for("generate")


class TestComplete:
    def test_routes_to_ollama_with_resolved_model(self, monkeypatch):
        monkeypatch.setattr(llm, "settings", _fake_settings(llm_provider="ollama"))
        calls = {}

        def fake_complete_ollama(prompt, system, model, max_tokens):
            calls["args"] = (prompt, system, model, max_tokens)
            return "ollama response"

        monkeypatch.setattr(llm, "_complete_ollama", fake_complete_ollama)

        result = llm.complete("hello", "be nice", purpose="generate", max_tokens=50)

        assert result == "ollama response"
        assert calls["args"] == ("hello", "be nice", "llama3.1:8b", 50)

    def test_routes_to_anthropic_with_judge_model(self, monkeypatch):
        monkeypatch.setattr(llm, "settings", _fake_settings(llm_provider="anthropic"))
        calls = {}

        def fake_complete_anthropic(prompt, system, model, max_tokens):
            calls["args"] = (prompt, system, model, max_tokens)
            return "claude response"

        monkeypatch.setattr(llm, "_complete_anthropic", fake_complete_anthropic)

        result = llm.complete("hello", purpose="grade", max_tokens=50)

        assert result == "claude response"
        assert calls["args"] == ("hello", "", "claude-haiku-4-5-20251001", 50)
