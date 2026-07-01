"""Tests for api.py endpoints.

Uses FastAPI's TestClient (no real server needed). External calls
(classify_variant, answer_question) are mocked — the API layer is
just routing/serialisation, so that's all we need to test here.
"""

import pytest
from fastapi.testclient import TestClient

from src.variant_audit import api

client = TestClient(api.app, raise_server_exceptions=True)


@pytest.fixture
def fake_classify(monkeypatch):
    result = {
        "variant": "rs80357906",
        "classification": "Classification: Pathogenic\nCriteria used: PVS1",
        "evidence_used": [
            {"hgvs": "BRCA1:c.5266dup", "clinical_significance": "Pathogenic", "review_status": "reviewed by expert panel"}
        ],
        "criteria_cited": ["acmg_criteria.md"],
    }
    monkeypatch.setattr(api, "classify_variant", lambda v: result)
    return result


@pytest.fixture
def fake_ask(monkeypatch):
    monkeypatch.setattr(api, "answer_question", lambda q: ("the answer", ["acmg_criteria.md"]))


class TestHealthz:
    def test_returns_200(self):
        resp = client.get("/healthz")
        assert resp.status_code == 200

    def test_returns_ok_true(self):
        resp = client.get("/healthz")
        assert resp.json() == {"ok": True}


class TestClassifyEndpoint:
    def test_returns_200_for_valid_request(self, fake_classify):
        resp = client.post("/classify", json={"variant": "rs80357906"})
        assert resp.status_code == 200

    def test_returns_all_four_fields(self, fake_classify):
        resp = client.post("/classify", json={"variant": "rs80357906"})
        body = resp.json()
        assert set(body.keys()) == {"variant", "classification", "evidence_used", "criteria_cited"}

    def test_passes_variant_to_classify_variant(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(api, "classify_variant", lambda v: calls.update({"variant": v}) or {
            "variant": v, "classification": "", "evidence_used": [], "criteria_cited": []
        })
        client.post("/classify", json={"variant": "rs123"})
        assert calls["variant"] == "rs123"

    def test_returns_422_when_variant_missing(self):
        resp = client.post("/classify", json={})
        assert resp.status_code == 422

    def test_returns_422_when_body_is_empty(self):
        resp = client.post("/classify", json=None)
        assert resp.status_code == 422


class TestAskEndpoint:
    def test_returns_200_for_valid_request(self, fake_ask):
        resp = client.post("/ask", json={"question": "What is PVS1?"})
        assert resp.status_code == 200

    def test_returns_answer_and_sources(self, fake_ask):
        resp = client.post("/ask", json={"question": "What is PVS1?"})
        body = resp.json()
        assert body["answer"] == "the answer"
        assert body["sources"] == ["acmg_criteria.md"]

    def test_returns_422_when_question_missing(self):
        resp = client.post("/ask", json={})
        assert resp.status_code == 422


@pytest.mark.integration
class TestClassifyEndpointIntegration:
    def test_real_variant_returns_valid_classification(self):
        resp = client.post("/classify", json={"variant": "rs80357906"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["variant"] == "rs80357906"
        assert len(body["evidence_used"]) > 0
        assert any(
            tier in body["classification"]
            for tier in ["Pathogenic", "Likely Pathogenic", "Uncertain Significance", "Likely Benign", "Benign"]
        )
