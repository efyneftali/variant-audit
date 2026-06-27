"""Tests for retrieval.py.

Unit tests mock QdrantClient.query_points to check semantic_search's own logic
(top_k fallback, mapping ScoredPoint -> RetrievedChunk) without needing a live
Qdrant. The integration test hits the real Qdrant + real embedding model
against the variant_audit collection already populated by ingest_directory.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.variant_audit import retrieval


def _fake_scored_point(text, source, score, chunk_index=0):
    return SimpleNamespace(payload={"text": text, "source": source, "chunk_index": chunk_index}, score=score)


@pytest.fixture
def fake_client(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(retrieval, "QdrantClient", lambda url: client)
    monkeypatch.setattr(retrieval, "settings", SimpleNamespace(qdrant_url="http://fake", collection="fake", top_k=5))
    monkeypatch.setattr(retrieval.embeddings, "embed_query", lambda q: [0.1, 0.2, 0.3])
    return client


class TestSemanticSearch:
    def test_returns_retrieved_chunks_from_points(self, fake_client):
        fake_client.query_points.return_value = SimpleNamespace(
            points=[_fake_scored_point("hello world", "a.md", 0.9)]
        )

        results = retrieval.semantic_search("query")

        assert len(results) == 1
        chunk = results[0]
        assert isinstance(chunk, retrieval.RetrievedChunk)
        assert chunk.text == "hello world"
        assert chunk.source == "a.md"
        assert chunk.score == 0.9

    def test_preserves_order_of_points(self, fake_client):
        fake_client.query_points.return_value = SimpleNamespace(
            points=[
                _fake_scored_point("first", "a.md", 0.9),
                _fake_scored_point("second", "b.md", 0.5),
            ]
        )

        results = retrieval.semantic_search("query")

        assert [r.text for r in results] == ["first", "second"]

    def test_empty_results_returns_empty_list(self, fake_client):
        fake_client.query_points.return_value = SimpleNamespace(points=[])
        assert retrieval.semantic_search("query") == []

    def test_uses_settings_top_k_when_not_given(self, fake_client):
        fake_client.query_points.return_value = SimpleNamespace(points=[])
        retrieval.semantic_search("query")

        _, kwargs = fake_client.query_points.call_args
        assert kwargs["limit"] == 5

    def test_explicit_top_k_overrides_settings(self, fake_client):
        fake_client.query_points.return_value = SimpleNamespace(points=[])
        retrieval.semantic_search("query", top_k=2)

        _, kwargs = fake_client.query_points.call_args
        assert kwargs["limit"] == 2

    def test_embeds_query_and_passes_vector_to_qdrant(self, fake_client):
        fake_client.query_points.return_value = SimpleNamespace(points=[])
        retrieval.semantic_search("what is pathogenic")

        _, kwargs = fake_client.query_points.call_args
        assert kwargs["query"] == [0.1, 0.2, 0.3]


@pytest.mark.integration
class TestSemanticSearchIntegration:
    def test_finds_relevant_chunk_in_real_corpus(self):
        results = retrieval.semantic_search("five-tier classification scale", top_k=1)

        assert len(results) == 1
        assert results[0].source in {"acmg_criteria.md", "classification_basics.md"}
        assert 0.0 <= results[0].score <= 1.0

    def test_results_are_sorted_by_descending_score(self):
        results = retrieval.semantic_search("variant pathogenicity evidence", top_k=3)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
