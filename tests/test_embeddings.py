"""Unit tests for embeddings.py.

Most tests mock the SentenceTransformer model — they check that embed_texts/
embed_query/embedding_dim call it correctly, not that the model itself
produces good vectors. One test (marked `integration`) loads the real model
to catch actual breakage (wrong kwarg, model name typo, etc).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.variant_audit import embeddings


@pytest.fixture
def fake_model(monkeypatch):
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 384
    model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    monkeypatch.setattr(embeddings, "_model", lambda: model)
    return model


class TestEmbeddingDim:
    def test_returns_model_dimension(self, fake_model):
        assert embeddings.embedding_dim() == 384


class TestEmbedTexts:
    def test_encodes_with_normalization(self, fake_model):
        embeddings.embed_texts(["hello"])
        args, kwargs = fake_model.encode.call_args
        assert args[0] == ["hello"]
        assert kwargs["normalize_embeddings"] is True

    def test_returns_plain_lists_not_numpy(self, fake_model):
        result = embeddings.embed_texts(["hello"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert all(isinstance(x, float) for x in result[0])


class TestEmbedQuery:
    def test_returns_first_embedding(self, fake_model):
        result = embeddings.embed_query("hello")
        assert result == [0.1, 0.2, 0.3]

    def test_passes_single_item_batch(self, fake_model):
        embeddings.embed_query("hello")
        args, _ = fake_model.encode.call_args
        assert args[0] == ["hello"]


@pytest.mark.integration
class TestRealModel:
    """Loads the actual sentence-transformers model — slower, network/cache dependent."""

    def test_embed_query_is_normalized_and_matches_dim(self):
        vec = embeddings.embed_query("BRCA1 pathogenic missense variant")
        assert len(vec) == embeddings.embedding_dim()
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)

    def test_similar_sentences_score_higher_than_unrelated(self):
        a, b, c = embeddings.embed_texts(
            [
                "The variant is classified as pathogenic.",
                "This variant is likely pathogenic.",
                "The weather today is sunny and warm.",
            ]
        )
        sim_related = np.dot(a, b)
        sim_unrelated = np.dot(a, c)
        assert sim_related > sim_unrelated
