"""Tests for ingestion.py.

chunk_text is pure logic, tested with plain unit tests. ingest_directory needs
a running Qdrant + real embedding model, so it's covered by an `integration`
test against a throwaway collection (cleaned up after) rather than mocks.
"""

from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

from src.variant_audit import ingestion


def _words(n):
    return " ".join(f"word{i}" for i in range(n))


class TestChunkText:
    def test_single_chunk_when_shorter_than_chunk_size(self):
        text = _words(5)
        chunks = ingestion.chunk_text(text, chunk_size=10, overlap=2)
        assert chunks == [text]

    def test_empty_text_returns_no_chunks(self):
        assert ingestion.chunk_text("", chunk_size=10, overlap=2) == []

    def test_first_transition_has_exact_configured_overlap(self):
        """The first stride is governed exactly by chunk_size/overlap; later chunks
        may be shifted backward to avoid a tiny trailing fragment (see the next test),
        so only the first transition is guaranteed to match overlap exactly."""
        text = _words(25)
        chunks = ingestion.chunk_text(text, chunk_size=10, overlap=2)

        assert chunks[0].split()[-2:] == chunks[1].split()[:2]

    def test_consecutive_chunks_share_at_least_the_overlap(self):
        text = _words(25)
        chunks = ingestion.chunk_text(text, chunk_size=10, overlap=2)

        for a, b in zip(chunks, chunks[1:]):
            shared = set(a.split()) & set(b.split())
            assert len(shared) >= 2

    def test_no_word_is_dropped(self):
        text = _words(25)
        chunks = ingestion.chunk_text(text, chunk_size=10, overlap=2)

        seen = set()
        for chunk in chunks:
            seen.update(chunk.split())
        assert seen == {f"word{i}" for i in range(25)}

    def test_last_chunk_is_full_size_not_a_tiny_fragment(self):
        text = _words(25)
        chunks = ingestion.chunk_text(text, chunk_size=10, overlap=2)

        assert all(len(chunk.split()) == 10 for chunk in chunks)

    def test_no_duplicate_chunks_from_shifted_last_window(self):
        text = _words(25)
        chunks = ingestion.chunk_text(text, chunk_size=10, overlap=2)
        assert len(chunks) == len(set(chunks))

    def test_uses_settings_defaults_when_not_given(self, monkeypatch):
        monkeypatch.setattr(ingestion, "settings", SimpleNamespace(chunk_size=10, chunk_overlap=2))

        text = _words(25)
        assert ingestion.chunk_text(text) == ingestion.chunk_text(text, chunk_size=10, overlap=2)


@pytest.mark.integration
class TestIngestDirectory:
    @pytest.fixture
    def corpus_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ingestion,
            "settings",
            SimpleNamespace(
                chunk_size=512,
                chunk_overlap=64,
                qdrant_url="http://localhost:6333",
                collection="test_ingestion_scratch",
            ),
        )
        (tmp_path / "one.md").write_text("Pathogenic variants in BRCA1 increase cancer risk.")
        (tmp_path / "two.txt").write_text("Benign variants are not expected to cause disease.")
        (tmp_path / "ignored.json").write_text('{"not": "ingested"}')

        yield tmp_path

        client = QdrantClient(url="http://localhost:6333")
        if client.collection_exists("test_ingestion_scratch"):
            client.delete_collection("test_ingestion_scratch")

    def test_ingests_md_and_txt_only(self, corpus_dir):
        count = ingestion.ingest_directory(str(corpus_dir))
        assert count == 2

    def test_rerunning_does_not_duplicate_points(self, corpus_dir):
        ingestion.ingest_directory(str(corpus_dir))
        ingestion.ingest_directory(str(corpus_dir))

        client = QdrantClient(url="http://localhost:6333")
        assert client.count("test_ingestion_scratch").count == 2

    def test_payload_has_text_source_and_chunk_index(self, corpus_dir):
        ingestion.ingest_directory(str(corpus_dir))

        client = QdrantClient(url="http://localhost:6333")
        points, _ = client.scroll("test_ingestion_scratch", limit=10, with_payload=True)
        sources = {p.payload["source"] for p in points}

        assert sources == {"one.md", "two.txt"}
        assert all(p.payload["chunk_index"] == 0 for p in points)
        assert all(isinstance(p.payload["text"], str) and p.payload["text"] for p in points)
