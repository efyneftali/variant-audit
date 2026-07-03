"""Tests for mcp_tools/alphamissense.py.

Unlike the other tools, the lookup layer here is a local SQLite DB, not a
network call — so unit tests seed a tiny real SQLite file (via tmp_path) with a
couple of fake rows and monkeypatch DB_PATH to it, rather than mocking requests.
get_gene_consequence is still mocked, since that's the real network dependency.

The integration test hits the real local DB (must be built first via
`python -m variant_audit.mcp_tools.alphamissense --build`) plus the real
Ensembl API.
"""

import sqlite3
from pathlib import Path

import pytest

from src.variant_audit.mcp_tools import alphamissense


@pytest.fixture
def fake_db(tmp_path, monkeypatch):
    db_path = tmp_path / "alphamissense.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE variants (
            chrom TEXT, pos INTEGER, ref TEXT, alt TEXT,
            transcript_id TEXT, protein_variant TEXT,
            am_pathogenicity REAL, am_class TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO variants VALUES ('chr17', 43063903, 'G', 'A', 'ENST00000357654.9', 'A566V', 0.9871, 'likely_pathogenic')"
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(alphamissense, "DB_PATH", db_path)
    return db_path


class TestGetAlphamissenseScore:
    def test_not_applicable_when_ensembl_has_no_record(self, monkeypatch, fake_db):
        monkeypatch.setattr(alphamissense, "get_gene_consequence", lambda v: {"found": False})

        result = alphamissense.get_alphamissense_score("rsFAKE")

        assert result == {"variant": "rsFAKE", "found": False, "applicable": False}

    def test_not_applicable_for_non_missense_consequence(self, monkeypatch, fake_db):
        monkeypatch.setattr(
            alphamissense, "get_gene_consequence",
            lambda v: {"found": True, "most_severe_consequence": "frameshift_variant", "chrom": "17", "start": 1, "ref": "-", "alt": "G"},
        )

        result = alphamissense.get_alphamissense_score("rs80357906")

        assert result == {"variant": "rs80357906", "found": False, "applicable": False}

    def test_found_returns_score_for_known_missense_variant(self, monkeypatch, fake_db):
        monkeypatch.setattr(
            alphamissense, "get_gene_consequence",
            lambda v: {"found": True, "most_severe_consequence": "missense_variant", "chrom": "17", "start": 43063903, "ref": "G", "alt": "A"},
        )

        result = alphamissense.get_alphamissense_score("rs28897696")

        assert result == {
            "variant": "rs28897696",
            "found": True,
            "applicable": True,
            "am_pathogenicity": 0.9871,
            "am_class": "likely_pathogenic",
            "transcript_id": "ENST00000357654.9",
            "protein_variant": "A566V",
        }

    def test_not_found_when_missense_but_absent_from_table(self, monkeypatch, fake_db):
        monkeypatch.setattr(
            alphamissense, "get_gene_consequence",
            lambda v: {"found": True, "most_severe_consequence": "missense_variant", "chrom": "17", "start": 999999, "ref": "A", "alt": "T"},
        )

        result = alphamissense.get_alphamissense_score("rsUNSEEN")

        assert result == {"variant": "rsUNSEEN", "found": False, "applicable": True}

    def test_missing_ref_or_alt_returns_not_found_but_applicable_without_querying_db(self, monkeypatch):
        # if ref/alt can't be resolved, skip the DB entirely -- must not require a DB to exist
        monkeypatch.setattr(
            alphamissense, "get_gene_consequence",
            lambda v: {"found": True, "most_severe_consequence": "missense_variant", "chrom": "17", "start": 1, "ref": None, "alt": None},
        )
        monkeypatch.setattr(alphamissense, "DB_PATH", Path("/nonexistent/path.sqlite"))

        result = alphamissense.get_alphamissense_score("rsTEST")

        assert result == {"variant": "rsTEST", "found": False, "applicable": True}

    def test_accepts_a_pre_fetched_consequence_and_skips_ensembl_call(self, monkeypatch, fake_db):
        called = {"ensembl": False}
        monkeypatch.setattr(alphamissense, "get_gene_consequence", lambda v: called.update({"ensembl": True}))

        consequence = {"found": True, "most_severe_consequence": "missense_variant", "chrom": "17", "start": 43063903, "ref": "G", "alt": "A"}
        alphamissense.get_alphamissense_score("rs28897696", consequence=consequence)

        assert called["ensembl"] is False

    def test_adds_chr_prefix_only_when_missing(self, monkeypatch, fake_db):
        # bare chrom, like Ensembl returns ("17" not "chr17")
        monkeypatch.setattr(
            alphamissense, "get_gene_consequence",
            lambda v: {"found": True, "most_severe_consequence": "missense_variant", "chrom": "17", "start": 43063903, "ref": "G", "alt": "A"},
        )

        result = alphamissense.get_alphamissense_score("rs28897696")

        assert result["found"] is True  # only matches the seeded row if "chr17" was queried, not "17"

    def test_raises_file_not_found_when_db_missing(self, monkeypatch):
        monkeypatch.setattr(
            alphamissense, "get_gene_consequence",
            lambda v: {"found": True, "most_severe_consequence": "missense_variant", "chrom": "17", "start": 43063903, "ref": "G", "alt": "A"},
        )
        monkeypatch.setattr(alphamissense, "DB_PATH", Path("/nonexistent/path.sqlite"))

        with pytest.raises(FileNotFoundError, match="--build"):
            alphamissense.get_alphamissense_score("rs28897696")


@pytest.mark.integration
class TestGetAlphamissenseScoreIntegration:
    def test_known_missense_variant_returns_a_score(self):
        result = alphamissense.get_alphamissense_score("rs28897696")

        assert result["applicable"] is True
        assert result["found"] is True
        assert 0.0 <= result["am_pathogenicity"] <= 1.0
        assert result["am_class"] in ("likely_benign", "ambiguous", "likely_pathogenic")

    def test_non_missense_variant_is_not_applicable(self):
        result = alphamissense.get_alphamissense_score("rs80357906")
        assert result == {"variant": "rs80357906", "found": False, "applicable": False}
