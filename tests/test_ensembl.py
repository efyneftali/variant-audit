"""Tests for mcp_tools/ensembl.py.

Unit tests mock requests.get (the VEP call) to check the parsing/error-handling
logic without hitting the network — including the primary-transcript-selection
logic, since a variant can hit many transcripts with different consequences.

The integration test hits the real Ensembl REST API.
"""

import logging
from unittest.mock import MagicMock

import pytest
import requests

from src.variant_audit.mcp_tools import ensembl


def _fake_response(payload, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


@pytest.fixture
def fake_requests_get(monkeypatch):
    mock_get = MagicMock()
    monkeypatch.setattr(ensembl.requests, "get", mock_get)
    return mock_get


class TestGetGeneConsequence:
    def test_not_found_returns_clean_result_on_400(self, fake_requests_get):
        resp = _fake_response({"error": "No variant found with ID 'rsFAKE'"}, status_code=400)
        resp.text = "No variant found with ID 'rsFAKE'"
        fake_requests_get.return_value = resp

        result = ensembl.get_gene_consequence("rsFAKE")

        assert result == {"variant": "rsFAKE", "found": False}

    def test_maps_top_level_and_primary_transcript_fields(self, fake_requests_get):
        fake_requests_get.return_value = _fake_response(
            [
                {
                    "most_severe_consequence": "frameshift_variant",
                    "seq_region_name": "17",
                    "start": 43057066,
                    "end": 43057065,
                    "allele_string": "-/G",
                    "transcript_consequences": [
                        {
                            "biotype": "nonsense_mediated_decay",
                            "consequence_terms": ["frameshift_variant"],
                            "impact": "HIGH",
                            "gene_symbol": "WRONG",
                            "variant_allele": "G",
                        },
                        {
                            "biotype": "protein_coding",
                            "consequence_terms": ["frameshift_variant"],
                            "impact": "HIGH",
                            "gene_symbol": "BRCA1",
                            "variant_allele": "G",
                        },
                    ],
                }
            ]
        )

        result = ensembl.get_gene_consequence("rs80357906")

        assert result == {
            "variant": "rs80357906",
            "found": True,
            "most_severe_consequence": "frameshift_variant",
            "impact": "HIGH",
            "gene_symbol": "BRCA1",
            "chrom": "17",
            "start": 43057066,
            "end": 43057065,
            "allele_string": "-/G",
            "ref": "-",
            "alt": "G",
        }

    def test_prefers_protein_coding_transcript_matching_most_severe(self, fake_requests_get):
        # a non-protein-coding transcript matches most_severe; the protein_coding
        # one on this variant has a different (lesser) consequence.
        fake_requests_get.return_value = _fake_response(
            [
                {
                    "most_severe_consequence": "missense_variant",
                    "transcript_consequences": [
                        {"biotype": "protein_coding", "consequence_terms": ["synonymous_variant"], "impact": "LOW", "gene_symbol": "GENEA"},
                        {"biotype": "nonsense_mediated_decay", "consequence_terms": ["missense_variant"], "impact": "MODERATE", "gene_symbol": "GENEB"},
                    ],
                }
            ]
        )

        result = ensembl.get_gene_consequence("rsTEST")

        assert result["gene_symbol"] == "GENEB"
        assert result["impact"] == "MODERATE"

    def test_falls_back_to_first_transcript_when_none_match(self, fake_requests_get):
        fake_requests_get.return_value = _fake_response(
            [
                {
                    "most_severe_consequence": "intergenic_variant",
                    "transcript_consequences": [
                        {"biotype": "protein_coding", "consequence_terms": ["missense_variant"], "impact": "MODERATE", "gene_symbol": "GENEA"},
                    ],
                }
            ]
        )

        result = ensembl.get_gene_consequence("rsTEST")

        assert result["gene_symbol"] == "GENEA"

    def test_no_transcript_consequences_returns_none_fields(self, fake_requests_get):
        fake_requests_get.return_value = _fake_response(
            [{"most_severe_consequence": "intergenic_variant", "transcript_consequences": []}]
        )

        result = ensembl.get_gene_consequence("rsTEST")

        assert result["found"] is True
        assert result["gene_symbol"] is None
        assert result["impact"] is None

    def test_ref_alt_derived_from_multiallelic_site_match_primary_transcript(self, fake_requests_get):
        # rs28897696-style site: 3 possible alts, each with its own consequence;
        # ref/alt must track whichever allele `_primary_transcript_consequence` picked.
        fake_requests_get.return_value = _fake_response(
            [
                {
                    "most_severe_consequence": "missense_variant",
                    "allele_string": "G/A/C/T",
                    "transcript_consequences": [
                        {"biotype": "protein_coding", "consequence_terms": ["missense_variant"], "impact": "MODERATE", "gene_symbol": "BRCA1", "variant_allele": "A"},
                        {"biotype": "protein_coding", "consequence_terms": ["missense_variant"], "impact": "MODERATE", "gene_symbol": "BRCA1", "variant_allele": "C"},
                    ],
                }
            ]
        )

        result = ensembl.get_gene_consequence("rs28897696")

        assert result["ref"] == "G"
        assert result["alt"] == "A"  # first matching transcript_consequence, not just any allele

    def test_ref_alt_are_none_when_allele_string_missing(self, fake_requests_get):
        fake_requests_get.return_value = _fake_response(
            [{"most_severe_consequence": "intergenic_variant", "transcript_consequences": []}]
        )

        result = ensembl.get_gene_consequence("rsTEST")

        assert result["ref"] is None
        assert result["alt"] is None

    def test_requests_json_content_type_for_correct_id(self, fake_requests_get):
        fake_requests_get.return_value = _fake_response(
            [{"most_severe_consequence": "x", "transcript_consequences": []}]
        )

        ensembl.get_gene_consequence("rs28897696")

        args, kwargs = fake_requests_get.call_args
        assert args[0] == f"{ensembl.VEP_BASE}/rs28897696"
        assert kwargs["params"]["content-type"] == "application/json"


class TestErrorHandling:
    def test_network_failure_raises_runtime_error(self, fake_requests_get, caplog):
        fake_requests_get.side_effect = requests.exceptions.ConnectionError("simulated failure")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="Ensembl VEP call failed"):
                ensembl.get_gene_consequence("rs28897696")

        assert "ensembl vep call failed" in caplog.text
        assert "rs28897696" in caplog.text

    def test_network_failure_preserves_original_exception(self, fake_requests_get):
        original = requests.exceptions.Timeout("simulated timeout")
        fake_requests_get.side_effect = original

        with pytest.raises(RuntimeError) as exc_info:
            ensembl.get_gene_consequence("rs28897696")

        assert exc_info.value.__cause__ is original

    def test_non_400_http_error_status_raises_runtime_error(self, fake_requests_get):
        resp = _fake_response({}, status_code=500)
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        fake_requests_get.return_value = resp

        with pytest.raises(RuntimeError, match="Ensembl VEP call failed"):
            ensembl.get_gene_consequence("rs28897696")

    def test_non_json_response_raises_runtime_error(self, fake_requests_get):
        resp = _fake_response({}, status_code=200)
        resp.json.side_effect = ValueError("no JSON")
        resp.text = "<html>error</html>"
        fake_requests_get.return_value = resp

        with pytest.raises(RuntimeError, match="unexpected response"):
            ensembl.get_gene_consequence("rs28897696")

    def test_empty_result_list_raises_runtime_error(self, fake_requests_get, caplog):
        fake_requests_get.return_value = _fake_response([])

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="unexpected response shape"):
                ensembl.get_gene_consequence("rs28897696")


@pytest.mark.integration
class TestGetGeneConsequenceIntegration:
    def test_known_variant_returns_consequence_and_gene(self):
        result = ensembl.get_gene_consequence("rs80357906")

        assert result["found"] is True
        assert result["most_severe_consequence"]
        assert result["gene_symbol"] == "BRCA1"

    def test_unknown_rsid_returns_not_found(self):
        result = ensembl.get_gene_consequence("rs999999999999999")
        assert result == {"variant": "rs999999999999999", "found": False}
