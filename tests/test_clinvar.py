"""Tests for mcp_tools/clinvar.py.

Unit tests mock requests.get (two calls: esearch then esummary) to check the
parsing/mapping logic without hitting the network. The integration test hits
the real NCBI E-utilities API.
"""

import logging
from unittest.mock import MagicMock

import pytest
import requests

from src.variant_audit.mcp_tools import clinvar


def _fake_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


@pytest.fixture
def fake_requests_get(monkeypatch):
    mock_get = MagicMock()
    monkeypatch.setattr(clinvar.requests, "get", mock_get)
    return mock_get


class TestGetClinvarRecord:
    def test_not_found_returns_clean_result(self, fake_requests_get):
        fake_requests_get.return_value = _fake_response({"esearchresult": {"idlist": []}})

        result = clinvar.get_clinvar_record("rs99999999999999")

        assert result == {"variant": "rs99999999999999", "found": False}

    def test_not_found_does_not_call_esummary(self, fake_requests_get):
        fake_requests_get.return_value = _fake_response({"esearchresult": {"idlist": []}})

        clinvar.get_clinvar_record("rs99999999999999")

        assert fake_requests_get.call_count == 1

    def test_single_match_maps_fields_correctly(self, fake_requests_get):
        fake_requests_get.side_effect = [
            _fake_response({"esearchresult": {"idlist": ["123"]}}),
            _fake_response(
                {
                    "result": {
                        "uids": ["123"],
                        "123": {
                            "title": "NM_000000.1(GENE):c.1A>G (p.Met1Val)",
                            "germline_classification": {
                                "description": "Pathogenic",
                                "review_status": "reviewed by expert panel",
                            },
                        },
                    }
                }
            ),
        ]

        result = clinvar.get_clinvar_record("rsTEST")

        assert result == {
            "variant": "rsTEST",
            "found": True,
            "matches": [
                {
                    "hgvs": "NM_000000.1(GENE):c.1A>G (p.Met1Val)",
                    "clinical_significance": "Pathogenic",
                    "review_status": "reviewed by expert panel",
                }
            ],
        }

    def test_multiple_matches_all_surfaced_not_collapsed(self, fake_requests_get):
        fake_requests_get.side_effect = [
            _fake_response({"esearchresult": {"idlist": ["1", "2"]}}),
            _fake_response(
                {
                    "result": {
                        "uids": ["1", "2"],
                        "1": {
                            "title": "variant A",
                            "germline_classification": {"description": "Pathogenic", "review_status": "x"},
                        },
                        "2": {
                            "title": "variant B",
                            "germline_classification": {"description": "Benign", "review_status": "y"},
                        },
                    }
                }
            ),
        ]

        result = clinvar.get_clinvar_record("rsAMBIGUOUS")

        assert len(result["matches"]) == 2
        assert result["matches"][0]["clinical_significance"] == "Pathogenic"
        assert result["matches"][1]["clinical_significance"] == "Benign"

    def test_esummary_id_param_joins_all_ids_from_esearch(self, fake_requests_get):
        fake_requests_get.side_effect = [
            _fake_response({"esearchresult": {"idlist": ["1", "2", "3"]}}),
            _fake_response({"result": {"uids": []}}),
        ]

        clinvar.get_clinvar_record("rsTEST")

        _, esummary_kwargs = fake_requests_get.call_args_list[1]
        assert esummary_kwargs["params"]["id"] == "1,2,3"

    def test_esearch_uses_variant_as_term(self, fake_requests_get):
        fake_requests_get.return_value = _fake_response({"esearchresult": {"idlist": []}})

        clinvar.get_clinvar_record("rs28897696")

        _, esearch_kwargs = fake_requests_get.call_args_list[0]
        assert esearch_kwargs["params"]["term"] == "rs28897696"
        assert esearch_kwargs["params"]["db"] == "clinvar"


class TestErrorHandling:
    def test_esearch_network_failure_raises_runtime_error(self, fake_requests_get, caplog):
        fake_requests_get.side_effect = requests.exceptions.ConnectionError("simulated failure")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="esearch failed"):
                clinvar.get_clinvar_record("rs28897696")

        assert "clinvar esearch failed" in caplog.text
        assert "rs28897696" in caplog.text

    def test_esearch_network_failure_preserves_original_exception(self, fake_requests_get):
        original = requests.exceptions.Timeout("simulated timeout")
        fake_requests_get.side_effect = original

        with pytest.raises(RuntimeError) as exc_info:
            clinvar.get_clinvar_record("rs28897696")

        assert exc_info.value.__cause__ is original

    def test_esearch_http_error_status_raises_runtime_error(self, fake_requests_get, caplog):
        resp = _fake_response({})
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        fake_requests_get.return_value = resp

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="esearch failed"):
                clinvar.get_clinvar_record("rs28897696")

    def test_esearch_unexpected_shape_raises_runtime_error(self, fake_requests_get, caplog):
        fake_requests_get.return_value = _fake_response({"unexpected": "shape"})

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="unexpected response"):
                clinvar.get_clinvar_record("rs28897696")

        assert "clinvar esearch returned unexpected shape" in caplog.text

    def test_esummary_network_failure_raises_runtime_error(self, fake_requests_get, caplog):
        fake_requests_get.side_effect = [
            _fake_response({"esearchresult": {"idlist": ["123"]}}),
            requests.exceptions.ConnectionError("simulated failure"),
        ]

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="esummary failed"):
                clinvar.get_clinvar_record("rs28897696")

        assert "clinvar esummary failed" in caplog.text

    def test_esummary_unexpected_shape_raises_runtime_error(self, fake_requests_get, caplog):
        fake_requests_get.side_effect = [
            _fake_response({"esearchresult": {"idlist": ["123"]}}),
            _fake_response({"result": {"uids": ["123"], "123": {"title": "x"}}}),  # missing germline_classification
        ]

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="unexpected response"):
                clinvar.get_clinvar_record("rs28897696")

        assert "clinvar esummary returned unexpected shape" in caplog.text


@pytest.mark.integration
class TestGetClinvarRecordIntegration:
    def test_known_rsid_with_ambiguous_alleles(self):
        result = clinvar.get_clinvar_record("rs28897696")

        assert result["found"] is True
        assert len(result["matches"]) >= 1
        for match in result["matches"]:
            assert match["hgvs"]
            assert match["clinical_significance"]

    def test_unknown_rsid_returns_not_found(self):
        result = clinvar.get_clinvar_record("rs99999999999999")
        assert result == {"variant": "rs99999999999999", "found": False}
