"""Tests for mcp_tools/gnomad.py.

Unit tests mock requests.post (the single GraphQL call) to check the
parsing/error-handling logic without hitting the network. gnomAD reports a
genuinely unknown variant as HTTP 200 + {"data": {"variant": null}}, which is
distinct from a real GraphQL/schema error ({"errors": [...]} with no "data"
key) — both are exercised below.

The integration test hits the real gnomAD GraphQL API.
"""

import logging
from unittest.mock import MagicMock

import pytest
import requests

from src.variant_audit.mcp_tools import gnomad


def _fake_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


@pytest.fixture
def fake_requests_post(monkeypatch):
    mock_post = MagicMock()
    monkeypatch.setattr(gnomad.requests, "post", mock_post)
    return mock_post


class TestGetAlleleFrequency:
    def test_not_found_returns_clean_result(self, fake_requests_post):
        fake_requests_post.return_value = _fake_response(
            {"data": {"variant": None}, "errors": [{"message": "Variant not found"}]}
        )

        result = gnomad.get_allele_frequency("rs99999999999")

        assert result == {"variant": "rs99999999999", "found": False}

    def test_found_prefers_genome_frequency(self, fake_requests_post):
        fake_requests_post.return_value = _fake_response(
            {
                "data": {
                    "variant": {
                        "variantId": "17-43057062-T-TG",
                        "chrom": "17",
                        "pos": 43057062,
                        "ref": "T",
                        "alt": "TG",
                        "genome": {"af": 5.25e-05, "ac": 8, "an": 152180},
                        "exome": {"af": 6.9e-05, "ac": 101, "an": 1461732},
                    }
                }
            }
        )

        result = gnomad.get_allele_frequency("rs80357906")

        assert result["found"] is True
        assert result["allele_freq"] == 5.25e-05
        assert result["genome"]["ac"] == 8
        assert result["exome"]["an"] == 1461732

    def test_found_falls_back_to_exome_when_no_genome_data(self, fake_requests_post):
        fake_requests_post.return_value = _fake_response(
            {"data": {"variant": {"genome": None, "exome": {"af": 1.2e-04, "ac": 5, "an": 40000}}}}
        )

        result = gnomad.get_allele_frequency("rsTEST")

        assert result["allele_freq"] == 1.2e-04
        assert result["genome"] is None

    def test_found_with_no_frequency_data_at_all(self, fake_requests_post):
        fake_requests_post.return_value = _fake_response({"data": {"variant": {"genome": None, "exome": None}}})

        result = gnomad.get_allele_frequency("rsTEST")

        assert result["allele_freq"] is None

    def test_query_sends_rsid_and_dataset_variables(self, fake_requests_post):
        fake_requests_post.return_value = _fake_response({"data": {"variant": None}})

        gnomad.get_allele_frequency("rs28897696")

        _, kwargs = fake_requests_post.call_args
        assert kwargs["json"]["variables"] == {"rsid": "rs28897696", "dataset": gnomad.GNOMAD_DATASET}


class TestErrorHandling:
    def test_network_failure_raises_runtime_error(self, fake_requests_post, caplog):
        fake_requests_post.side_effect = requests.exceptions.ConnectionError("simulated failure")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="gnomAD query failed"):
                gnomad.get_allele_frequency("rs28897696")

        assert "gnomad query failed" in caplog.text
        assert "rs28897696" in caplog.text

    def test_network_failure_preserves_original_exception(self, fake_requests_post):
        original = requests.exceptions.Timeout("simulated timeout")
        fake_requests_post.side_effect = original

        with pytest.raises(RuntimeError) as exc_info:
            gnomad.get_allele_frequency("rs28897696")

        assert exc_info.value.__cause__ is original

    def test_http_error_status_raises_runtime_error(self, fake_requests_post):
        resp = _fake_response({})
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        fake_requests_post.return_value = resp

        with pytest.raises(RuntimeError, match="gnomAD query failed"):
            gnomad.get_allele_frequency("rs28897696")

    def test_schema_error_with_no_data_key_raises_runtime_error(self, fake_requests_post, caplog):
        fake_requests_post.return_value = _fake_response(
            {"errors": [{"message": 'Cannot query field "af" on type "VariantPopulation".'}]}
        )

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="returned an error"):
                gnomad.get_allele_frequency("rs28897696")

        assert "gnomad graphql error" in caplog.text

    def test_unexpected_shape_raises_runtime_error(self, fake_requests_post, caplog):
        fake_requests_post.return_value = _fake_response({"data": {"variant": {"genome": "not-a-dict"}}})

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="unexpected response shape"):
                gnomad.get_allele_frequency("rs28897696")

    def test_non_json_response_raises_runtime_error(self, fake_requests_post):
        resp = MagicMock()
        resp.json.side_effect = ValueError("no JSON")
        resp.text = "<html>error</html>"
        fake_requests_post.return_value = resp

        with pytest.raises(RuntimeError, match="unexpected response"):
            gnomad.get_allele_frequency("rs28897696")


@pytest.mark.integration
class TestGetAlleleFrequencyIntegration:
    def test_known_variant_returns_frequency(self):
        result = gnomad.get_allele_frequency("rs80357906")

        assert result["found"] is True
        assert result["allele_freq"] is not None
        assert result["allele_freq"] >= 0

    def test_unknown_rsid_returns_not_found(self):
        result = gnomad.get_allele_frequency("rs99999999999")
        assert result == {"variant": "rs99999999999", "found": False}
