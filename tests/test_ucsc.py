"""Tests for mcp_tools/ucsc.py.

UCSC has no rsID lookup — get_genomic_context resolves coordinates via Ensembl
first, then queries UCSC. Unit tests mock get_gene_consequence and
requests.get to check the coordinate-conversion and parsing logic without
hitting the network. The integration test hits the real Ensembl + UCSC APIs.
"""

import logging
from unittest.mock import MagicMock

import pytest
import requests

from src.variant_audit.mcp_tools import ucsc


def _fake_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


def _track_response(track, value, chrom="chr17", start=0, end=1):
    return {track: [{"chrom": chrom, "start": start, "end": end, "value": value}], "itemsReturned": 1}


@pytest.fixture
def fake_requests_get(monkeypatch):
    mock_get = MagicMock()
    monkeypatch.setattr(ucsc.requests, "get", mock_get)
    return mock_get


@pytest.fixture
def fake_ensembl_found(monkeypatch):
    consequence = {"variant": "rs28897696", "found": True, "chrom": "17", "start": 43063903, "end": 43063903}
    monkeypatch.setattr(ucsc, "get_gene_consequence", lambda v: consequence)
    return consequence


class TestGetGenomicContext:
    def test_not_found_when_ensembl_has_no_record(self, monkeypatch, fake_requests_get):
        monkeypatch.setattr(ucsc, "get_gene_consequence", lambda v: {"found": False})

        result = ucsc.get_genomic_context("rsFAKE")

        assert result == {"variant": "rsFAKE", "found": False}
        fake_requests_get.assert_not_called()

    def test_accepts_a_pre_fetched_consequence_and_skips_ensembl_call(self, monkeypatch, fake_requests_get):
        called = {"ensembl": False}
        monkeypatch.setattr(ucsc, "get_gene_consequence", lambda v: called.update({"ensembl": True}))
        fake_requests_get.return_value = _fake_response(_track_response(ucsc.PHYLOP_TRACK, 1.0))

        consequence = {"found": True, "chrom": "17", "start": 100, "end": 100}
        ucsc.get_genomic_context("rs28897696", consequence=consequence)

        assert called["ensembl"] is False

    def test_maps_phylop_and_phastcons_scores(self, fake_ensembl_found, fake_requests_get):
        fake_requests_get.side_effect = [
            _fake_response(_track_response(ucsc.PHYLOP_TRACK, 6.179, start=43063902, end=43063903)),
            _fake_response(_track_response(ucsc.PHASTCONS_TRACK, 1, start=43063902, end=43063903)),
        ]

        result = ucsc.get_genomic_context("rs28897696")

        assert result == {
            "variant": "rs28897696",
            "found": True,
            "phylop": 6.179,
            "phastcons": 1,
            "chrom": "17",
            "start": 43063903,
            "end": 43063903,
        }

    def test_converts_ensembl_snv_coordinates_to_ucsc_0based_half_open(self, fake_ensembl_found, fake_requests_get):
        fake_requests_get.return_value = _fake_response(_track_response(ucsc.PHYLOP_TRACK, 1.0))

        ucsc.get_genomic_context("rs28897696")

        args, kwargs = fake_requests_get.call_args_list[0]
        assert kwargs["params"]["chrom"] == "chr17"
        assert kwargs["params"]["start"] == 43063902  # ensembl start (43063903) - 1
        assert kwargs["params"]["end"] == 43063903

    def test_handles_ensembl_insertion_coordinates_as_a_single_point(self, monkeypatch, fake_requests_get):
        # Ensembl represents a pure insertion with end < start (zero-length ref span)
        monkeypatch.setattr(
            ucsc, "get_gene_consequence",
            lambda v: {"found": True, "chrom": "17", "start": 43057066, "end": 43057065},
        )
        fake_requests_get.return_value = _fake_response(_track_response(ucsc.PHYLOP_TRACK, 1.46348))

        ucsc.get_genomic_context("rs80357906")

        args, kwargs = fake_requests_get.call_args_list[0]
        assert kwargs["params"]["start"] == 43057065
        assert kwargs["params"]["end"] == 43057066

    def test_adds_chr_prefix_only_when_missing(self, fake_ensembl_found, fake_requests_get):
        fake_requests_get.return_value = _fake_response(_track_response(ucsc.PHYLOP_TRACK, 1.0))

        ucsc.get_genomic_context("rs28897696")

        _, kwargs = fake_requests_get.call_args_list[0]
        assert kwargs["params"]["chrom"] == "chr17"

    def test_no_coverage_at_locus_returns_none_score_not_error(self, fake_ensembl_found, fake_requests_get):
        fake_requests_get.side_effect = [
            _fake_response({ucsc.PHYLOP_TRACK: []}),
            _fake_response({ucsc.PHASTCONS_TRACK: []}),
        ]

        result = ucsc.get_genomic_context("rs28897696")

        assert result["found"] is True
        assert result["phylop"] is None
        assert result["phastcons"] is None


class TestErrorHandling:
    def test_network_failure_raises_runtime_error(self, fake_ensembl_found, fake_requests_get, caplog):
        fake_requests_get.side_effect = requests.exceptions.ConnectionError("simulated failure")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="UCSC query failed"):
                ucsc.get_genomic_context("rs28897696")

        assert "ucsc query failed" in caplog.text

    def test_network_failure_preserves_original_exception(self, fake_ensembl_found, fake_requests_get):
        original = requests.exceptions.Timeout("simulated timeout")
        fake_requests_get.side_effect = original

        with pytest.raises(RuntimeError) as exc_info:
            ucsc.get_genomic_context("rs28897696")

        assert exc_info.value.__cause__ is original

    def test_http_error_status_raises_runtime_error(self, fake_ensembl_found, fake_requests_get):
        resp = _fake_response({})
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        fake_requests_get.return_value = resp

        with pytest.raises(RuntimeError, match="UCSC query failed"):
            ucsc.get_genomic_context("rs28897696")

    def test_api_error_body_raises_runtime_error(self, fake_ensembl_found, fake_requests_get, caplog):
        fake_requests_get.return_value = _fake_response(
            {"error": "can not find track=bogus name for endpoint", "statusCode": 400}
        )

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="returned an error"):
                ucsc.get_genomic_context("rs28897696")

        assert "ucsc api error" in caplog.text

    def test_unexpected_shape_raises_runtime_error(self, fake_ensembl_found, fake_requests_get, caplog):
        fake_requests_get.return_value = _fake_response({ucsc.PHYLOP_TRACK: [{"no_value_field": True}]})

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="unexpected response shape"):
                ucsc.get_genomic_context("rs28897696")

    def test_non_json_response_raises_runtime_error(self, fake_ensembl_found, fake_requests_get):
        resp = MagicMock()
        resp.json.side_effect = ValueError("no JSON")
        resp.text = "<html>error</html>"
        fake_requests_get.return_value = resp

        with pytest.raises(RuntimeError, match="unexpected response"):
            ucsc.get_genomic_context("rs28897696")

    def test_ensembl_lookup_failure_propagates(self, monkeypatch, fake_requests_get):
        def raise_error(v):
            raise RuntimeError("Ensembl VEP call failed for variant 'rs28897696'")

        monkeypatch.setattr(ucsc, "get_gene_consequence", raise_error)

        with pytest.raises(RuntimeError, match="Ensembl VEP call failed"):
            ucsc.get_genomic_context("rs28897696")

        fake_requests_get.assert_not_called()


@pytest.mark.integration
class TestGetGenomicContextIntegration:
    def test_known_variant_returns_conservation_scores(self):
        result = ucsc.get_genomic_context("rs28897696")

        assert result["found"] is True
        assert result["phylop"] is not None
        assert result["phastcons"] is not None

    def test_unknown_rsid_returns_not_found(self):
        result = ucsc.get_genomic_context("rs999999999999999")
        assert result == {"variant": "rs999999999999999", "found": False}
