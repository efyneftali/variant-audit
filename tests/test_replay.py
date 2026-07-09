"""Tests for evals/replay.py -- the record-and-replay layer (VA-39).

Replay must be offline and loud about gaps; record must retry rather than accept
a hole; install() must redirect the graph module without touching src/.
"""

from unittest.mock import MagicMock

import pytest

from evals import replay
from src.variant_audit import graph


@pytest.fixture
def fixtures_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "FIXTURES", tmp_path)
    return tmp_path


class TestFixtureIO:
    def test_save_then_load_roundtrips(self, fixtures_dir):
        payload = {"variant": "rsTEST", "found": True, "matches": []}
        replay.save_fixture("clinvar", "rsTEST", payload)

        assert replay.load_fixture("clinvar", "rsTEST") == payload

    def test_missing_fixture_raises_loudly(self, fixtures_dir):
        with pytest.raises(replay.FixtureMissing, match="rsNOPE"):
            replay.load_fixture("gnomad", "rsNOPE")

    def test_save_is_keyed_by_tool_and_variant(self, fixtures_dir):
        replay.save_fixture("clinvar", "rsTEST", {"tool": "clinvar"})
        replay.save_fixture("gnomad", "rsTEST", {"tool": "gnomad"})

        assert replay.load_fixture("clinvar", "rsTEST")["tool"] == "clinvar"
        assert replay.load_fixture("gnomad", "rsTEST")["tool"] == "gnomad"


class TestReplayInstall:
    def test_install_redirects_graph_tools_to_fixtures(self, fixtures_dir):
        replay.save_fixture("clinvar", "rsTEST", {"variant": "rsTEST", "found": True})
        with replay.replay_tools():
            assert graph.get_clinvar_record("rsTEST") == {"variant": "rsTEST", "found": True}

    def test_replay_ignores_consequence_kwarg(self, fixtures_dir):
        # ucsc/alphamissense take consequence=; replay keys on the variant alone.
        replay.save_fixture("ucsc", "rsTEST", {"variant": "rsTEST", "found": False})
        with replay.replay_tools():
            assert graph.get_genomic_context("rsTEST", consequence={"anything": 1})["found"] is False

    def test_context_manager_restores_the_originals(self, fixtures_dir):
        original = graph.get_clinvar_record
        with replay.replay_tools():
            assert graph.get_clinvar_record is not original
        assert graph.get_clinvar_record is original


class TestRecordRetry:
    def test_retries_transient_failure_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(replay.time, "sleep", lambda *_: None)  # don't actually back off
        flaky = MagicMock(side_effect=[RuntimeError("rate limited"), {"ok": True}])

        result = replay._call_with_retry(flaky, "rsTEST", desc="clinvar rsTEST")

        assert result == {"ok": True}
        assert flaky.call_count == 2

    def test_gives_up_after_max_retries(self, monkeypatch):
        monkeypatch.setattr(replay.time, "sleep", lambda *_: None)
        always_fails = MagicMock(side_effect=RuntimeError("down"))

        with pytest.raises(RuntimeError, match="record failed after"):
            replay._call_with_retry(always_fails, "rsTEST", desc="clinvar rsTEST")

        assert always_fails.call_count == replay.RECORD_RETRIES

    def test_record_variant_writes_all_five_fixtures(self, fixtures_dir, monkeypatch):
        monkeypatch.setattr(replay.time, "sleep", lambda *_: None)
        # patch the real callables record_variant closes over
        monkeypatch.setattr(replay, "get_clinvar_record", lambda v: {"t": "clinvar"})
        monkeypatch.setattr(replay, "get_allele_frequency", lambda v: {"t": "gnomad"})
        monkeypatch.setattr(replay, "get_gene_consequence", lambda v: {"t": "ensembl"})
        monkeypatch.setattr(replay, "get_genomic_context", lambda v, consequence=None: {"t": "ucsc"})
        monkeypatch.setattr(replay, "get_alphamissense_score", lambda v, consequence=None: {"t": "alphamissense"})

        replay.record_variant("rsTEST")

        for tool in replay._TOOLS:
            assert replay.load_fixture(tool, "rsTEST")["t"] == tool
