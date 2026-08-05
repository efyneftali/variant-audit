"""Tests for the groundedness calibration harness (VA-35, step 4).

The load-bearing invariants are: (1) the labeler is *blind* -- it never renders a
case's id, provenance, or intended answer; (2) labels round-trip and labeling is
resumable; (3) an injected defect is genuinely ungrounded (its fabricated code is
absent from the criteria text). Interactive I/O isn't tested -- the pure functions
under it are.
"""

import json
from types import SimpleNamespace

import evals.calibration.build_cases as build_cases
import evals.calibration.label as label


def _real_case():
    return {
        "id": "gv-001",
        "variant": "rs123",
        "gene": "BRCA1",
        "difficulty": "easy",
        "classification": "Classification: Pathogenic\nCriteria used: PVS1",
        "evidence_text": "-- gnomAD --\nfrequency 0.00005",
        "criteria_text": "[acmg] PVS1 applies to null variants.",
        "criteria_sources": ["acmg_criteria.md"],
        "_provenance": {"kind": "real"},
    }


class TestBlindness:
    def test_blind_view_excludes_id_and_provenance(self):
        view = label.blind_view(_real_case())
        assert "id" not in view
        assert "_provenance" not in view
        assert "intended" not in view
        # but the genetics context the expert needs IS shown
        assert view["variant"] == "rs123"
        assert view["classification"].startswith("Classification: Pathogenic")

    def test_injected_case_view_hides_intended_verdict(self):
        injected = build_cases.inject_fabricated_code(_real_case())
        view = label.blind_view(injected)
        blob = json.dumps(view)
        # nothing in the rendered view reveals it's a planted ungrounded case
        assert "intended" not in blob
        assert "fab" not in blob  # the id suffix must not leak
        assert injected["_provenance"]["intended"] == "ungrounded"  # still tracked in the file

    def test_render_never_prints_the_id(self):
        injected = build_cases.inject_fabricated_code(_real_case())  # id ends in '#fab'
        rendered = label.render_case(injected, position=1, total=10)
        assert "#fab" not in rendered
        assert injected["id"] not in rendered


class TestInjectedDefect:
    def test_fabricated_code_is_absent_from_criteria(self):
        case = _real_case()  # criteria mention only PVS1
        injected = build_cases.inject_fabricated_code(case)
        code = injected["_provenance"]["fabricated_code"]
        # the injected code really is missing from the criteria -> genuinely ungrounded
        assert code not in build_cases.graph._acmg_codes(case["criteria_text"])
        assert code in injected["classification"]

    def test_absent_code_picks_a_missing_code(self):
        assert build_cases.absent_code("[acmg] PVS1 and BA1 only.") not in ("PVS1", "BA1")

    def test_provenance_marks_intended_ungrounded(self):
        injected = build_cases.inject_fabricated_code(_real_case())
        assert injected["_provenance"]["kind"] == "injected_fabricated_code"
        assert injected["_provenance"]["intended"] == "ungrounded"
        assert injected["_provenance"]["from_id"] == "gv-001"


class TestLabelRoundTrip:
    def test_append_and_load(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        rec = label.make_label_record(_real_case(), "grounded", "PVS1 is present", "", "efy")
        label.append_label(path, rec)
        loaded = label.load_labels(path)
        assert loaded["gv-001"]["label"] == "grounded"
        assert loaded["gv-001"]["rationale"] == "PVS1 is present"
        assert loaded["gv-001"]["labeler"] == "efy"

    def test_pending_skips_already_labeled(self, tmp_path):
        cases = [_real_case(), {**_real_case(), "id": "gv-002"}]
        labels = {"gv-001": {"id": "gv-001", "label": "grounded"}}
        todo = label.pending(cases, labels, seed=1)
        assert [c["id"] for c in todo] == ["gv-002"]

    def test_pending_order_is_deterministic_for_a_seed(self, tmp_path):
        cases = [{**_real_case(), "id": f"gv-{i:03d}"} for i in range(10)]
        first = [c["id"] for c in label.pending(cases, {}, seed=7)]
        second = [c["id"] for c in label.pending(cases, {}, seed=7)]
        assert first == second


class TestBuildRealCase:
    def test_freezes_judge_facing_artifact_without_a_verdict(self, monkeypatch):
        # stub the agent nodes so no LLM/Qdrant is needed
        monkeypatch.setattr(build_cases.graph, "gather_evidence", lambda s: {"evidence": {"clinvar": {"matches": []}}})
        monkeypatch.setattr(
            build_cases.graph, "retrieve_criteria",
            lambda s: {"criteria": [SimpleNamespace(text="PVS1 applies.", source="acmg.md", score=0.9)]},
        )
        monkeypatch.setattr(
            build_cases.graph, "classify",
            lambda s: {"classification": "Classification: Pathogenic\nCriteria used: PVS1"},
        )
        case = build_cases.build_real_case({"id": "gv-001", "variant": "rs123", "gene": "BRCA1", "difficulty": "easy"})

        assert case["_provenance"] == {"kind": "real"}
        assert case["classification"].startswith("Classification: Pathogenic")
        assert "PVS1 applies." in case["criteria_text"]
        # a case must never carry a groundedness verdict -- that's what makes it blind
        assert "grounded" not in case
        assert "label" not in case
