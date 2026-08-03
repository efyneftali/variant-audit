"""Tests for the judge calibration scorer (VA-35, stretch).

The math (Cohen's kappa, the confusion split) and the selection rule (cheapest
candidate clearing the threshold) are what matter and what's easy to get subtly
wrong. The paid API path is never exercised -- run_candidate is driven with a
stubbed groundedness_verdict so no model is called.
"""

import math

import evals.calibration.score_judges as sj


class TestCohenKappa:
    def test_perfect_agreement_is_one(self):
        a = ["grounded", "ungrounded", "grounded"]
        assert sj.cohen_kappa(a, a) == 1.0

    def test_chance_level_is_about_zero(self):
        # judge alternates independently of the human -> kappa near 0
        human = ["grounded", "ungrounded"] * 10
        judge = ["grounded", "grounded", "ungrounded", "ungrounded"] * 5
        assert abs(sj.cohen_kappa(human, judge)) < 0.15

    def test_total_disagreement_is_negative(self):
        human = ["grounded", "ungrounded", "grounded", "ungrounded"]
        judge = ["ungrounded", "grounded", "ungrounded", "grounded"]
        assert sj.cohen_kappa(human, judge) < 0

    def test_single_label_everywhere_edge_case(self):
        # both rated everything grounded: pe == 1; perfect -> 1.0, not NaN
        assert sj.cohen_kappa(["grounded"] * 5, ["grounded"] * 5) == 1.0

    def test_empty_is_nan(self):
        assert math.isnan(sj.cohen_kappa([], []))


class TestConfusion:
    def test_splits_the_two_error_kinds(self):
        human = ["grounded", "ungrounded", "grounded", "ungrounded"]
        judge = ["ungrounded", "grounded", "grounded", "ungrounded"]
        c = sj.confusion(human, judge)
        assert c["false_ungrounded"] == 1   # human grounded, judge ungrounded
        assert c["missed_ungrounded"] == 1  # human ungrounded, judge grounded
        assert c["agree_grounded"] == 1
        assert c["agree_ungrounded"] == 1


class TestPickJudge:
    def _score(self, name, kappa):
        return {"candidate": name, "kappa": kappa, "provider": sj.CANDIDATES[name][0],
                "model": sj.CANDIDATES[name][1], "paid": sj.CANDIDATES[name][2], "n": 10,
                "accuracy": 0.9, "confusion": {}}

    def test_prefers_cheapest_that_clears(self):
        # local and haiku both clear; local is cheaper -> pick local
        scores = [self._score("local", 0.72), self._score("haiku", 0.88), self._score("sonnet", 0.91)]
        assert sj.pick_judge(scores, 0.6)["candidate"] == "local"

    def test_skips_below_threshold_for_cheaper(self):
        # local fails threshold, haiku clears -> haiku (not local)
        scores = [self._score("local", 0.41), self._score("haiku", 0.82), self._score("sonnet", 0.90)]
        assert sj.pick_judge(scores, 0.6)["candidate"] == "haiku"

    def test_none_when_nobody_clears(self):
        scores = [self._score("local", 0.2), self._score("haiku", 0.5)]
        assert sj.pick_judge(scores, 0.6) is None


class TestRunCandidateCaches:
    def test_stubbed_judge_writes_and_reuses_cache(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def fake_verdict(variant, classification, evidence_text, criteria_text, *, provider=None, model=None):
            calls["n"] += 1
            return (True, "ok") if "PVS1" in classification else (False, "bad")

        monkeypatch.setattr(sj.graph, "groundedness_verdict", fake_verdict)
        monkeypatch.setattr(sj, "RUNS_DIR", tmp_path)

        cases = [
            {"id": "c1", "variant": "rs1", "classification": "PVS1", "evidence_text": "", "criteria_text": ""},
            {"id": "c2", "variant": "rs2", "classification": "nope", "evidence_text": "", "criteria_text": ""},
        ]
        first = sj.run_candidate("local", cases, force=False)
        assert first["c1"]["label"] == "grounded"
        assert first["c2"]["label"] == "ungrounded"
        assert calls["n"] == 2

        # second run: everything cached -> no new judge calls
        second = sj.run_candidate("local", cases, force=False)
        assert calls["n"] == 2
        assert second["c1"]["label"] == "grounded"

    def test_force_rejudges(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        monkeypatch.setattr(sj.graph, "groundedness_verdict",
                            lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), (True, "ok"))[1])
        monkeypatch.setattr(sj, "RUNS_DIR", tmp_path)
        cases = [{"id": "c1", "variant": "rs1", "classification": "x", "evidence_text": "", "criteria_text": ""}]
        sj.run_candidate("local", cases, force=False)
        sj.run_candidate("local", cases, force=True)
        assert calls["n"] == 2  # forced re-judge, not served from cache


class TestScoreCandidate:
    def test_aligns_verdicts_to_human_order(self):
        verdicts = {"c1": {"label": "grounded"}, "c2": {"label": "ungrounded"}, "c3": {"label": "grounded"}}
        labeled = [("c1", "grounded"), ("c2", "ungrounded"), ("c3", "ungrounded")]
        s = sj.score_candidate("local", verdicts, labeled)
        assert s["n"] == 3
        assert abs(s["accuracy"] - 2 / 3) < 1e-9
        assert s["confusion"]["missed_ungrounded"] == 1


class TestRenderReport:
    def test_records_pick_and_kappa(self):
        scores = [sj.score_candidate("local", {"c1": {"label": "grounded"}}, [("c1", "grounded")])]
        pick = scores[0]
        md = sj.render_report(scores, pick, 0.6, n_labeled=1)
        assert "Selected judge: **local**" in md
        assert "κ" in md

    def test_flags_when_none_clear(self):
        scores = [sj.score_candidate("local", {"c1": {"label": "ungrounded"}}, [("c1", "grounded")])]
        md = sj.render_report(scores, None, 0.6, n_labeled=1)
        assert "No candidate cleared" in md
