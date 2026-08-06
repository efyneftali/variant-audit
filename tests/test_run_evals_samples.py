"""Tests for the multi-sample generation-variance loop (run_evals.eval_classification_multi).

The aggregation math is what's worth testing -- stability partitioning, the two
accuracy numbers, and that Wilson stays at the true row count. The k underlying
runs are stubbed (no LLM), each returning a scripted single-run report, so we can
assert on exact predictions.
"""

import evals.run_evals as run_evals


def _single_run_report(rows: list[dict]) -> dict:
    """Minimal eval_classification-shaped report for the fields _multi reads."""
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "harm_weighted_cost_mean": 0.0,
        "abstention": {"n": 0, "correct": 0, "accuracy": 1.0},
        "n_errors": 0,
        "n_unparseable": 0,
        "rows": rows,
    }


def _row(id_, gold, predicted):
    return {"id": id_, "variant": f"rs{id_}", "gold": gold, "predicted": predicted, "correct": predicted == gold}


def _stub_runs(monkeypatch, scripted_runs):
    """Make eval_classification return the next scripted run on each call."""
    it = iter(scripted_runs)
    monkeypatch.setattr(
        run_evals, "eval_classification", lambda dataset, temperature=None: _single_run_report(next(it))
    )


def test_partitions_stable_and_unstable_rows(monkeypatch):
    # 3 variants across 3 samples:
    #   A: P/P/P  gold P  -> stable-correct
    #   B: P/P/P  gold VUS -> stable-wrong
    #   C: VUS/P/VUS gold VUS -> unstable (coin-flip)
    _stub_runs(monkeypatch, [
        [_row("A", "P", "P"), _row("B", "VUS", "P"), _row("C", "VUS", "VUS")],
        [_row("A", "P", "P"), _row("B", "VUS", "P"), _row("C", "VUS", "P")],
        [_row("A", "P", "P"), _row("B", "VUS", "P"), _row("C", "VUS", "VUS")],
    ])

    result = run_evals.eval_classification_multi(dataset=[None], k=3)

    st = result["stability"]
    assert st["stable_correct"] == 1
    assert st["stable_wrong"] == 1 and st["stable_wrong_ids"] == ["B"]
    assert st["unstable"] == 1 and st["unstable_ids"] == ["C"]


def test_mean_sd_and_majority_vote_differ(monkeypatch):
    # Two runs, same two variants:
    #   run1 accuracy 1.0, run2 accuracy 0.5 -> mean 0.75, sd > 0
    #   A stable-correct (P/P); B flips (VUS then P) but modal over 2 ties ->
    #   Counter first-seen keeps VUS (correct), so majority-vote counts B correct.
    _stub_runs(monkeypatch, [
        [_row("A", "P", "P"), _row("B", "VUS", "VUS")],
        [_row("A", "P", "P"), _row("B", "VUS", "P")],
    ])

    result = run_evals.eval_classification_multi(dataset=[None], k=2)

    assert result["per_run_accuracies"] == [1.0, 0.5]
    assert result["mean_accuracy"] == 0.75
    assert result["accuracy_sd"] > 0.0
    # majority vote: A correct, B modal=VUS (tie broken to first-seen) = correct -> 1.0
    assert result["majority_vote_accuracy"] == 1.0
    # Wilson uses the true row count (2), not k*n (4).
    assert result["n"] == 2


def test_predictions_preserved_for_audit(monkeypatch):
    _stub_runs(monkeypatch, [
        [_row("A", "P", "P")],
        [_row("A", "P", "VUS")],
        [_row("A", "P", "P")],
    ])

    result = run_evals.eval_classification_multi(dataset=[None], k=3)

    (item,) = result["per_item"]
    assert item["predictions"] == ["P", "VUS", "P"]
    assert item["correct_count"] == 2
    assert item["majority_label"] == "P"


def test_single_sample_has_zero_sd(monkeypatch):
    _stub_runs(monkeypatch, [[_row("A", "P", "P")]])

    result = run_evals.eval_classification_multi(dataset=[None], k=1)

    assert result["accuracy_sd"] == 0.0
    assert result["mean_accuracy"] == 1.0


def test_forwards_temperature_to_every_underlying_run(monkeypatch):
    """VA-41: the same temperature must reach all k samples, not just the first --
    otherwise the measured SD is jitter across a mix of settings, not at one T."""
    seen_temperatures = []

    def fake_eval_classification(dataset, temperature=None):
        seen_temperatures.append(temperature)
        return _single_run_report([_row("A", "P", "P")])

    monkeypatch.setattr(run_evals, "eval_classification", fake_eval_classification)

    run_evals.eval_classification_multi(dataset=[None], k=3, temperature=0.3)

    assert seen_temperatures == [0.3, 0.3, 0.3]
