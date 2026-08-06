"""Tests for the multi-sample generation-variance loop (run_evals.eval_classification_multi).

The aggregation math is what's worth testing -- stability partitioning, the two
accuracy numbers, and that Wilson stays at the true row count. The k underlying
runs are stubbed (no LLM), each returning a scripted single-run report, so we can
assert on exact predictions.
"""

import time

import evals.run_evals as run_evals
from src.variant_audit import graph


class TestMeasureLlmCalls:
    """_measure_llm_calls (VA-41 part 2) wraps graph.llm.complete to count calls
    and time them -- the cost/latency instrumentation eval_classification relies
    on. Pure, no agent-loop dependency, so it's worth testing directly."""

    def test_counts_calls_and_restores_original_after_the_block(self, monkeypatch):
        real = lambda *a, **kw: "ok"  # noqa: E731 - trivial stand-in, not the real llm.complete
        monkeypatch.setattr(graph.llm, "complete", real)

        with run_evals._measure_llm_calls() as stats:
            graph.llm.complete("p1")
            graph.llm.complete("p2")
            assert stats["n_calls"] == 2

        assert graph.llm.complete is real  # unwrapped again once the block exits

    def test_sums_wall_clock_time_across_calls(self, monkeypatch):
        def slow(*a, **kw):
            time.sleep(0.01)
            return "ok"

        monkeypatch.setattr(graph.llm, "complete", slow)

        with run_evals._measure_llm_calls() as stats:
            graph.llm.complete("p1")
            graph.llm.complete("p2")

        assert stats["n_calls"] == 2
        assert stats["total_seconds"] >= 0.02

    def test_returns_the_wrapped_call_result_unchanged(self, monkeypatch):
        monkeypatch.setattr(graph.llm, "complete", lambda *a, **kw: "the classification text")

        with run_evals._measure_llm_calls():
            result = graph.llm.complete("p")

        assert result == "the classification text"


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
        "llm_calls": n,
        "llm_seconds": float(n),
        "wall_seconds": float(n),
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


def test_aggregates_cost_and_latency_across_runs(monkeypatch):
    """VA-41 part 2: best-of-k cost/latency must sum the k underlying runs'
    llm_calls/wall_seconds, not just report one run's figures."""
    _stub_runs(monkeypatch, [
        [_row("A", "P", "P")],
        [_row("A", "P", "P")],
        [_row("A", "P", "P")],
    ])
    # _single_run_report(rows) sets llm_calls = wall_seconds = llm_seconds = len(rows) = 1 per run

    result = run_evals.eval_classification_multi(dataset=[None], k=3)

    assert result["llm_calls_total"] == 3
    assert result["wall_seconds_total"] == 3.0
    assert result["wall_seconds_per_run"] == [1.0, 1.0, 1.0]
    # 3 calls total / (k=3 * n_items=1) = 1 call per variant per sample
    assert result["llm_calls_per_variant_mean"] == 1.0
    assert result["wall_seconds_per_variant_mean"] == 1.0


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
