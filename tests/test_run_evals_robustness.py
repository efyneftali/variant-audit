"""Tests for eval_robustness's diff/flip logic (VA-13). The perturbation and
LLM calls are stubbed -- what's worth testing here is the accuracy-delta math
and flip detection, not the paraphrase table (see test_perturb.py) or the
agent loop itself (needs a real model).
"""
import evals.run_evals as run_evals


def _report(rows: list[dict], *, llm_calls: int = 1, wall_seconds: float = 1.0) -> dict:
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "harm_weighted_cost_mean": 0.5,
        "abstention": {"n": 1, "correct": 1, "accuracy": 1.0},
        "rows": rows,
        "llm_calls": llm_calls,
        "llm_seconds": float(llm_calls),
        "wall_seconds": wall_seconds,
    }


def _row(id_, correct):
    return {"id": id_, "variant": f"rs{id_}", "gold": "P", "predicted": "P" if correct else "VUS", "correct": correct}


def test_reports_accuracy_delta_between_clean_and_perturbed(monkeypatch):
    calls = iter([
        _report([_row("A", True), _row("B", True)]),   # baseline: 1.0
        _report([_row("A", True), _row("B", False)]),  # perturbed: 0.5
    ])
    monkeypatch.setattr(run_evals, "eval_classification", lambda dataset: next(calls))
    monkeypatch.setattr(run_evals.replay, "install", lambda: None)
    monkeypatch.setattr(run_evals.perturb, "install", lambda: None)

    result = run_evals.eval_robustness(dataset=[None])

    assert result["baseline_accuracy"] == 1.0
    assert result["perturbed_accuracy"] == 0.5
    assert result["accuracy_delta"] == -0.5


def test_flags_rows_that_flip_correct_to_wrong(monkeypatch):
    calls = iter([
        _report([_row("A", True), _row("B", True), _row("C", False)]),
        _report([_row("A", True), _row("B", False), _row("C", True)]),
    ])
    monkeypatch.setattr(run_evals, "eval_classification", lambda dataset: next(calls))
    monkeypatch.setattr(run_evals.replay, "install", lambda: None)
    monkeypatch.setattr(run_evals.perturb, "install", lambda: None)

    result = run_evals.eval_robustness(dataset=[None])

    assert result["flips_correct_to_wrong"] == ["B"]
    assert result["flips_wrong_to_correct"] == ["C"]


def test_no_delta_when_nothing_changes(monkeypatch):
    same = [_row("A", True), _row("B", False)]
    calls = iter([_report(same), _report(same)])
    monkeypatch.setattr(run_evals, "eval_classification", lambda dataset: next(calls))
    monkeypatch.setattr(run_evals.replay, "install", lambda: None)
    monkeypatch.setattr(run_evals.perturb, "install", lambda: None)

    result = run_evals.eval_robustness(dataset=[None])

    assert result["accuracy_delta"] == 0.0
    assert result["flips_correct_to_wrong"] == []
    assert result["flips_wrong_to_correct"] == []


def test_restores_clean_replay_after_the_perturbed_pass(monkeypatch):
    install_calls = []
    monkeypatch.setattr(run_evals, "eval_classification", lambda dataset: _report([_row("A", True)]))
    monkeypatch.setattr(run_evals.replay, "install", lambda: install_calls.append("replay"))
    monkeypatch.setattr(run_evals.perturb, "install", lambda: install_calls.append("perturb"))

    run_evals.eval_robustness(dataset=[None])

    # replay (baseline) -> perturb (perturbed pass) -> replay again (cleanup)
    assert install_calls == ["replay", "perturb", "replay"]
