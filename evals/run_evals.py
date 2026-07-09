"""Eval harness runner — THE headline of the project.

Grades the agent (the system under test) against the golden dataset. Two kinds
of metrics, evaluated separately (a key point: retrieval and generation fail
differently, so measure them apart):

  RETRIEVAL (deterministic, cheap — run constantly):
    recall@k, MRR — did the right ACMG criteria get retrieved?

  CLASSIFICATION (vs ClinVar ground truth):
    accuracy on the 3-class label, with ASYMMETRIC (harm-weighted) error cost,
    and abstention correctness on VUS/adversarial cases.

  GENERATION (LLM-as-judge — costs tokens; calibrate it, see Day 13):
    faithfulness (is the justification grounded?), and judge–human agreement (kappa).

  ROBUSTNESS (Day 14):
    perturb inputs, measure whether quality holds.

Usage (planned):
    python evals/run_evals.py --skip-generation     # retrieval+classification only (cheap)
    python evals/run_evals.py                        # full run

TODO(day-12): eval_retrieval + a JSON report.
TODO(day-13): eval_generation_judge + calibration tracking.
TODO(day-14): eval_robustness + statistical (multi-run) variance.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.variant_audit import graph  # noqa: E402

DATASET = Path(__file__).parent / "golden_dataset.jsonl"
REPORTS = Path(__file__).parent / "reports"

LABELS = ("P", "VUS", "B")

# 5-tier LLM output -> collapsed 3-class gold-label vocabulary (see DATASET.md
# "Defining 'correct'" -- LP/P and LB/B disagreement doesn't change management).
TIER_TO_LABEL = {
    "pathogenic": "P",
    "likely pathogenic": "P",
    "uncertain significance": "VUS",
    "likely benign": "B",
    "benign": "B",
}

# Asymmetric harm cost [gold][predicted] (see DATASET.md "Defining 'correct'":
# calling a true P "benign" is the catastrophic error). Diagonal is always 0.
#   P->B  = 10 : false reassurance on a real pathogenic variant -- no monitoring/
#                intervention when it was warranted. The worst outcome.
#   P->VUS =  3 : misses the actionable call, but doesn't assert safety and
#                 still flags for human follow-up.
#   B->P  =  5 : false alarm on a real benign variant -- unnecessary anxiety/
#                intervention, serious but a follow-up workup can catch it.
#   B->VUS =  1 : mild over-caution.
#   VUS->P/VUS->B = 2 : overconfident call on genuinely ambiguous evidence --
#                       premature either direction, but not the worst case.
HARM_COST = {
    "P": {"P": 0, "VUS": 3, "B": 10},
    "B": {"B": 0, "VUS": 1, "P": 5},
    "VUS": {"VUS": 0, "P": 2, "B": 2},
}


def load_dataset() -> list[dict]:
    """Read golden_dataset.jsonl into a list of example dicts."""
    with DATASET.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def eval_retrieval(dataset: list[dict], k: int = 5) -> dict:
    """recall@k + MRR: how often the right criteria/source appears in the top-k."""
    raise NotImplementedError("TODO(day-12): run retrieval per example, compute metrics")


def _predicted_label(classification_text: str) -> str:
    """Pull the 3-class label out of graph.ask()'s 'Classification: <tier>' line.

    An unparseable/unrecognized tier is treated as a non-confident call (VUS)
    rather than silently miscounted -- same standard the abstention rubric
    already holds the model to.
    """
    for line in classification_text.splitlines():
        if line.strip().lower().startswith("classification:"):
            tier = line.split(":", 1)[1].strip().lower()
            return TIER_TO_LABEL.get(tier, "VUS")
    return "VUS"


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval (95% by default) for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def eval_classification(dataset: list[dict]) -> dict:
    """3-class accuracy vs ClinVar, harm-weighted error cost, abstention correctness."""
    confusion = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    abstain_rows = [row for row in dataset if row.get("expected_behavior") == "abstain"]
    abstain_correct = 0
    correct = 0
    total_cost = 0
    rows_out = []
    errors = []

    for row in dataset:
        variant = row["variant"]
        gold = row["gold_label"]
        try:
            result = graph.ask(variant)
            predicted = _predicted_label(result.get("classification", ""))
        except Exception as exc:  # noqa: BLE001 - live tool/network calls, isolate one bad row
            errors.append({"id": row["id"], "variant": variant, "error": str(exc)})
            continue

        is_correct = predicted == gold
        confusion[gold][predicted] += 1
        correct += is_correct
        total_cost += HARM_COST[gold][predicted]
        if row.get("expected_behavior") == "abstain":
            abstain_correct += predicted == "VUS"
        rows_out.append(
            {"id": row["id"], "variant": variant, "gold": gold, "predicted": predicted, "correct": is_correct}
        )

    n = len(rows_out)
    accuracy = correct / n if n else 0.0
    ci_lo, ci_hi = wilson_ci(correct, n)

    return {
        "n": n,
        "n_errors": len(errors),
        "accuracy": accuracy,
        "accuracy_ci_95": {"lo": ci_lo, "hi": ci_hi},
        "harm_weighted_cost_mean": total_cost / n if n else 0.0,
        "harm_weighted_cost_total": total_cost,
        "confusion_matrix": confusion,
        "abstention": {
            "n": len(abstain_rows),
            "correct": abstain_correct,
            "accuracy": abstain_correct / len(abstain_rows) if abstain_rows else 0.0,
        },
        "rows": rows_out,
        "errors": errors,
    }


def eval_generation_judge(dataset: list[dict]) -> dict:
    """LLM-as-judge faithfulness/relevance. Track judge–human agreement (kappa)."""
    raise NotImplementedError("TODO(day-13): judge outputs; compare to your hand-labels")


def eval_robustness(dataset: list[dict]) -> dict:
    """Perturb inputs (paraphrase/noise/formatting); measure quality delta."""
    raise NotImplementedError("TODO(day-14): perturb, re-run, report stability")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    # TODO: run the sections, assemble a report dict, write REPORTS/eval_<ts>.json,
    #       print a scorecard. Treat reports like test results — keep the history.
    raise NotImplementedError("TODO(day-12+): orchestrate the eval run and write a report")


if __name__ == "__main__":
    main()
