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

TODO(day-12): load_dataset + eval_retrieval + eval_classification + a JSON report.
TODO(day-13): eval_generation_judge + calibration tracking.
TODO(day-14): eval_robustness + statistical (multi-run) variance.
"""

import argparse
import json
from pathlib import Path

DATASET = Path(__file__).parent / "golden_dataset.jsonl"
REPORTS = Path(__file__).parent / "reports"


def load_dataset() -> list[dict]:
    """Read golden_dataset.jsonl into a list of example dicts."""
    raise NotImplementedError("TODO(day-12): parse the JSONL file")


def eval_retrieval(dataset: list[dict], k: int = 5) -> dict:
    """recall@k + MRR: how often the right criteria/source appears in the top-k."""
    raise NotImplementedError("TODO(day-12): run retrieval per example, compute metrics")


def eval_classification(dataset: list[dict]) -> dict:
    """3-class accuracy vs ClinVar, harm-weighted error cost, abstention correctness."""
    raise NotImplementedError("TODO(day-12): run the agent, score vs gold_label + expected_behavior")


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
