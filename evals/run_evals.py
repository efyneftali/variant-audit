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

TODO(day-12): a JSON report tying eval_retrieval + eval_classification together.
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

# Sentinel for a classification() response the extractor genuinely could not
# read a tier out of. Never silently coerced into a real label (see
# _predicted_label) -- a parse failure is a different failure mode than a
# model abstention, and conflating them would hide harness bugs as VUS calls.
UNPARSEABLE = "UNPARSEABLE"

# 5-tier LLM output -> collapsed 3-class gold-label vocabulary (see DATASET.md
# "Defining 'correct'" -- LP/P and LB/B disagreement doesn't change management).
TIER_TO_LABEL = {
    "pathogenic": "P",
    "likely pathogenic": "P",
    "uncertain significance": "VUS",
    "likely benign": "B",
    "benign": "B",
}

# evidence_type -> the ACMG code(s) that evidence type should surface via RAG
# (see DATASET.md "Retrieval proxy answer key"). Rows don't carry a per-row
# "relevant criterion" label, so this stands in as the recall@k answer key.
# `conflicting_evidence_sources` (1 row) is deliberately absent: it names a
# disagreement between sources, not a specific criterion, so there's no
# single code to check for -- that row is excluded from retrieval scoring
# (still covered by eval_classification via its abstain label).
EVIDENCE_TYPE_TO_CRITERIA = {
    "frameshift/LoF": ["PVS1"],
    "population_frequency": ["BA1", "BS1"],
    "missense/computational": ["PP3", "BP4"],
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


def _retrieval_query(variant: str, clinvar_record: dict) -> str:
    """Mirror graph.retrieve_criteria's query construction exactly (graph.py
    ~line 78-85) -- the eval has to search with the same query the agent
    actually uses, or recall@k measures a different pipeline than production."""
    matches = clinvar_record.get("matches", [])
    if matches:
        significances = ", ".join(m["clinical_significance"] for m in matches)
        hgvs_names = ", ".join(m["hgvs"] for m in matches)
        return f"{hgvs_names} {significances}"
    return variant


def _covers_criterion(chunk_text: str, expected_codes: list[str]) -> bool:
    lowered = chunk_text.lower()
    return any(code.lower() in lowered for code in expected_codes)


def eval_retrieval(dataset: list[dict], k: int = 5) -> dict:
    """recall@k + MRR: did a top-k chunk cover the expected ACMG criterion?

    Answer key is the evidence_type -> ACMG code proxy in
    EVIDENCE_TYPE_TO_CRITERIA (no per-row "relevant criterion" label exists).
    Rows whose evidence_type isn't in that map are skipped, not scored as
    misses -- there's no defined answer to check against.
    """
    scored_rows = [row for row in dataset if row.get("evidence_type") in EVIDENCE_TYPE_TO_CRITERIA]
    skipped_ids = [row["id"] for row in dataset if row.get("evidence_type") not in EVIDENCE_TYPE_TO_CRITERIA]

    hits = 0
    reciprocal_ranks = []
    rows_out = []

    for row in scored_rows:
        variant = row["variant"]
        expected = EVIDENCE_TYPE_TO_CRITERIA[row["evidence_type"]]
        clinvar_record = graph.get_clinvar_record(variant)
        query = _retrieval_query(variant, clinvar_record)
        chunks = graph.semantic_search(query, top_k=k)

        rank = next(
            (i for i, chunk in enumerate(chunks, start=1) if _covers_criterion(chunk.text, expected)),
            None,
        )
        hit = rank is not None
        hits += hit
        reciprocal_ranks.append(1 / rank if hit else 0.0)
        rows_out.append(
            {
                "id": row["id"],
                "variant": variant,
                "evidence_type": row["evidence_type"],
                "expected_criteria": expected,
                "hit": hit,
                "rank": rank,
            }
        )

    n = len(scored_rows)
    return {
        "k": k,
        "n": n,
        "n_skipped": len(skipped_ids),
        "skipped_ids": skipped_ids,
        "recall_at_k": hits / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
        "rows": rows_out,
    }


def _extract_verdict_line_tier(classification_text: str) -> str | None:
    """Strict pass: a line that (after stripping markdown emphasis) starts
    with 'classification:', same shape classify.SYSTEM_PROMPT asks for."""
    for line in classification_text.splitlines():
        stripped = line.strip().strip("*").strip()
        if stripped.lower().startswith("classification:"):
            tier = stripped.split(":", 1)[1].strip().strip("*").rstrip(".").strip().lower()
            if tier in TIER_TO_LABEL:
                return tier
    return None


def _extract_fallback_tier(classification_text: str) -> str | None:
    """Bounded fallback for when the model doesn't emit the verdict line
    cleanly (markdown wrapping, extra preamble, wrong case): scan the whole
    response for a known tier phrase. Longest phrases first so 'likely
    pathogenic' isn't mis-caught as a bare 'pathogenic' substring hit."""
    lowered = classification_text.lower()
    for phrase in sorted(TIER_TO_LABEL, key=len, reverse=True):
        if phrase in lowered:
            return phrase
    return None


def _predicted_label(classification_text: str) -> str:
    """Pull the 3-class label out of graph.ask()'s classification text.

    Two bounded, deliberate passes (strict verdict line, then a whole-text
    phrase scan) -- not an open-ended regex hoping to catch prose. If neither
    finds a known tier, return UNPARSEABLE rather than guessing: a parse
    failure must stay visible to the caller, not get silently scored as a
    VUS call the model never made.
    """
    tier = _extract_verdict_line_tier(classification_text) or _extract_fallback_tier(classification_text)
    if tier is None:
        return UNPARSEABLE
    return TIER_TO_LABEL[tier]


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
    """3-class accuracy vs ClinVar, harm-weighted error cost, abstention correctness.

    UNPARSEABLE predictions (see _predicted_label) stay out of the 3x3
    confusion matrix -- they're not a real P/VUS/B call, adding a 4th
    row/column would corrupt its semantics -- but they still count against
    accuracy (never assume an answer we can't read was correct) and are
    charged the worst-case harm cost for that gold label (an answer a
    downstream system can't parse can't safely be assumed benign). They're
    always tallied separately (n_unparseable / unparseable_rows) so the
    failure is visible rather than smeared into the aggregate.
    """
    confusion = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    abstain_rows = [row for row in dataset if row.get("expected_behavior") == "abstain"]
    abstain_correct = 0
    correct = 0
    total_cost = 0
    rows_out = []
    errors = []
    unparseable_rows = []

    for row in dataset:
        variant = row["variant"]
        gold = row["gold_label"]
        try:
            result = graph.ask(variant)
            predicted = _predicted_label(result.get("classification", ""))
        except Exception as exc:  # noqa: BLE001 - live tool/network calls, isolate one bad row
            errors.append({"id": row["id"], "variant": variant, "error": str(exc)})
            continue

        if predicted == UNPARSEABLE:
            unparseable_rows.append({"id": row["id"], "variant": variant, "gold": gold})
            is_correct = False
            total_cost += max(HARM_COST[gold].values())
        else:
            is_correct = predicted == gold
            confusion[gold][predicted] += 1
            total_cost += HARM_COST[gold][predicted]
            correct += is_correct

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
        "n_unparseable": len(unparseable_rows),
        "unparseable_rows": unparseable_rows,
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
