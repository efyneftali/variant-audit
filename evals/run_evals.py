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

The five external evidence tools (ClinVar/gnomAD/Ensembl/UCSC/AlphaMissense) are
served from frozen fixtures via evals/replay.py, NOT the live network -- that's
what makes this number trustworthy (see VA-39). Live runs used to lose rows to
NCBI/gnomAD rate limits and score recall@k over whatever survived. Freeze the
fixtures once with --record; every run after is offline and reproducible.

Usage:
    python evals/run_evals.py --record               # one-time: hit the network, freeze fixtures, then score
    python evals/run_evals.py --limit 5              # smoke test on 5 rows first (fast, offline)
    python evals/run_evals.py --skip-generation       # retrieval+classification only (cheap)
    python evals/run_evals.py                         # full run, offline against fixtures, temperature=0.0 (VA-41 decision)
    python evals/run_evals.py --samples 5             # 5x per variant at temp=0.0 -- SD should be ~0, that's expected
    python evals/run_evals.py --samples 5 --temperature 0.7   # reproduce the VA-41 voting investigation at a non-zero temp

Every run drops a timestamped report in evals/reports/ -- never overwritten, so
a later "did this prompt change help" comparison is a diff between two files.

TODO(day-13): eval_generation_judge + calibration tracking.
TODO(day-14): eval_robustness + statistical (multi-run) variance.
"""

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import perturb, replay  # noqa: E402
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

    get_clinvar_record deliberately raises (not a graceful {"found": False})
    on a real API failure -- network error, timeout, rate limit -- as opposed
    to a genuinely-unknown variant (see mcp_tools/clinvar.py). NCBI's eutils
    rate-limits under sustained load (see DATASET.md's build notes -- this bit
    the dataset build too), so a live run WILL hit this. One bad row is
    isolated into `errors`, same as eval_classification, rather than crashing
    the whole retrieval pass.
    """
    scored_rows = [row for row in dataset if row.get("evidence_type") in EVIDENCE_TYPE_TO_CRITERIA]
    skipped_ids = [row["id"] for row in dataset if row.get("evidence_type") not in EVIDENCE_TYPE_TO_CRITERIA]

    hits = 0
    reciprocal_ranks = []
    rows_out = []
    errors = []

    for row in scored_rows:
        variant = row["variant"]
        expected = EVIDENCE_TYPE_TO_CRITERIA[row["evidence_type"]]
        try:
            clinvar_record = graph.get_clinvar_record(variant)
            query = _retrieval_query(variant, clinvar_record)
            chunks = graph.semantic_search(query, top_k=k)
        except Exception as exc:  # noqa: BLE001 - live network calls, isolate one bad row
            errors.append({"id": row["id"], "variant": variant, "error": str(exc)})
            continue

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

    n = len(rows_out)
    return {
        "k": k,
        "n": n,
        "n_skipped": len(skipped_ids),
        "skipped_ids": skipped_ids,
        "n_errors": len(errors),
        "errors": errors,
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


@contextmanager
def _measure_llm_calls():
    """Count LLM calls and sum their wall-clock time for the duration of the
    block -- the cost/latency half of the VA-41 voting decision ("voting is N
    times the calls" needs an actual N and an actual elapsed time, not a guess).

    Wraps graph.llm.complete transparently for every caller (classify,
    grade_evidence, check_grounded) without touching src/ -- same pattern as
    replay.py rebinding the five tool names, scoped to the eval process only.
    """
    real_complete = graph.llm.complete
    stats = {"n_calls": 0, "total_seconds": 0.0}

    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        try:
            return real_complete(*args, **kwargs)
        finally:
            stats["n_calls"] += 1
            stats["total_seconds"] += time.perf_counter() - start

    graph.llm.complete = wrapped
    try:
        yield stats
    finally:
        graph.llm.complete = real_complete


def eval_classification(dataset: list[dict], *, temperature: float | None = 0.0) -> dict:
    """3-class accuracy vs ClinVar, harm-weighted error cost, abstention correctness.

    temperature: forwarded to graph.ask() -- the classify() generation call only.
        Defaults to 0.0, the VA-41 decision (TEMPERATURE_VOTING_DECISION.md).
        Pass an explicit value (or None for the old provider-default behavior)
        to reproduce the sweep/voting investigation.

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

    wall_start = time.perf_counter()
    with _measure_llm_calls() as llm_stats:
        for row in dataset:
            variant = row["variant"]
            gold = row["gold_label"]
            try:
                result = graph.ask(variant, temperature=temperature)
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
    wall_seconds = time.perf_counter() - wall_start

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
        # cost/latency instrumentation (VA-41 part 2): wall_seconds is real
        # elapsed time for this pass (LLM calls + retrieval + fixture reads);
        # llm_calls/llm_seconds isolate just the model-call portion of that.
        "wall_seconds": wall_seconds,
        "llm_calls": llm_stats["n_calls"],
        "llm_seconds": llm_stats["total_seconds"],
    }


def eval_classification_multi(dataset: list[dict], k: int, *, temperature: float | None = 0.0) -> dict:
    """Run eval_classification k times and report generation-run variance.

    temperature: forwarded to every underlying eval_classification run -- same
        value for all k samples, so the SD measured here is run-to-run jitter AT
        that temperature, not a mix of settings. Defaults to 0.0 (the VA-41
        decision), which makes k>1 samples somewhat redundant by construction --
        pass an explicit non-zero value to reproduce the voting investigation.

    With the five tools frozen to fixtures (VA-39), the agent loop's only
    remaining source of run-to-run drift is the LLM itself. Repeating each
    variant k times isolates exactly that jitter -- the thing the Wilson
    interval structurally can't see. Wilson answers "how much would accuracy
    move on a different sample of *rows*?"; the across-sample SD here answers
    "how much does it move on the *same* rows just because the model is
    stochastic?". They're different uncertainties; we report both side by side.

    Two accuracy numbers, because they answer different questions:
      - mean_accuracy +/- accuracy_sd: what a single run typically scores and
        how much any one run can swing. This is the honest point estimate --
        one run gave 0.811 and another 0.676 off identical fixtures.
      - majority_vote_accuracy: score each variant by its modal label across the
        k samples, then take accuracy over those. Answers "if I let the model
        vote with itself, how good is it?" -- robust to single-sample flips.

    Per-variant it also partitions every row into stable-correct / stable-wrong
    / unstable (a coin-flip across samples) -- the rigorous, at-scale version of
    the by-hand abstention partition (a stable-wrong row needs a better model; a
    flip is a consistency problem).
    """
    runs = [eval_classification(dataset, temperature=temperature) for _ in range(k)]

    # variant id -> the k predicted labels, in run order (raw per-sample outputs,
    # kept so any specific sample stays auditable from the report).
    per_item: dict[str, dict] = {}
    for run in runs:
        for row in run["rows"]:
            entry = per_item.setdefault(
                row["id"],
                {"id": row["id"], "variant": row["variant"], "gold": row["gold"], "predictions": []},
            )
            entry["predictions"].append(row["predicted"])

    per_item_rows = []
    majority_confusion = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    stable_correct = stable_wrong = unstable = 0
    majority_correct = 0

    for entry in per_item.values():
        preds = entry["predictions"]
        gold = entry["gold"]
        # modal label; Counter breaks count ties by first-seen (run order).
        modal = Counter(preds).most_common(1)[0][0]
        correct_count = sum(p == gold for p in preds)
        is_stable = len(set(preds)) == 1
        is_majority_correct = modal == gold

        majority_correct += is_majority_correct
        if is_stable and is_majority_correct:
            stable_correct += 1
        elif is_stable:
            stable_wrong += 1
        else:
            unstable += 1
        # majority-vote confusion only over real P/VUS/B modal calls (an
        # UNPARSEABLE modal label has no cell, same rule as eval_classification).
        if gold in majority_confusion and modal in LABELS:
            majority_confusion[gold][modal] += 1

        per_item_rows.append(
            {
                "id": entry["id"],
                "variant": entry["variant"],
                "gold": gold,
                "predictions": preds,
                "correct_count": correct_count,
                "k": len(preds),
                "majority_label": modal,
                "majority_correct": is_majority_correct,
                "stable": is_stable,
            }
        )

    per_run_accuracies = [run["accuracy"] for run in runs]
    n_items = len(per_item_rows)
    mean_accuracy = statistics.fmean(per_run_accuracies) if per_run_accuracies else 0.0
    # sample SD (needs k>=2); a single sample has no spread to report.
    accuracy_sd = statistics.stdev(per_run_accuracies) if len(per_run_accuracies) > 1 else 0.0
    majority_vote_accuracy = majority_correct / n_items if n_items else 0.0
    # Wilson stays at the true row count (n_items), reported on the mean accuracy
    # -- it's the sampling CI, deliberately NOT inflated to k*n as if repeats
    # were independent rows.
    ci_lo, ci_hi = wilson_ci(round(mean_accuracy * n_items), n_items)

    return {
        "k": k,
        "n": n_items,
        "per_run_accuracies": per_run_accuracies,
        "mean_accuracy": mean_accuracy,
        "accuracy_sd": accuracy_sd,
        "accuracy_ci_95": {"lo": ci_lo, "hi": ci_hi},
        "majority_vote_accuracy": majority_vote_accuracy,
        "harm_weighted_cost_mean_over_runs": statistics.fmean(
            run["harm_weighted_cost_mean"] for run in runs
        ) if runs else 0.0,
        "abstention_accuracy_mean_over_runs": statistics.fmean(
            run["abstention"]["accuracy"] for run in runs
        ) if runs else 0.0,
        "n_errors_per_run": [run["n_errors"] for run in runs],
        "n_unparseable_per_run": [run["n_unparseable"] for run in runs],
        "stability": {
            "stable_correct": stable_correct,
            "stable_wrong": stable_wrong,
            "unstable": unstable,
            "unstable_ids": [r["id"] for r in per_item_rows if not r["stable"]],
            "stable_wrong_ids": [
                r["id"] for r in per_item_rows if r["stable"] and not r["majority_correct"]
            ],
        },
        "majority_confusion_matrix": majority_confusion,
        # cost/latency instrumentation (VA-41 part 2): best-of-k is k independent
        # eval_classification passes, so these totals scale ~k x a single-sample
        # baseline by construction -- the "_mean" figures are the per-variant unit
        # cost (should hold ~constant across k), the "_total" figures are what the
        # whole harness run actually paid to get the majority-vote answer.
        "llm_calls_total": sum(run["llm_calls"] for run in runs),
        "llm_seconds_total": sum(run["llm_seconds"] for run in runs),
        "wall_seconds_total": sum(run["wall_seconds"] for run in runs),
        "wall_seconds_per_run": [run["wall_seconds"] for run in runs],
        "llm_calls_per_variant_mean": (
            sum(run["llm_calls"] for run in runs) / (k * n_items) if n_items else 0.0
        ),
        "wall_seconds_per_variant_mean": (
            sum(run["wall_seconds"] for run in runs) / (k * n_items) if n_items else 0.0
        ),
        "per_item": per_item_rows,
        # a representative single run, so the report keeps the classic
        # classification block (confusion matrix etc.) alongside the samples view.
        "representative_run": runs[0] if runs else None,
    }


def eval_generation_judge(dataset: list[dict]) -> dict:
    """LLM-as-judge faithfulness/relevance. Track judge–human agreement (kappa)."""
    raise NotImplementedError("TODO(day-13): judge outputs; compare to your hand-labels")


def eval_robustness(dataset: list[dict], *, baseline: dict | None = None) -> dict:
    """Perturb inputs (paraphrase/noise/formatting); measure quality delta (VA-13).

    Reruns eval_classification against the SAME 37 rows and SAME underlying
    facts under perturb.install() (paraphrased clinical_significance/
    review_status/hgvs/most_severe_consequence/am_class, plus case/whitespace
    noise -- see evals/perturb.py), and diffs it against a clean-fixture
    baseline. Fixture lookups stay exact-match on the untouched variant id, so
    nothing drops out; only the free text classify() reads -- and the RAG
    query retrieve_criteria builds from that same text -- changes surface
    form. If quality holds, that's real robustness; if accuracy drops, the
    pipeline was leaning on exact phrasing rather than the evidence itself.

    Uses the default (temperature=0.0, VA-41) config for both passes -- this
    measures robustness AT the decided operating point, not some other one.

    baseline: an already-computed eval_classification(dataset) result at
        temperature=0.0 to reuse instead of recomputing. temp=0 is
        deterministic, so main() passes its own classification_result here
        when it already ran one -- recomputing would just burn another full
        pass (~10+ min locally) for a byte-identical number.
    """
    replay.install()
    if baseline is None:
        baseline = eval_classification(dataset)

    perturb.install()
    try:
        perturbed = eval_classification(dataset)
    finally:
        replay.install()  # restore clean fixtures for anything that runs after

    baseline_correct = {r["id"]: r["correct"] for r in baseline["rows"]}
    perturbed_correct = {r["id"]: r["correct"] for r in perturbed["rows"]}
    common_ids = sorted(set(baseline_correct) & set(perturbed_correct))

    flips_to_wrong = [i for i in common_ids if baseline_correct[i] and not perturbed_correct[i]]
    flips_to_correct = [i for i in common_ids if not baseline_correct[i] and perturbed_correct[i]]

    return {
        "n": len(common_ids),
        "baseline_accuracy": baseline["accuracy"],
        "perturbed_accuracy": perturbed["accuracy"],
        "accuracy_delta": perturbed["accuracy"] - baseline["accuracy"],
        "baseline_harm_weighted_cost_mean": baseline["harm_weighted_cost_mean"],
        "perturbed_harm_weighted_cost_mean": perturbed["harm_weighted_cost_mean"],
        "baseline_abstention_accuracy": baseline["abstention"]["accuracy"],
        "perturbed_abstention_accuracy": perturbed["abstention"]["accuracy"],
        "flips_correct_to_wrong": flips_to_wrong,
        "flips_wrong_to_correct": flips_to_correct,
        "baseline_llm_calls": baseline["llm_calls"],
        "perturbed_llm_calls": perturbed["llm_calls"],
        "baseline_wall_seconds": baseline["wall_seconds"],
        "perturbed_wall_seconds": perturbed["wall_seconds"],
    }


def _print_scorecard(report: dict) -> None:
    print("=" * 60)
    print(f"variant-audit eval report -- {report['timestamp']}")
    limit_note = f" (--limit {report['limit']})" if report["limit"] else ""
    print(f"dataset: {report['n_used']}/{report['n_dataset']} rows{limit_note}")
    temp_note = "provider default" if report.get("temperature") is None else report["temperature"]
    print(f"classify() temperature: {temp_note}")
    print("=" * 60)

    retrieval = report.get("retrieval")
    if retrieval:
        print(f"\n-- retrieval (k={retrieval['k']}) --")
        print(
            f"recall@{retrieval['k']}: {retrieval['recall_at_k']:.3f}   "
            f"MRR: {retrieval['mrr']:.3f}   "
            f"n={retrieval['n']} ({retrieval['n_skipped']} skipped, no answer key)"
        )
        if retrieval["n_errors"]:
            ids = [row["id"] for row in retrieval["errors"]]
            print(f"ERRORS (skipped, not scored): {retrieval['n_errors']} row(s) -- {ids}")

    classification = report.get("classification")
    if classification:
        acc = classification["accuracy"]
        ci = classification["accuracy_ci_95"]
        print("\n-- classification --")
        print(f"accuracy: {acc:.3f}, 95% CI [{ci['lo']:.3f}, {ci['hi']:.3f}], n={classification['n']}")
        print(
            f"harm-weighted cost: {classification['harm_weighted_cost_mean']:.3f} mean "
            f"/ {classification['harm_weighted_cost_total']} total"
        )
        ab = classification["abstention"]
        print(f"abstention (expected_behavior=abstain): {ab['correct']}/{ab['n']} correct ({ab['accuracy']:.3f})")
        if "llm_calls" in classification and classification["n"]:
            print(
                f"cost/latency: {classification['llm_calls']} LLM calls, "
                f"{classification['llm_seconds']:.1f}s in-model / {classification['wall_seconds']:.1f}s wall "
                f"for {classification['n']} variants "
                f"({classification['llm_calls'] / classification['n']:.2f} calls/variant)"
            )
        if classification["n_unparseable"]:
            ids = [row["id"] for row in classification["unparseable_rows"]]
            print(f"UNPARSEABLE: {classification['n_unparseable']} row(s) -- {ids}")
        if classification["n_errors"]:
            ids = [row["id"] for row in classification["errors"]]
            print(f"ERRORS (skipped, not scored): {classification['n_errors']} row(s) -- {ids}")

        print("\nconfusion matrix (rows=gold, cols=predicted):")
        print("        " + "".join(f"{p:>6}" for p in LABELS))
        for gold in LABELS:
            row = classification["confusion_matrix"][gold]
            print(f"  {gold:>4}  " + "".join(f"{row[pred]:>6}" for pred in LABELS))

    _print_samples(report.get("classification_samples"))
    _print_robustness(report.get("robustness"))
    print()


def _print_samples(samples: dict | None) -> None:
    """Print the multi-sample generation-variance block, when --samples k>1."""
    if not samples:
        return
    ci = samples["accuracy_ci_95"]
    per_run = ", ".join(f"{a:.3f}" for a in samples["per_run_accuracies"])
    print(f"\n-- classification: generation variance (k={samples['k']} samples/variant) --")
    print(
        f"mean accuracy: {samples['mean_accuracy']:.3f} +/- {samples['accuracy_sd']:.3f} SD "
        f"(across-sample); per-run [{per_run}]"
    )
    print(f"Wilson 95% CI [{ci['lo']:.3f}, {ci['hi']:.3f}] (sampling over n={samples['n']} rows -- different question than SD)")
    print(f"majority-vote accuracy: {samples['majority_vote_accuracy']:.3f} (modal label per variant)")
    print(
        f"harm-weighted cost: {samples['harm_weighted_cost_mean_over_runs']:.3f} mean-of-run-means   "
        f"abstention: {samples['abstention_accuracy_mean_over_runs']:.3f} mean"
    )
    print(
        f"cost/latency (best-of-{samples['k']}, i.e. {samples['k']}x a single-sample pass): "
        f"{samples['llm_calls_total']} LLM calls total "
        f"({samples['llm_calls_per_variant_mean']:.2f} calls/variant/sample), "
        f"{samples['wall_seconds_total']:.1f}s wall total "
        f"({samples['wall_seconds_per_variant_mean']:.2f}s/variant/sample)"
    )
    per_run_wall = ", ".join(f"{s:.1f}" for s in samples["wall_seconds_per_run"])
    print(f"  per-run wall seconds: [{per_run_wall}]")
    st = samples["stability"]
    print(
        f"stability: {st['stable_correct']} stable-correct, {st['stable_wrong']} stable-wrong, "
        f"{st['unstable']} unstable (coin-flip) of {samples['n']}"
    )
    if st["stable_wrong_ids"]:
        print(f"  stable-wrong (need a better model): {st['stable_wrong_ids']}")
    if st["unstable_ids"]:
        print(f"  unstable (consistency problem): {st['unstable_ids']}")

    print("\nmajority-vote confusion matrix (rows=gold, cols=modal predicted):")
    print("        " + "".join(f"{p:>6}" for p in LABELS))
    for gold in LABELS:
        row = samples["majority_confusion_matrix"][gold]
        print(f"  {gold:>4}  " + "".join(f"{row[pred]:>6}" for pred in LABELS))


def _print_robustness(robustness: dict | None) -> None:
    """Print the paraphrase/formatting-noise robustness block, when --robustness."""
    if not robustness:
        return
    print(f"\n-- robustness: paraphrase + formatting noise (VA-13, n={robustness['n']}) --")
    print(
        f"accuracy: {robustness['baseline_accuracy']:.3f} clean -> "
        f"{robustness['perturbed_accuracy']:.3f} perturbed "
        f"(delta {robustness['accuracy_delta']:+.3f})"
    )
    print(
        f"harm-weighted cost: {robustness['baseline_harm_weighted_cost_mean']:.3f} clean -> "
        f"{robustness['perturbed_harm_weighted_cost_mean']:.3f} perturbed"
    )
    print(
        f"abstention accuracy: {robustness['baseline_abstention_accuracy']:.3f} clean -> "
        f"{robustness['perturbed_abstention_accuracy']:.3f} perturbed"
    )
    print(
        f"cost/latency: {robustness['baseline_llm_calls']} calls / {robustness['baseline_wall_seconds']:.1f}s clean, "
        f"{robustness['perturbed_llm_calls']} calls / {robustness['perturbed_wall_seconds']:.1f}s perturbed"
    )
    if robustness["flips_correct_to_wrong"]:
        print(f"  flipped correct -> wrong under noise: {robustness['flips_correct_to_wrong']}")
    if robustness["flips_wrong_to_correct"]:
        print(f"  flipped wrong -> correct under noise: {robustness['flips_wrong_to_correct']}")
    if not robustness["flips_correct_to_wrong"] and not robustness["flips_wrong_to_correct"]:
        print("  no rows flipped either direction")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-generation", action="store_true", help="skip the LLM-judge generation eval")
    parser.add_argument("--k", type=int, default=5, help="top-k for retrieval")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only run the first N dataset rows -- smoke test the pipeline before a full run",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="one-time: hit the live tools for the used rows and (re)freeze fixtures before scoring",
    )
    parser.add_argument(
        "--samples", type=int, default=1, metavar="K",
        help="run classification K times per variant (3-5) to measure generation variance -- "
             "k*N LLM calls, so opt-in; default 1",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, metavar="T",
        help="sampling temperature for the classify() generation call. Defaults to 0.0 -- the "
             "VA-41 decision (see TEMPERATURE_VOTING_DECISION.md): beat both a temperature sweep "
             "and best-of-5 voting on accuracy, harm cost, AND latency simultaneously. Override "
             "to reproduce that investigation (pair with --samples for the variance/voting view).",
    )
    parser.add_argument(
        "--robustness", action="store_true",
        help="VA-13: paraphrase/formatting-noise the evidence text (evals/perturb.py) and report "
             "the accuracy delta vs clean fixtures -- 2x eval_classification passes, so opt-in. "
             "Always runs at temperature=0.0 (the VA-41 decision) regardless of --temperature, "
             "since this measures robustness AT the deployed operating point.",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")

    full_dataset = load_dataset()
    dataset = full_dataset[: args.limit] if args.limit else full_dataset
    print(f"Loaded {len(full_dataset)} rows" + (f", using first {len(dataset)} (--limit)" if args.limit else "") + ".")

    if args.record:
        # The only path that touches the network. Freeze exactly the rows we're
        # about to score, so --record --limit N stays self-consistent.
        replay.record_all([row["variant"] for row in dataset])

    # Every scoring path reads from fixtures: no live tool calls, no dropped rows.
    replay.install()

    print("Running eval_retrieval (cheap, no LLM calls)...")
    retrieval_result = eval_retrieval(dataset, k=args.k)

    if args.samples > 1:
        print(
            f"Running eval_classification {args.samples}x per variant "
            f"({len(dataset) * args.samples} agent-loop runs total -- this is the slow, k*N part)..."
        )
        samples_result = eval_classification_multi(dataset, k=args.samples, temperature=args.temperature)
        classification_result = samples_result.pop("representative_run")
    else:
        print(f"Running eval_classification ({len(dataset)} rows through the full agent loop -- this is the slow part)...")
        classification_result = eval_classification(dataset, temperature=args.temperature)
        samples_result = None

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(DATASET),
        "n_dataset": len(full_dataset),
        "n_used": len(dataset),
        "limit": args.limit,
        "tool_source": "fixtures (record+replay)" if args.record else "fixtures (replay)",
        "k": args.k,
        "samples": args.samples,
        "temperature": args.temperature,
        "retrieval": retrieval_result,
        "classification": classification_result,
    }
    if samples_result is not None:
        report["classification_samples"] = samples_result

    if not args.skip_generation:
        try:
            report["generation"] = eval_generation_judge(dataset)
        except NotImplementedError:
            print("eval_generation_judge isn't implemented yet (Day 13) -- skipped.")

    if args.robustness:
        # reuse the classification pass just run above as the baseline when it
        # already matches robustness's fixed temperature=0.0/single-sample
        # contract -- temp=0 is deterministic, so recomputing would just burn
        # another full pass for the identical number.
        reusable_baseline = classification_result if (args.temperature == 0.0 and args.samples == 1) else None
        extra_runs = len(dataset) if reusable_baseline is None else 0
        print(
            f"Running eval_robustness (perturbed pass over {len(dataset)} variants"
            + (f" + {extra_runs} for a fresh temperature=0.0 baseline" if extra_runs else " -- reusing the baseline above")
            + ")..."
        )
        report["robustness"] = eval_robustness(dataset, baseline=reusable_baseline)

    REPORTS.mkdir(exist_ok=True)
    report_path = REPORTS / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2))

    _print_scorecard(report)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
