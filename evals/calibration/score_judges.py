"""Score candidate groundedness judges against the human labels (VA-35, stretch).

Runs each candidate judge (local llama / Haiku / Sonnet) over the SAME frozen
cases you hand-labeled, computes Cohen's kappa between each judge and you, and
picks the cheapest candidate that clears your agreement threshold. The number and
the full table land in JUDGE_CALIBRATION.md.

Cost discipline (Haiku and Sonnet are paid API calls):
  - Default runs ONLY the local (free) candidate. Opt into paid ones explicitly:
        --candidates local,haiku,sonnet
  - Each candidate's raw per-case verdicts are cached under judge_runs/. A re-run
    reads the cache instead of re-calling the API, so re-scoring or moving the
    threshold costs nothing. Force a fresh run with --force.
  - The deterministic pre-check settles the clear cases with no model call at all,
    so a paid candidate is only billed for the cases it actually has to judge.

Order of operations:
  1. python evals/calibration/build_cases.py     # freeze cases
  2. python evals/calibration/label.py           # you label them, blind
  3. python evals/calibration/score_judges.py --candidates local,haiku,sonnet

Usage:
    python evals/calibration/score_judges.py                       # free: local only
    python evals/calibration/score_judges.py --candidates local,haiku,sonnet
    python evals/calibration/score_judges.py --threshold 0.6 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evals.calibration.label import load_cases, load_labels  # noqa: E402
from src.variant_audit import graph  # noqa: E402
from src.variant_audit.config import settings  # noqa: E402

HERE = Path(__file__).resolve().parent
CASES_DEFAULT = HERE / "cases.jsonl"
LABELS_DEFAULT = HERE / "labels.jsonl"
SPLIT_DEFAULT = HERE / "split.json"
RUNS_DIR = HERE / "judge_runs"
REPORT_DEFAULT = Path(__file__).resolve().parent.parent.parent / "JUDGE_CALIBRATION.md"

# name -> (provider, model, paid?, cost_rank). cost_rank breaks threshold ties
# toward the cheaper judge -- no reason to pay for Sonnet if Haiku already clears.
CANDIDATES: dict[str, tuple[str, str, bool, int]] = {
    "local": ("ollama", settings.ollama_judge_model, False, 0),
    "haiku": ("anthropic", "claude-haiku-4-5-20251001", True, 1),
    "sonnet": ("anthropic", "claude-sonnet-4-6", True, 2),
}

# Substantial agreement (Landis & Koch). Your call -- override with --threshold.
DEFAULT_THRESHOLD = 0.6


# --- metrics -------------------------------------------------------------------

def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two aligned label sequences. Category-agnostic.

    kappa = (po - pe) / (1 - pe). When pe == 1 (both raters used a single label
    for everything) kappa is undefined; return 1.0 on perfect agreement, else 0.0.
    """
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def confusion(human: list[str], judge: list[str]) -> dict:
    """2x2 counts from the judge's perspective, using human labels as truth.
    'ungrounded' is the positive (interesting) class."""
    c = {"false_ungrounded": 0, "missed_ungrounded": 0, "agree_grounded": 0, "agree_ungrounded": 0}
    for h, j in zip(human, judge):
        if h == j == "grounded":
            c["agree_grounded"] += 1
        elif h == j == "ungrounded":
            c["agree_ungrounded"] += 1
        elif h == "grounded" and j == "ungrounded":
            c["false_ungrounded"] += 1  # judge flagged a grounded call -> over-strict
        elif h == "ungrounded" and j == "grounded":
            c["missed_ungrounded"] += 1  # judge missed a real problem -> dangerous
    return c


# --- running one candidate (cached) --------------------------------------------

def _verdict_label(case: dict, provider: str, model: str) -> tuple[str, str]:
    """Run the shared judge on one frozen case; return ('grounded'|'ungrounded', reason)."""
    grounded, reason = graph.groundedness_verdict(
        case["variant"], case["classification"], case["evidence_text"], case["criteria_text"],
        provider=provider, model=model,
    )
    return ("grounded" if grounded else "ungrounded"), reason


def run_candidate(name: str, cases: list[dict], *, force: bool) -> dict[str, dict]:
    """Judge every case with one candidate, caching raw verdicts to judge_runs/{name}.jsonl.
    Returns {case_id: {"label": ..., "reason": ...}}. Cached ids are not re-judged
    (so paid re-runs cost nothing) unless force=True."""
    provider, model, _paid, _rank = CANDIDATES[name]
    cache_path = RUNS_DIR / f"{name}.jsonl"
    cached: dict[str, dict] = {}
    if cache_path.exists() and not force:
        for line in cache_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cached[rec["id"]] = rec

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases if c["id"] not in cached]
    if todo:
        print(f"  {name}: judging {len(todo)} case(s) via {provider}/{model}"
              f"{' [PAID]' if _paid else ' [free]'}...")
        with cache_path.open("a") as f:
            for case in todo:
                label, reason = _verdict_label(case, provider, model)
                rec = {"id": case["id"], "label": label, "reason": reason}
                cached[case["id"]] = rec
                f.write(json.dumps(rec) + "\n")
    else:
        print(f"  {name}: all {len(cases)} case(s) cached, no calls made.")
    return cached


# --- scoring & selection -------------------------------------------------------

def score_candidate(name: str, verdicts: dict[str, dict], labeled: list[tuple[str, str]]) -> dict:
    """labeled is [(case_id, human_label)]. Align judge verdicts to it and score."""
    human = [hl for _cid, hl in labeled]
    judge = [verdicts[cid]["label"] for cid, _hl in labeled]
    return {
        "candidate": name,
        "provider": CANDIDATES[name][0],
        "model": CANDIDATES[name][1],
        "paid": CANDIDATES[name][2],
        "n": len(labeled),
        "accuracy": sum(h == j for h, j in zip(human, judge)) / len(labeled) if labeled else float("nan"),
        "kappa": cohen_kappa(human, judge),
        "confusion": confusion(human, judge),
    }


def pick_judge(scores: list[dict], threshold: float) -> dict | None:
    """The cheapest candidate that clears the threshold; None if none do."""
    clearing = [s for s in scores if s["kappa"] >= threshold]
    if not clearing:
        return None
    return min(clearing, key=lambda s: CANDIDATES[s["candidate"]][3])


# --- report --------------------------------------------------------------------

def render_report(scores: list[dict], pick: dict | None, threshold: float, *, n_labeled: int,
                  partition: str, in_sample: bool) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if in_sample:
        rigor = (
            f"⚠️ **IN-SAMPLE κ** over all {n_labeled} labeled cases — no holdout. This is only "
            "an honest number if the judge prompt was **not** tuned against these cases; the "
            "moment you inspect per-case disagreements and edit the prompt, re-report on a "
            "holdout (`make_split.py` → `--split test`), because in-sample κ is an upper bound."
        )
    else:
        rigor = (
            f"**Holdout κ** on the `{partition}` split ({n_labeled} cases the judge prompt was "
            "never tuned against). Dev-split cases are excluded — tune the prompt there, report here."
        )
    lines = [
        "# Judge calibration (VA-35)",
        "",
        f"_Generated {now} · threshold κ ≥ {threshold:.2f} (substantial agreement, Landis & Koch)._",
        "",
        rigor,
        "",
        "Cohen's κ is agreement between each candidate judge and the human labels, "
        "corrected for chance. `false_ungrounded` = judge flagged a grounded call "
        "(over-strict); `missed_ungrounded` = judge passed a real problem (the "
        "dangerous error).",
        "",
        "| candidate | provider / model | n | accuracy | κ | false_ungrounded | missed_ungrounded | clears? |",
        "|-----------|------------------|---|----------|---|------------------|-------------------|---------|",
    ]
    for s in sorted(scores, key=lambda x: x["kappa"], reverse=True):
        cm = s["confusion"]
        clears = "✅" if s["kappa"] >= threshold else "—"
        lines.append(
            f"| {s['candidate']}{' (paid)' if s['paid'] else ''} "
            f"| `{s['provider']}/{s['model']}` | {s['n']} | {s['accuracy']:.3f} "
            f"| **{s['kappa']:.3f}** | {cm['false_ungrounded']} | {cm['missed_ungrounded']} | {clears} |"
        )
    lines.append("")
    if pick is not None:
        model_env = "OLLAMA_JUDGE_MODEL" if pick["provider"] == "ollama" else "ANTHROPIC_JUDGE_MODEL"
        lines += [
            f"## Selected judge: **{pick['candidate']}** (κ = {pick['kappa']:.3f})",
            "",
            f"`{pick['provider']}/{pick['model']}` — the cheapest candidate clearing "
            f"κ ≥ {threshold:.2f}. Set it as the judge with "
            f"`JUDGE_PROVIDER={pick['provider']}` and `{model_env}={pick['model']}`.",
        ]
    else:
        best = max(scores, key=lambda s: s["kappa"]) if scores else None
        lines += [
            f"## No candidate cleared κ ≥ {threshold:.2f}",
            "",
            (f"Best was **{best['candidate']}** at κ = {best['kappa']:.3f}. "
             "Either lower the threshold (with eyes open), add labeled cases, or "
             "improve the judge prompt before trusting it.") if best else "No scores.",
        ]
    reproduce = (
        "python evals/calibration/score_judges.py --candidates local,haiku,sonnet"
        + ("" if in_sample and partition == "all" else f" --split {partition}")
    )
    lines += ["", f"_Reproduce: `{reproduce}`. Raw per-case verdicts are cached in "
              "`evals/calibration/judge_runs/`._", ""]
    return "\n".join(lines)


# --- driver --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidate groundedness judges vs human labels.")
    parser.add_argument("--cases", type=Path, default=CASES_DEFAULT)
    parser.add_argument("--labels", type=Path, default=LABELS_DEFAULT)
    parser.add_argument("--candidates", default="local",
                        help="comma-separated: local,haiku,sonnet (default: local only — paid ones are opt-in)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    parser.add_argument("--split", choices=["test", "dev", "all"], default=None,
                        help="which partition of split.json to score. Default: 'test' (holdout) if a "
                             "split is frozen, else 'all' (in-sample). Report the headline number on 'test'.")
    parser.add_argument("--force", action="store_true", help="ignore cache and re-judge (re-pays for paid candidates)")
    args = parser.parse_args()

    names = [n.strip() for n in args.candidates.split(",") if n.strip()]
    unknown = [n for n in names if n not in CANDIDATES]
    if unknown:
        sys.exit(f"Unknown candidate(s): {unknown}. Choose from {list(CANDIDATES)}.")

    if not args.labels.exists():
        sys.exit(f"No labels at {args.labels}. Label the cases first: python evals/calibration/label.py")
    cases = {c["id"]: c for c in load_cases(args.cases)}
    labels = load_labels(args.labels)
    labeled = [(cid, labels[cid]["label"]) for cid in labels if cid in cases]
    if not labeled:
        sys.exit("No labeled cases overlap with the cases file — nothing to score.")

    # Resolve the holdout partition. A frozen split defaults us to the test set;
    # scoring 'all' (or with no split at all) is in-sample and gets stamped as such.
    frozen = json.loads(SPLIT_DEFAULT.read_text()) if SPLIT_DEFAULT.exists() else None
    partition = args.split or ("test" if frozen else "all")
    if partition != "all" and frozen is None:
        sys.exit(f"--split {partition} needs a frozen split. Create one: python evals/calibration/make_split.py")
    if partition == "all":
        selected_ids = None
        in_sample = True
    else:
        selected_ids = set(frozen[partition])
        in_sample = partition == "dev"  # dev is what you tune on -> also in-sample if reported
    if selected_ids is not None:
        labeled = [(cid, lbl) for cid, lbl in labeled if cid in selected_ids]
        if not labeled:
            sys.exit(f"No labeled cases in the '{partition}' partition.")
    scoring_cases = [cases[cid] for cid, _ in labeled]

    banner = (
        f"IN-SAMPLE (all {len(labeled)} cases — no holdout)" if partition == "all"
        else f"holdout '{partition}' split ({len(labeled)} cases)"
    )
    paid = [n for n in names if CANDIDATES[n][2]]
    if paid:
        print(f"NOTE: {paid} are paid API calls (up to {len(scoring_cases)} judged cases each, "
              "minus whatever the free pre-check settles). Cached after the first run.\n")

    print(f"Scoring {banner} across: {names}")
    scores = []
    for name in names:
        verdicts = run_candidate(name, scoring_cases, force=args.force)
        scores.append(score_candidate(name, verdicts, labeled))

    pick = pick_judge(scores, args.threshold)
    report = render_report(scores, pick, args.threshold, n_labeled=len(labeled),
                           partition=partition, in_sample=in_sample)
    args.report.write_text(report)

    print("\n" + report)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
