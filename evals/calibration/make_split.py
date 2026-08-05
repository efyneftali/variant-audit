"""Freeze a dev/test split of the labeled calibration cases (VA-35 rigor).

Tuning the judge prompt against the same cases you then report kappa on is
training on the test set — the number comes out optimistically biased. This
freezes a holdout: you iterate on the judge prompt against the **dev** split
only, and report the headline kappa on the **test** split, which the prompt has
never seen.

The split is:
  - stratified by your human label, so both grounded and ungrounded are
    represented in each partition (a random split of 45 can starve the minority
    class);
  - deterministic given the seed, and written once to split.json. It refuses to
    overwrite an existing split without --force — re-rolling the split until the
    number looks good is itself a form of cheating, so the split is frozen like
    the dataset is.

Usage:
    python evals/calibration/make_split.py                 # 40% holdout, seed 20260717
    python evals/calibration/make_split.py --holdout 0.33
    python evals/calibration/make_split.py --force         # re-roll (records the reason)

After freezing, score the holdout with:
    python evals/calibration/score_judges.py --split test --candidates local,haiku,sonnet
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evals.calibration.label import load_cases, load_labels  # noqa: E402

HERE = Path(__file__).resolve().parent
CASES_DEFAULT = HERE / "cases.jsonl"
LABELS_DEFAULT = HERE / "labels.jsonl"
SPLIT_DEFAULT = HERE / "split.json"

DEFAULT_HOLDOUT = 0.40
DEFAULT_SEED = 20260717


def make_split(labeled: list[tuple[str, str]], holdout_frac: float, seed: int) -> dict:
    """Stratified dev/test split. `labeled` is [(case_id, human_label)].
    Returns {"dev": [...ids], "test": [...ids]}, both sorted."""
    by_label: dict[str, list[str]] = defaultdict(list)
    for cid, label in labeled:
        by_label[label].append(cid)

    rng = random.Random(seed)
    dev, test = [], []
    for label in sorted(by_label):
        ids = sorted(by_label[label])
        rng.shuffle(ids)
        n_test = round(len(ids) * holdout_frac)
        test.extend(ids[:n_test])
        dev.extend(ids[n_test:])
    return {"dev": sorted(dev), "test": sorted(test)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a dev/test split of the labeled calibration cases.")
    parser.add_argument("--cases", type=Path, default=CASES_DEFAULT)
    parser.add_argument("--labels", type=Path, default=LABELS_DEFAULT)
    parser.add_argument("--out", type=Path, default=SPLIT_DEFAULT)
    parser.add_argument("--holdout", type=float, default=DEFAULT_HOLDOUT, help="fraction held out for test (default 0.40)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="overwrite an existing frozen split")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        sys.exit(
            f"A split already exists at {args.out}. It's frozen on purpose — re-rolling it "
            "to chase a better number defeats the holdout. Use --force only if you have a "
            "real reason (and note it in the commit)."
        )
    if not args.labels.exists():
        sys.exit(f"No labels at {args.labels}. Label the cases first.")

    cases = {c["id"] for c in load_cases(args.cases)}
    labels = load_labels(args.labels)
    labeled = [(cid, labels[cid]["label"]) for cid in labels if cid in cases]
    if not labeled:
        sys.exit("No labeled cases overlap with the cases file.")

    split = make_split(labeled, args.holdout, args.seed)
    label_of = {cid: labels[cid]["label"] for cid, _ in labeled}

    def counts(ids: list[str]) -> dict:
        c: dict[str, int] = defaultdict(int)
        for cid in ids:
            c[label_of[cid]] += 1
        return dict(c)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "holdout_frac": args.holdout,
        "n_labeled": len(labeled),
        "dev": split["dev"],
        "test": split["test"],
        "dev_label_counts": counts(split["dev"]),
        "test_label_counts": counts(split["test"]),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Froze split at {args.out}")
    print(f"  dev  : {len(split['dev'])} cases  {counts(split['dev'])}   (tune the judge prompt here)")
    print(f"  test : {len(split['test'])} cases  {counts(split['test'])}   (report headline kappa here — never tune on it)")
    print("\nNext: python evals/calibration/score_judges.py --split test --candidates local,haiku,sonnet")


if __name__ == "__main__":
    main()
