"""Blind hand-labeling tool for the groundedness calibration set (VA-35, step 4).

You label each case grounded / ungrounded yourself, using your genetics judgment,
BEFORE any model verdict exists to anchor on. Blindness here is structural, not a
promise: cases.jsonl contains no judge verdict, this tool never imports the judge,
and it renders only an allowlist of fields -- never the case id (whose suffix
could flag an injected case) or the `_provenance` metadata.

The loop is resumable: already-labeled ids are skipped, so you can stop with `q`
and pick up where you left off. Cases are shown in a shuffled order so planted
and natural cases interleave.

Usage:
    python evals/calibration/label.py                        # label into labels.jsonl
    python evals/calibration/label.py --labeler efy          # tag who labeled
    python evals/calibration/label.py --out labels_pass2.jsonl --seed 99
        # a second independent pass -> measure your own self-consistency (intra-rater kappa)

For each case: [g]rounded  [u]ngrounded  [s]kip (decide later)  [q]uit & save.
Then a one-line reason, and (for ungrounded) the offending criterion/claim.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES_DEFAULT = HERE / "cases.jsonl"
LABELS_DEFAULT = HERE / "labels.jsonl"

# The ONLY fields ever shown to the labeler. Everything else in a case record --
# `id` (its suffix flags injected cases), `_provenance`, `intended` -- is withheld
# so the human judges the artifact on its merits, blind.
_SHOWN_FIELDS = ("variant", "gene", "difficulty", "classification", "evidence_text", "criteria_text", "criteria_sources")

VALID_LABELS = {"grounded", "ungrounded"}


def load_cases(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("#")]


def load_labels(path: Path) -> dict[str, dict]:
    """Existing labels keyed by case id -- for resuming and for skipping done ids."""
    if not path.exists():
        return {}
    labels: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                rec = json.loads(line)
                labels[rec["id"]] = rec
    return labels


def blind_view(case: dict) -> dict:
    """The subset of a case a labeler may see. Excludes id / provenance / intended
    so nothing hints at the intended answer."""
    return {k: case[k] for k in _SHOWN_FIELDS if k in case}


def pending(cases: list[dict], labels: dict[str, dict], seed: int) -> list[dict]:
    """Unlabeled cases in a shuffled, provenance-independent order."""
    todo = [c for c in cases if c["id"] not in labels]
    random.Random(seed).shuffle(todo)
    return todo


def render_case(case: dict, position: int, total: int) -> str:
    v = blind_view(case)
    lines = [
        "=" * 72,
        f"Case {position} of {total}"
        + (f"   variant={v['variant']}" if v.get("variant") else "")
        + (f"   gene={v['gene']}" if v.get("gene") else "")
        + (f"   difficulty={v['difficulty']}" if v.get("difficulty") else ""),
        "=" * 72,
        "\n-- Classification under review --",
        v.get("classification", "").strip(),
        "\n-- Evidence the agent was given --",
        v.get("evidence_text", "").strip(),
        "\n-- Retrieved ACMG criteria the agent was given --",
        v.get("criteria_text", "").strip(),
    ]
    return "\n".join(lines)


def make_label_record(case: dict, label: str, rationale: str, offending: str, labeler: str) -> dict:
    return {
        "id": case["id"],
        "variant": case.get("variant"),
        "label": label,
        "rationale": rationale.strip(),
        "offending_criterion": offending.strip(),
        "labeler": labeler,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
    }


def append_label(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _prompt_verdict() -> str | None:
    """Return 'grounded' / 'ungrounded', 'SKIP', or None to quit."""
    while True:
        choice = input("\n[g]rounded  [u]ngrounded  [s]kip  [q]uit > ").strip().lower()
        if choice in ("g", "grounded"):
            return "grounded"
        if choice in ("u", "ungrounded"):
            return "ungrounded"
        if choice in ("s", "skip"):
            return "SKIP"
        if choice in ("q", "quit"):
            return None
        print("  (please answer g, u, s, or q)")


def run(cases_path: Path, labels_path: Path, labeler: str, seed: int) -> None:
    cases = load_cases(cases_path)
    labels = load_labels(labels_path)
    todo = pending(cases, labels, seed)

    print(f"{len(cases)} cases total, {len(labels)} already labeled, {len(todo)} to go.")
    if not todo:
        print("Nothing left to label. Done.")
        return

    labeled_now = 0
    for i, case in enumerate(todo, start=1):
        print(render_case(case, position=len(labels) + labeled_now + 1, total=len(cases)))
        verdict = _prompt_verdict()
        if verdict is None:
            print(f"\nStopped. Labeled {labeled_now} this session; run again to continue.")
            return
        if verdict == "SKIP":
            print("  skipped -- will reappear next run.")
            continue
        rationale = input("  reason (one line): ").strip()
        offending = ""
        if verdict == "ungrounded":
            offending = input("  offending criterion/claim (optional): ").strip()
        append_label(labels_path, make_label_record(case, verdict, rationale, offending, labeler))
        labeled_now += 1

    print(f"\nAll cases labeled. {labeled_now} this session; labels in {labels_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind-label groundedness calibration cases.")
    parser.add_argument("--input", type=Path, default=CASES_DEFAULT, help="cases.jsonl to label")
    parser.add_argument("--out", type=Path, default=LABELS_DEFAULT, help="labels output (resumable)")
    parser.add_argument("--labeler", default="", help="your name/initials, recorded on each label")
    parser.add_argument("--seed", type=int, default=42, help="presentation-order shuffle seed")
    args = parser.parse_args()

    labeler = args.labeler or input("Labeler (name/initials): ").strip() or "anon"
    if not args.input.exists():
        sys.exit(f"No cases at {args.input}. Build them first: python evals/calibration/build_cases.py")
    run(args.input, args.out, labeler, args.seed)


if __name__ == "__main__":
    main()
