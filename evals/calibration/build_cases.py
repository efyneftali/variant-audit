"""Freeze a groundedness calibration set for blind hand-labeling (VA-35, step 4).

A "groundedness case" is exactly what check_grounded judges: a variant, the
classification text the agent produced, and the evidence + ACMG criteria it was
given. This script assembles those cases from the frozen fixtures (evals/replay)
so they are reproducible, and writes them to cases.jsonl WITHOUT any label and
WITHOUT ever running the judge. The point is to hand a human a stable artifact to
label blind, before any model verdict exists to anchor on.

Two provenances of case:
  - "real": a genuine agent classification (gather -> retrieve -> classify).
    This is the natural distribution, and it is mostly grounded.
  - "injected_fabricated_code": a copy of a real case with a citation appended to
    an ACMG code that is provably absent from the retrieved criteria text. These
    are known-ungrounded by construction and exist only to give the set enough
    of the minority class that judge-vs-human agreement is measurable. They carry
    an `intended` verdict so a later step can check the human agreed -- but the
    human never sees that (label.py renders an allowlist of fields only).

Running this needs the same services a normal eval run needs: the generation
model (Ollama or Anthropic, via LLM_PROVIDER) for classify(), and Qdrant for
retrieval. It does NOT hit the five evidence tools -- those are replayed from
fixtures, so a missing fixture is a hard error (record it first).

Usage:
    python evals/calibration/build_cases.py                 # all 37 real + 8 injected
    python evals/calibration/build_cases.py --limit 30      # first 30 variants
    python evals/calibration/build_cases.py --injected 0    # real cases only
    python evals/calibration/build_cases.py --seed 7        # which variants get injected
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evals import replay  # noqa: E402
from src.variant_audit import graph  # noqa: E402

DATASET = Path(__file__).resolve().parent.parent / "golden_dataset.jsonl"
OUT_DEFAULT = Path(__file__).resolve().parent / "cases.jsonl"

# The canonical ACMG criterion codes. Used only to pick a code that is provably
# absent from a case's criteria text when injecting a known-ungrounded defect.
_ALL_ACMG_CODES = (
    ["PVS1", "BA1"]
    + [f"PS{i}" for i in range(1, 5)]
    + [f"PM{i}" for i in range(1, 7)]
    + [f"PP{i}" for i in range(1, 6)]
    + [f"BS{i}" for i in range(1, 5)]
    + [f"BP{i}" for i in range(1, 8)]
)


def load_dataset(limit: int | None = None) -> list[dict]:
    with DATASET.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def build_real_case(row: dict) -> dict:
    """Drive gather -> retrieve -> classify under replay and freeze the result.

    Stops at classify(): the judge (check_grounded) is deliberately NOT run, so
    nothing in the frozen case can anchor the human labeler.
    """
    state: dict = {"variant": row["variant"]}
    state.update(graph.gather_evidence(state))
    state.update(graph.retrieve_criteria(state))
    state.update(graph.classify(state))

    evidence_text, criteria_text = graph._build_evidence_and_criteria_text(state)
    return {
        "id": row["id"],
        "variant": row["variant"],
        "gene": row.get("gene"),
        "difficulty": row.get("difficulty"),
        "classification": state["classification"],
        "evidence_text": evidence_text,
        "criteria_text": criteria_text,
        "criteria_sources": sorted({c.source for c in state["criteria"]}),
        # underscore-prefixed: analysis metadata, never rendered to the labeler.
        "_provenance": {"kind": "real"},
    }


def absent_code(criteria_text: str) -> str | None:
    """An ACMG code that does NOT appear in `criteria_text`, or None if (absurdly)
    every code is already present. Reuses the judge's own extractor so 'present'
    means exactly what the deterministic pre-check means."""
    present = graph._acmg_codes(criteria_text)
    for code in _ALL_ACMG_CODES:
        if code not in present:
            return code
    return None


def inject_fabricated_code(case: dict) -> dict | None:
    """Copy a real case and append a citation to a criterion absent from the
    criteria text -- a clean, known-ungrounded example. Returns None if no absent
    code exists (so the caller can skip rather than emit a bogus case)."""
    code = absent_code(case["criteria_text"])
    if code is None:
        return None
    fabricated = dict(case)
    fabricated["id"] = f"{case['id']}#fab"
    fabricated["classification"] = (
        f"{case['classification'].rstrip()}\n"
        f"Additional criterion: {code} also applies here."
    )
    fabricated["_provenance"] = {
        "kind": "injected_fabricated_code",
        "from_id": case["id"],
        "fabricated_code": code,
        "intended": "ungrounded",
    }
    return fabricated


def build(rows: list[dict], n_injected: int, seed: int) -> list[dict]:
    cases: list[dict] = []
    skipped: list[str] = []
    for row in rows:
        try:
            cases.append(build_real_case(row))
        except Exception as exc:  # noqa: BLE001 - isolate a bad row, don't sink the pass
            skipped.append(f"{row['id']} ({row['variant']}): {exc}")
            print(f"  ! skipped {row['id']} ({row['variant']}): {exc}", file=sys.stderr)

    # Inject defects into a deterministic spread of the real cases we got.
    rng = random.Random(seed)
    pool = list(cases)
    rng.shuffle(pool)
    injected = 0
    for base in pool:
        if injected >= n_injected:
            break
        defect = inject_fabricated_code(base)
        if defect is not None:
            cases.append(defect)
            injected += 1

    if skipped:
        print(f"\nSkipped {len(skipped)} row(s); the rest were frozen.", file=sys.stderr)
    print(f"Built {len(cases)} cases ({len(cases) - injected} real, {injected} injected).")
    return cases


def write_cases(cases: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(
            "# groundedness calibration cases -- label these blind with "
            f"evals/calibration/label.py (built {datetime.now(timezone.utc).isoformat()})\n"
        )
        for case in cases:
            f.write(json.dumps(case) + "\n")
    print(f"Wrote {len(cases)} cases to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze groundedness cases for blind labeling.")
    parser.add_argument("--limit", type=int, default=None, help="only build the first N dataset variants")
    parser.add_argument("--injected", type=int, default=8, help="how many known-ungrounded defect cases to add (0 to disable)")
    parser.add_argument("--seed", type=int, default=13, help="which real cases get an injected defect")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT, help="output path")
    args = parser.parse_args()

    replay.install()  # five evidence tools -> fixtures; nothing hits the network
    rows = load_dataset(args.limit)
    cases = build(rows, n_injected=args.injected, seed=args.seed)
    write_cases(cases, args.out)


if __name__ == "__main__":
    main()
