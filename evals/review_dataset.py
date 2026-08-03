"""Walk the golden dataset one variant at a time, for human review (read-only).

Prints each row's gold-label metadata next to the frozen fixture evidence the
agent actually sees (ClinVar / gnomAD / VEP / UCSC / AlphaMissense), so you can
eyeball whether the gold_label and expected_behavior are defensible against the
evidence. It never writes anything -- reviewing the dataset, not editing it
(DATASET.md: "never silently edit").

Usage:
    python evals/review_dataset.py                      # all 37, one per screen
    python evals/review_dataset.py --label VUS          # only gold_label VUS
    python evals/review_dataset.py --behavior abstain   # only the abstain subset
    python evals/review_dataset.py --difficulty hardest
    python evals/review_dataset.py --source synthetic
    python evals/review_dataset.py --id gv-028          # jump to one row

At each variant: Enter = next, q = quit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import replay  # noqa: E402

DATASET = Path(__file__).resolve().parent / "golden_dataset.jsonl"


def load_rows() -> list[dict]:
    return [json.loads(line) for line in DATASET.open() if line.strip()]


def _fixture(tool: str, variant: str) -> dict | None:
    try:
        return replay.load_fixture(tool, variant)
    except replay.FixtureMissing:
        return None


def _clinvar(fx: dict | None) -> str:
    if not fx:
        return "(no fixture)"
    matches = fx.get("matches", [])
    if not matches:
        return "no ClinVar record"
    return "; ".join(
        f"{m.get('clinical_significance','?')} [{m.get('review_status','?')}]" for m in matches
    )


def _gnomad(fx: dict | None) -> str:
    if not fx:
        return "(no fixture)"
    if not fx.get("found"):
        return "absent from gnomAD"
    af = fx.get("allele_freq")
    return f"AF = {af:.4%}" if isinstance(af, (int, float)) else json.dumps(fx)[:120]


def _ensembl(fx: dict | None) -> str:
    if not fx:
        return "(no fixture)"
    if not fx.get("found"):
        return "not resolved by VEP"
    return f"{fx.get('most_severe_consequence','?')} (impact={fx.get('impact','?')}, gene={fx.get('gene_symbol','?')})"


def _ucsc(fx: dict | None) -> str:
    if not fx or not fx.get("found"):
        return "(no fixture)" if not fx else "no conservation"
    return f"phyloP={fx.get('phylop')}, phastCons={fx.get('phastcons')}"


def _alphamissense(fx: dict | None) -> str:
    if not fx:
        return "(no fixture)"
    if not fx.get("applicable", True):
        return "not applicable (non-missense)"
    if not fx.get("found"):
        return "missense, but not in the AlphaMissense table"
    return f"score={fx.get('am_pathogenicity')} (class={fx.get('am_class')})"


def render(row: dict, position: int, total: int) -> str:
    v = row["variant"]
    lines = [
        "=" * 78,
        f"[{position}/{total}]  {row['id']}   {row['gene']}   {v}",
        "=" * 78,
        f"  gold_label        : {row['gold_label']}",
        f"  expected_behavior : {row['expected_behavior']}",
        f"  difficulty        : {row['difficulty']}   (review_stars={row.get('review_stars')})",
        f"  evidence_type     : {row['evidence_type']}",
        f"  clinvar_significance (gold provenance): {row.get('clinvar_significance')}",
        f"  hgvs_name         : {row.get('hgvs_name')}",
        f"  source            : {row['source']}"
        + (f"   synthetic_rule={row['synthetic_rule']}" if row.get("synthetic_rule") else ""),
        "",
        "  -- frozen evidence the agent sees (evals/fixtures) --",
        f"    ClinVar       : {_clinvar(_fixture('clinvar', v))}",
        f"    gnomAD        : {_gnomad(_fixture('gnomad', v))}",
        f"    Ensembl VEP   : {_ensembl(_fixture('ensembl', v))}",
        f"    UCSC          : {_ucsc(_fixture('ucsc', v))}",
        f"    AlphaMissense : {_alphamissense(_fixture('alphamissense', v))}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk the golden dataset for review (read-only).")
    parser.add_argument("--label", choices=["P", "VUS", "B"], help="filter by gold_label")
    parser.add_argument("--behavior", choices=["classify", "abstain"], help="filter by expected_behavior")
    parser.add_argument("--difficulty", choices=["easy", "hard", "hardest"], help="filter by difficulty")
    parser.add_argument("--source", choices=["clinvar", "synthetic"], help="filter by source")
    parser.add_argument("--id", help="jump to a single row id (e.g. gv-028)")
    args = parser.parse_args()

    rows = load_rows()
    if args.id:
        rows = [r for r in rows if r["id"] == args.id]
    if args.label:
        rows = [r for r in rows if r["gold_label"] == args.label]
    if args.behavior:
        rows = [r for r in rows if r["expected_behavior"] == args.behavior]
    if args.difficulty:
        rows = [r for r in rows if r["difficulty"] == args.difficulty]
    if args.source:
        rows = [r for r in rows if r["source"] == args.source]

    if not rows:
        sys.exit("No rows match those filters.")

    total = len(rows)
    print(f"Reviewing {total} row(s). Enter = next, q = quit.\n")
    for i, row in enumerate(rows, start=1):
        print(render(row, i, total))
        if i < total:
            if input("\n  [Enter] next / [q] quit > ").strip().lower() == "q":
                print("Stopped.")
                return
    print("\nEnd of set.")


if __name__ == "__main__":
    main()
