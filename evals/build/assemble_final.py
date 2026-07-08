"""Stage 5: assemble the final evals/golden_dataset.jsonl from the validated
real anchor (36 rows) + the rule-mined synthetic entry (1 row).

Schema note: `variant` holds the rsID, not HGVS -- graph.py's gather_evidence
passes this string unmodified to gnomAD/Ensembl/UCSC/AlphaMissense, all of
which require an rsID. `hgvs_name` carries the human-readable ClinVar name
for provenance/readability only; it is never fed to a tool.
"""
import json
from pathlib import Path

REAL = Path(__file__).parent / "validated_selection.jsonl"
SYNTHETIC = Path(__file__).parent / "synthetic_selection.jsonl"
OUT = Path(__file__).parent.parent / "golden_dataset.jsonl"

DIFFICULTY_ORDER = {"easy": 0, "hard": 1, "hardest": 2}
LABEL_ORDER = {"P": 0, "B": 1, "VUS": 2}


def to_row(rec: dict, source: str) -> dict:
    row = {
        "variant": rec["rsid"],
        "hgvs_name": rec["name"],
        "gene": rec["gene"],
        "gold_label": rec["gold_label"],
        "clinvar_significance": rec["clinvar_significance"],
        "review_stars": rec["review_stars"],
        "evidence_type": rec["evidence_type"],
        "difficulty": rec["difficulty"],
        "expected_behavior": rec["expected_behavior"],
        "source": source,
    }
    if rec.get("allele_freq") is not None:
        row["allele_freq"] = rec["allele_freq"]
    if rec.get("vep_most_severe_consequence"):
        row["vep_most_severe_consequence"] = rec["vep_most_severe_consequence"]
    if rec.get("synthetic_rule"):
        row["synthetic_rule"] = rec["synthetic_rule"]
    return row


def main() -> None:
    real = [json.loads(l) for l in open(REAL)]
    synthetic = [json.loads(l) for l in open(SYNTHETIC)] if SYNTHETIC.exists() else []

    real.sort(key=lambda r: (DIFFICULTY_ORDER[r["difficulty"]], LABEL_ORDER[r["gold_label"]], r["gene"]))

    rows = []
    for i, rec in enumerate(real, start=1):
        row = to_row(rec, "clinvar")
        row["id"] = f"gv-{i:03d}"
        rows.append(row)
    for i, rec in enumerate(synthetic, start=1):
        row = to_row(rec, "synthetic")
        row["id"] = f"syn-{i:03d}"
        rows.append(row)

    # id first for readability
    ordered_rows = []
    for row in rows:
        ordered = {"id": row.pop("id")}
        ordered.update(row)
        ordered_rows.append(ordered)

    with open(OUT, "w") as f:
        for row in ordered_rows:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(ordered_rows)} rows ({len(real)} real + {len(synthetic)} synthetic) -> {OUT}")


if __name__ == "__main__":
    main()
