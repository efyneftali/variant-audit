"""Stage 2: narrow the 2.88M in-scope candidates to a curated gene list, then
bucket by difficulty tier, so we have a manageable pool to live-validate and
hand-pick from.

Gene list is deliberately chosen to spread mechanism (so different ACMG
criteria get exercised, per DATASET.md):
  - LoF/PVS1 genes (frameshift/nonsense/splice matter most): BRCA1, BRCA2,
    MLH1, MSH2, MSH6, PMS2, APC, PTEN, TP53, LDLR, VHL, NF1, RB1
  - Missense/computational-driven genes (AlphaMissense matters most): MYH7,
    KCNQ1, SCN5A, RYR1, COL1A1, FBN1, GJB2
  - Common-variant/population-frequency genes (BA1/BS1 candidates): CFTR,
    HFE, MTHFR, F5, SERPINA1, APOE

Buckets (difficulty tier, per DATASET.md):
  - easy:    review_stars >= 2, not conflicting, gold_label in {P, B}
  - hard:    review_stars in {0, 1}, not conflicting, gold_label in {P, B}
  - hardest_vus_confident: review_stars >= 2, not conflicting, gold_label == VUS
  - hardest_abstain: conflicting == True (our abstention moat)
"""
import json
from collections import defaultdict
from pathlib import Path

CANDIDATES = Path(__file__).parent / "candidates.jsonl"
OUT = Path(__file__).parent / "pool.json"

GENES = {
    # LoF-mechanism
    "BRCA1", "BRCA2", "MLH1", "MSH2", "MSH6", "PMS2", "APC", "PTEN", "TP53",
    "LDLR", "VHL", "NF1", "RB1",
    # missense/computational-mechanism
    "MYH7", "KCNQ1", "SCN5A", "RYR1", "COL1A1", "FBN1", "GJB2",
    # common-variant/frequency-mechanism
    "CFTR", "HFE", "MTHFR", "F5", "SERPINA1", "APOE",
}


def bucket_for(rec: dict) -> str | None:
    if rec["gene"] not in GENES:
        return None
    if rec["conflicting"]:
        return "hardest_abstain"
    if rec["gold_label"] == "VUS":
        return "hardest_vus_confident" if rec["review_stars"] >= 2 else "hardest_abstain_lowstar_vus"
    if rec["gold_label"] in ("P", "B"):
        if rec["review_stars"] >= 2:
            return "easy"
        return "hard"
    return None


def main() -> None:
    buckets: dict[str, list[dict]] = defaultdict(list)
    with open(CANDIDATES) as f:
        for line in f:
            rec = json.loads(line)
            b = bucket_for(rec)
            if b:
                buckets[b].append(rec)

    for name, items in buckets.items():
        by_gene = defaultdict(int)
        for it in items:
            by_gene[it["gene"]] += 1
        print(f"{name}: {len(items)} total, by gene: {dict(sorted(by_gene.items()))}")

    with open(OUT, "w") as f:
        json.dump(buckets, f)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
