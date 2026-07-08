"""Stage 1: filter ClinVar's variant_summary.txt.gz down to a candidate pool.

Filters applied (each documented in DATASET.md):
  1. Assembly == GRCh38 (the file has one row per variant per assembly; GRCh37
     rows are exact duplicates of GRCh38 rows and would double-count).
  2. Has an rsID (RS# (dbSNP) != -1) — four of the five MCP tools
     (gnomAD, Ensembl/VEP, UCSC, AlphaMissense) take only an rsID as input.
  3. ClinicalSignificance maps cleanly to the 3-class scheme (P/B/VUS) or is a
     conflicting-interpretations call (kept separately for the abstention set).
     Rows that are risk factor / drug response / association / protective /
     not provided are dropped — they don't fit the collapse.

Output: evals/build/candidates.jsonl — one row per (deduped) variant, with
review stars computed from ReviewStatus text and gold_label already collapsed.
"""
import csv
import gzip
import json
import sys
from pathlib import Path

IN_PATH_GZ = Path(__file__).parent / "variant_summary.txt.gz"
IN_PATH_TXT = Path(__file__).parent / "variant_summary.txt"
OUT_PATH = Path(__file__).parent / "candidates.jsonl"


def _open_input():
    if IN_PATH_TXT.exists():
        return open(IN_PATH_TXT, "rt", encoding="utf-8", errors="replace")
    return gzip.open(IN_PATH_GZ, "rt", encoding="utf-8", errors="replace")

# ClinVar's ReviewStatus text -> gold star rating.
# https://www.ncbi.nlm.nih.gov/clinvar/docs/review_status/
STAR_MAP = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "no assertion criteria provided": 0,
    "no assertion provided": 0,
    "no classification provided": 0,
    "no classification for the single variant": 0,
}

PATHOGENIC_TERMS = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN_TERMS = {"Benign", "Likely benign", "Benign/Likely benign"}
VUS_TERMS = {"Uncertain significance"}


def stars_for(review_status: str) -> int:
    return STAR_MAP.get(review_status.strip().lower(), -1)


def collapse_label(clinical_significance: str) -> str | None:
    """Collapse raw ClinicalSignificance to P/B/VUS, or None if out of scope."""
    # ClinicalSignificance can be multi-valued across submissions, e.g.
    # "Pathogenic|Uncertain significance" for a variant with discordant
    # historical calls — treat any pipe-delimited or "Conflicting" value as
    # conflicting, handled by the caller via is_conflicting().
    if clinical_significance in PATHOGENIC_TERMS:
        return "P"
    if clinical_significance in BENIGN_TERMS:
        return "B"
    if clinical_significance in VUS_TERMS:
        return "VUS"
    return None


def is_conflicting(clinical_significance: str) -> bool:
    return (
        "conflicting" in clinical_significance.lower()
        or "|" in clinical_significance
    )


def main() -> None:
    seen_variation_ids: set[str] = set()
    n_total = 0
    n_grch38 = 0
    n_has_rsid = 0
    n_in_scope = 0
    kept = []

    with _open_input() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_total += 1
            if row["Assembly"] != "GRCh38":
                continue
            n_grch38 += 1

            rs = row["RS# (dbSNP)"].strip()
            if rs in ("-1", "", "na"):
                continue
            n_has_rsid += 1

            variation_id = row["VariationID"]
            if variation_id in seen_variation_ids:
                continue
            seen_variation_ids.add(variation_id)

            sig = row["ClinicalSignificance"].strip()
            conflicting = is_conflicting(sig)
            label = None if conflicting else collapse_label(sig)
            if not conflicting and label is None:
                continue  # out-of-scope category (risk factor, drug response, ...)
            n_in_scope += 1

            stars = stars_for(row["ReviewStatus"])
            kept.append({
                "variation_id": variation_id,
                "rsid": f"rs{rs}",
                "name": row["Name"],
                "gene": row["GeneSymbol"],
                "clinvar_significance": sig,
                "review_stars": stars,
                "conflicting": conflicting,
                "gold_label": label,  # None if conflicting
                "review_status_raw": row["ReviewStatus"],
                "last_evaluated": row["LastEvaluated"],
            })

    with open(OUT_PATH, "w") as out:
        for rec in kept:
            out.write(json.dumps(rec) + "\n")

    print(f"total rows read:        {n_total}", file=sys.stderr)
    print(f"GRCh38 rows:            {n_grch38}", file=sys.stderr)
    print(f"with rsID:              {n_has_rsid}", file=sys.stderr)
    print(f"in-scope (deduped):     {n_in_scope}", file=sys.stderr)
    print(f"wrote:                  {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
