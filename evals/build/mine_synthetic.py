"""Stage 4: rule-mined synthetic augmentation.

"Synthetic" here does NOT mean fabricated variants -- every tool in this
pipeline requires a real, resolvable rsID, so an invented variant would just
fail to run. Instead: mine the full 2.88M-row candidate pool for a specific
adversarial pattern using an automated rule, rather than hand-picking one at
a time. Real rsID, real ClinVar provenance; the *selection* is programmatic.

Rule: "look-alike trap" -- a variant ClinVar calls Pathogenic/Likely
pathogenic on a single submitter's say-so (review_stars <= 1, not
conflicting), but which gnomAD shows is actually common in the population
(allele_freq > 0.01). BA1/BS1 says a variant this common in the general
population can't be causing a rare Mendelian disease -- so a naive agent
that trusts the ClinVar label at face value gets this wrong, while one that
weighs the frequency evidence catches the contradiction. This tests
something the hand-picked 36 don't: agreement between evidence sources, not
just single-source difficulty.

Correct agent behavior: flag as VUS/uncertain (or explicitly note the
label-frequency conflict) rather than parroting "Pathogenic".
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from variant_audit.mcp_tools.ensembl import get_gene_consequence
from variant_audit.mcp_tools.gnomad import get_allele_frequency

CANDIDATES = Path(__file__).parent / "candidates.jsonl"
ALREADY_USED = Path(__file__).parent / "validated_selection.jsonl"
OUT = Path(__file__).parent / "synthetic_selection.jsonl"
LOG = Path(__file__).parent / "synthetic_log.txt"

TARGET = 5
SAMPLE_BUDGET = 250  # cap on gnomAD calls if hits are rare
FREQ_THRESHOLD = 0.01
BASE_SLEEP = 1.5
MAX_RETRIES = 4

random.seed(42)


def _is_429(exc: Exception) -> bool:
    cause = exc.__cause__
    resp = getattr(cause, "response", None)
    return resp is not None and resp.status_code == 429


def query_with_backoff(fn, rsid, log):
    delay = BASE_SLEEP
    for attempt in range(MAX_RETRIES):
        time.sleep(delay)
        try:
            return fn(rsid)
        except Exception as e:
            if _is_429(e) and attempt < MAX_RETRIES - 1:
                delay *= 3
                log.write(f"RETRY {rsid}: 429, backing off to {delay:.1f}s\n")
                continue
            raise
    raise RuntimeError("unreachable")


def main() -> None:
    used_rsids = {json.loads(l)["rsid"] for l in open(ALREADY_USED)}

    pool = []
    with open(CANDIDATES) as f:
        for line in f:
            rec = json.loads(line)
            if (
                rec["gold_label"] == "P"
                and not rec["conflicting"]
                and rec["review_stars"] <= 1
                and rec["rsid"] not in used_rsids
            ):
                pool.append(rec)
    print(f"candidate pool for look-alike mining: {len(pool)}", file=sys.stderr)

    random.shuffle(pool)
    pool = pool[:SAMPLE_BUDGET]

    log = open(LOG, "w")
    hits = []
    checked = 0
    for rec in pool:
        if len(hits) >= TARGET:
            break
        checked += 1
        rsid = rec["rsid"]
        try:
            freq_rec = query_with_backoff(get_allele_frequency, rsid, log)
        except Exception as e:
            log.write(f"SKIP {rsid}: gnomAD call failed after retries {e}\n")
            continue
        af = freq_rec.get("allele_freq")
        if af is None or af <= FREQ_THRESHOLD:
            continue  # not a hit -- rare/unobserved, no contradiction

        # Confirm the rsID actually resolves and matches the intended gene
        # before locking it in (same bar as the hand-picked anchor set).
        try:
            vep = query_with_backoff(get_gene_consequence, rsid, log)
        except Exception as e:
            log.write(f"SKIP {rsid}: VEP call failed after retries {e}\n")
            continue
        if not vep.get("found") or (vep.get("gene_symbol") or "").upper() != rec["gene"].upper():
            log.write(f"SKIP {rsid}: VEP not found / gene mismatch\n")
            continue

        rec_out = {
            **rec,
            "gold_label": "VUS",  # BA1-style contradiction -> not a confident P
            "evidence_type": "conflicting_evidence_sources",
            "difficulty": "hardest",
            "expected_behavior": "abstain",
            "allele_freq": af,
            "vep_most_severe_consequence": vep.get("most_severe_consequence"),
            "source": "synthetic",
            "synthetic_rule": "look_alike_trap_frequency_vs_label",
        }
        hits.append(rec_out)
        log.write(f"HIT {rsid} ({rec['gene']}): allele_freq={af}, clinvar={rec['clinvar_significance']}\n")

    with open(OUT, "w") as f:
        for rec in hits:
            f.write(json.dumps(rec) + "\n")

    log.write(f"\nchecked {checked} candidates, found {len(hits)} hits\n")
    log.close()
    print(f"found {len(hits)}/{TARGET} look-alike traps after checking {checked} candidates -> {OUT}")


if __name__ == "__main__":
    main()
