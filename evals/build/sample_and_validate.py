"""Stage 3: sample a target composition from pool.json, live-validate each
candidate against the pipeline's own Ensembl VEP tool (the hard, unconditional
dependency in graph.py's gather_evidence), and enrich the survivors with
gnomAD / AlphaMissense evidence where relevant to the intended evidence_type.

A candidate is dropped (and the next one in its cell tried) if:
  - Ensembl VEP doesn't resolve the rsID (found=False) -- the agent would get
    no consequence evidence for reasons unrelated to intended difficulty.
  - Ensembl's gene_symbol doesn't match ClinVar's GeneSymbol -- rsIDs are
    genome-wide identifiers; a mismatch usually means the ClinVar record's
    HGVS transcript disagrees with VEP's default transcript choice enough
    that this rsID isn't a clean instance of the intended gene/mechanism.

Mechanism -> gene cell -> desired evidence_type:
  LoF genes    + Pathogenic -> frameshift/LoF   (PVS1 exercised)
  Missense gen + P/B        -> missense/computational (AlphaMissense exercised)
  Freq genes   + Benign     -> population_frequency (BA1/BS1 exercised)
"""
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from variant_audit.mcp_tools.ensembl import get_gene_consequence
from variant_audit.mcp_tools.gnomad import get_allele_frequency
from variant_audit.mcp_tools.alphamissense import get_alphamissense_score

POOL = Path(__file__).parent / "pool.json"
OUT = Path(__file__).parent / "validated_selection.jsonl"
LOG = Path(__file__).parent / "validation_log.txt"

random.seed(17)

LOF_GENES = {"BRCA1", "BRCA2", "MLH1", "MSH2", "MSH6", "PMS2", "APC", "PTEN",
             "TP53", "LDLR", "VHL", "NF1", "RB1"}
MISSENSE_GENES = {"MYH7", "KCNQ1", "SCN5A", "RYR1", "COL1A1", "FBN1", "GJB2"}
FREQ_GENES = {"CFTR", "HFE", "MTHFR", "F5", "SERPINA1", "APOE"}

LOF_CONSEQUENCES = {
    "frameshift_variant", "stop_gained", "splice_donor_variant",
    "splice_acceptor_variant", "stop_lost", "start_lost",
}

# (bucket, gold_label, gene_set, count, evidence_type)
PLAN = [
    ("easy", "P", LOF_GENES, 3, "frameshift/LoF"),
    ("easy", "P", MISSENSE_GENES, 3, "missense/computational"),
    ("easy", "B", FREQ_GENES, 3, "population_frequency"),
    ("easy", "B", MISSENSE_GENES, 3, "missense/computational"),
    ("hard", "P", LOF_GENES, 3, "frameshift/LoF"),
    ("hard", "P", MISSENSE_GENES, 3, "missense/computational"),
    ("hard", "B", FREQ_GENES, 3, "population_frequency"),
    ("hard", "B", MISSENSE_GENES, 3, "missense/computational"),
    ("hardest_vus_confident", "VUS", LOF_GENES | MISSENSE_GENES, 6, "mixed"),
    ("hardest_abstain", "VUS", LOF_GENES | MISSENSE_GENES | FREQ_GENES, 6, "conflicting"),
]


def evidence_type_from_consequence(most_severe: str | None, fallback: str) -> str:
    if most_severe in LOF_CONSEQUENCES:
        return "frameshift/LoF"
    if most_severe == "missense_variant":
        return "missense/computational"
    return fallback


def main() -> None:
    with open(POOL) as f:
        buckets = json.load(f)

    log = open(LOG, "w")
    selected = []
    used_rsids = set()
    gene_use_count = defaultdict(int)

    for bucket, gold_label_filter, gene_set, count, evidence_hint in PLAN:
        items = [
            r for r in buckets[bucket]
            if r["gene"] in gene_set
            and (r["gold_label"] == gold_label_filter or (bucket == "hardest_abstain"))
            and r["rsid"] not in used_rsids
        ]
        random.shuffle(items)
        # round-robin by gene to spread across the gene set, cap repeats
        items.sort(key=lambda r: gene_use_count[r["gene"]])

        picked = 0
        for rec in items:
            if picked >= count:
                break
            if gene_use_count[rec["gene"]] >= 3:
                continue
            rsid = rec["rsid"]
            time.sleep(0.3)
            try:
                vep = get_gene_consequence(rsid)
            except Exception as e:
                log.write(f"DROP {rsid} ({rec['gene']}): VEP raised {e}\n")
                continue
            if not vep.get("found"):
                log.write(f"DROP {rsid} ({rec['gene']}): VEP not found\n")
                continue
            if (vep.get("gene_symbol") or "").upper() != rec["gene"].upper():
                log.write(
                    f"DROP {rsid} ({rec['gene']}): VEP gene_symbol mismatch "
                    f"({vep.get('gene_symbol')})\n"
                )
                continue

            evidence_type = evidence_type_from_consequence(
                vep.get("most_severe_consequence"), evidence_hint
            )

            enrichment = {}
            if evidence_hint == "population_frequency":
                time.sleep(0.3)
                try:
                    freq = get_allele_frequency(rsid)
                    allele_freq = freq.get("allele_freq")
                except Exception as e:
                    log.write(f"WARN {rsid}: gnomAD call failed {e}\n")
                    continue
                # BA1/BS1 only applies if the variant is genuinely common --
                # ClinVar's "Benign" here could just as easily mean "synonymous,
                # no functional impact" (BP7), which is a different evidence
                # type entirely. Don't accept a population_frequency slot on a
                # variant gnomAD shows as rare or unobserved.
                if allele_freq is None or allele_freq < 0.01:
                    log.write(
                        f"DROP {rsid} ({rec['gene']}): population_frequency slot but "
                        f"allele_freq={allele_freq} (not common)\n"
                    )
                    continue
                enrichment["allele_freq"] = allele_freq
            if evidence_type == "missense/computational":
                time.sleep(0.3)
                try:
                    am = get_alphamissense_score(rsid, consequence=vep)
                    enrichment["alphamissense"] = am if am.get("found") else None
                except Exception as e:
                    log.write(f"WARN {rsid}: AlphaMissense call failed {e}\n")

            rec_out = {
                **rec,
                "evidence_type": evidence_type,
                "difficulty": (
                    "easy" if bucket == "easy" else
                    "hard" if bucket == "hard" else
                    "hardest"
                ),
                "expected_behavior": "abstain" if bucket == "hardest_abstain" else "classify",
                "gold_label": rec["gold_label"] or "VUS",
                "vep_most_severe_consequence": vep.get("most_severe_consequence"),
                **enrichment,
            }
            selected.append(rec_out)
            used_rsids.add(rsid)
            gene_use_count[rec["gene"]] += 1
            picked += 1
            log.write(f"KEEP {rsid} ({rec['gene']}, {bucket}, {evidence_type})\n")

        if picked < count:
            log.write(f"!! bucket {bucket}/{gold_label_filter}/{evidence_hint}: only found {picked}/{count}\n")

    with open(OUT, "w") as f:
        for rec in selected:
            f.write(json.dumps(rec) + "\n")

    log.close()
    print(f"selected {len(selected)} candidates -> {OUT}")
    print(f"log -> {LOG}")


if __name__ == "__main__":
    main()
