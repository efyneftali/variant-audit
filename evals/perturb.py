"""VA-13 adversarial-robustness perturbation layer (Day 14).

Rewrites the free-text fields of the frozen evidence fixtures into a
paraphrased / formatting-noisy surface form -- same facts, different words --
so eval_robustness can measure whether classify()'s answer holds when the
evidence is phrased or capitalized differently. That's realistic noise: real
ClinVar submitters/tools vary in wording and formatting constantly without
changing what's actually true about a variant.

Deliberately narrow: only prose/categorical text fields (clinical_significance,
review_status, hgvs, most_severe_consequence, am_class) get paraphrased, plus
light case/whitespace noise on top. Numeric facts (allele frequencies,
positions, conservation scores) and the `found`/`variant` keys are never
touched -- mutating those would change the evidence itself, not its surface
form, and corrupt the ground truth the accuracy delta is measured against.
Deterministic per (tool, variant): a robustness run is reproducible and
diffable across code/prompt changes, not a fresh coin flip each time.

Installed the same way replay.py installs fixture replay: rebinds the graph
module's five tool names for the eval process only, wrapping replay's own
fixture reader so the two layer cleanly (replay reads the exact-match frozen
fixture -- unaffected by any of this -- then this module mutates the free
text before gather_evidence sees it).
"""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import replay  # noqa: E402
from src.variant_audit import graph  # noqa: E402

# Meaning-preserving paraphrases for the closed vocabularies actually observed
# in evals/fixtures/{clinvar,ensembl,alphamissense}/*.json (see build_cases.py
# equivalents / DATASET.md for where these tiers come from). Index 0 is always
# the original string -- rng.choice over the whole list means "no paraphrase"
# is itself a valid noisy sample, same as real-world data isn't perturbed 100%
# of the time.
CLINICAL_SIGNIFICANCE_PARAPHRASE = {
    "Pathogenic": ["Pathogenic", "classified as pathogenic", "disease-causing (pathogenic)"],
    "Likely pathogenic": ["Likely pathogenic", "considered likely pathogenic", "probably disease-causing"],
    "Benign": ["Benign", "classified as benign", "not disease-causing (benign)"],
    "Likely benign": ["Likely benign", "considered likely benign", "probably not disease-causing"],
    "Uncertain significance": [
        "Uncertain significance", "of uncertain clinical significance", "significance not yet established",
    ],
    "Benign/Likely benign": ["Benign/Likely benign", "benign to likely benign", "benign, possibly likely benign"],
    "Pathogenic/Likely pathogenic": [
        "Pathogenic/Likely pathogenic", "pathogenic to likely pathogenic", "pathogenic, possibly likely pathogenic",
    ],
    "Conflicting classifications of pathogenicity": [
        "Conflicting classifications of pathogenicity",
        "submitters disagree on pathogenicity classification",
        "pathogenicity classification is contested among submitters",
    ],
}

REVIEW_STATUS_PARAPHRASE = {
    "reviewed by expert panel": [
        "reviewed by expert panel", "expert-panel reviewed", "vetted by an expert review panel",
    ],
    "criteria provided, multiple submitters, no conflicts": [
        "criteria provided, multiple submitters, no conflicts",
        "multiple submitters agree, criteria provided",
        "criteria met; several submitters, no disagreement",
    ],
    "criteria provided, single submitter": [
        "criteria provided, single submitter",
        "one submitter, criteria provided",
        "single-submitter assertion meeting criteria",
    ],
    "criteria provided, conflicting classifications": [
        "criteria provided, conflicting classifications",
        "criteria provided but submitters conflict",
        "conflicting classifications despite criteria being provided",
    ],
    "no assertion criteria provided": [
        "no assertion criteria provided", "assertion criteria not provided", "submitted without assertion criteria",
    ],
}

CONSEQUENCE_PARAPHRASE = {
    "missense_variant": ["missense_variant", "a missense variant", "missense change"],
    "frameshift_variant": ["frameshift_variant", "a frameshift variant", "frameshift-causing change"],
    "stop_gained": ["stop_gained", "a premature stop codon (nonsense)", "stop-gained (nonsense) variant"],
    "stop_lost": ["stop_lost", "loss of the stop codon", "stop-lost variant"],
    "synonymous_variant": ["synonymous_variant", "a synonymous (silent) variant", "silent/synonymous change"],
    "intron_variant": ["intron_variant", "an intronic variant", "variant within an intron"],
    "splice_acceptor_variant": [
        "splice_acceptor_variant", "a splice-acceptor variant", "disrupts the splice acceptor site",
    ],
    "splice_donor_variant": ["splice_donor_variant", "a splice-donor variant", "disrupts the splice donor site"],
    "5_prime_UTR_variant": ["5_prime_UTR_variant", "a 5' UTR variant", "variant in the 5' untranslated region"],
    "upstream_gene_variant": ["upstream_gene_variant", "an upstream gene variant", "variant upstream of the gene"],
}

AM_CLASS_PARAPHRASE = {
    "likely_pathogenic": ["likely_pathogenic", "likely pathogenic", "AlphaMissense-predicted likely pathogenic"],
    "likely_benign": ["likely_benign", "likely benign", "AlphaMissense-predicted likely benign"],
}


def _paraphrase(value: str, table: dict[str, list[str]], rng: random.Random) -> str:
    options = table.get(value)
    return rng.choice(options) if options else value


def _case_and_whitespace_noise(text: str, rng: random.Random) -> str:
    """Formatting noise: title/upper-case a random word, insert a stray double
    space -- the kind of drift real free-text fields accumulate without
    changing meaning."""
    words = text.split(" ")
    if words and rng.random() < 0.5:
        i = rng.randrange(len(words))
        words[i] = words[i].upper() if rng.random() < 0.5 else words[i].capitalize()
    noisy = " ".join(words)
    if rng.random() < 0.5:
        idx = rng.randrange(len(noisy) + 1)
        noisy = noisy[:idx] + " " + noisy[idx:]
    return noisy


def _perturb_clinvar(record: dict, rng: random.Random) -> dict:
    out = copy.deepcopy(record)
    for match in out.get("matches", []):
        if "clinical_significance" in match:
            match["clinical_significance"] = _case_and_whitespace_noise(
                _paraphrase(match["clinical_significance"], CLINICAL_SIGNIFICANCE_PARAPHRASE, rng), rng
            )
        if "review_status" in match:
            match["review_status"] = _case_and_whitespace_noise(
                _paraphrase(match["review_status"], REVIEW_STATUS_PARAPHRASE, rng), rng
            )
        if "hgvs" in match:
            match["hgvs"] = _case_and_whitespace_noise(match["hgvs"], rng)
    return out


def _perturb_ensembl(record: dict, rng: random.Random) -> dict:
    out = copy.deepcopy(record)
    if out.get("found"):
        if "most_severe_consequence" in out:
            out["most_severe_consequence"] = _paraphrase(out["most_severe_consequence"], CONSEQUENCE_PARAPHRASE, rng)
        if "gene_symbol" in out:
            out["gene_symbol"] = _case_and_whitespace_noise(out["gene_symbol"], rng)
    return out


def _perturb_alphamissense(record: dict, rng: random.Random) -> dict:
    out = copy.deepcopy(record)
    if out.get("applicable") and "am_class" in out:
        out["am_class"] = _paraphrase(out["am_class"], AM_CLASS_PARAPHRASE, rng)
    return out


# gnomad/ucsc carry no prose worth perturbing -- numeric fields only.
_PERTURBERS = {
    "clinvar": _perturb_clinvar,
    "ensembl": _perturb_ensembl,
    "alphamissense": _perturb_alphamissense,
}


def _make_perturbed_reader(tool: str):
    perturber = _PERTURBERS.get(tool)

    def wrapper(variant: str, *_args, **_kwargs) -> dict:
        raw = replay.load_fixture(tool, variant)
        if perturber is None:
            return raw
        rng = random.Random(f"{tool}:{variant}")  # deterministic per (tool, variant)
        return perturber(raw, rng)

    wrapper.__name__ = f"perturbed_{tool}"
    return wrapper


def install() -> None:
    """Rebind the five tool names to fixture-then-perturb readers. Process-
    scoped and idempotent, same contract as replay.install() -- call
    replay.install() again afterward to restore clean fixtures."""
    for tool, (attr, _real) in replay._TOOLS.items():
        setattr(graph, attr, _make_perturbed_reader(tool))
