"""Ensembl / VEP tool — variant consequence and gene annotation.

Source: Ensembl REST API (https://rest.ensembl.org), VEP endpoint.
Tells you what the variant *does* (missense, frameshift, splice, ...) and which
gene/transcript — needed to know which ACMG criteria even apply (e.g. PVS1 is
loss-of-function specific).
"""
import logging

import requests

logger = logging.getLogger(__name__)

VEP_BASE = "https://rest.ensembl.org/vep/human/id"
REQUEST_TIMEOUT = 40


def get_gene_consequence(variant: str) -> dict:
    """Look up a variant's molecular consequence and gene/transcript context via VEP.

    Args:
        variant: an rsID, e.g. "rs28897696".
    Returns:
        {variant, found, most_severe_consequence, impact, gene_symbol, chrom,
         start, end, allele_string}; found=False (with no other keys) when
        Ensembl has no variant matching this ID (unknown or malformed rsID).

    Raises:
        RuntimeError: if the VEP call fails (network error, an unexpected
        non-400 status, or an unexpected response shape) — logged with enough
        detail to diagnose which stage failed and why.
    """
    url = f"{VEP_BASE}/{variant}"
    logger.info("ensembl vep variant=%s", variant)
    try:
        resp = requests.get(url, params={"content-type": "application/json"}, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 400:
            logger.info("ensembl vep not found variant=%s body=%s", variant, resp.text[:200])
            return {"variant": variant, "found": False}
        resp.raise_for_status()
        results = resp.json()
    except requests.exceptions.RequestException as e:
        logger.exception("ensembl vep call failed variant=%s url=%s", variant, url)
        raise RuntimeError(f"Ensembl VEP call failed for variant '{variant}'") from e
    except ValueError as e:
        logger.exception("ensembl vep returned non-JSON response variant=%s body=%s", variant, resp.text[:500])
        raise RuntimeError(f"Ensembl VEP returned an unexpected response for variant '{variant}'") from e

    try:
        top = results[0]
        most_severe = top.get("most_severe_consequence")
        primary = _primary_transcript_consequence(most_severe, top.get("transcript_consequences", []))
        return {
            "variant": variant,
            "found": True,
            "most_severe_consequence": most_severe,
            "impact": primary.get("impact"),
            "gene_symbol": primary.get("gene_symbol"),
            "chrom": top.get("seq_region_name"),
            "start": top.get("start"),
            "end": top.get("end"),
            "allele_string": top.get("allele_string"),
        }
    except (KeyError, IndexError, TypeError) as e:
        logger.exception("ensembl vep returned unexpected shape variant=%s body=%s", variant, results)
        raise RuntimeError(f"Ensembl VEP returned an unexpected response shape for variant '{variant}'") from e


def _primary_transcript_consequence(most_severe: str | None, transcripts: list) -> dict:
    """Pick the transcript_consequences entry that best represents the overall call.

    A variant can hit many transcripts (and, at multiallelic sites, multiple
    alleles) with different consequences; prefer a protein_coding transcript
    whose consequence_terms include the overall most_severe_consequence.
    """
    candidates = [
        t for t in transcripts
        if most_severe in t.get("consequence_terms", []) and t.get("biotype") == "protein_coding"
    ]
    if not candidates:
        candidates = [t for t in transcripts if most_severe in t.get("consequence_terms", [])]
    if not candidates:
        candidates = transcripts
    return candidates[0] if candidates else {}
