"""UCSC Genome Browser tool — genomic context (conservation).

Source: UCSC REST API, https://api.genome.ucsc.edu (no auth). Conservation
supports ACMG computational evidence (PP3/BP4): highly conserved positions
support a deleterious effect, poorly conserved ones support benign.

Unlike ClinVar/gnomAD/Ensembl, UCSC has no rsID lookup — it only answers
"what's at this genomic position?". So this tool first calls Ensembl VEP
(mcp_tools.ensembl.get_gene_consequence) to resolve the rsID to coordinates,
then queries UCSC at that locus.
"""
import logging

import requests

from .ensembl import get_gene_consequence

logger = logging.getLogger(__name__)

UCSC_API_URL = "https://api.genome.ucsc.edu/getData/track"
GENOME = "hg38"
PHYLOP_TRACK = "phyloP100way"       # per-base conservation score (higher = more conserved)
PHASTCONS_TRACK = "phastCons100way"  # 0-1 probability the base is in a conserved element
REQUEST_TIMEOUT = 20


def get_genomic_context(variant: str, consequence: dict | None = None) -> dict:
    """Return conservation scores at a variant's locus (resolved via Ensembl).

    Args:
        variant: an rsID, e.g. "rs28897696".
        consequence: an already-fetched get_gene_consequence(variant) result.
            Pass this when the caller already looked the variant up in Ensembl
            (e.g. graph.gather_evidence, which needs it for its own "ensembl"
            evidence key anyway) to avoid a second, slow VEP round-trip.
            Fetched internally if omitted.
    Returns:
        {variant, found, phylop, phastcons, chrom, start, end}; found=False
        (with no other keys) when Ensembl can't resolve the variant to
        coordinates in the first place — UCSC is never even queried.

    Raises:
        RuntimeError: if Ensembl's own lookup fails (see get_gene_consequence),
        or if the UCSC call fails (network error, bad status, a real API
        error, or an unexpected response shape).
    """
    if consequence is None:
        consequence = get_gene_consequence(variant)

    if not consequence.get("found"):
        return {"variant": variant, "found": False}

    chrom, start, end = consequence.get("chrom"), consequence.get("start"), consequence.get("end")
    if chrom is None or start is None or end is None:
        return {"variant": variant, "found": False}

    ucsc_chrom = chrom if str(chrom).startswith("chr") else f"chr{chrom}"
    if end < start:
        # Ensembl represents a pure insertion this way (zero-length reference
        # span) — treat it as a single point at the insertion site.
        ucsc_start, ucsc_end = start - 1, start
    else:
        ucsc_start, ucsc_end = start - 1, end

    phylop = _query_track(PHYLOP_TRACK, ucsc_chrom, ucsc_start, ucsc_end, variant)
    phastcons = _query_track(PHASTCONS_TRACK, ucsc_chrom, ucsc_start, ucsc_end, variant)

    return {
        "variant": variant,
        "found": True,
        "phylop": phylop,
        "phastcons": phastcons,
        "chrom": chrom,
        "start": start,
        "end": end,
    }


def _query_track(track: str, chrom: str, start: int, end: int, variant: str) -> float | None:
    """Query one UCSC wig track over [start, end) (0-based, half-open) and return its score.

    Returns None when the track has no data at this locus (a legitimate,
    non-error result — e.g. an unplaced contig or gap in coverage).
    """
    logger.info("ucsc query variant=%s track=%s chrom=%s start=%d end=%d", variant, track, chrom, start, end)
    try:
        resp = requests.get(
            UCSC_API_URL,
            params={"genome": GENOME, "track": track, "chrom": chrom, "start": start, "end": end},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
    except requests.exceptions.RequestException as e:
        logger.exception("ucsc query failed variant=%s track=%s", variant, track)
        raise RuntimeError(f"UCSC query failed for variant '{variant}' (track={track})") from e
    except ValueError as e:
        logger.exception("ucsc returned non-JSON response variant=%s track=%s body=%s", variant, track, resp.text[:500])
        raise RuntimeError(f"UCSC returned an unexpected response for variant '{variant}'") from e

    if "error" in body:
        logger.error("ucsc api error variant=%s track=%s error=%s", variant, track, body["error"])
        raise RuntimeError(f"UCSC query returned an error for variant '{variant}' (track={track}): {body['error']}")

    items = body.get(track, [])
    if not items:
        return None

    try:
        return items[0]["value"]
    except (KeyError, IndexError, TypeError) as e:
        logger.exception("ucsc returned unexpected shape variant=%s track=%s body=%s", variant, track, body)
        raise RuntimeError(f"UCSC returned an unexpected response shape for variant '{variant}' (track={track})") from e
