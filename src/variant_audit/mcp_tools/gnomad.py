"""gnomAD tool — population allele frequency.

Source: gnomAD (https://gnomad.broadinstitute.org), GraphQL API.
Frequency is core ACMG evidence: very common -> benign (BA1/BS1); absent/rare ->
supports pathogenic (PM2).
"""
import logging

import requests

logger = logging.getLogger(__name__)

GNOMAD_API_URL = "https://gnomad.broadinstitute.org/api"
GNOMAD_DATASET = "gnomad_r4"
REQUEST_TIMEOUT = 40

VARIANT_QUERY = """
query VariantRsid($rsid: String!, $dataset: DatasetId!) {
  variant(rsid: $rsid, dataset: $dataset) {
    variantId
    chrom
    pos
    ref
    alt
    genome { af ac an }
    exome { af ac an }
  }
}
"""


def get_allele_frequency(variant: str) -> dict:
    """Look up a variant's population allele frequency in gnomAD.

    Args:
        variant: an rsID, e.g. "rs28897696".
    Returns:
        {variant, found, allele_freq, genome, exome} — allele_freq prefers the
        genome frequency, falling back to exome; found=False (with no other
        keys) when gnomAD has never observed this variant, or the rsID doesn't
        resolve to a variant at all.

    Raises:
        RuntimeError: if the gnomAD API call fails (network error, bad status,
        a real GraphQL error, or an unexpected response shape) — logged with
        enough detail to diagnose which stage failed and why.
    """
    logger.info("gnomad query variant=%s dataset=%s", variant, GNOMAD_DATASET)
    try:
        resp = requests.post(
            GNOMAD_API_URL,
            json={"query": VARIANT_QUERY, "variables": {"rsid": variant, "dataset": GNOMAD_DATASET}},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
    except requests.exceptions.RequestException as e:
        logger.exception("gnomad query failed variant=%s url=%s", variant, GNOMAD_API_URL)
        raise RuntimeError(f"gnomAD query failed for variant '{variant}'") from e
    except ValueError as e:
        logger.exception("gnomad returned non-JSON response variant=%s body=%s", variant, resp.text[:500])
        raise RuntimeError(f"gnomAD returned an unexpected response for variant '{variant}'") from e

    # gnomAD reports real errors as HTTP 200 + a GraphQL "errors" array. A
    # genuinely unknown variant looks like {"data": {"variant": null}}; a
    # malformed query/schema error looks like {"errors": [...]} with no "data"
    # key at all — that's the case we still need to raise on.
    if not isinstance(body.get("data"), dict):
        errors = body.get("errors", [])
        logger.error("gnomad graphql error variant=%s errors=%s", variant, errors)
        raise RuntimeError(f"gnomAD query returned an error for variant '{variant}': {errors}")

    record = body["data"].get("variant")
    if record is None:
        return {"variant": variant, "found": False}

    try:
        genome = record.get("genome")
        exome = record.get("exome")
        allele_freq = genome["af"] if genome else (exome["af"] if exome else None)
    except (KeyError, TypeError) as e:
        logger.exception("gnomad returned unexpected shape variant=%s body=%s", variant, body)
        raise RuntimeError(f"gnomAD returned an unexpected response shape for variant '{variant}'") from e

    return {
        "variant": variant,
        "found": True,
        "allele_freq": allele_freq,
        "genome": genome,
        "exome": exome,
    }
