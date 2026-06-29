"""ClinVar tool — known clinical significance for a variant.

Source: NCBI ClinVar. API docs: https://www.ncbi.nlm.nih.gov/clinvar/docs/
NOTE: ClinVar labels are also your eval *ground truth* — in the agent, treat a
ClinVar assertion as one piece of evidence, not the final answer (otherwise the
eval is circular).

TODO(day-4): implement this one first — it's the "one MCP tool end to end" gate.
"""
import logging

import requests

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUTILS_SEARCH = "/esearch.fcgi"
EUTILS_ESUMMARY = "/esummary.fcgi"
REQUEST_TIMEOUT = 10


def get_clinvar_record(variant: str) -> dict:
    """Look up a variant's ClinVar record.

    Args:
        variant: HGVS or rsID (decide on a canonical input form and document it).
    Returns:
        structured dict, e.g. {clinical_significance, review_status, ...}, or a
        clear "not found" result. Handle unknown variants gracefully.

    Raises:
        RuntimeError: if the esearch/esummary call fails (network error, bad
        status, or an unexpected response shape) — logged with enough detail
        to diagnose which call failed and why.
    """
    search_url = EUTILS_BASE + EUTILS_SEARCH
    logger.info("clinvar esearch variant=%s", variant)
    try:
        search_resp = requests.get(
            search_url,
            params={"db": "clinvar", "term": variant, "retmode": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        search_resp.raise_for_status()
        ids_list = search_resp.json()["esearchresult"]["idlist"]
    except requests.exceptions.RequestException as e:
        logger.exception("clinvar esearch failed variant=%s url=%s", variant, search_url)
        raise RuntimeError(f"ClinVar esearch failed for variant '{variant}'") from e
    except (KeyError, ValueError) as e:
        logger.exception(
            "clinvar esearch returned unexpected shape variant=%s body=%s",
            variant,
            search_resp.text[:500],
        )
        raise RuntimeError(f"ClinVar esearch returned an unexpected response for variant '{variant}'") from e

    if not ids_list:
        return {"variant": variant, "found": False}

    ids_str = ",".join(ids_list)
    esummary_url = EUTILS_BASE + EUTILS_ESUMMARY
    logger.info("clinvar esummary variant=%s ids=%s", variant, ids_str)
    try:
        esummary_res = requests.get(
            esummary_url,
            params={"db": "clinvar", "id": ids_str, "retmode": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        esummary_res.raise_for_status()
        summary = esummary_res.json()["result"]

        matches = []
        for uid in summary["uids"]:
            record = summary[uid]
            classification = record["germline_classification"]
            matches.append({
                "hgvs": record["title"],
                "clinical_significance": classification["description"],
                "review_status": classification["review_status"],
            })
    except requests.exceptions.RequestException as e:
        logger.exception("clinvar esummary failed variant=%s ids=%s url=%s", variant, ids_str, esummary_url)
        raise RuntimeError(f"ClinVar esummary failed for variant '{variant}'") from e
    except (KeyError, ValueError) as e:
        logger.exception(
            "clinvar esummary returned unexpected shape variant=%s ids=%s body=%s",
            variant,
            ids_str,
            esummary_res.text[:500],
        )
        raise RuntimeError(f"ClinVar esummary returned an unexpected response for variant '{variant}'") from e

    return {"variant": variant, "found": True, "matches": matches}
