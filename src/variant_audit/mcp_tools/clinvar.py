"""ClinVar tool — known clinical significance for a variant.

Source: NCBI ClinVar. API docs: https://www.ncbi.nlm.nih.gov/clinvar/docs/
NOTE: ClinVar labels are also your eval *ground truth* — in the agent, treat a
ClinVar assertion as one piece of evidence, not the final answer (otherwise the
eval is circular).

TODO(day-4): implement this one first — it's the "one MCP tool end to end" gate.
"""


def get_clinvar_record(variant: str) -> dict:
    """Look up a variant's ClinVar record.

    Args:
        variant: HGVS or rsID (decide on a canonical input form and document it).
    Returns:
        structured dict, e.g. {clinical_significance, review_status, ...}, or a
        clear "not found" result. Handle unknown variants gracefully.
    """
    raise NotImplementedError("TODO(day-4): query ClinVar and return a structured record")
