"""gnomAD tool — population allele frequency.

Source: gnomAD (https://gnomad.broadinstitute.org), GraphQL API.
Frequency is core ACMG evidence: very common -> benign (BA1/BS1); absent/rare ->
supports pathogenic (PM2).

TODO(day-7): implement.
"""


def get_allele_frequency(variant: str) -> dict:
    """Return population allele frequency info for a variant (or 'not observed')."""
    raise NotImplementedError("TODO(day-7): query gnomAD and return frequency data")
