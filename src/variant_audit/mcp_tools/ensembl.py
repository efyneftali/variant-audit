"""Ensembl / VEP tool — variant consequence and gene annotation.

Source: Ensembl REST API (https://rest.ensembl.org), VEP endpoint.
Tells you what the variant *does* (missense, frameshift, splice, ...) and which
gene/transcript — needed to know which ACMG criteria even apply (e.g. PVS1 is
loss-of-function specific).

TODO(day-7): implement. (This is the first thing to drop if you're behind — see cut-list.)
"""


def get_gene_consequence(variant: str) -> dict:
    """Return the variant's molecular consequence + gene/transcript context."""
    raise NotImplementedError("TODO(day-7): query Ensembl VEP and return consequence")
