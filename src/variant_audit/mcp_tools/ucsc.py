"""UCSC Genome Browser tool — genomic context (conservation, genes, regulation).

Source: UCSC REST API, https://api.genome.ucsc.edu (no auth, rate-limited ~1 req/sec).
Endpoints you'll likely use: /getData/track (e.g. phyloP/phastCons conservation),
/getData/sequence. Conservation supports ACMG computational evidence (PP3/BP4).

TODO(day-7): implement.
"""


def get_genomic_context(variant: str) -> dict:
    """Return conservation scores / nearby genes / regulatory context for a variant's locus."""
    raise NotImplementedError("TODO(day-7): call the UCSC REST API and return context")
