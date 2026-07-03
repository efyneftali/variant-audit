"""MCP server exposing our tools (ClinVar, etc.) over the Model Context Protocol.

Run with the .venv-mcp interpreter (the `mcp` SDK needs Python >=3.10, while
the rest of this project's .venv is pinned to 3.9):

    PYTHONPATH=src .venv-mcp/bin/python -m variant_audit.mcp_server

Inspect/test it with the MCP Inspector (needs Node.js):

    npx @modelcontextprotocol/inspector \\
        --  env PYTHONPATH=src .venv-mcp/bin/python -m variant_audit.mcp_server
"""

from mcp.server.fastmcp import FastMCP

from .mcp_tools.clinvar import get_clinvar_record
from .mcp_tools.ensembl import get_gene_consequence
from .mcp_tools.gnomad import get_allele_frequency
from .mcp_tools.ucsc import get_genomic_context

mcp = FastMCP("variant-audit")


@mcp.tool()
def clinvar_lookup(variant: str) -> dict:
    """Look up a variant's known ClinVar clinical significance.

    Args:
        variant: an rsID, e.g. "rs28897696".
    """
    return get_clinvar_record(variant)


@mcp.tool()
def gnomad_lookup(variant: str) -> dict:
    """Look up a variant's population allele frequency in gnomAD.

    Args:
        variant: an rsID, e.g. "rs28897696".
    """
    return get_allele_frequency(variant)


@mcp.tool()
def ensembl_lookup(variant: str) -> dict:
    """Look up a variant's molecular consequence and gene context via Ensembl VEP.

    Args:
        variant: an rsID, e.g. "rs28897696".
    """
    return get_gene_consequence(variant)


@mcp.tool()
def ucsc_lookup(variant: str) -> dict:
    """Look up a variant's conservation scores (phyloP/phastCons) via UCSC.

    Args:
        variant: an rsID, e.g. "rs28897696".
    """
    return get_genomic_context(variant)


if __name__ == "__main__":
    mcp.run()
