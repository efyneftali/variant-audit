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

mcp = FastMCP("variant-audit")


@mcp.tool()
def clinvar_lookup(variant: str) -> dict:
    """Look up a variant's known ClinVar clinical significance.

    Args:
        variant: an rsID, e.g. "rs28897696".
    """
    return get_clinvar_record(variant)


if __name__ == "__main__":
    mcp.run()
