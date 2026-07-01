"""Tests for mcp_server.py."""

import asyncio

import pytest

from src.variant_audit import mcp_server


@pytest.fixture
def fake_clinvar_record(monkeypatch):
    fake_result = {"variant": "rs28897696", "found": True, "matches": [{"hgvs": "x", "clinical_significance": "Pathogenic", "review_status": "y"}]}
    calls = {}

    def fake_get_clinvar_record(variant):
        calls["variant"] = variant
        return fake_result

    monkeypatch.setattr(mcp_server, "get_clinvar_record", fake_get_clinvar_record)
    return calls, fake_result


class TestClinvarLookupTool:
    def test_delegates_to_get_clinvar_record_with_same_argument(self, fake_clinvar_record):
        calls, _ = fake_clinvar_record
        mcp_server.clinvar_lookup("rs28897696")
        assert calls["variant"] == "rs28897696"

    def test_returns_get_clinvar_record_result_unchanged(self, fake_clinvar_record):
        _, fake_result = fake_clinvar_record
        result = mcp_server.clinvar_lookup("rs28897696")
        assert result == fake_result

    def test_tool_is_registered_with_correct_name_and_schema(self):
        tools = asyncio.run(mcp_server.mcp.list_tools())
        names = [t.name for t in tools]
        assert "clinvar_lookup" in names

        tool = next(t for t in tools if t.name == "clinvar_lookup")
        assert tool.inputSchema["required"] == ["variant"]
        assert tool.inputSchema["properties"]["variant"]["type"] == "string"


@pytest.mark.integration
class TestMcpServerIntegration:
    def test_real_stdio_round_trip_returns_real_clinvar_record(self):
        import os
        import sys

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def run():
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "variant_audit.mcp_server"],
                env={**os.environ, "PYTHONPATH": "src"},
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tools = await session.list_tools()
                    assert "clinvar_lookup" in [t.name for t in tools.tools]

                    result = await session.call_tool("clinvar_lookup", {"variant": "rs28897696"})
                    return result.content[0].text

        text = asyncio.run(run())
        assert "rs28897696" in text
        assert "clinical_significance" in text
