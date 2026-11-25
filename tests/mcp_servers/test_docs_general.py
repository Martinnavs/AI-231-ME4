"""Test docs.py docs_catalog / docs_enumerate tool registration (ticket 221a)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client

from voice_agent.mcp_servers.docs import build_server


async def test_general_tools_with_db(_ingested: Path, tmp_path: Path) -> None:
    async with Client(build_server(tmp_path, db=_ingested)) as client:
        result = await client.call_tool("docs_catalog", {})
        assert result is not None
        entries = result.structured_content["result"]
        assert entries
        assert all("rel_key" in e and "line" in e for e in entries)
        result = await client.call_tool("docs_enumerate", {"query": "install"})
        rows = result.structured_content["result"]
        assert any(r["rel_key"] == "guide.md" for r in rows)


async def test_general_tools_no_db(tmp_path: Path) -> None:
    async with Client(build_server(tmp_path)) as client:
        result = await client.call_tool("docs_catalog", {})
        assert result.structured_content["result"] == []
        result = await client.call_tool("docs_enumerate", {"query": "install"})
        assert result.structured_content["result"] == []


async def test_enumerate_limit_clamped(_ingested: Path, tmp_path: Path) -> None:
    async with Client(build_server(tmp_path, db=_ingested)) as client:
        result = await client.call_tool("docs_enumerate", {"query": "install", "limit": 1})
        rows = result.structured_content["result"]
        assert len(rows) <= 1


@pytest.fixture(scope="module")
def _ingested(tmp_path_factory: pytest.TempPathFactory) -> Path:
    src = tmp_path_factory.mktemp("src")
    db = tmp_path_factory.mktemp("db") / "test.db"
    (src / "guide.md").write_text("# Setup\n\nInstall the package.\n", encoding="utf-8")
    from voice_agent.rag.ingest import ingest

    ingest(src, db)
    return db
