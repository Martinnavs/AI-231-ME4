"""Test docs.py file_context tool registration (ticket 209)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client

from voice_agent.mcp_servers.docs import build_server


async def test_file_context_with_db(_ingested: Path, tmp_path: Path) -> None:
    async with Client(build_server(tmp_path, db=_ingested)) as client:
        result = await client.call_tool("file_context", {"query": "install"})
        assert result is not None
        assert result.structured_content is not None


async def test_file_context_no_db(tmp_path: Path) -> None:
    async with Client(build_server(tmp_path)) as client:
        result = await client.call_tool("file_context", {"query": "install"})
        assert result.structured_content is not None
        assert result.structured_content["result"] == []


@pytest.fixture(scope="module")
def _ingested(tmp_path_factory: pytest.TempPathFactory) -> Path:
    src = tmp_path_factory.mktemp("src")
    db = tmp_path_factory.mktemp("db") / "test.db"
    (src / "guide.md").write_text("# Setup\n\nInstall the package.\n", encoding="utf-8")
    from voice_agent.rag.ingest import ingest

    ingest(src, db)
    return db
