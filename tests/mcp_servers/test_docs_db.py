"""Tests for docs server with --db (ticket 107)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client

from voice_agent.mcp_servers.docs import build_server
from voice_agent.rag.ingest import ingest


@pytest.fixture(scope="module")
def _ingested(tmp_path_factory: pytest.TempPathFactory) -> Path:
    src = tmp_path_factory.mktemp("src")
    db = tmp_path_factory.mktemp("db") / "test.db"

    (src / "guide.md").write_text(
        "# Setup\n\nInstall the package on 2026-03-15 following standard steps.\n\n"
        "## Usage\n\nRun the agent with the --verbose flag for detailed output.\n",
        encoding="utf-8",
    )
    (src / "notes.txt").write_text(
        "Meeting notes from November 18, 2026.\n",
        encoding="utf-8",
    )
    ingest(src, db)
    return db


async def test_search_docs_via_db(_ingested: Path, tmp_path: Path) -> None:
    async with Client(build_server(tmp_path, db=_ingested)) as client:
        result = await client.call_tool("search_docs", {"query": "install", "top_k": 2})
        hits = result.structured_content["result"] if result.structured_content else []
        assert len(hits) >= 1
        assert any("install" in h["text"].lower() or "package" in h["text"].lower() for h in hits)


async def test_docs_status_via_db(_ingested: Path, tmp_path: Path) -> None:
    async with Client(build_server(tmp_path, db=_ingested)) as client:
        status = (await client.call_tool("docs_status", {})).structured_content
        assert status is not None
        assert status["ready"] is True
        assert status["indexing"] is False
        assert len(status["files"]) >= 2
        assert status["retriever"] == "fts5-db"


async def test_no_failures_in_db_mode(_ingested: Path, tmp_path: Path) -> None:
    async with Client(build_server(tmp_path, db=_ingested)) as client:
        status = (await client.call_tool("docs_status", {})).structured_content
        # No failures since both files ingested ok
        assert status is not None
        assert status["failures"] == []
