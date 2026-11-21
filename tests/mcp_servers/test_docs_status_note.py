"""docs_status publishes a note for a truncated table (WP5)."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

from voice_agent.mcp_servers import docs as docs_mod
from voice_agent.rag import canonical


def test_docs_status_notes_a_truncated_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICE_AGENT_DOCS_REFRESH_WAIT_S", "10")
    rows = "".join(f"item{i},{i}\n" for i in range(canonical.MAX_CSV_ROWS + 1000))
    (tmp_path / "big.csv").write_text("name,qty\n" + rows, encoding="utf-8")
    (tmp_path / "small.csv").write_text("name,qty\nbolt,4\n", encoding="utf-8")
    server = docs_mod.build_server(tmp_path, retriever_mode="bm25")

    async def scenario() -> dict[str, Any]:
        async with Client(server) as client:
            for _ in range(100):
                status = (await client.call_tool("docs_status", {})).structured_content
                if status["ready"]:
                    break
                await asyncio.sleep(0.1)
            return status

    files = {entry["name"]: entry for entry in asyncio.run(scenario())["files"]}
    assert files["big.csv"]["note"] == canonical.TABLE_TRUNCATED_NOTE
    assert "note" not in files["small.csv"]
