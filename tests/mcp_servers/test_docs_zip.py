import asyncio
import stat
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

from voice_agent.mcp_servers import docs as docs_mod
from voice_agent.rag import zipingest


def _make_zip(path: Path) -> Path:
    link = zipfile.ZipInfo("link.md")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("pack/readme.md", "# Pack\n\nplatypus venom spur\n")
        zf.writestr(link, "/etc/passwd")
    return path


async def _ready_status(client: Client) -> dict[str, Any]:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        st = (await client.call_tool("docs_status", {})).structured_content
        if st["ready"] and not st["indexing"]:
            return dict(st)
        await asyncio.sleep(0.05)
    raise AssertionError("index never became ready")


async def _search(client: Client, query: str) -> list[dict[str, Any]]:
    res = (await client.call_tool("search_docs", {"query": query})).structured_content
    return list(res["result"])


def test_a_dropped_zip_is_indexed_and_refusals_are_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOICE_AGENT_DOCS_REFRESH_WAIT_S", "0")
    _make_zip(tmp_path / "drop.zip")
    server = docs_mod.build_server(tmp_path, retriever_mode="bm25")

    async def scenario() -> None:
        async with Client(server) as client:
            st = await _ready_status(client)
            assert [f["name"] for f in st["files"]] == ["drop/pack/readme.md"]
            reasons = [f["reason"] for f in st["failures"]]
            assert len(reasons) == 1 and reasons[0].startswith("zip: ") and "symlink" in reasons[0]
            assert (await _search(client, "platypus venom"))[0]["source"] == "drop/pack/readme.md"

    asyncio.run(scenario())


def test_an_external_zip_is_ingested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICE_AGENT_DOCS_REFRESH_WAIT_S", "0")
    outside = _make_zip(tmp_path / "away.zip")
    corpus = tmp_path / "corpus"
    server = docs_mod.build_server(corpus, retriever_mode="bm25", zips=[outside])

    async def scenario() -> None:
        async with Client(server) as client:
            await _ready_status(client)
            assert (await _search(client, "platypus venom"))[0]["source"] == "away/pack/readme.md"

    asyncio.run(scenario())


def test_a_crashing_ingest_is_reported_and_indexing_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOICE_AGENT_DOCS_REFRESH_WAIT_S", "0")

    def boom(corpus: Path, extra: list[Path] | None = None) -> list[tuple[str, str]]:
        raise RuntimeError("secret /path/detail")

    monkeypatch.setattr(zipingest, "ingest_zips", boom)
    (tmp_path / "a.md").write_text("# A\n\nkiwi fruit\n")
    server = docs_mod.build_server(tmp_path, retriever_mode="bm25")

    async def scenario() -> None:
        async with Client(server) as client:
            st = await _ready_status(client)
            assert [f["name"] for f in st["files"]] == ["a.md"]
            assert st["failures"] == [
                {"name": "zip", "status": "unreadable", "reason": "zip: ingest failed"}
            ]

    asyncio.run(scenario())


def test_a_zip_corpus_argument_serves_a_folder_next_to_it(tmp_path: Path) -> None:
    corpus, zips = docs_mod._resolve_corpus(tmp_path / "day.ZIP")
    assert corpus == tmp_path / "day_docs" and zips == [tmp_path / "day.ZIP"]
    assert docs_mod._resolve_corpus(tmp_path / "folder") == (tmp_path / "folder", [])
