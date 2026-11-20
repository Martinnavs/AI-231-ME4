import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

from voice_agent.mcp_servers import docs as docs_mod
from voice_agent.rag import docindex


async def _call(server: Any, tool: str, **args: Any) -> Any:
    async with Client(server) as client:
        res = await client.call_tool(tool, args)
    return res.structured_content


def _slow_big(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    real = docindex._load_chunks_for_mode

    def slow(raw: bytes, **kwargs: Any) -> Any:
        if kwargs.get("source") == "big.pdf":
            time.sleep(seconds)
            kwargs = {**kwargs, "kind": "md", "mode": "identity"}
        return real(raw, **kwargs)

    monkeypatch.setattr(docindex, "_load_chunks_for_mode", slow)


def test_tools_serve_snapshot_while_a_big_file_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOICE_AGENT_DOCS_REFRESH_WAIT_S", "0.3")
    (tmp_path / "small.md").write_text("# Small\n\nzebra stripes are black and white\n")
    _slow_big(monkeypatch, 2.0)
    server = docs_mod.build_server(tmp_path, retriever_mode="bm25")

    async def scenario() -> None:
        async with Client(server) as client:
            for _ in range(100):
                st = (await client.call_tool("docs_status", {})).structured_content
                if st["ready"]:
                    break
                await asyncio.sleep(0.05)
            assert st["ready"] and not st["indexing"]
            (tmp_path / "big.pdf").write_bytes(b"# Big\n\nplatypus venom spur\n")
            for _ in range(100):
                st = (await client.call_tool("docs_status", {})).structured_content
                if st["parsing"] == "big.pdf":
                    break
                await asyncio.sleep(0.05)
            assert st["parsing"] == "big.pdf"
            assert st["indexing"] is True
            t0 = time.monotonic()
            res = (
                await client.call_tool("search_docs", {"query": "zebra stripes"})
            ).structured_content["result"]
            assert time.monotonic() - t0 < 0.5
            assert res and res[0]["source"] == "small.md"
            deadline = time.monotonic() + 10
            hits: list[Any] = []
            while time.monotonic() < deadline:
                hits = (
                    await client.call_tool("search_docs", {"query": "platypus venom"})
                ).structured_content["result"]
                if hits:
                    break
                await asyncio.sleep(0.1)
            assert hits and hits[0]["source"] == "big.pdf"
            st = (await client.call_tool("docs_status", {})).structured_content
            assert st["indexing"] is False

    asyncio.run(scenario())


def test_status_answers_immediately_before_the_first_refresh_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "big.pdf").write_bytes(b"# Big\n\nplatypus venom spur\n")
    _slow_big(monkeypatch, 1.5)
    monkeypatch.setenv("VOICE_AGENT_DOCS_REFRESH_WAIT_S", "0")
    server = docs_mod.build_server(tmp_path, retriever_mode="bm25")

    async def scenario() -> None:
        t0 = time.monotonic()
        st = await _call(server, "docs_status")
        assert time.monotonic() - t0 < 0.5
        assert st["ready"] is False
        assert st["indexing"] is True
        assert st["files"] == []

    asyncio.run(scenario())


def test_a_failing_refresh_sets_index_error_and_the_worker_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOICE_AGENT_DOCS_REFRESH_WAIT_S", "0")
    calls = {"n": 0}
    real = docindex.DocIndex.refresh

    def flaky(self: docindex.DocIndex) -> docindex.RefreshReport:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("secret /path/detail")
        return real(self)

    monkeypatch.setattr(docindex.DocIndex, "refresh", flaky)
    (tmp_path / "a.md").write_text("# A\n\nkiwi fruit\n")
    server = docs_mod.build_server(tmp_path, retriever_mode="bm25")

    async def scenario() -> None:
        async with Client(server) as client:
            seen_error = False
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                st = (await client.call_tool("docs_status", {})).structured_content
                if st.get("index_error"):
                    assert "secret" not in st["index_error"]
                    seen_error = True
                if seen_error and not st.get("index_error") and st["files"]:
                    return
                await asyncio.sleep(0.1)
            raise AssertionError("worker did not recover")

    asyncio.run(scenario())
    assert calls["n"] >= 3
