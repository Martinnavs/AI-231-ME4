import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from mcp.client.stdio import get_default_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "rag_corpus"


def _transport(corpus: Path, *flags: str) -> StdioTransport:
    return StdioTransport(
        command=sys.executable,
        args=["-m", "voice_agent.mcp_servers.docs", "--corpus", str(corpus), *flags],
        cwd=str(REPO_ROOT),
        env={
            **get_default_environment(),
            "VOICE_AGENT_DOCS_REFRESH_WAIT_S": "10",
            "VOICE_AGENT_DOCS_POLL_S": "0.2",
        },
    )


def _zip_bomb_docx(path: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/bomb.txt", "A" * 50_000)


async def test_pdf_pages_docx_sections_and_a_zip_bomb_over_stdio(tmp_path: Path) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    for name in ("handbook.pdf", "handbook.docx"):
        shutil.copy(FIXTURES / name, corpus / name)
    _zip_bomb_docx(corpus / "bomb.docx")

    async with Client(_transport(corpus, "--retriever", "bm25")) as client:
        meal = (
            await client.call_tool("search_docs", {"query": "daily meal allowance expense claims"})
        ).structured_content["result"]
        assert meal[0]["source"] == "handbook.pdf"
        assert meal[0]["page"] == 2
        assert "350 pesos" in meal[0]["text"]

        first_day = (
            await client.call_tool("search_docs", {"query": "new hires first day laptop badge"})
        ).structured_content["result"]
        assert first_day[0]["source"] == "handbook.docx"
        assert first_day[0]["section"] == "Onboarding checklist"

        status = (await client.call_tool("docs_status", {})).structured_content
        assert {f["name"] for f in status["files"]} == {
            "handbook.pdf",
            "handbook.docx",
            "bomb.docx",
        }
        failures = {f["name"]: f for f in status["failures"]}
        assert set(failures) == {"bomb.docx"}
        assert failures["bomb.docx"]["status"] == "unreadable"
        assert "ratio" in failures["bomb.docx"]["reason"].lower()


async def test_missing_sidecar_serves_fts5_over_stdio_and_says_why_when_forced(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    shutil.copy(FIXTURES / "router_reset.md", corpus / "router_reset.md")
    missing = tmp_path / "no-such-python"

    async with Client(
        _transport(corpus, "--retriever", "auto", "--embed-python", str(missing))
    ) as client:
        hits = (
            await client.call_tool("search_docs", {"query": "factory reset pinhole twelve seconds"})
        ).structured_content["result"]
        assert hits and hits[0]["source"] == "router_reset.md"
        status = (await client.call_tool("docs_status", {})).structured_content
        assert status["retriever"] == "fts5:file"

    async with Client(
        _transport(corpus, "--retriever", "hybrid", "--embed-python", str(missing))
    ) as client:
        hits = (
            await client.call_tool("search_docs", {"query": "factory reset pinhole twelve seconds"})
        ).structured_content["result"]
        assert hits and hits[0]["source"] == "router_reset.md"
        status = (await client.call_tool("docs_status", {})).structured_content
        assert status["retriever"].startswith("bm25 (hybrid unavailable:")


async def test_a_short_file_is_searchable_or_named_never_silently_dropped(tmp_path: Path) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "wifi_note.txt").write_text("Office Wi-Fi password: hunter2.\n", encoding="utf-8")

    async with Client(_transport(corpus, "--retriever", "bm25")) as client:
        hits: list[dict[str, Any]] = (
            await client.call_tool("search_docs", {"query": "office wifi password"})
        ).structured_content["result"]
        status = (await client.call_tool("docs_status", {})).structured_content

    named = {f["name"] for f in status["failures"]}
    assert any(h["source"] == "wifi_note.txt" for h in hits)
    assert "wifi_note.txt" not in named
