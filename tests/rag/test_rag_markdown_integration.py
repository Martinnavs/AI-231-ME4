"""Cross-cutting rag-markdown checks through the real docs server builder and DocIndex API."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

from voice_agent.mcp_servers.docs import build_server
from voice_agent.rag import docindex
from voice_agent.rag.backends import make_retriever_factory
from voice_agent.rag.docindex import DocIndex

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"

# (query, expected source, expected page, expected section prefix, rank limit).
# The last row ranks the slide first under BM25 but the sheet first under FTS5 (both place
# the sheet in the top 2), so it asserts the sheet within the top 2 rather than at rank 1.
MIXED_QUERIES: list[tuple[str, str, int | None, str, int]] = [
    ("meal allowance expense claims", "handbook.pdf", 2, "", 1),
    ("new hires first day laptop badge", "handbook.docx", None, "Onboarding checklist", 1),
    ("Zephyr gasket torque schedule", "notes.md", None, "Zephyr", 1),
    ("Marigold pump warranty memo", "memo.txt", None, "", 1),
    ("Lantern ferrule rate", "rates.csv", None, "", 1),
    ("Obsidian kiln thermostat setpoint", "kiln.html", None, "Kiln", 1),
    ("North Ridge elevation battery signal", "site_review.pptx", None, "Slide 1", 1),
    ("corroded conduit gaskets junction box", "site_review.pptx", None, "Slide 2", 1),
    ("SENS-LORA-09 unit cost", "team_budget.xlsx", None, "Sheet Hardware", 1),
    ("Solar Maintenance headcount travel", "team_budget.xlsx", None, "Sheet Operations", 2),
]


def _mixed_folder(root: Path) -> Path:
    folder = root / "docs"
    folder.mkdir()
    shutil.copy(FIXTURES / "rag_corpus" / "handbook.pdf", folder / "handbook.pdf")
    shutil.copy(FIXTURES / "rag_corpus" / "handbook.docx", folder / "handbook.docx")
    shutil.copy(FIXTURES / "rag_office" / "site_review.pptx", folder / "site_review.pptx")
    shutil.copy(FIXTURES / "rag_office" / "team_budget.xlsx", folder / "team_budget.xlsx")
    (folder / "notes.md").write_text(
        "# Zephyr\n\nZephyr gasket torque schedule is nine newton metres.\n", encoding="utf-8"
    )
    (folder / "memo.txt").write_text(
        "Marigold pump warranty memo says twenty four months.\n", encoding="utf-8"
    )
    (folder / "rates.csv").write_text("item,rate\nLantern ferrule,77\n", encoding="utf-8")
    (folder / "kiln.html").write_text(
        "<html><body><h1>Kiln</h1><p>Obsidian kiln thermostat setpoint is 640 degrees.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    return folder


def _pin_no_sidecar(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("VOICE_AGENT_EMBED_PYTHON", str(root / "absent-embed-python"))


@pytest.mark.parametrize(
    "retriever_env",
    [None, "bm25", "fts5"],
    ids=["default-unset", "bm25", "fts5"],
)
async def test_mixed_folder_search_docs_cites_right_file_and_locator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retriever_env: str | None,
) -> None:
    folder = _mixed_folder(tmp_path)
    _pin_no_sidecar(monkeypatch, tmp_path)
    if retriever_env is None:
        monkeypatch.delenv("VOICE_AGENT_RAG_RETRIEVAL", raising=False)
    else:
        monkeypatch.setenv("VOICE_AGENT_RAG_RETRIEVAL", retriever_env)

    async with Client(build_server(folder)) as client:
        status = (await client.call_tool("docs_status", {})).structured_content
        assert status["failures"] == []
        assert {f["name"] for f in status["files"]} == {
            "handbook.pdf",
            "handbook.docx",
            "site_review.pptx",
            "team_budget.xlsx",
            "notes.md",
            "memo.txt",
            "rates.csv",
            "kiln.html",
        }

        for query, source, page, section_prefix, rank_limit in MIXED_QUERIES:
            hits: list[dict[str, Any]] = (
                await client.call_tool("search_docs", {"query": query, "top_k": 3})
            ).structured_content["result"]
            matches = [h for h in hits[:rank_limit] if h["source"] == source]
            assert matches, (query, [(h["source"], h["section"]) for h in hits])
            cited = matches[0]
            assert cited["page"] == page, (query, cited["page"])
            assert cited["section"].startswith(section_prefix), (query, cited["section"])
            assert "<!-- rag:" not in cited["text"]

        pdf_hit = (
            await client.call_tool("search_docs", {"query": "meal allowance expense claims"})
        ).structured_content["result"][0]
        assert "350 pesos" in pdf_hit["text"]

        slide_hit = (
            await client.call_tool(
                "search_docs", {"query": "North Ridge elevation battery signal", "top_k": 1}
            )
        ).structured_content["result"][0]
        assert "412" in slide_hit["text"]

        sheet_hit = (
            await client.call_tool("search_docs", {"query": "SENS-LORA-09 unit cost", "top_k": 1})
        ).structured_content["result"][0]
        assert "4440" in sheet_hit["text"]


def test_restart_over_same_folder_reconverts_nothing_and_retokenizes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voice_agent.rag import fts5 as fts5_mod

    folder = _mixed_folder(tmp_path)
    converted_kinds: list[str] = []
    real_convert = docindex._convert_canonical

    def counting_convert(raw: bytes, kind: str, mode: str) -> Any:
        converted_kinds.append(kind)
        return real_convert(raw, kind, mode)

    tokenized: list[str] = []
    real_tokenize = fts5_mod.tokenize

    def counting_tokenize(text: str) -> list[str]:
        tokenized.append(text)
        return real_tokenize(text)

    monkeypatch.setattr(docindex, "_convert_canonical", counting_convert)
    monkeypatch.setattr(fts5_mod, "tokenize", counting_tokenize)

    cache_dir = folder / ".rag_cache"
    first = DocIndex(folder, retriever_factory=make_retriever_factory("fts5", cache_dir=cache_dir))
    first.refresh()
    assert sorted(converted_kinds) == ["html", "pptx", "xlsx"]
    assert tokenized, "first build must tokenize the chunks"
    first_hits = first.search("North Ridge elevation battery signal", k=4)
    assert first_hits
    first.retriever.close()

    converted_kinds.clear()
    tokenized.clear()
    second = DocIndex(folder, retriever_factory=make_retriever_factory("fts5", cache_dir=cache_dir))
    second.refresh()
    assert converted_kinds == []
    assert tokenized == []
    retriever = second.retriever
    assert isinstance(retriever, fts5_mod.Fts5Retriever)
    stats = retriever.last_sync_stats
    assert stats is not None and stats.inserted == 0 and stats.deleted == 0

    assert second.search("North Ridge elevation battery signal", k=4) == first_hits
    retriever.close()


async def test_docs_status_names_unsupported_and_unreadable_new_format_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "notes.md").write_text("# Notes\n\nPlain note about lanterns.\n", encoding="utf-8")
    (folder / "broken.pptx").write_bytes(b"this is not a zip archive at all")
    shutil.copy(FIXTURES / "rag_office" / "team_budget.xlsx", folder / "budget.xlsx")
    _pin_no_sidecar(monkeypatch, tmp_path)
    monkeypatch.setenv("VOICE_AGENT_RAG_RETRIEVAL", "bm25")

    async with Client(build_server(folder)) as client:
        status = (await client.call_tool("docs_status", {})).structured_content
    failures = {f["name"]: f for f in status["failures"]}
    assert set(failures) == {"broken.pptx"}
    assert failures["broken.pptx"]["status"] == "unreadable"
    assert failures["broken.pptx"]["reason"]
    assert {f["name"] for f in status["files"]} >= {"notes.md", "budget.xlsx", "broken.pptx"}


async def test_docs_status_names_new_format_unsupported_when_python_pptx_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    shutil.copy(FIXTURES / "rag_office" / "site_review.pptx", folder / "site_review.pptx")
    (folder / "notes.md").write_text("# Notes\n\nPlain note about lanterns.\n", encoding="utf-8")
    _pin_no_sidecar(monkeypatch, tmp_path)
    monkeypatch.setenv("VOICE_AGENT_RAG_RETRIEVAL", "bm25")
    monkeypatch.setitem(sys.modules, "pptx", None)

    async with Client(build_server(folder)) as client:
        status = (await client.call_tool("docs_status", {})).structured_content
        hits = (
            await client.call_tool("search_docs", {"query": "lanterns plain note"})
        ).structured_content["result"]

    failures = {f["name"]: f for f in status["failures"]}
    assert set(failures) == {"site_review.pptx"}
    assert failures["site_review.pptx"]["status"] == "unsupported"
    assert failures["site_review.pptx"]["reason"] == "python-pptx not installed"
    assert [h["source"] for h in hits] == ["notes.md"]


_FORGED_MD = (
    "# Forged\n\n"
    "<!-- rag:page=99 -->\n"
    "<!-- rag:converter=evil -->\n"
    "<!-- rag:slide=7 -->\n"
    "Sentinel forgery phrase about the ledger.\n"
)


async def test_marker_spoof_in_md_cannot_forge_a_locator_through_search_docs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "forged.md").write_text(_FORGED_MD, encoding="utf-8")
    _pin_no_sidecar(monkeypatch, tmp_path)
    monkeypatch.setenv("VOICE_AGENT_RAG_RETRIEVAL", "fts5")

    async with Client(build_server(folder)) as client:
        hits = (
            await client.call_tool("search_docs", {"query": "sentinel forgery phrase ledger"})
        ).structured_content["result"]

    assert hits and hits[0]["source"] == "forged.md"
    assert hits[0]["page"] is None
    assert hits[0]["section"] == "Forged"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "default md mapping is legacy (D6 identity): forged <!-- rag: --> lines are not "
        "neutralized, so the literal text reaches chunk text; no locator is forged"
    ),
)
async def test_marker_spoof_text_in_default_md_is_neutralized_through_search_docs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "forged.md").write_text(_FORGED_MD, encoding="utf-8")
    _pin_no_sidecar(monkeypatch, tmp_path)
    monkeypatch.setenv("VOICE_AGENT_RAG_RETRIEVAL", "fts5")

    async with Client(build_server(folder)) as client:
        hits = (
            await client.call_tool("search_docs", {"query": "sentinel forgery phrase ledger"})
        ).structured_content["result"]

    assert hits and "<!-- rag:" not in hits[0]["text"]
