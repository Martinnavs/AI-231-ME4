"""FTS5 backend checks through the docs server builder (search_docs / docs_status)."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

from doc_chat.corpus import tokenize
from voice_agent.mcp_servers.docs import build_server
from voice_agent.rag import fts5 as fts5_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_CORPUS = REPO_ROOT / "tests" / "fixtures" / "rag_corpus"

FIXTURE_QUERIES = [
    "daily meal allowance expense claims",
    "new hires first day laptop badge",
    "factory reset pinhole twelve seconds",
    "office wifi password",
]

ADVERSARIAL_QUERIES = [
    '"alpha beta',
    "alpha*",
    "NEAR(alpha beta, 3)",
    "alpha AND NOT beta OR gamma",
    "title:alpha",
    "^alpha",
    "((((alpha))))",
    "alpha\x00beta",
    '"" OR 1=1 --',
    'alpha"); DROP TABLE fts; --',
    "{alpha} [beta] ~gamma ?delta",
    "alpha " * 1700,
    "x" * 10_000 + " alpha",
    "“alpha” ‘beta’",
]


def _write_alpha_corpus(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rows = [
        "alpha beta gamma delta alpha calibration",
        "beta gamma sensor reading and alpha drift",
        "delta epsilon widget torque alpha",
        "gamma ray widget beta inventory",
        "alpha widget shelf label gamma",
        "unrelated kitchen recipe with lemons",
    ]
    for idx, body in enumerate(rows):
        (folder / f"doc_{idx}.md").write_text(f"# Doc {idx}\n\n{body}.\n", encoding="utf-8")


async def _search_all(server: Any, queries: Sequence[str], top_k: int = 4) -> list[list[dict]]:
    async with Client(server) as client:
        out: list[list[dict]] = []
        for query in queries:
            result = await client.call_tool("search_docs", {"query": query, "top_k": top_k})
            out.append(result.structured_content["result"])
        return out


async def _status(server: Any) -> dict[str, Any]:
    async with Client(server) as client:
        return dict((await client.call_tool("docs_status", {})).structured_content)


async def test_fts5_mode_end_to_end_over_fixtures_with_private_cache_perms(tmp_path: Path) -> None:
    corpus = _copy_fixtures(tmp_path / "fixtures_copy")
    bm25_corpus = _copy_fixtures(tmp_path / "bm25_copy")

    fts5_hits = await _search_all(build_server(corpus, retriever_mode="fts5"), FIXTURE_QUERIES)
    bm25_hits = await _search_all(
        build_server(bm25_corpus, retriever_mode="bm25"),
        FIXTURE_QUERIES,
    )

    assert fts5_hits[0] and fts5_hits[0][0]["source"] == "handbook.pdf"
    assert "350 pesos" in fts5_hits[0][0]["text"]
    assert fts5_hits[2] and fts5_hits[2][0]["source"] == "router_reset.md"
    assert [sorted((h["source"], h["page"]) for h in hits) for hits in fts5_hits] == [
        sorted((h["source"], h["page"]) for h in hits) for hits in bm25_hits
    ]

    db_dir = corpus / ".rag_cache" / "fts5"
    db_file = db_dir / "index.sqlite3"
    assert db_file.is_file()
    assert stat.S_IMODE(db_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(db_dir.stat().st_mode) == 0o700


def _copy_fixtures(dest: Path) -> Path:
    shutil.copytree(FIXTURES_CORPUS, dest)
    return dest


async def test_db_tamper_mid_session_serves_bm25_results_without_raising(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    folder = tmp_path / "docs"
    _write_alpha_corpus(folder)
    shutil.copytree(folder, tmp_path / "bm25_docs")

    fts5_server = build_server(folder, retriever_mode="fts5")
    before = await _search_all(fts5_server, ["alpha widget", "beta gamma"])
    assert before[0]
    assert (await _status(fts5_server))["retriever"] == "fts5:file"

    db_file = folder / ".rag_cache" / "fts5" / "index.sqlite3"
    with open(db_file, "r+b") as fh:
        fh.write(b"\x00corrupted mid-session\x00" * 64)

    queries = ["alpha widget", "beta gamma", "delta epsilon widget"]
    after = await _search_all(fts5_server, queries)
    expected = await _search_all(
        build_server(tmp_path / "bm25_docs", retriever_mode="bm25"), queries
    )

    assert after == expected
    status = await _status(fts5_server)
    assert status["retriever"].startswith("bm25 (fts5 unavailable:")
    stderr = capsys.readouterr().err
    assert stderr.count("fts5 unavailable") == 1


async def test_concurrent_search_docs_threads_match_serial_results(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    _write_alpha_corpus(folder)
    server = build_server(folder, retriever_mode="fts5")
    queries = ["alpha widget", "beta gamma", "delta epsilon widget", "gamma ray widget"]
    serial = await _search_all(server, queries)

    workers = 8
    barrier = threading.Barrier(workers)

    def _worker() -> list[list[dict]]:
        barrier.wait(timeout=30)
        return asyncio.run(_search_all(server, queries * 2))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker) for _ in range(workers)]
        results = [future.result(timeout=120) for future in futures]

    for per_thread in results:
        assert per_thread == serial * 2


async def test_read_only_docs_folder_serves_in_memory_fts5(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores directory permission bits")
    folder = tmp_path / "docs"
    _write_alpha_corpus(folder)
    folder.chmod(0o555)
    for entry in folder.iterdir():
        entry.chmod(0o444)
    try:
        server = build_server(folder, retriever_mode="fts5")
        status = await _status(server)
        hits = (await _search_all(server, ["alpha widget"]))[0]
    finally:
        folder.chmod(0o755)
        for entry in folder.iterdir():
            entry.chmod(0o644)

    assert status["retriever"] == "fts5:memory"
    assert hits and hits[0]["source"] in {"doc_0.md", "doc_4.md", "doc_2.md"}
    assert not (folder / ".rag_cache").exists()
    assert capsys.readouterr().err.count("using memory store") == 1


async def test_adversarial_queries_through_search_docs_never_raise_or_add_terms(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "docs"
    _write_alpha_corpus(folder)
    server = build_server(folder, retriever_mode="fts5")

    async with Client(server) as client:
        non_empty = 0
        for query in ADVERSARIAL_QUERIES:
            adversarial = (
                await client.call_tool("search_docs", {"query": query, "top_k": 4})
            ).structured_content["result"]
            plain_query = " ".join(tokenize(query[:2000]))
            plain = (
                await client.call_tool("search_docs", {"query": plain_query, "top_k": 4})
            ).structured_content["result"]
            assert adversarial == plain, query
            non_empty += bool(adversarial)

    assert non_empty >= 3
    assert (await _status(server))["retriever"] == "fts5:file"


async def test_probe_failure_through_search_docs_is_identical_to_bm25(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "fixtures_copy"
    shutil.copytree(FIXTURES_CORPUS, corpus)
    bm25_corpus = _copy_fixtures(tmp_path / "bm25_copy")

    monkeypatch.setattr(
        fts5_mod,
        "fts5_available",
        lambda tokenizer=fts5_mod.PRETOK_TOKENIZER: (False, "forced probe failure"),
    )
    fallback_server = build_server(corpus, retriever_mode="fts5")
    queries = [*FIXTURE_QUERIES, "lemon kitchen recipe"]
    fallback = await _search_all(fallback_server, queries)
    fallback_again = await _search_all(fallback_server, queries)
    expected = await _search_all(build_server(bm25_corpus, retriever_mode="bm25"), queries)

    assert fallback == expected
    assert fallback_again == expected
    status = await _status(fallback_server)
    assert status["retriever"] == "bm25 (fts5 unavailable: forced probe failure)"
    assert status["failures"] == []
    assert capsys.readouterr().err.count("fts5 unavailable (forced probe failure)") == 1


async def test_edited_and_deleted_files_leave_no_stale_fts5_hits_through_search_docs(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text("# A\n\nalpha uniquewordone widget.\n", encoding="utf-8")
    (folder / "b.md").write_text("# B\n\nbeta uniquewordtwo gizmo.\n", encoding="utf-8")
    (folder / "c.md").write_text("# C\n\ngamma stable reference.\n", encoding="utf-8")
    server = build_server(folder, retriever_mode="fts5")

    assert (await _search_all(server, ["uniquewordone"]))[0][0]["source"] == "a.md"
    assert (await _search_all(server, ["uniquewordtwo"]))[0][0]["source"] == "b.md"

    (folder / "a.md").write_text("# A\n\nalpha replacement sprocket.\n", encoding="utf-8")
    os.utime(folder / "a.md", ns=(10**18, 10**18 + 10**9))
    (folder / "b.md").unlink()

    stale, edited, gone = await _search_all(server, ["uniquewordone", "sprocket", "uniquewordtwo"])
    assert stale == []
    assert [h["source"] for h in edited] == ["a.md"]
    assert gone == []
    names = {f["name"] for f in (await _status(server))["files"]}
    assert names == {"a.md", "c.md"}
