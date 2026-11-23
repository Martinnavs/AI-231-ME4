"""Offline tests for voice_agent.rag.backends (lexical retriever selection)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from voice_agent.rag import backends
from voice_agent.rag.docindex import Bm25Retriever, Chunk, DocIndex
from voice_agent.rag.fts5 import Fts5Retriever

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


def _chunk(idx: int, source: str, header: str, text: str) -> Chunk:
    return Chunk(
        id=f"{source}#{idx}",
        file_sha256=f"f{idx}",
        chunk_sha256=f"c{idx}",
        position=idx,
        page=None,
        section=header,
        file_date="2026-11-19",
        source=source,
        header=header,
        text=text,
    )


def test_no_heavy_module_level_imports_in_backends() -> None:
    src_path = SRC_ROOT / "voice_agent" / "rag" / "backends.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"), filename=str(src_path))
    forbidden = {
        "numpy",
        "rank_bm25",
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = set(alias.name.split("."))
                assert not (parts & forbidden), f"forbidden module-level import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod_parts = set((node.module or "").split("."))
            assert not (mod_parts & forbidden), f"forbidden module-level import: {node.module}"
            for alias in node.names:
                assert alias.name not in forbidden, f"forbidden module-level import: {alias.name}"


def test_resolve_mode_fts5_validation_and_rejects_hybrid() -> None:
    assert backends.ALLOWED_MODES == ("bm25", "fts5", "auto")
    assert backends.DEFAULT_LEXICAL == "fts5"

    assert backends.resolve_mode("fts5", {}) == "fts5"
    assert backends.resolve_mode("FTS5", {}) == "fts5"
    assert backends.resolve_mode(None, {"VOICE_AGENT_RAG_RETRIEVAL": "fts5"}) == "fts5"
    assert backends.resolve_mode("bm25", {"VOICE_AGENT_RAG_RETRIEVAL": "fts5"}) == "bm25"
    with pytest.raises(ValueError, match="bm25.*fts5.*auto"):
        backends.resolve_mode("unknown", {})
    with pytest.raises(ValueError, match="bm25.*fts5.*auto"):
        backends.resolve_mode(None, {"VOICE_AGENT_RAG_RETRIEVAL": "dense"})
    with pytest.raises(ValueError, match="unknown retriever mode"):
        backends.resolve_mode("hybrid", {})
    with pytest.raises(ValueError, match="unknown lexical mode"):
        backends.make_retriever_factory(mode="auto", lexical="dense")  # type: ignore[arg-type]


def test_bm25_and_fts5_factories_and_docindex(tmp_path: Path) -> None:
    bm25_factory = backends.make_retriever_factory(mode="bm25", cache_dir=tmp_path / ".rag_cache")
    assert bm25_factory is Bm25Retriever

    mem_factory = backends.make_retriever_factory(mode="fts5", cache_dir=None)
    mem_r1 = mem_factory()
    mem_r2 = mem_factory()
    assert isinstance(mem_r1, Fts5Retriever)
    assert mem_r1 is mem_r2
    assert mem_r1.status() == "fts5:memory"
    mem_r1.close()

    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "solar.md").write_text(
        "# Solar\n\nPhotovoltaic inverters convert DC power into AC grid power.\n",
        encoding="utf-8",
    )
    cache_dir = corpus / ".rag_cache"
    file_factory = backends.make_retriever_factory(
        mode="fts5",
        cache_dir=cache_dir,
        fts5_tokenizer="porter",
    )
    r1 = file_factory()
    try:
        assert isinstance(r1, Fts5Retriever)
        assert file_factory() is r1
        assert r1.status() == "fts5:file"
        assert (cache_dir / "fts5" / "index.sqlite3").is_file()

        idx = DocIndex(corpus, retriever_factory=file_factory)
        idx.refresh()
        hits = idx.search("inverters", k=2)
        assert hits
        assert hits[0]["source"] == "solar.md"
    finally:
        r1.close()


@pytest.mark.parametrize("lexical", ["bm25", "fts5"])
def test_auto_with_lexical_override_builds_the_named_lexical_backend(
    tmp_path: Path,
    lexical: str,
) -> None:
    corpus = tmp_path / f"docs_{lexical}"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nTelemetry calibration guide.\n", encoding="utf-8")
    factory = backends.make_retriever_factory(
        mode="auto",
        cache_dir=corpus / ".rag_cache",
        lexical=lexical,  # type: ignore[arg-type]
    )
    expected = Fts5Retriever if lexical == "fts5" else Bm25Retriever
    retriever = factory()
    try:
        assert isinstance(retriever, expected)
        idx = DocIndex(corpus, retriever_factory=factory)
        idx.refresh()
        hits = idx.search("telemetry calibration", k=2)
        assert hits
        assert hits[0]["source"] == "a.md"
    finally:
        close_fn = getattr(retriever, "close", None)
        if callable(close_fn):
            close_fn()


def test_auto_defaults_to_fts5_and_explicit_bm25_is_honoured(tmp_path: Path) -> None:
    auto_factory = backends.make_retriever_factory(mode="auto", cache_dir=tmp_path / ".rag_cache")
    retriever = auto_factory()
    try:
        assert isinstance(retriever, Fts5Retriever)
        assert auto_factory() is retriever
    finally:
        retriever.close()

    assert backends.make_retriever_factory(mode="auto", lexical="bm25") is Bm25Retriever
