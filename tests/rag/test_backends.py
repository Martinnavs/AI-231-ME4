"""Offline tests for voice_agent.rag.backends and hybrid lexical arm wiring."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from voice_agent.rag import backends, hybrid
from voice_agent.rag.docindex import Bm25Retriever, Chunk, DocIndex
from voice_agent.rag.fts5 import Fts5Retriever
from voice_agent.rag.hybrid import HybridRetriever

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
FAKE_SIDECAR = Path(__file__).resolve().with_name("fake_sidecar.py")


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
        "hybrid",
        "embed_cache",
        "embed_client",
        "embedder_sidecar",
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


def test_resolve_mode_fts5_validation_and_hybrid_reexports() -> None:
    assert backends.ALLOWED_MODES == ("bm25", "fts5", "hybrid", "auto")
    assert backends.DEFAULT_LEXICAL == "fts5"
    assert hybrid.ALLOWED_MODES is backends.ALLOWED_MODES
    assert hybrid.resolve_mode is backends.resolve_mode

    assert backends.resolve_mode("fts5", {}) == "fts5"
    assert backends.resolve_mode("FTS5", {}) == "fts5"
    assert backends.resolve_mode(None, {"VOICE_AGENT_RAG_RETRIEVAL": "fts5"}) == "fts5"
    assert backends.resolve_mode("bm25", {"VOICE_AGENT_RAG_RETRIEVAL": "fts5"}) == "bm25"
    assert hybrid.resolve_mode("fts5", {}) == "fts5"

    with pytest.raises(ValueError, match="bm25.*fts5.*hybrid.*auto"):
        backends.resolve_mode("unknown", {})
    with pytest.raises(ValueError, match="bm25.*fts5.*hybrid.*auto"):
        backends.resolve_mode(None, {"VOICE_AGENT_RAG_RETRIEVAL": "dense"})
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
def test_hybrid_rrf_with_configured_lexical_arm_and_fake_sidecar(
    tmp_path: Path,
    lexical: str,
) -> None:
    chunks = [
        _chunk(0, "a.md", "a.md", "alpha beta gamma delta"),
        _chunk(1, "b.md", "b.md", "epsilon zeta eta theta"),
        _chunk(2, "c.md", "c.md", "xenon lamp calibration specification"),
    ]
    cache_dir = tmp_path / f"cache_{lexical}"
    factory = backends.make_retriever_factory(
        mode="hybrid",
        cache_dir=cache_dir,
        embed_python=sys.executable,
        script=FAKE_SIDECAR,
        background=False,
        rrf_k=60,
        lexical=lexical,  # type: ignore[arg-type]
    )
    retriever = factory()
    assert isinstance(retriever, HybridRetriever)
    assert factory() is retriever
    if lexical == "fts5":
        assert isinstance(retriever._bm25, Fts5Retriever)
    else:
        assert isinstance(retriever._bm25, Bm25Retriever)

    try:
        retriever.build(chunks)
        assert retriever.wait_ready(timeout=5.0)
        assert retriever.ready

        # Lexical arm returns only chunk 2; fake_sidecar ranks [0, 1, 2].
        ranked = retriever.rank("xenon", 3)
        expected = {
            2: 1.0 / 61 + 1.0 / 63,
            0: 1.0 / 61,
            1: 1.0 / 62,
        }
        assert [idx for idx, _ in ranked] == [2, 0, 1]
        for idx, score in ranked:
            assert score == pytest.approx(expected[idx])
    finally:
        retriever.close()


@pytest.mark.parametrize("lexical", ["bm25", "fts5"])
def test_hybrid_docindex_end_to_end_with_fake_sidecar(tmp_path: Path, lexical: str) -> None:
    corpus = tmp_path / f"docs_{lexical}"
    corpus.mkdir()
    (corpus / "alpha.md").write_text("# Alpha\n\nGeneral system overview.\n", encoding="utf-8")
    (corpus / "xenon.md").write_text(
        "# Xenon\n\nXenon arc lamp startup voltage specification.\n",
        encoding="utf-8",
    )

    factory = backends.make_retriever_factory(
        mode="hybrid",
        cache_dir=corpus / ".rag_cache",
        embed_python=sys.executable,
        script=FAKE_SIDECAR,
        background=False,
        lexical=lexical,  # type: ignore[arg-type]
    )
    retriever = factory()
    assert isinstance(retriever, HybridRetriever)
    try:
        idx = DocIndex(corpus, retriever_factory=factory)
        idx.refresh()
        assert retriever.wait_ready(timeout=5.0)
        assert retriever.ready
        hits = idx.search("xenon voltage", k=2)
        assert hits
        assert hits[0]["source"] == "xenon.md"
    finally:
        retriever.close()


def test_hybrid_make_retriever_factory_lexical_factory_override(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    custom_fts5 = Fts5Retriever(cache_dir / "fts5" / "index.sqlite3", root=tmp_path)
    factory = hybrid.make_retriever_factory(
        mode="hybrid",
        cache_dir=cache_dir,
        embed_python=sys.executable,
        script=FAKE_SIDECAR,
        background=False,
        lexical_factory=lambda: custom_fts5,
    )
    retriever = factory()
    assert isinstance(retriever, HybridRetriever)
    assert retriever._bm25 is custom_fts5
    retriever.close()


@pytest.mark.parametrize("lexical", ["bm25", "fts5"])
def test_auto_with_missing_thin_env_does_not_import_hybrid_subprocess(
    tmp_path: Path,
    lexical: str,
) -> None:
    missing_python = tmp_path / "no-such-venv" / "bin" / "python"
    cache_dir = tmp_path / ".rag_cache"

    # Note: rag/__init__.py still imports hybrid eagerly until F05, so we stub the
    # package containers with __path__ pointing to src/ to isolate backends.py itself
    # (this works identically before and after F05).
    script = f"""
import json
import sys
import types
from pathlib import Path

src_root = Path({str(SRC_ROOT)!r})
sys.path.insert(0, str(src_root))

pkg_va = types.ModuleType("voice_agent")
pkg_va.__path__ = [str(src_root / "voice_agent")]
pkg_rag = types.ModuleType("voice_agent.rag")
pkg_rag.__path__ = [str(src_root / "voice_agent" / "rag")]
sys.modules["voice_agent"] = pkg_va
sys.modules["voice_agent.rag"] = pkg_rag

from voice_agent.rag import backends
from voice_agent.rag.docindex import Chunk

factory = backends.make_retriever_factory(
    mode="auto",
    cache_dir=Path({str(cache_dir)!r}),
    embed_python=None,
    lexical={lexical!r},
)
hybrid_loaded_after_factory = "voice_agent.rag.hybrid" in sys.modules
numpy_loaded_after_factory = "numpy" in sys.modules
retriever = factory()
retriever.build([
    Chunk(
        id="a.md#0",
        file_sha256="f0",
        chunk_sha256="c0",
        position=0,
        page=None,
        section="a.md",
        file_date="2026-11-19",
        source="a.md",
        header="a.md",
        text="telemetry calibration guide",
    )
])
hits = retriever.rank("telemetry", 2)
close_fn = getattr(retriever, "close", None)
if callable(close_fn):
    close_fn()

print(json.dumps({{
    "hybrid_loaded_after_factory": hybrid_loaded_after_factory,
    "numpy_loaded_after_factory": numpy_loaded_after_factory,
    "hybrid_loaded": "voice_agent.rag.hybrid" in sys.modules,
    "numpy_loaded": "numpy" in sys.modules,
    "rank_bm25_loaded": "rank_bm25" in sys.modules,
    "retriever_type": type(retriever).__name__,
    "hits": hits,
}}))
"""
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    env["VOICE_AGENT_EMBED_PYTHON"] = str(missing_python)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    payload = json.loads(proc.stdout.strip())
    assert payload["hybrid_loaded_after_factory"] is False
    assert payload["numpy_loaded_after_factory"] is False
    assert payload["hybrid_loaded"] is False
    if lexical == "fts5":
        assert payload["numpy_loaded"] is False
        assert payload["rank_bm25_loaded"] is False
    assert payload["retriever_type"] == ("Fts5Retriever" if lexical == "fts5" else "Bm25Retriever")
    assert len(payload["hits"]) == 1
    assert payload["hits"][0][0] == 0


def test_auto_with_existing_thin_env_imports_hybrid_subprocess(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".rag_cache"
    script = f"""
import json
import sys
import types
from pathlib import Path

src_root = Path({str(SRC_ROOT)!r})
sys.path.insert(0, str(src_root))

pkg_va = types.ModuleType("voice_agent")
pkg_va.__path__ = [str(src_root / "voice_agent")]
pkg_rag = types.ModuleType("voice_agent.rag")
pkg_rag.__path__ = [str(src_root / "voice_agent" / "rag")]
sys.modules["voice_agent"] = pkg_va
sys.modules["voice_agent.rag"] = pkg_rag

from voice_agent.rag import backends

factory = backends.make_retriever_factory(
    mode="auto",
    cache_dir=Path({str(cache_dir)!r}),
    embed_python=sys.executable,
    script=Path({str(FAKE_SIDECAR)!r}),
    background=False,
)
retriever = factory()
try:
    print(json.dumps({{
        "hybrid_loaded": "voice_agent.rag.hybrid" in sys.modules,
        "retriever_type": type(retriever).__name__,
    }}))
finally:
    retriever.close()
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout.strip())
    assert payload["hybrid_loaded"] is True
    assert payload["retriever_type"] == "HybridRetriever"


def test_hybrid_default_lexical_is_bm25_while_non_hybrid_stays_fts5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert backends.DEFAULT_HYBRID_LEXICAL == "bm25"
    assert backends.DEFAULT_LEXICAL == "fts5"

    captured: list[dict[str, object]] = []

    def fake_hybrid_factory(*args: object, **kwargs: object) -> object:
        captured.append(dict(kwargs))
        lexical_factory = kwargs["lexical_factory"]
        assert callable(lexical_factory)
        return lexical_factory

    monkeypatch.setattr(hybrid, "make_retriever_factory", fake_hybrid_factory)

    hybrid_factory = backends.make_retriever_factory(
        mode="hybrid",
        cache_dir=tmp_path / ".rag_cache",
    )
    assert hybrid_factory is Bm25Retriever
    assert captured[-1]["lexical_factory"] is Bm25Retriever

    auto_hybrid_factory = backends.make_retriever_factory(
        mode="auto",
        cache_dir=tmp_path / ".rag_cache",
        embed_python=sys.executable,
    )
    assert auto_hybrid_factory is Bm25Retriever
    assert captured[-1]["lexical_factory"] is Bm25Retriever

    calls_before = len(captured)
    auto_lexical_factory = backends.make_retriever_factory(
        mode="auto",
        cache_dir=tmp_path / ".rag_cache",
        embed_python=tmp_path / "no-such-venv" / "bin" / "python",
    )
    assert len(captured) == calls_before
    retriever = auto_lexical_factory()
    try:
        assert isinstance(retriever, Fts5Retriever)
    finally:
        close_fn = getattr(retriever, "close", None)
        if callable(close_fn):
            close_fn()


def test_explicit_lexical_override_respected_across_hybrid_and_auto(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_hybrid_factory(*args: object, **kwargs: object) -> object:
        captured.append(dict(kwargs))
        return kwargs["lexical_factory"]

    monkeypatch.setattr(hybrid, "make_retriever_factory", fake_hybrid_factory)

    for mode, embed_python in [
        ("hybrid", None),
        ("auto", sys.executable),
    ]:
        factory = backends.make_retriever_factory(
            mode=mode,  # type: ignore[arg-type]
            cache_dir=tmp_path / f"cache_{mode}",
            embed_python=embed_python,
            lexical="fts5",
        )
        assert callable(factory)
        retriever = factory()
        try:
            assert isinstance(retriever, Fts5Retriever)
        finally:
            close_fn = getattr(retriever, "close", None)
            if callable(close_fn):
                close_fn()
        assert captured[-1]["lexical_factory"] is factory

    bm25_factory = backends.make_retriever_factory(
        mode="auto",
        cache_dir=tmp_path / "cache_no_thin",
        embed_python=tmp_path / "no-such-venv" / "bin" / "python",
        lexical="bm25",
    )
    assert bm25_factory is Bm25Retriever
