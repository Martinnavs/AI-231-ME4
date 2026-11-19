"""Default retriever selection matrix: env var x thin env x lexical arm (no sidecar started)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from voice_agent.rag import backends
from voice_agent.rag.docindex import Bm25Retriever
from voice_agent.rag.fts5 import Fts5Retriever

REPO_ROOT = Path(__file__).resolve().parents[2]
FAKE_SIDECAR = REPO_ROOT / "tests" / "rag" / "fake_sidecar.py"

ENV_CASES: list[tuple[str | None, str]] = [
    (None, "auto"),
    ("", "auto"),
    ("   ", "auto"),
    ("bm25", "bm25"),
    ("BM25", "bm25"),
    ("  fts5 ", "fts5"),
    ("Hybrid", "hybrid"),
    ("auto", "auto"),
]


@pytest.mark.parametrize(("raw", "expected"), ENV_CASES)
def test_env_var_resolves_to_mode(raw: str | None, expected: str) -> None:
    env: dict[str, str] = {} if raw is None else {"VOICE_AGENT_RAG_RETRIEVAL": raw}
    assert backends.resolve_mode(None, env) == expected


def test_unknown_env_value_raises_and_names_allowed_modes() -> None:
    with pytest.raises(ValueError, match="bm25, fts5, hybrid, auto"):
        backends.resolve_mode(None, {"VOICE_AGENT_RAG_RETRIEVAL": "faiss"})


@pytest.mark.parametrize(
    ("flag", "env_raw", "expected"),
    [
        ("fts5", "bm25", "fts5"),
        ("bm25", "fts5", "bm25"),
        ("auto", "hybrid", "auto"),
        (None, "fts5", "fts5"),
        (None, None, "auto"),
    ],
)
def test_flag_beats_env_and_env_beats_auto(
    flag: str | None,
    env_raw: str | None,
    expected: str,
) -> None:
    env: dict[str, str] = {} if env_raw is None else {"VOICE_AGENT_RAG_RETRIEVAL": env_raw}
    assert backends.resolve_mode(flag, env) == expected


def _kind(factory: Any) -> str:
    if factory is Bm25Retriever:
        return "bm25"
    if getattr(factory, "__module__", "") == "voice_agent.rag.hybrid":
        return "hybrid"
    retriever = factory()
    try:
        return {"Fts5Retriever": "fts5", "Bm25Retriever": "bm25", "HybridRetriever": "hybrid"}[
            type(retriever).__name__
        ]
    finally:
        close_fn = getattr(retriever, "close", None)
        if callable(close_fn):
            close_fn()


@pytest.mark.parametrize(
    ("mode", "thin_env_present", "lexical", "expected_kind"),
    [
        ("bm25", False, "bm25", "bm25"),
        ("bm25", True, "bm25", "bm25"),
        ("fts5", False, "bm25", "fts5"),
        ("fts5", True, "bm25", "fts5"),
        ("auto", False, "bm25", "bm25"),
        ("auto", False, "fts5", "fts5"),
        ("auto", True, "bm25", "hybrid"),
        ("hybrid", False, "bm25", "hybrid"),
    ],
)
def test_mode_thin_env_and_lexical_arm_select_expected_backend(
    tmp_path: Path,
    mode: str,
    thin_env_present: bool,
    lexical: str,
    expected_kind: str,
) -> None:
    embed_python: Path | str | None
    if thin_env_present:
        embed_python = sys.executable
    else:
        embed_python = tmp_path / "no-such-venv" / "bin" / "python"
    factory = backends.make_retriever_factory(
        mode=mode,  # type: ignore[arg-type]
        cache_dir=tmp_path / ".rag_cache",
        embed_python=embed_python,
        script=FAKE_SIDECAR if thin_env_present else None,
        background=False,
        lexical=lexical,  # type: ignore[arg-type]
    )
    assert _kind(factory) == expected_kind


def test_auto_without_thin_env_uses_the_lexical_default_and_never_hybrid(tmp_path: Path) -> None:
    factory = backends.make_retriever_factory(
        mode="auto",
        cache_dir=tmp_path / ".rag_cache",
        embed_python=tmp_path / "no-such-venv" / "bin" / "python",
    )
    assert getattr(factory, "__module__", "") != "voice_agent.rag.hybrid"
    expected = Fts5Retriever if backends.DEFAULT_LEXICAL == "fts5" else Bm25Retriever
    retriever = factory()
    try:
        assert isinstance(retriever, expected)
    finally:
        close_fn = getattr(retriever, "close", None)
        if callable(close_fn):
            close_fn()


def test_lexical_default_is_fts5_after_adoption() -> None:
    # User-approved adoption (F09, 2026-11-19): the non-hybrid default is FTS5 with rank-bm25
    # as the automatic fallback; the hybrid lexical arm default is covered by test_backends.py.
    assert backends.DEFAULT_LEXICAL == "fts5"


@pytest.mark.parametrize("mode", ["bm25", "fts5", "hybrid", "auto"])
def test_unknown_lexical_arm_raises_for_every_mode(tmp_path: Path, mode: str) -> None:
    with pytest.raises(ValueError, match="unknown lexical mode"):
        backends.make_retriever_factory(
            mode=mode,  # type: ignore[arg-type]
            cache_dir=tmp_path / ".rag_cache",
            embed_python=tmp_path / "no-such-venv" / "bin" / "python",
            lexical="faiss",  # type: ignore[arg-type]
        )


def test_unknown_mode_raises_from_factory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown retriever mode"):
        backends.make_retriever_factory(
            mode="faiss",  # type: ignore[arg-type]
            cache_dir=tmp_path / ".rag_cache",
        )
