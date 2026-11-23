"""Default retriever selection matrix: env var x lexical arm (lexical-only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voice_agent.rag import backends
from voice_agent.rag.docindex import Bm25Retriever
from voice_agent.rag.fts5 import Fts5Retriever

ENV_CASES: list[tuple[str | None, str]] = [
    (None, "auto"),
    ("", "auto"),
    ("   ", "auto"),
    ("bm25", "bm25"),
    ("BM25", "bm25"),
    ("  fts5 ", "fts5"),
    ("auto", "auto"),
]


@pytest.mark.parametrize(("raw", "expected"), ENV_CASES)
def test_env_var_resolves_to_mode(raw: str | None, expected: str) -> None:
    env: dict[str, str] = {} if raw is None else {"VOICE_AGENT_RAG_RETRIEVAL": raw}
    assert backends.resolve_mode(None, env) == expected


@pytest.mark.parametrize("raw", ["faiss", "hybrid", "Hybrid"])
def test_unknown_or_removed_env_value_raises_and_names_allowed_modes(raw: str) -> None:
    with pytest.raises(ValueError, match="bm25, fts5, auto"):
        backends.resolve_mode(None, {"VOICE_AGENT_RAG_RETRIEVAL": raw})


@pytest.mark.parametrize(
    ("flag", "env_raw", "expected"),
    [
        ("fts5", "bm25", "fts5"),
        ("bm25", "fts5", "bm25"),
        ("auto", "fts5", "auto"),
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
    retriever = factory()
    try:
        return {"Fts5Retriever": "fts5", "Bm25Retriever": "bm25"}[type(retriever).__name__]
    finally:
        close_fn = getattr(retriever, "close", None)
        if callable(close_fn):
            close_fn()


@pytest.mark.parametrize(
    ("mode", "lexical", "expected_kind"),
    [
        ("bm25", "bm25", "bm25"),
        ("fts5", "bm25", "fts5"),
        ("auto", "bm25", "bm25"),
        ("auto", "fts5", "fts5"),
    ],
)
def test_mode_and_lexical_arm_select_expected_backend(
    tmp_path: Path,
    mode: str,
    lexical: str,
    expected_kind: str,
) -> None:
    factory = backends.make_retriever_factory(
        mode=mode,  # type: ignore[arg-type]
        cache_dir=tmp_path / ".rag_cache",
        lexical=lexical,  # type: ignore[arg-type]
    )
    assert _kind(factory) == expected_kind


def test_auto_without_a_lexical_override_uses_the_fts5_default(tmp_path: Path) -> None:
    factory = backends.make_retriever_factory(mode="auto", cache_dir=tmp_path / ".rag_cache")
    expected = Fts5Retriever if backends.DEFAULT_LEXICAL == "fts5" else Bm25Retriever
    retriever = factory()
    try:
        assert isinstance(retriever, expected)
    finally:
        close_fn = getattr(retriever, "close", None)
        if callable(close_fn):
            close_fn()


def test_lexical_default_is_fts5_after_adoption() -> None:
    # User-approved adoption (F09, 2026-11-19): the default lexical backend is FTS5 with
    # rank-bm25 as the automatic fallback.
    assert backends.DEFAULT_LEXICAL == "fts5"


@pytest.mark.parametrize("mode", ["bm25", "fts5", "auto"])
def test_unknown_lexical_arm_raises_for_every_mode(tmp_path: Path, mode: str) -> None:
    with pytest.raises(ValueError, match="unknown lexical mode"):
        backends.make_retriever_factory(
            mode=mode,  # type: ignore[arg-type]
            cache_dir=tmp_path / ".rag_cache",
            lexical="faiss",  # type: ignore[arg-type]
        )


def test_unknown_mode_raises_from_factory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown retriever mode"):
        backends.make_retriever_factory(
            mode="faiss",  # type: ignore[arg-type]
            cache_dir=tmp_path / ".rag_cache",
        )
