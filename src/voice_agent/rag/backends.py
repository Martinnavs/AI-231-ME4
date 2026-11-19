"""Lightweight retriever backend selection and factory construction."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

from voice_agent.rag.docindex import Bm25Retriever, Retriever
from voice_agent.rag.fts5 import Fts5Retriever, Fts5TokenizerMode

RetrieverMode = Literal["bm25", "fts5", "hybrid", "auto"]
LexicalMode = Literal["bm25", "fts5"]
ALLOWED_MODES: tuple[RetrieverMode, ...] = ("bm25", "fts5", "hybrid", "auto")
ALLOWED_LEXICAL_MODES: tuple[LexicalMode, ...] = ("bm25", "fts5")
DEFAULT_LEXICAL: LexicalMode = "fts5"
DEFAULT_HYBRID_LEXICAL: LexicalMode = "bm25"
DEFAULT_EMBED_PYTHON = "~/.venvs/rag-embed/bin/python"


def resolve_mode(
    flag: str | None = None,
    env: Mapping[str, str] | None = None,
) -> RetrieverMode:
    """Resolve retrieval mode: ``flag`` beats ``VOICE_AGENT_RAG_RETRIEVAL`` beats ``auto``."""
    env_map = os.environ if env is None else env
    raw: str | None
    if flag is not None:
        raw = flag.strip()
    else:
        env_val = env_map.get("VOICE_AGENT_RAG_RETRIEVAL")
        raw = env_val.strip() if env_val is not None and env_val.strip() else None

    if raw is None:
        return "auto"
    normalized = raw.lower()
    if normalized not in ALLOWED_MODES:
        allowed = ", ".join(ALLOWED_MODES)
        raise ValueError(f"unknown retriever mode {raw!r}; expected one of: {allowed}")
    return normalized


def _resolve_lexical(lexical: str) -> LexicalMode:
    normalized = lexical.strip().lower()
    if normalized not in ALLOWED_LEXICAL_MODES:
        allowed = ", ".join(ALLOWED_LEXICAL_MODES)
        raise ValueError(f"unknown lexical mode {lexical!r}; expected one of: {allowed}")
    return normalized


def _resolve_embed_python(embed_python: str | Path | None) -> Path:
    raw_python = (
        str(embed_python).strip()
        if embed_python is not None and str(embed_python).strip()
        else (os.environ.get("VOICE_AGENT_EMBED_PYTHON") or "").strip() or DEFAULT_EMBED_PYTHON
    )
    return Path(raw_python).expanduser()


def _make_lexical_factory(
    lexical: LexicalMode,
    cache_dir: Path | str | None,
    *,
    fts5_tokenizer: Fts5TokenizerMode,
) -> Callable[[], Retriever]:
    if lexical == "bm25":
        return Bm25Retriever

    eff_cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else None
    db_path = eff_cache_dir / "fts5" / "index.sqlite3" if eff_cache_dir is not None else None
    root = eff_cache_dir.parent if eff_cache_dir is not None else Path(".")
    retriever = Fts5Retriever(db_path, root=root, tokenizer=fts5_tokenizer)

    def _fts5_factory() -> Retriever:
        return retriever

    return _fts5_factory


def make_retriever_factory(
    mode: RetrieverMode = "auto",
    cache_dir: Path | str | None = None,
    embed_python: str | Path | None = None,
    model: str | None = None,
    model_dir: Path | str | None = None,
    script: Path | str | None = None,
    *,
    background: bool = True,
    rrf_k: int = 60,
    threads: int | None = None,
    lexical: LexicalMode | None = None,
    fts5_tokenizer: Fts5TokenizerMode = "pretok",
) -> Callable[[], Retriever]:
    """Build a ``Retriever`` factory without importing hybrid/numpy unless needed."""
    resolved_mode = resolve_mode(mode, {})
    explicit_lexical = _resolve_lexical(lexical) if lexical is not None else None

    if resolved_mode == "bm25":
        return Bm25Retriever
    if resolved_mode == "fts5":
        return _make_lexical_factory("fts5", cache_dir, fts5_tokenizer=fts5_tokenizer)

    if resolved_mode == "auto" and not _resolve_embed_python(embed_python).exists():
        non_hybrid_lexical = (
            explicit_lexical if explicit_lexical is not None else _resolve_lexical(DEFAULT_LEXICAL)
        )
        return _make_lexical_factory(
            non_hybrid_lexical,
            cache_dir,
            fts5_tokenizer=fts5_tokenizer,
        )

    hybrid_lexical = (
        explicit_lexical
        if explicit_lexical is not None
        else _resolve_lexical(DEFAULT_HYBRID_LEXICAL)
    )
    lexical_factory = _make_lexical_factory(
        hybrid_lexical,
        cache_dir,
        fts5_tokenizer=fts5_tokenizer,
    )

    from voice_agent.rag import hybrid as _hybrid
    from voice_agent.rag.embedder_sidecar import DEFAULT_MODEL, DEFAULT_THREADS

    return _hybrid.make_retriever_factory(
        mode=resolved_mode,
        cache_dir=cache_dir,
        embed_python=embed_python,
        model=DEFAULT_MODEL if model is None else model,
        model_dir=model_dir,
        script=script,
        background=background,
        rrf_k=rrf_k,
        threads=DEFAULT_THREADS if threads is None else threads,
        lexical_factory=lexical_factory,
    )
