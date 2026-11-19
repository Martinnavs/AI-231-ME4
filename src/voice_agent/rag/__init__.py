"""Document index package for local RAG retrieval."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voice_agent.rag.backends import make_retriever_factory, resolve_mode
    from voice_agent.rag.docindex import (
        Bm25Retriever,
        Chunk,
        DocIndex,
        FileRecord,
        RefreshReport,
        Retriever,
    )
    from voice_agent.rag.hybrid import HybridRetriever

__all__ = [
    "Bm25Retriever",
    "Chunk",
    "DocIndex",
    "FileRecord",
    "HybridRetriever",
    "RefreshReport",
    "Retriever",
    "make_retriever_factory",
    "resolve_mode",
]

_LAZY_EXPORTS: dict[str, str] = {
    "Bm25Retriever": "voice_agent.rag.docindex",
    "Chunk": "voice_agent.rag.docindex",
    "DocIndex": "voice_agent.rag.docindex",
    "FileRecord": "voice_agent.rag.docindex",
    "HybridRetriever": "voice_agent.rag.hybrid",
    "RefreshReport": "voice_agent.rag.docindex",
    "Retriever": "voice_agent.rag.docindex",
    "make_retriever_factory": "voice_agent.rag.backends",
    "resolve_mode": "voice_agent.rag.backends",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
