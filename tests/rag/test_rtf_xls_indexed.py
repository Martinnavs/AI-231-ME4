"""rtf and xls files must survive the full index path (cache keys, converter labels, markers)."""

from __future__ import annotations

import shutil
from pathlib import Path

from voice_agent.rag.docindex import DocIndex

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "legacy"


def test_rtf_and_xls_are_indexed_and_searchable_with_and_without_cache(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("ffc.rtf", "ffc.xls"):
        shutil.copy(FIXTURES / name, corpus / name)
    for _ in range(2):  # the second pass reads the canonical cache
        index = DocIndex(corpus, canonical_cache_dir=tmp_path / "cache")
        index.refresh()
        assert index.failures() == []
        assert {r.name: r.status for r in index.records()} == {"ffc.rtf": "ok", "ffc.xls": "ok"}
    assert any("ffc.rtf" in hit["source"] for hit in index.search("file format commons"))
