"""Version bumps make stale table caches re-parse (WP5)."""

from __future__ import annotations

from pathlib import Path

from voice_agent.rag import canonical
from voice_agent.rag import docindex as docindex_mod
from voice_agent.rag.docindex import DocIndex


def test_versions_are_bumped() -> None:
    assert (docindex_mod.LEGACY_PARSE_VERSION, canonical.POSTPROC_VERSION) == (2, 2)


def test_old_text_payload_is_rejected_and_new_one_is_read() -> None:
    parse = docindex_mod._parse_legacy_payload
    assert parse('{"v":1,"text":"a: x"}', "csv") is None
    assert parse('{"v":2,"text":"a: x"}', "csv") == ("a: x", None)


def test_csv_cache_entry_from_before_the_bump_is_reparsed(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "w.csv").write_text("Title,m\n2020 Q1,5\n", encoding="utf-8")
    first = DocIndex(root, converters={})
    first.refresh()
    (path,) = sorted((root / ".rag_cache" / "canonical").glob("*.md"))
    path.write_text('{"v":1,"text":"STALE old-format line"}', encoding="utf-8")
    second = DocIndex(root, converters={})
    second.refresh()
    assert [chunk.text for chunk in second.chunks] == [chunk.text for chunk in first.chunks]
    assert "STALE" not in second.chunks[0].text
