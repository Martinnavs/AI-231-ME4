"""Big text/csv files are indexed from a 64 MB prefix instead of being skipped."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_agent.rag import docindex as docindex_mod
from voice_agent.rag.docindex import DocIndex


def _lines(count: int, *, marker: str | None = None, marker_at: int | None = None) -> str:
    """Return ``count`` lines, optionally inserting ``marker`` at 0-based index ``marker_at``."""
    lines: list[str] = []
    for i in range(count):
        if marker_at is not None and i == marker_at:
            lines.append(f"line {i} {marker}")
        else:
            lines.append(f"line {i} with enough words for chunking and indexing")
    return "\n".join(lines) + "\n"


def _csv_text(rows: int) -> str:
    return "id,name\n" + "\n".join(f"{i},item{i}" for i in range(rows)) + "\n"


# ── prefix kinds: csv and txt ────────────────────────────────────────────


def test_big_csv_is_ok_with_prefix_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docindex_mod, "MAX_FILE_BYTES", 4096)
    root = tmp_path / "docs"
    root.mkdir()
    # Write a CSV > 4 KB
    (root / "big.csv").write_text(_csv_text(600), encoding="utf-8")
    index = DocIndex(root, converters={}, canonical_cache_dir=None)
    index.refresh()
    assert index.failures() == []
    records = list(index.records())
    assert len(records) == 1
    assert records[0].status == "ok"
    assert records[0].error is not None
    assert "MB" in records[0].error
    assert "truncated" in records[0].error


def test_big_txt_is_ok_with_prefix_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docindex_mod, "MAX_FILE_BYTES", 4096)
    root = tmp_path / "docs"
    root.mkdir()
    (root / "big.txt").write_text(_lines(300), encoding="utf-8")
    index = DocIndex(root, converters={}, canonical_cache_dir=None)
    index.refresh()
    assert index.failures() == []
    records = list(index.records())
    assert records[0].status == "ok"
    assert records[0].error is not None
    assert docindex_mod.PREFIX_TRUNCATED_NOTE in records[0].error


def test_prefix_contains_first_lines_not_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(docindex_mod, "MAX_FILE_BYTES", 2048)
    root = tmp_path / "docs"
    root.mkdir()
    text = _lines(500, marker="GOLDEN_MARKER", marker_at=10)
    (root / "data.txt").write_text(text, encoding="utf-8")
    index = DocIndex(root, converters={}, canonical_cache_dir=None)
    index.refresh()
    hits = index.search("GOLDEN_MARKER")
    assert any("GOLDEN_MARKER" in hit["text"] for hit in hits), "first lines should be searchable"

    # The very last line should NOT be indexed
    hits_end = index.search("line 499")
    assert not any("line 499" in hit["text"] for hit in hits_end), "last lines should be cut"


def test_no_split_line_at_cut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docindex_mod, "MAX_FILE_BYTES", 1500)
    root = tmp_path / "docs"
    root.mkdir()
    # Lines of ~80 chars each; the cut should land on a newline
    long_lines = (
        "\n".join(
            f"item_{i:04d} has a long description that makes this line over eighty characters"
            for i in range(100)
        )
        + "\n"
    )
    (root / "data.txt").write_text(long_lines, encoding="utf-8")
    index = DocIndex(root, converters={}, canonical_cache_dir=None)
    index.refresh()
    records = list(index.records())
    assert records[0].status == "ok"
    # All indexed chunk text should contain complete lines (each line starts with "item_")
    for chunk in index._entries.values():
        for c in chunk.chunks:
            for line in c.text.strip().split("\n"):
                stripped = line.strip()
                if stripped:
                    assert stripped.startswith("item_"), f"split line found: {stripped!r}"


# ── non-prefix kinds stay too_large ──────────────────────────────────────


def test_big_pdf_still_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docindex_mod, "MAX_FILE_BYTES", 4096)
    root = tmp_path / "docs"
    root.mkdir()
    # Fake PDF bytes larger than the limit
    (root / "big.pdf").write_bytes(b"%PDF\n" + b"x" * 8192)
    index = DocIndex(root, converters={}, canonical_cache_dir=None)
    index.refresh()
    records = list(index.records())
    assert records[0].status == "too_large"


def test_big_docx_still_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docindex_mod, "MAX_FILE_BYTES", 4096)
    root = tmp_path / "docs"
    root.mkdir()
    (root / "big.docx").write_bytes(b"PK" + b"x" * 8192)
    index = DocIndex(root, converters={}, canonical_cache_dir=None)
    index.refresh()
    records = list(index.records())
    assert records[0].status == "too_large"


# ── unchanged second refresh ────────────────────────────────────────────


def test_unchanged_second_refresh_reports_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(docindex_mod, "MAX_FILE_BYTES", 2048)
    root = tmp_path / "docs"
    root.mkdir()
    (root / "big.txt").write_text(_lines(300), encoding="utf-8")
    index = DocIndex(root, converters={}, canonical_cache_dir=None)
    report1 = index.refresh()
    assert report1.added == ["big.txt"]

    report2 = index.refresh()
    assert report2.unchanged == 1
    assert report2.added == []
    assert report2.changed == []


def test_real_file_size_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The FileRecord.size must reflect the real file size, not the prefix length."""
    monkeypatch.setattr(docindex_mod, "MAX_FILE_BYTES", 2048)
    root = tmp_path / "docs"
    root.mkdir()
    text = _lines(400)
    (root / "big.txt").write_text(text, encoding="utf-8")
    real_size = (root / "big.txt").stat().st_size
    assert real_size > 2048

    index = DocIndex(root, converters={}, canonical_cache_dir=None)
    index.refresh()
    records = list(index.records())
    assert records[0].size == real_size
