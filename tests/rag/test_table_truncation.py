"""A table cut at the row cap is recorded on its FileRecord (WP5)."""

from __future__ import annotations

import os
from pathlib import Path

from voice_agent.rag import canonical
from voice_agent.rag.docindex import DocIndex


def _notes(index: DocIndex) -> dict[str, str | None]:
    return {record.name: record.error for record in index.records()}


def test_csv_over_the_cap_carries_the_note_and_keeps_it_when_touched(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    rows = "".join(f"item{i},{i}\n" for i in range(canonical.MAX_CSV_ROWS + 1000))
    (root / "big.csv").write_text("name,qty\n" + rows, encoding="utf-8")
    (root / "small.csv").write_text("name,qty\nbolt,4\n", encoding="utf-8")
    index = DocIndex(root, converters={}, canonical_cache_dir=None)
    index.refresh()
    assert _notes(index) == {"big.csv": canonical.TABLE_TRUNCATED_NOTE, "small.csv": None}
    assert index.failures() == []
    stat = (root / "big.csv").stat()
    os.utime(root / "big.csv", (stat.st_atime + 10, stat.st_mtime + 10))
    index.refresh()
    assert _notes(index)["big.csv"] == canonical.TABLE_TRUNCATED_NOTE
