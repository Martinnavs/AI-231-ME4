"""Big CSV tables: 30k rows index fully; 60k rows are truncated with notice."""

from __future__ import annotations

import time
from pathlib import Path

from voice_agent.rag.canonical import MAX_CSV_ROWS
from voice_agent.rag.docindex import DocIndex


def _csv_text(rows: int, *, marker_at: int | None = None) -> str:
    lines = ["id,name,val"]
    for i in range(rows):
        if marker_at is not None and i == marker_at - 1:
            lines.append(f"{i},zzq{marker_at},{i}")
        else:
            lines.append(f"{i},item{i},{i}")
    return "\n".join(lines) + "\n"


def test_30k_csv_indexes_without_truncation(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "data.csv").write_text(_csv_text(30_000, marker_at=25_000), encoding="utf-8")
    index = DocIndex(root, converters={}, canonical_cache_dir=None)
    index.refresh()
    assert index.failures() == []
    records = list(index.records())
    assert len(records) == 1
    assert records[0].error is None
    hits = index.search("zzq25000 name")
    assert any("zzq25000" in hit["text"] for hit in hits)


def test_60k_csv_is_flagged_truncated(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "big.csv").write_text(_csv_text(60_000), encoding="utf-8")
    index = DocIndex(root, converters={}, canonical_cache_dir=None)
    index.refresh()
    assert index.failures() == []
    records = list(index.records())
    assert len(records) == 1
    assert records[0].error is not None
    assert str(MAX_CSV_ROWS) in records[0].error


def test_60k_csv_indexes_under_time_budget(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "big.csv").write_text(_csv_text(60_000), encoding="utf-8")
    t0 = time.perf_counter()
    index = DocIndex(root, converters={}, canonical_cache_dir=None)
    index.refresh()
    elapsed = time.perf_counter() - t0
    assert elapsed < 30
