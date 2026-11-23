"""Tests for load_file extraction from DocIndex._refresh_locked."""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from voice_agent.rag import DocIndex
from voice_agent.rag.docindex import Bm25Retriever, FileRecord, _IndexedEntry, load_file


def _pptx_bytes() -> bytes:
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    sl = prs.slides.add_slide(prs.slide_layouts[1])
    sl.shapes.title.text = "T"
    sl.placeholders[1].text = "Body text here."
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def _xlsx_bytes() -> bytes:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.append(["A", "B"])
    ws.append(["1", "2"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


_N = "2026-01-01T00:00:00+00:00"
_C = {"txt": "legacy", "md": "legacy", "csv": "legacy", "pptx": "direct", "xlsx": "direct"}


def test_txt_matches_docindex(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("Hello world on 2026-01-15 with enough text.\n")
    lf = load_file(f, "a.txt", converters=_C, canonical_cache=None, prev=None, now_iso=_N)
    assert lf.status == "ok" and not lf.unchanged and lf.note is None
    idx = DocIndex(tmp_path)
    idx.refresh()
    for a, b in zip(lf.chunks, idx.chunks, strict=False):
        assert a.id == b.id and a.text == b.text and a.header == b.header


def test_pptx_matches_docindex(tmp_path: Path) -> None:
    f = tmp_path / "s.pptx"
    f.write_bytes(_pptx_bytes())
    lf = load_file(f, "s.pptx", converters=_C, canonical_cache=None, prev=None, now_iso=_N)
    assert lf.status == "ok"
    idx = DocIndex(tmp_path)
    idx.refresh()
    assert len(lf.chunks) == len([c for c in idx.chunks if c.source == "s.pptx"])


def test_error_cases(tmp_path: Path) -> None:
    f = tmp_path / "f.xyz"
    f.write_bytes(b"data")
    lf = load_file(f, "f.xyz", converters=_C, canonical_cache=None, prev=None, now_iso=_N)
    assert lf.status == "unsupported" and lf.error is not None and ".xyz" in lf.error

    f2 = tmp_path / "c.pptx"
    f2.write_bytes(b"\x00\x01random\xff\xfe")
    lf2 = load_file(f2, "c.pptx", converters=_C, canonical_cache=None, prev=None, now_iso=_N)
    assert lf2.status == "unreadable"


def test_oversize_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_agent.rag import docindex as m

    monkeypatch.setattr(m, "MAX_FILE_BYTES", 100)
    f = tmp_path / "b.xlsx"
    f.write_bytes(_xlsx_bytes())
    lf = load_file(f, "b.xlsx", converters=_C, canonical_cache=None, prev=None, now_iso=_N)
    assert lf.status == "too_large"

    monkeypatch.setattr(m, "MAX_FILE_BYTES", 50)
    f2 = tmp_path / "b.txt"
    f2.write_text("Line " * 200 + "\n")
    lf2 = load_file(f2, "b.txt", converters=_C, canonical_cache=None, prev=None, now_iso=_N)
    assert lf2.status == "ok" and lf2.note is not None and "truncated" in lf2.note.lower()


def test_prev_unchanged(tmp_path: Path) -> None:
    f = tmp_path / "st.txt"
    f.write_text("Stable content.\n")
    lf1 = load_file(f, "st.txt", converters=_C, canonical_cache=None, prev=None, now_iso=_N)
    assert lf1.status == "ok" and not lf1.unchanged
    st = f.stat()
    prev = _IndexedEntry(
        record=FileRecord(
            path=str(f),
            name="st.txt",
            mtime=lf1.mtime,
            ingested_at=_N,
            sha256=lf1.sha256,
            size=lf1.size,
            kind=lf1.kind,
            dates_in_text=lf1.dates,
            status="ok",
            error=None,
        ),
        size=lf1.size,
        mtime_ns=st.st_mtime_ns + 1,
        chunks=lf1.chunks,
        ctime_ns=st.st_ctime_ns,
        canonical_key=lf1.canonical_key,
    )
    time.sleep(0.05)
    lf2 = load_file(f, "st.txt", converters=_C, canonical_cache=None, prev=prev, now_iso=_N)
    assert lf2.status == "ok" and lf2.unchanged is True


def test_refresh_reports_a_file_that_cannot_be_stat_ed_like_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("hello world", encoding="utf-8")
    real_stat = Path.stat

    def fake_stat(self: Path, *a: Any, **k: Any) -> Any:
        if self.name == "bad.txt" and sys._getframe(1).f_code.co_name == "_refresh_locked":
            raise PermissionError("denied")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", fake_stat)
    index = DocIndex(tmp_path, retriever_factory=Bm25Retriever)
    report = index.refresh()
    assert ("bad.txt", "unreadable", "could not be read (PermissionError)") in report.failures
