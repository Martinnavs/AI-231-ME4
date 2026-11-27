"""Ticket 251: a mixed PDF keeps its native text pages.

A PDF where only *some* pages have a text layer must not be classified as
scanned: every page with at least the existing 20-character threshold is
returned, status is ``scanned`` only when NO page has text, and the note
lists how many pages had no text plus their page numbers (capped at 20) so a
later OCR phase can pick them up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_loaders_binary import _build_pdf_bytes

from voice_agent.rag import DocIndex, docindex

pytest.importorskip("pypdfium2")

ENV = "VOICE_AGENT_RAG_PDF_ENGINE"
ENGINES = ["pypdf", "pdfium"]

LONG_TEXT = (
    "This page carries plenty of native text to clear the "
    "twenty character threshold used for sparse page detection."
)
BLANK = ""


def _load(monkeypatch: pytest.MonkeyPatch, name: str, data: bytes) -> Any:
    monkeypatch.setenv(ENV, name)
    return docindex.load_pdf_pages(data)


@pytest.mark.parametrize("name", ENGINES)
def test_three_text_three_blank_keeps_text_pages(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    got = _load(monkeypatch, name, _build_pdf_bytes([LONG_TEXT, BLANK] * 3))
    assert [n for n, _ in got[0]] == [1, 3, 5]
    assert got[1] == "ok"
    assert got[2] == "ok (3 pages without text: 2, 4, 6)"


def test_three_text_three_blank_indexes_three_pages(tmp_path: Path) -> None:
    (tmp_path / "mixed.pdf").write_bytes(_build_pdf_bytes([LONG_TEXT, BLANK] * 3))
    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == ["mixed.pdf"]
    rec = idx.records()[0]
    assert rec.status == "ok"
    assert len(idx.chunks) == 3


@pytest.mark.parametrize("name", ENGINES)
def test_majority_sparse_pdf_keeps_text_pages(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    # 6 of 10 pages are sparse: the old "> half" rule discarded the 4 text
    # pages and returned scanned; they must survive now.
    pages = [BLANK, LONG_TEXT, BLANK, LONG_TEXT, BLANK, BLANK, LONG_TEXT, BLANK, BLANK, LONG_TEXT]
    got = _load(monkeypatch, name, _build_pdf_bytes(pages))
    assert [n for n, _ in got[0]] == [2, 4, 7, 10]
    assert got[1] == "ok"
    assert got[2] == "ok (6 pages without text: 1, 3, 5, 6, 8, 9)"


@pytest.mark.parametrize("name", ENGINES)
def test_all_blank_pdf_is_scanned_with_no_pages(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    got = _load(monkeypatch, name, _build_pdf_bytes([BLANK, BLANK, BLANK]))
    assert got == ([], "scanned", "scanned: 3 pages without text")


def test_all_blank_pdf_indexes_no_pages(tmp_path: Path) -> None:
    (tmp_path / "blank.pdf").write_bytes(_build_pdf_bytes([BLANK, BLANK]))
    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == ["blank.pdf"]
    rec = idx.records()[0]
    assert rec.status == "ok"
    assert len(idx.chunks) == 0
    assert "scanned" in (rec.error or "").lower()


@pytest.mark.parametrize("name", ENGINES)
def test_all_text_pdf_unchanged(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    got = _load(monkeypatch, name, _build_pdf_bytes([LONG_TEXT, LONG_TEXT]))
    assert [n for n, _ in got[0]] == [1, 2]
    assert got[1] == "ok"
    assert got[2] is None


@pytest.mark.parametrize("name", ENGINES)
def test_note_caps_listed_page_numbers_at_20(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    # 21 sparse pages + 1 text page: the count is 21 but only the first 20
    # page numbers are listed in the note.
    got = _load(monkeypatch, name, _build_pdf_bytes([BLANK] * 21 + [LONG_TEXT]))
    assert [n for n, _ in got[0]] == [22]
    assert got[1] == "ok"
    assert got[2] is not None
    assert got[2].startswith("ok (21 pages without text: ")
    listed = got[2].split(": ", 1)[1].rsplit(")", 1)[0]
    assert listed == ", ".join(str(n) for n in range(1, 21))
