"""Output-size bounds and line-start escaping for DOCX tables."""

from __future__ import annotations

import random
import string
import time

import pytest
from test_docx_tables import _tc, _tr
from test_loaders_binary import _build_docx_bytes, _docx_p, _wrap_document_xml

from voice_agent.rag import docindex
from voice_agent.rag.docindex import load_docx


def _rnd(n: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    return "".join(rng.choices(string.ascii_letters, k=n))


def _docx(body: str) -> bytes:
    return _build_docx_bytes(_wrap_document_xml(body))


def _header_amp_docx(hlen: int, rows: int) -> bytes:
    hdr = _tr(_tc(_rnd(hlen), grid_span=64), tbl_header=True)
    data = _tr(_tc("x") * 64) * rows
    return _docx(f"<w:tbl>{hdr}{data}</w:tbl>")


def test_header_amplification_is_bounded_and_fast() -> None:
    hlen = 10_000
    data = _header_amp_docx(hlen, 400)
    start = time.monotonic()
    text, status, _err = load_docx(data)
    assert time.monotonic() - start < 5
    assert len(text) <= docindex.DOCX_OUTPUT_BUDGET_CHARS
    if status == "ok":
        assert _rnd(hlen)[: docindex.MAX_DOCX_HEADER_CHARS + 1] not in text


def test_header_amplification_rejected_with_small_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docindex, "DOCX_OUTPUT_BUDGET_CHARS", 5_000)
    text, status, err = load_docx(_header_amp_docx(10_000, 400))
    assert (text, status, err) == ("", "unreadable", "docx text exceeds the size limit")


def test_budget_also_counts_plain_paragraphs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docindex, "DOCX_OUTPUT_BUDGET_CHARS", 100)
    _text, status, err = load_docx(_docx(_docx_p("a" * 500)))
    assert status == "unreadable"
    assert err == "docx text exceeds the size limit"


def test_vmerge_continuation_values_cut_to_header_limit() -> None:
    big = _rnd(200_000)
    restart = _tr(_tc(big, vmerge="restart") + _tc("h2"))
    cont = _tr(_tc("", vmerge="continue") + _tc("v")) * 20
    hdr = _tr(_tc("A") + _tc("B"), tbl_header=True)
    text, status, _ = load_docx(_docx(f"<w:tbl>{hdr}{restart}{cont}</w:tbl>"))
    assert status == "ok"
    lines = text.split("\n")
    assert len(lines) == 21
    assert lines[0].startswith("A: " + big[: docindex.MAX_DOCX_CELL_CHARS] + ";")
    for line in lines[1:]:
        assert line.startswith("A: " + big[: docindex.MAX_DOCX_HEADER_CHARS] + "; B: v")


def test_large_cell_is_cut() -> None:
    hdr = _tr(_tc("A") + _tc("B"), tbl_header=True)
    row = _tr(_tc(_rnd(10_000, 1)) + _tc("ok"))
    text, status, _ = load_docx(_docx(f"<w:tbl>{hdr}{row}</w:tbl>"))
    assert status == "ok"
    cell = _rnd(10_000, 1)
    assert cell[: docindex.MAX_DOCX_CELL_CHARS] in text
    assert cell[: docindex.MAX_DOCX_CELL_CHARS + 1] not in text


def test_total_row_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docindex, "_DOCX_MAX_TOTAL_TABLE_ROWS", 30)
    hdr = _tr(_tc("A") + _tc("B"), tbl_header=True)
    tbl = "<w:tbl>" + hdr + _tr(_tc("1") + _tc("2")) * 20 + "</w:tbl>"
    text, status, _ = load_docx(_docx(tbl * 3))
    assert status == "ok"
    assert text.count("A: 1; B: 2") == 30


def test_normal_table_unchanged() -> None:
    hdr = _tr(_tc("Date") + _tc("Qty"), tbl_header=True)
    text, status, _ = load_docx(_docx(f"<w:tbl>{hdr}{_tr(_tc('#N/A') + _tc('5'))}</w:tbl>"))
    assert status == "ok"
    assert text == "Date: #N/A; Qty: 5"


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("# H", "\\# H: 1"),
        ("### H", "\\### H: 1"),
        ("```x", "\\```x: 1"),
        ("~~~", "\\~~~: 1"),
        ("<!-- rag:page=9", "\\<!-- rag:page=9: 1"),
        ("<!-- Slide number: 2", "\\<!-- Slide number: 2: 1"),
        ("Date", "Date: 1"),
        ("A-", "A-: 1"),
        ("#N/A", "#N/A: 1"),
        ("``x", "``x: 1"),
    ],
)
def test_table_row_line_start_escape(header: str, expected: str) -> None:
    hdr = _tr(_tc(header.replace("<", "&lt;")) + _tc("Z"), tbl_header=True)
    text, status, _ = load_docx(_docx(f"<w:tbl>{hdr}{_tr(_tc('1') + _tc(''))}</w:tbl>"))
    assert status == "ok"
    assert text == expected


def test_footnote_line_escape_helper() -> None:
    assert docindex._escape_docx_line_start("# x") == "\\# x"
    assert docindex._escape_docx_line_start("[fn 1] # x") == "[fn 1] # x"
