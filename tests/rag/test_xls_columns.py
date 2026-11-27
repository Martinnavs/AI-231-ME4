"""Ticket 253: read_xls keeps column positions, honours datemode, single-line cells.

xlwt is not installed in the root venv, so instead of generating real .xls
bytes we install a fake ``xlrd`` module in ``sys.modules`` whose
``open_workbook`` returns synthetic workbooks/sheets/cells. Ctype constants
and date math come from the real installed xlrd, so the reader is exercised
with the genuine 1900/1904 conversions. All fixtures are synthetic.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from voice_agent.rag import readers


class _FakeCell:
    def __init__(self, ctype: int, value: Any) -> None:
        self.ctype = ctype
        self.value = value


class _FakeSheet:
    def __init__(self, name: str, rows: list[list[_FakeCell]]) -> None:
        self.name = name
        self.nrows = len(rows)
        self.ncols = max((len(row) for row in rows), default=0)
        self._rows = [list(row) + [_FakeCell(0, None)] * (self.ncols - len(row)) for row in rows]

    def cell(self, row: int, col: int) -> _FakeCell:
        return self._rows[row][col]


class _FakeWorkbook:
    def __init__(self, sheets: list[_FakeSheet], datemode: int = 0) -> None:
        self._sheets = sheets
        self.nsheets = len(sheets)
        self.datemode = datemode

    def sheet_by_index(self, idx: int) -> _FakeSheet:
        return self._sheets[idx]


class _XlsFixture:
    """Builds fake cells and reads a fake workbook through ``readers.read_xls``."""

    def __init__(self, xlrd: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        self._xlrd = xlrd
        self._monkeypatch = monkeypatch

    def text(self, value: str) -> _FakeCell:
        return _FakeCell(self._xlrd.XL_CELL_TEXT, value)

    def num(self, value: float) -> _FakeCell:
        return _FakeCell(self._xlrd.XL_CELL_NUMBER, value)

    def boolean(self, value: bool) -> _FakeCell:
        return _FakeCell(self._xlrd.XL_CELL_BOOLEAN, value)

    def date(self, value: float) -> _FakeCell:
        return _FakeCell(self._xlrd.XL_CELL_DATE, value)

    def empty(self) -> _FakeCell:
        return _FakeCell(self._xlrd.XL_CELL_EMPTY, None)

    def read(self, wb: _FakeWorkbook) -> Any:
        fake: Any = types.ModuleType("xlrd")
        fake.XL_CELL_EMPTY = self._xlrd.XL_CELL_EMPTY
        fake.XL_CELL_TEXT = self._xlrd.XL_CELL_TEXT
        fake.XL_CELL_NUMBER = self._xlrd.XL_CELL_NUMBER
        fake.XL_CELL_BOOLEAN = self._xlrd.XL_CELL_BOOLEAN
        fake.XL_CELL_DATE = self._xlrd.XL_CELL_DATE
        fake.xldate_as_datetime = self._xlrd.xldate_as_datetime
        fake.open_workbook = lambda file_contents=None, on_demand=False: wb
        self._monkeypatch.setitem(sys.modules, "xlrd", fake)
        self._monkeypatch.delenv("VOICE_AGENT_RAG_TABLE_HEADERS", raising=False)
        return readers.read_xls(b"")


@pytest.fixture
def xls(monkeypatch: pytest.MonkeyPatch) -> _XlsFixture:
    xlrd = pytest.importorskip("xlrd")
    return _XlsFixture(xlrd, monkeypatch)


# --- Column positions ------------------------------------------------------


def test_blank_middle_cell_keeps_column_position(xls: _XlsFixture) -> None:
    """A blank middle cell must not shift the values right of it."""
    wb = _FakeWorkbook(
        [
            _FakeSheet(
                "Data",
                [
                    [xls.text("name"), xls.text("qty"), xls.text("total")],
                    [xls.text("widget"), xls.empty(), xls.num(50.0)],
                ],
            )
        ]
    )
    res = xls.read(wb)
    assert res.status == "ok"
    assert res.fallback is None
    assert res.text == "## Data\n\n| name | qty | total |\n| widget |  | 50 |"


def test_trailing_blanks_trimmed_to_last_non_blank(xls: _XlsFixture) -> None:
    """Rows render one cell per column up to the last non-blank column."""
    wb = _FakeWorkbook(
        [
            _FakeSheet(
                "Data",
                [
                    [xls.text("a"), xls.empty(), xls.empty()],
                    [xls.text("x"), xls.empty(), xls.text("y")],
                ],
            )
        ]
    )
    res = xls.read(wb)
    assert res.status == "ok"
    assert res.text == "## Data\n\n| a |\n| x |  | y |"


def test_all_blank_rows_are_skipped(xls: _XlsFixture) -> None:
    wb = _FakeWorkbook(
        [
            _FakeSheet(
                "Data",
                [
                    [xls.text("h1"), xls.text("h2")],
                    [xls.empty(), xls.empty()],
                    [xls.empty()],
                    [xls.text("a"), xls.text("b")],
                ],
            )
        ]
    )
    res = xls.read(wb)
    assert res.status == "ok"
    assert res.text == "## Data\n\n| h1 | h2 |\n| a | b |"


def test_wholly_blank_sheet_is_empty(xls: _XlsFixture) -> None:
    wb = _FakeWorkbook(
        [_FakeSheet("Blank", [[xls.empty(), xls.empty()], [xls.empty(), xls.empty()]])]
    )
    res = xls.read(wb)
    assert res.status == "empty"


# --- Datemode ---------------------------------------------------------------


def test_date_honours_datemode_1(xls: _XlsFixture) -> None:
    """1904-based workbooks must render 1904-epoch dates (serial 1.0)."""
    wb = _FakeWorkbook([_FakeSheet("D", [[xls.date(1.0)]])], datemode=1)
    res = xls.read(wb)
    assert res.status == "ok"
    assert res.text == "## D\n\n| 1904-01-02T00:00:00 |"


def test_date_datemode_0_unchanged(xls: _XlsFixture) -> None:
    """1900-based workbooks keep the classic rendering for the same serial."""
    wb = _FakeWorkbook([_FakeSheet("D", [[xls.date(1.0)]])])
    res = xls.read(wb)
    assert res.status == "ok"
    assert res.text == "## D\n\n| 1900-01-01T00:00:00 |"


# --- Single-line cells -------------------------------------------------------


def test_newline_cell_stays_on_one_row(xls: _XlsFixture) -> None:
    """\\r and \\n inside a cell collapse to single spaces; the row is not split."""
    wb = _FakeWorkbook(
        [
            _FakeSheet(
                "Data",
                [
                    [xls.text("a\nb\r\nc"), xls.text("d\re")],
                    [xls.text("plain"), xls.text("q")],
                ],
            )
        ]
    )
    res = xls.read(wb)
    assert res.status == "ok"
    assert res.text == "## Data\n\n| a b c | d e |\n| plain | q |"


# --- Byte-identical dense output ---------------------------------------------


def test_dense_sheet_output_is_byte_identical(xls: _XlsFixture) -> None:
    """A sheet with no blank cells and no newlines renders exactly as before.

    The expected string below is precisely what the pre-253 reader produced
    for this sheet (every cell included, in order, pipes escaped), so equality
    proves the fix changes nothing for dense data.
    """
    wb = _FakeWorkbook(
        [
            _FakeSheet(
                "Data",
                [
                    [
                        xls.text("name"),
                        xls.text("count"),
                        xls.text("ratio"),
                        xls.text("flag"),
                        xls.text("date"),
                    ],
                    [
                        xls.text("alpha"),
                        xls.num(3.0),
                        xls.num(0.5),
                        xls.boolean(True),
                        xls.date(100.0),
                    ],
                    [
                        xls.text("beta|pipe"),
                        xls.num(7.0),
                        xls.num(2.0),
                        xls.boolean(False),
                        xls.date(101.0),
                    ],
                ],
            )
        ]
    )
    res = xls.read(wb)
    assert res.status == "ok"
    assert res.fallback is None
    assert res.text == (
        "## Data\n\n"
        "| name | count | ratio | flag | date |\n"
        "| alpha | 3 | 0.5 | TRUE | 1900-04-09T00:00:00 |\n"
        "| beta\\|pipe | 7 | 2 | FALSE | 1900-04-10T00:00:00 |"
    )
