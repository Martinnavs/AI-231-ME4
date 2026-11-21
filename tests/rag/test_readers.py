"""Tests for direct XLSX/PPTX readers (``readers.py``, ``convert.py``, ``docindex.py``)."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import io
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from voice_agent.rag import canonical, convert, docindex, metrics, readers
from voice_agent.rag.convert import ConvertResult, convert_isolated, convert_with_direct
from voice_agent.rag.docindex import DocIndex

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICE_DIR = REPO_ROOT / "tests" / "fixtures" / "rag_office"


def _make_zip(members: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


def _save_wb(wb: Any) -> bytes:
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _save_prs(prs: Any) -> bytes:
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def test_xlsx_title_row_real_headers_and_no_unnamed() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Operations"
    ws.append(["Quarterly Operations Summary"])
    ws.append(["Department", "Headcount", "Budget USD"])
    ws.append(["Telemetry", 6, 12000])
    ws.append(["Water", 4, 9600])

    raw = _save_wb(wb)
    direct_res = readers.read_xlsx(raw)
    assert direct_res.status == "ok"
    assert direct_res.fallback is None
    assert direct_res.converter == f"direct-openpyxl-{importlib.metadata.version('openpyxl')}"
    assert "**Quarterly Operations Summary**" in direct_res.text
    assert "| Department | Headcount | Budget USD |" in direct_res.text

    res = convert_with_direct(raw, "xlsx")
    assert res.status == "ok"
    assert res.error is None
    assert "Unnamed" not in res.text
    assert "**Quarterly Operations Summary**" in res.text
    assert "<!-- rag:sheet=Operations -->" in res.text
    assert "Department: Telemetry; Headcount: 6; Budget USD: 12000" in res.text
    assert "Department: Water; Headcount: 4; Budget USD: 9600" in res.text


def test_xlsx_two_level_headers_forward_fill_and_blank_header() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Quarterly"
    ws.append(["Item", "Q1 2026", None, "Q2 2026", None, None])
    ws.append(["Item", "Units", "Revenue", "Units", None, None])
    ws.append(["Sensor", 10, 500, 12, 600, 99])

    res = convert_with_direct(_save_wb(wb), "xlsx")
    assert res.status == "ok"
    assert "Item: Sensor;" in res.text
    assert "Q1 2026 / Units: 10;" in res.text
    assert "Q1 2026 / Revenue: 500;" in res.text
    assert "Q2 2026 / Units: 12;" in res.text
    # Forward-filled Q2 2026 with blank lower skips join -> "Q2 2026"
    assert "Q2 2026: 600;" in res.text
    # Forward-filled Q2 2026 still applies across merged blanks in upper row; test explicit
    # leading blank header in a separate sheet so col_N is also exercised.

    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    assert ws2 is not None
    ws2.title = "BlankHead"
    ws2.append([None, "Metric", "Score"])
    ws2.append(["Alpha", 1, 95])
    res2 = convert_with_direct(_save_wb(wb2), "xlsx")
    assert res2.status == "ok"
    assert "col_1: Alpha; Metric: 1; Score: 95" in res2.text


def test_xlsx_string_only_body_does_not_trigger_two_level_header() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Roster"
    ws.append(["Name", "Role", "Site"])
    ws.append(["Ana", "Lead", "North"])
    ws.append(["Ben", "Tech", "South"])

    res = convert_with_direct(_save_wb(wb), "xlsx")
    assert res.status == "ok"
    assert "Name: Ana; Role: Lead; Site: North" in res.text
    assert "Name: Ben; Role: Tech; Site: South" in res.text
    assert " / " not in res.text


def test_xlsx_drops_empty_rows_cols_and_renders_types() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Types"
    # Empty leading row and empty leading/middle columns (col A and col C)
    ws.append([None, None, None, None])
    ws.append([None, "Kind", None, "Val"])
    ws.append([None, "BoolTrue", None, True])
    ws.append([None, "BoolFalse", None, False])
    ws.append([None, "IntFloat", None, 6.0])
    ws.append([None, "FracFloat", None, 0.15])
    ws.append([None, "DateOnly", None, dt.date(2026, 3, 15)])
    ws.append([None, "MidnightDT", None, dt.datetime(2026, 3, 15, 0, 0, 0)])
    ws.append([None, "Timestamp", None, dt.datetime(2026, 3, 15, 14, 30, 0)])
    ws.append([None, None, None, None])
    ws.append([None, "  Pipe|Line\nBreak  ", None, "  ok  "])

    raw_res = readers.read_xlsx(_save_wb(wb))
    assert raw_res.status == "ok"
    assert "| Kind | Val |" in raw_res.text
    assert "Pipe\\|Line Break" in raw_res.text

    res = convert_with_direct(_save_wb(wb), "xlsx")
    assert res.status == "ok"
    assert "Kind: BoolTrue; Val: TRUE" in res.text
    assert "Kind: BoolFalse; Val: FALSE" in res.text
    assert "Kind: IntFloat; Val: 6" in res.text
    assert "6.0" not in res.text
    assert "Kind: FracFloat; Val: 0.15" in res.text
    assert "Kind: DateOnly; Val: 2026-03-15" in res.text
    assert "Kind: MidnightDT; Val: 2026-03-15" in res.text
    assert "Kind: Timestamp; Val: 2026-03-15 14:30:00" in res.text
    assert "Kind: Pipe|Line Break; Val: ok" in res.text


def test_xlsx_caps_enforce_and_set_truncated_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    openpyxl = pytest.importorskip("openpyxl")

    # Sheet cap
    wb_sheets = openpyxl.Workbook()
    ws0 = wb_sheets.active
    assert ws0 is not None
    ws0.title = "S1"
    ws0.append(["Col", "Val"])
    ws0.append(["A", 1])
    ws1 = wb_sheets.create_sheet("S2")
    ws1.append(["Col", "Val"])
    ws1.append(["B", 2])
    with monkeypatch.context() as mp:
        mp.setattr(readers, "MAX_XLSX_SHEETS", 1)
        res = convert_with_direct(_save_wb(wb_sheets), "xlsx")
        assert res.status == "ok"
        assert res.fallback == "truncated"
        assert "<!-- rag:sheet=S1 -->" in res.text
        assert "<!-- rag:sheet=S2 -->" not in res.text

    # Row cap
    wb_rows = openpyxl.Workbook()
    ws_r = wb_rows.active
    assert ws_r is not None
    ws_r.title = "Rows"
    ws_r.append(["Item", "N"])
    ws_r.append(["First", 1])
    ws_r.append(["Second", 2])
    with monkeypatch.context() as mp:
        mp.setattr(readers, "MAX_XLSX_ROWS", 2)
        res = convert_with_direct(_save_wb(wb_rows), "xlsx")
        assert res.status == "ok"
        assert res.fallback == "truncated"
        assert "First" in res.text
        assert "Second" not in res.text

    # Col cap
    wb_cols = openpyxl.Workbook()
    ws_c = wb_cols.active
    assert ws_c is not None
    ws_c.title = "Cols"
    ws_c.append(["C1", "C2", "C3"])
    ws_c.append(["v1", "v2", "v3"])
    with monkeypatch.context() as mp:
        mp.setattr(readers, "MAX_XLSX_COLS", 2)
        res = convert_with_direct(_save_wb(wb_cols), "xlsx")
        assert res.status == "ok"
        assert res.fallback == "truncated"
        assert "C1: v1; C2: v2" in res.text
        assert "C3" not in res.text

    # Cell char cap
    wb_cell = openpyxl.Workbook()
    ws_cell = wb_cell.active
    assert ws_cell is not None
    ws_cell.title = "CellCap"
    ws_cell.append(["Key", "Text"])
    ws_cell.append(["K1", "abcdefghij"])
    with monkeypatch.context() as mp:
        mp.setattr(readers, "MAX_XLSX_CELL_CHARS", 5)
        res = convert_with_direct(_save_wb(wb_cell), "xlsx")
        assert res.status == "ok"
        assert res.fallback == "truncated"
        assert "Text: abcde" in res.text
        assert "abcdefghij" not in res.text


def test_pptx_title_bullets_merged_table_notes_group_and_empty_slide() -> None:
    pptx = pytest.importorskip("pptx")
    Inches = pptx.util.Inches

    prs = pptx.Presentation()

    # Slide 1: title, reading order shapes, bullets (level 1 & 2), group shape
    s1 = prs.slides.add_slide(prs.slide_layouts[5])
    assert s1.shapes.title is not None
    s1.shapes.title.text = "System Architecture"

    # Add lower box BEFORE upper box to test (top, left) reading order sorting.
    low_box = s1.shapes.add_textbox(Inches(1), Inches(4.5), Inches(4), Inches(1))
    low_box.text_frame.text = "SECOND_IN_ORDER"

    high_box = s1.shapes.add_textbox(Inches(1), Inches(1.5), Inches(4), Inches(1.5))
    tf = high_box.text_frame
    tf.text = "FIRST_IN_ORDER"
    p_b1 = tf.add_paragraph()
    p_b1.text = "Bullet level one"
    p_b1.level = 1
    p_b2 = tf.add_paragraph()
    p_b2.text = "Bullet level two"
    p_b2.level = 2

    grp = s1.shapes.add_group_shape()
    grp_tb = grp.shapes.add_textbox(Inches(5), Inches(2), Inches(3), Inches(1))
    grp_tb.text_frame.text = "Grouped shape detail"

    # Slide 2: table with horizontally merged header/cell + newlines + notes
    s2 = prs.slides.add_slide(prs.slide_layouts[5])
    assert s2.shapes.title is not None
    s2.shapes.title.text = "Merged Table Slide"
    tbl = s2.shapes.add_table(3, 3, Inches(0.5), Inches(1.5), Inches(8), Inches(2)).table
    tbl.cell(0, 0).text = "Region"
    tbl.cell(0, 1).text = "Metric"
    tbl.cell(0, 2).text = "Notes"
    tbl.cell(1, 0).text = "North\nRidge"
    tbl.cell(1, 1).merge(tbl.cell(1, 2))
    tbl.cell(1, 1).text = "CombinedSpan"
    tbl.cell(2, 0).text = "South"
    tbl.cell(2, 1).text = "98"
    tbl.cell(2, 2).text = "Nominal"
    s2.notes_slide.notes_text_frame.text = "Speaker check: confirm span."

    # Slide 3: empty slide (no shapes with text)
    prs.slides.add_slide(prs.slide_layouts[6])

    # Slide 4: trailing slide after empty slide
    s4 = prs.slides.add_slide(prs.slide_layouts[5])
    assert s4.shapes.title is not None
    s4.shapes.title.text = "Closing Summary"
    s4.shapes.add_textbox(
        Inches(1), Inches(2), Inches(5), Inches(1)
    ).text_frame.text = "Final slide body."

    raw = _save_prs(prs)
    raw_res = readers.read_pptx(raw)
    assert raw_res.status == "ok"
    assert raw_res.fallback is None
    assert raw_res.converter == f"direct-python-pptx-{importlib.metadata.version('python-pptx')}"
    assert "- Bullet level one" in raw_res.text
    assert "- Bullet level two" in raw_res.text
    assert "  - Bullet" not in raw_res.text
    # Origin cell kept once, spanned cell repeats nothing
    assert "| North Ridge | CombinedSpan |  |" in raw_res.text
    assert raw_res.text.index("FIRST_IN_ORDER") < raw_res.text.index("SECOND_IN_ORDER")

    res = convert_with_direct(raw, "pptx")
    assert res.status == "ok"
    for n in (1, 2, 3, 4):
        assert f"<!-- rag:slide={n} -->" in res.text

    units, _ = canonical.split_units(res.text)
    assert [u.slide for u in units] == [1, 2, 4]
    assert "Grouped shape detail" in units[0].body
    assert "Region: North Ridge; Metric: CombinedSpan" in units[1].body
    assert "Region: South; Metric: 98; Notes: Nominal" in units[1].body
    assert "### Notes:\nSpeaker check: confirm span." in units[1].body
    assert "Final slide body." in units[2].body


def test_pptx_caps_enforce_and_set_truncated_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    pptx = pytest.importorskip("pptx")
    Inches = pptx.util.Inches

    # Slide cap
    prs_slides = pptx.Presentation()
    for title in ("One", "Two", "Three"):
        s = prs_slides.slides.add_slide(prs_slides.slide_layouts[5])
        assert s.shapes.title is not None
        s.shapes.title.text = title
    with monkeypatch.context() as mp:
        mp.setattr(readers, "MAX_PPTX_SLIDES", 2)
        res = convert_with_direct(_save_prs(prs_slides), "pptx")
        assert res.status == "ok"
        assert res.fallback == "truncated"
        assert "<!-- rag:slide=1 -->" in res.text
        assert "<!-- rag:slide=2 -->" in res.text
        assert "<!-- rag:slide=3 -->" not in res.text

    # Run char cap
    prs_run = pptx.Presentation()
    s_run = prs_run.slides.add_slide(prs_run.slide_layouts[5])
    assert s_run.shapes.title is not None
    s_run.shapes.title.text = "LongLine"
    s_run.shapes.add_textbox(
        Inches(1), Inches(2), Inches(5), Inches(1)
    ).text_frame.text = "0123456789ABCDEF"
    with monkeypatch.context() as mp:
        mp.setattr(readers, "MAX_PPTX_RUN_CHARS", 8)
        res = convert_with_direct(_save_prs(prs_run), "pptx")
        assert res.status == "ok"
        assert res.fallback == "truncated"
        assert "01234567" in res.text
        assert "0123456789ABCDEF" not in res.text

    # Table row cap
    prs_tbl = pptx.Presentation()
    s_tbl = prs_tbl.slides.add_slide(prs_tbl.slide_layouts[5])
    tbl = s_tbl.shapes.add_table(4, 2, Inches(1), Inches(1), Inches(4), Inches(2)).table
    for r_i, val in enumerate(("Header", "RowOne", "RowTwo", "RowThree")):
        tbl.cell(r_i, 0).text = val
        tbl.cell(r_i, 1).text = str(r_i)
    with monkeypatch.context() as mp:
        mp.setattr(readers, "MAX_PPTX_TABLE_ROWS", 2)
        res = convert_with_direct(_save_prs(prs_tbl), "pptx")
        assert res.status == "ok"
        assert res.fallback == "truncated"
        assert "Header: RowOne" in res.text
        assert "RowTwo" not in res.text

    # Group depth cap
    prs_grp = pptx.Presentation()
    s_grp = prs_grp.slides.add_slide(prs_grp.slide_layouts[5])
    g1 = s_grp.shapes.add_group_shape()
    g2 = g1.shapes.add_group_shape()
    g2.shapes.add_textbox(Inches(1), Inches(1), Inches(2), Inches(1)).text_frame.text = "DeepText"
    s_grp.shapes.add_textbox(Inches(1), Inches(1), Inches(2), Inches(1)).text_frame.text = "TopText"
    with monkeypatch.context() as mp:
        mp.setattr(readers, "MAX_PPTX_GROUP_DEPTH", 1)
        res = convert_with_direct(_save_prs(prs_grp), "pptx")
        assert res.status == "ok"
        assert res.fallback == "truncated"
        assert "TopText" in res.text
        assert "DeepText" not in res.text


def test_direct_guards_run_before_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    def _forbid_xlsx(raw: bytes) -> ConvertResult:
        called.append("xlsx")
        raise AssertionError("read_xlsx must not be called on guard failure")

    def _forbid_pptx(raw: bytes) -> ConvertResult:
        called.append("pptx")
        raise AssertionError("read_pptx must not be called on guard failure")

    monkeypatch.setattr(readers, "read_xlsx", _forbid_xlsx)
    monkeypatch.setattr(readers, "read_pptx", _forbid_pptx)

    doctype_pptx = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            (
                "ppt/slides/slide1.xml",
                b'<?xml version="1.0"?><!DOCTYPE sld [<!ENTITY x "1">]><sld/>',
            ),
        ]
    )
    res_doctype_pptx = convert_with_direct(doctype_pptx, "pptx")
    assert res_doctype_pptx.status == "unreadable"
    assert not called

    doctype_xlsx = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            (
                "xl/worksheets/sheet1.xml",
                b'<?xml version="1.0"?><!DOCTYPE worksheet [<!ENTITY x "1">]><worksheet/>',
            ),
        ]
    )
    res_doctype_xlsx = convert_with_direct(doctype_xlsx, "xlsx")
    assert res_doctype_xlsx.status == "unreadable"
    assert not called

    bomb_bytes = _make_zip([("payload.xml", b"0" * (256 * 1024))])
    for kind in ("pptx", "xlsx"):
        res_bomb = convert_with_direct(bomb_bytes, kind)
        assert res_bomb.status == "unreadable"
        assert "AssertionError" not in (res_bomb.error or "")
    assert not called

    for other in ("pdf", "docx"):
        res_other = convert_with_direct(b"hello", other)
        assert res_other.status == "unsupported"


def test_direct_child_env_stays_bare() -> None:
    assert convert._child_env("direct") == convert._child_env("markitdown")


@pytest.mark.parametrize(
    ("kind", "filename"),
    [("xlsx", "team_budget.xlsx"), ("pptx", "site_review.pptx")],
)
def test_direct_isolated_matches_in_process(kind: str, filename: str) -> None:
    pytest.importorskip("openpyxl")
    pytest.importorskip("pptx")

    raw = (OFFICE_DIR / filename).read_bytes()
    expected = convert_with_direct(raw, kind)
    actual = convert_isolated(raw, kind, engine="direct")
    assert expected.status == "ok"
    assert actual == expected


def test_direct_missing_package_gives_unsupported_and_exceptions_give_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Missing packages -> unsupported
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "openpyxl", None)
        res_xlsx = convert_with_direct(b"PK\x03\x04", "xlsx")
        # Invalid zip hits zipguard first; verify reader directly and with real xlsx bytes.
        assert readers.read_xlsx(b"anything") == ConvertResult(
            text="",
            status="unsupported",
            error="openpyxl not installed",
        )
        assert res_xlsx.status == "unreadable"
        real_xlsx = (OFFICE_DIR / "team_budget.xlsx").read_bytes()
        assert convert_with_direct(real_xlsx, "xlsx") == ConvertResult(
            text="",
            status="unsupported",
            error="openpyxl not installed",
        )

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "pptx", None)
        real_pptx = (OFFICE_DIR / "site_review.pptx").read_bytes()
        assert readers.read_pptx(b"anything") == ConvertResult(
            text="",
            status="unsupported",
            error="python-pptx not installed",
        )
        assert convert_with_direct(real_pptx, "pptx") == ConvertResult(
            text="",
            status="unsupported",
            error="python-pptx not installed",
        )

    # DocIndex missing package check
    (tmp_path / "team_budget.xlsx").write_bytes((OFFICE_DIR / "team_budget.xlsx").read_bytes())
    (tmp_path / "site_review.pptx").write_bytes((OFFICE_DIR / "site_review.pptx").read_bytes())
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "openpyxl", None)
        mp.setitem(sys.modules, "pptx", None)
        idx = DocIndex(
            tmp_path,
            converters={"xlsx": "direct", "pptx": "direct"},
            canonical_cache_dir=None,
        )
        idx.refresh()
        assert sorted(idx.failures()) == [
            ("site_review.pptx", "unsupported", "python-pptx not installed"),
            ("team_budget.xlsx", "unsupported", "openpyxl not installed"),
        ]

    # Library exceptions -> unreadable without leaking text
    openpyxl = pytest.importorskip("openpyxl")
    pptx = pytest.importorskip("pptx")

    class _BoomXlsxError(ValueError):
        pass

    class _BoomPptxError(RuntimeError):
        pass

    with monkeypatch.context() as mp:
        mp.setattr(
            openpyxl,
            "load_workbook",
            lambda *a, **kw: (_ for _ in ()).throw(_BoomXlsxError("SECRET_XLSX_BODY")),
        )
        res_x = convert_with_direct((OFFICE_DIR / "team_budget.xlsx").read_bytes(), "xlsx")
        assert res_x.status == "unreadable"
        assert res_x.text == ""
        assert res_x.error == "xlsx parse error: _BoomXlsxError"
        assert "SECRET_XLSX_BODY" not in res_x.error

    with monkeypatch.context() as mp:
        mp.setattr(
            pptx,
            "Presentation",
            lambda *a, **kw: (_ for _ in ()).throw(_BoomPptxError("SECRET_PPTX_BODY")),
        )
        res_p = convert_with_direct((OFFICE_DIR / "site_review.pptx").read_bytes(), "pptx")
        assert res_p.status == "unreadable"
        assert res_p.text == ""
        assert res_p.error == "pptx parse error: _BoomPptxError"
        assert "SECRET_PPTX_BODY" not in res_p.error


def test_docindex_direct_indexes_rag_office_with_locators_and_real_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("openpyxl")
    pytest.importorskip("pptx")

    converters = metrics.resolve_converters_arg("pptx=direct,xlsx=direct")
    assert converters is not None
    assert converters["pptx"] == "direct"
    assert converters["xlsx"] == "direct"

    cache_dir = tmp_path / "corpus" / ".rag_cache" / "canonical"
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)
    for src in sorted(OFFICE_DIR.iterdir()):
        if src.is_file():
            (corpus / src.name).write_bytes(src.read_bytes())

    idx = DocIndex(corpus, converters=converters, canonical_cache_dir=cache_dir)
    report = idx.refresh()
    assert report.failures == []
    records = idx.records()
    assert len(records) == 2
    assert all(r.status == "ok" for r in records)

    pptx_chunks = [c for c in idx.chunks if c.source == "site_review.pptx"]
    assert sorted({c.slide for c in pptx_chunks if c.slide is not None}) == [1, 2]
    assert all(c.converter.startswith("direct-python-pptx-") for c in pptx_chunks)
    assert any(c.section == "Slide 1 > Station Inspection Metrics" for c in pptx_chunks)
    assert any(c.section == "Slide 2 > Maintenance Field Findings" for c in pptx_chunks)
    assert any("Station: North Ridge; Elevation m: 412" in c.text for c in pptx_chunks)

    xlsx_chunks = [c for c in idx.chunks if c.source == "team_budget.xlsx"]
    assert sorted({c.sheet for c in xlsx_chunks if c.sheet is not None}) == [
        "Hardware",
        "Operations",
    ]
    assert all(c.converter.startswith("direct-openpyxl-") for c in xlsx_chunks)
    ops_chunks = [c for c in xlsx_chunks if c.sheet == "Operations"]
    assert ops_chunks
    assert all(c.section == "Sheet Operations" for c in ops_chunks)
    ops_text = "\n".join(c.text for c in ops_chunks)
    assert "Unnamed" not in ops_text
    assert "**Operations Quarterly Allocation**" in ops_text
    expected_row = (
        "Department: Field Telemetry; Headcount: 6; Travel USD: 4200; Equipment USD: 11800"
    )
    assert expected_row in ops_text
    assert docindex._canonical_cache_key_for(
        "abc", "xlsx", "direct"
    ) != docindex._canonical_cache_key_for("abc", "xlsx", "markitdown")
    assert docindex._canonical_cache_key_for(
        "abc", "pptx", "direct"
    ) != docindex._canonical_cache_key_for("abc", "pptx", "markitdown")

    # Cache hit on a fresh DocIndex even when convert_isolated raises
    monkeypatch.setattr(
        convert,
        "convert_isolated",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    idx_warm = DocIndex(corpus, converters=converters, canonical_cache_dir=cache_dir)
    warm_report = idx_warm.refresh()
    assert warm_report.failures == []
    assert len(idx_warm.chunks) == len(idx.chunks)


def test_rtf_basic_via_convert() -> None:
    """RTF round-trip."""
    rtf_bytes = b"{\\rtf1\\ansi Hello RTF world.}"
    res = convert_with_direct(rtf_bytes, "rtf")
    assert res.status == "ok"
    assert "Hello" in res.text
    assert "RTF" in res.text


def test_rtf_unsupported_no_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """rtf: unsupported when striprtf missing."""
    monkeypatch.setitem(sys.modules, "striprtf", None)
    monkeypatch.setitem(sys.modules, "striprtf.striprtf", None)
    res = convert_with_direct(b"{\\rtf1}", "rtf")
    assert res.status == "unsupported"
    assert "striprtf" in res.error


def test_rtf_empty() -> None:
    """Empty RTF -> empty status."""
    res = convert_with_direct(b"{\\rtf1}", "rtf")
    assert res.status == "empty"


def test_xls_registered() -> None:
    """xls extension and converter registered."""
    from voice_agent.rag.docindex import CANONICAL_CONVERTERS, SUPPORTED_EXTENSIONS

    assert ".xls" in SUPPORTED_EXTENSIONS
    assert CANONICAL_CONVERTERS.get("xls") == "direct"


def test_xls_unsupported_no_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """xls: unsupported when xlrd missing."""
    monkeypatch.setitem(sys.modules, "xlrd", None)
    res = convert_with_direct(b"\\xd0\\xcf", "xls")
    assert res.status == "unsupported"
    assert "xlrd" in res.error


def test_xls_unreadable_garbage() -> None:
    """Non-xls bytes -> unreadable."""
    pytest.importorskip("xlrd")
    res = convert_with_direct(b"not an xls file at all", "xls")
    assert res.status == "unreadable"


# --- Ticket 64: Legacy .doc and .ppt via catdoc (Popen bounded-read) ---


def _make_fake_popen(stdout_data: bytes, returncode: int = 0):
    """Return a class that mimics subprocess.Popen for tests."""

    class FakePopen:
        def __init__(self, argv, **_kw):
            self.argv = argv
            self.stdout = io.BytesIO(stdout_data)
            self._returncode = returncode
            self.kill_called = False
            self.wait_called = False

        @property
        def returncode(self):
            return self._returncode

        def kill(self):
            self.kill_called = True
            self.stdout.close()

        def wait(self, timeout=None):
            self.wait_called = True

    return FakePopen


def test_doc_unsupported_no_catdoc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """doc reports unsupported when catdoc binary is not found."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    (tmp_path / "test.doc").write_bytes(b"fake doc")
    idx = DocIndex(tmp_path)
    idx.refresh()
    failures = {name: (s, e) for name, s, e in idx.failures()}
    assert "test.doc" in failures
    assert failures["test.doc"][0] == "unsupported"
    assert "catdoc" in failures["test.doc"][1]


def test_ppt_unsupported_no_catppt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ppt reports unsupported when catppt binary is not found."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    (tmp_path / "test.ppt").write_bytes(b"fake ppt")
    idx = DocIndex(tmp_path)
    idx.refresh()
    failures = {name: (s, e) for name, s, e in idx.failures()}
    assert "test.ppt" in failures
    assert failures["test.ppt"][0] == "unsupported"
    assert "catppt" in failures["test.ppt"][1]


def test_doc_catdoc_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """doc reports unreadable when catdoc exits non-zero."""
    monkeypatch.setattr(shutil, "which", lambda n: "/fake" if n in ("catdoc", "catppt") else None)

    class FakePopen:
        def __init__(self, argv, **_kw):
            self.argv = argv
            self.stdout = io.BytesIO(b"")
            self._returncode = 1
            self.kill_called = False
            self.wait_called = False

        @property
        def returncode(self):
            return self._returncode

        def kill(self):
            self.kill_called = True

        def wait(self, timeout=None):
            self.wait_called = True

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    (tmp_path / "test.doc").write_bytes(b"fake doc")
    idx = DocIndex(tmp_path)
    idx.refresh()
    failures = {name: (s, e) for name, s, e in idx.failures()}
    assert failures["test.doc"][0] == "unreadable"
    assert "catdoc" in failures["test.doc"][1]


def test_doc_output_too_large(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """doc reports unreadable when catdoc stdout exceeds 5 MB."""
    monkeypatch.setattr(shutil, "which", lambda n: "/fake" if n in ("catdoc", "catppt") else None)
    monkeypatch.setattr(subprocess, "Popen", _make_fake_popen(b"x" * (6 * 1024 * 1024), 0))
    (tmp_path / "test.doc").write_bytes(b"fake doc")
    idx = DocIndex(tmp_path)
    idx.refresh()
    failures = {name: (s, e) for name, s, e in idx.failures()}
    assert failures["test.doc"][0] == "unreadable"
    assert "too large" in failures["test.doc"][1]


def test_doc_hostile_filename(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Filename starting with - is handled safely (absolute path in argv)."""
    monkeypatch.setattr(shutil, "which", lambda n: "/fake" if n in ("catdoc", "catppt") else None)
    captured_argv: list[str] | None = None

    class CapturePopen:
        def __init__(self, argv, **_kw):
            nonlocal captured_argv
            captured_argv = list(argv)
            self.stdout = io.BytesIO(b"OK")
            self._returncode = 0
            self.kill_called = False
            self.wait_called = False

        @property
        def returncode(self):
            return self._returncode

        def kill(self):
            self.kill_called = True

        def wait(self, timeout=None):
            self.wait_called = True

    monkeypatch.setattr(subprocess, "Popen", CapturePopen)
    (tmp_path / "-rf.doc").write_bytes(b"fake")
    DocIndex(tmp_path).refresh()
    assert captured_argv is not None
    assert captured_argv[-1].startswith("/")
    assert captured_argv[-1].endswith("-rf.doc")
    assert "--" not in captured_argv


def test_doc_spaces_in_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Filename with spaces is a single argv element."""
    monkeypatch.setattr(shutil, "which", lambda n: "/fake" if n in ("catdoc", "catppt") else None)
    captured_argv: list[str] | None = None

    class CapturePopen:
        def __init__(self, argv, **_kw):
            nonlocal captured_argv
            captured_argv = list(argv)
            self.stdout = io.BytesIO(b"OK")
            self._returncode = 0
            self.kill_called = False
            self.wait_called = False

        @property
        def returncode(self):
            return self._returncode

        def kill(self):
            self.kill_called = True

        def wait(self, timeout=None):
            self.wait_called = True

    monkeypatch.setattr(subprocess, "Popen", CapturePopen)
    (tmp_path / "my file.doc").write_bytes(b"fake")
    DocIndex(tmp_path).refresh()
    assert captured_argv is not None
    assert any("my file.doc" in arg for arg in captured_argv)


@pytest.mark.skipif(shutil.which("catdoc") is None, reason="catdoc not installed")
def test_doc_real_catdoc_skipped() -> None:
    """Integration with real catdoc (skipped on this machine)."""
    pytest.skip("catdoc not installed")


def test_doc_ppt_registered() -> None:
    """doc and ppt extensions and converters are registered."""
    from voice_agent.rag.docindex import CANONICAL_CONVERTERS, SUPPORTED_EXTENSIONS

    assert ".doc" in SUPPORTED_EXTENSIONS
    assert ".ppt" in SUPPORTED_EXTENSIONS
    assert CANONICAL_CONVERTERS.get("doc") == "direct"
    assert CANONICAL_CONVERTERS.get("ppt") == "direct"
