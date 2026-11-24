"""Tests for header stacking (stack mode) and plain-text fallback (ticket 208)."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import pytest

from voice_agent.rag import convert, docindex, readers
from voice_agent.rag.ingest import ingest


def _build_eia_xlsx() -> bytes:
    """Build an EIA-shaped xlsx: 2 title rows, super-header, header, data."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "EIA 5.6.A"
    ws.append(["Electric Power Monthly"])
    ws.append(["August 2026"])
    ws.append(["", "Residential", "", "Commercial", ""])
    ws.append(["State", "Aug 2026", "Aug 2025", "Aug 2026", "Aug 2025"])
    ws.append(["Hawaii", 47.18, 38.53, 43.55, 34.93])
    ws.append(["Alaska", 51.22, 42.10, 40.11, 33.20])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _build_census_xlsx() -> bytes:
    """Build census-shaped xlsx: header over year row."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Census"
    ws.append(["Geographic Area", "Base", "Population Estimate", "", ""])
    ws.append(["", "", 2020, 2021, 2022])
    ws.append(["US", 1, 331449281, 331893745, 332407189])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _build_all_text_xlsx() -> bytes:
    """Build xlsx with all-text rows (no data row within 15 rows)."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Notes"
    for i in range(20):
        ws.append([f"Row {i}", f"Data {i}"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _build_cover_xlsx() -> bytes:
    """Build xlsx with a Cover sheet and a Sample_counts sheet."""
    import openpyxl

    wb = openpyxl.Workbook()
    cover = wb.active
    cover.title = "Cover"
    cover.append(["Sample_counts: number of firms sampled"])
    cover.append(["Total: 500"])
    ws2 = wb.create_sheet("Sample_counts")
    ws2.append(["Firm", "Count"])
    ws2.append(["A", 100])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# --- EIA-shaped test ---


def test_eia_xlsx_stack_mode() -> None:
    os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"] = "stack"
    try:
        raw = _build_eia_xlsx()
        res = readers.read_xlsx(raw)
        assert res.status == "ok"
        text = res.text
        assert "Residential / Aug 2026" in text
        assert "Commercial / Aug 2026" in text
        assert "47.18" in text
        assert "43.55" in text
    finally:
        del os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"]


def test_eia_xlsx_legacy_mode() -> None:
    os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"] = "legacy"
    try:
        raw = _build_eia_xlsx()
        res = readers.read_xlsx(raw)
        assert res.status == "ok"
        text = res.text
        assert "Residential / Aug 2026" not in text
    finally:
        del os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"]


def test_eia_xlsx_default_is_legacy() -> None:
    raw = _build_eia_xlsx()
    res = readers.read_xlsx(raw)
    assert res.status == "ok"
    assert "Residential / Aug 2026" not in res.text


# --- Census-shaped test ---


def test_census_xlsx_stack_mode() -> None:
    os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"] = "stack"
    try:
        raw = _build_census_xlsx()
        res = readers.read_xlsx(raw)
        assert res.status == "ok"
        text = res.text
        assert "Population Estimate / 2021" in text
    finally:
        del os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"]


# --- All-text fallback ---


def test_all_text_xlsx_falls_back_to_legacy() -> None:
    os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"] = "stack"
    try:
        raw = _build_all_text_xlsx()
        stack_res = readers.read_xlsx(raw)
        os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"] = "legacy"
        legacy_res = readers.read_xlsx(raw)
        assert stack_res.text == legacy_res.text
    finally:
        os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"] = "legacy"


# --- Cover caption ---


def test_cover_caption() -> None:
    os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"] = "stack"
    try:
        raw = _build_cover_xlsx()
        res = readers.read_xlsx(raw)
        assert res.status == "ok"
        text = res.text
        assert "Context: Sample_counts: number of firms sampled" in text
    finally:
        del os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"]


# --- xls column positions ---


def test_xls_column_positions_stacked() -> None:
    try:
        import xlwt  # noqa: F401
    except ImportError:
        pytest.skip("xlwt not installed")
    os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"] = "stack"
    try:
        import xlwt

        wb = xlwt.Workbook()
        ws = wb.add_sheet("Sheet1")
        ws.write(0, 0, "A")
        ws.write(0, 1, "B")
        ws.write(0, 2, "C")
        ws.write(1, 0, "x")
        ws.write(1, 1, "")
        ws.write(1, 2, "z")
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        raw = buf.read()
        res = readers.read_xls(raw)
        assert res.status == "ok"
        assert "x" in res.text
        assert "z" in res.text
    finally:
        del os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"]


# --- Cache key ---


def test_cache_key_differs_between_modes() -> None:
    os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"] = "legacy"
    legacy_key = docindex._canonical_cache_key_for("abc", "xlsx", "direct")
    os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"] = "stack"
    stack_key = docindex._canonical_cache_key_for("abc", "xlsx", "direct")
    assert legacy_key != stack_key
    os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"] = "legacy"
    legacy_xls_key = docindex._canonical_cache_key_for("abc", "xls", "direct")
    os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"] = "stack"
    stack_xls_key = docindex._canonical_cache_key_for("abc", "xls", "direct")
    assert legacy_xls_key != stack_xls_key
    os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"] = "legacy"
    html_key = docindex._canonical_cache_key_for("abc", "html", "direct")
    os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"] = "stack"
    html_key2 = docindex._canonical_cache_key_for("abc", "html", "direct")
    assert html_key == html_key2
    del os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"]


# --- Child env ---


def test_child_env_contains_table_headers_flag() -> None:
    os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"] = "stack"
    try:
        env = convert._child_env("direct")
        assert env.get("VOICE_AGENT_RAG_TABLE_HEADERS") == "stack"
    finally:
        del os.environ["VOICE_AGENT_RAG_TABLE_HEADERS"]


# --- looks_like_text ---


def test_looks_like_text() -> None:
    with tempfile.TemporaryDirectory() as td:
        ascii_path = Path(td) / "test.txt"
        ascii_path.write_text("Hello world", encoding="utf-8")
        assert docindex.looks_like_text(ascii_path) is True

        cut2_path = Path(td) / "cut2.bin"
        cut2_path.write_bytes("Hello \xc3".encode("latin-1"))
        assert docindex.looks_like_text(cut2_path) is True

        nul_path = Path(td) / "nul.bin"
        nul_path.write_bytes(b"Hello\x00World")
        assert docindex.looks_like_text(nul_path) is False

        png_path = Path(td) / "img.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        assert docindex.looks_like_text(png_path) is False


# --- Ingest: plain fallback and flag rebuild ---


def test_ingest_plain_fallback_and_rebuild() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        txt = td_path / "a.unknownext"
        txt.write_text("Hello from unknown extension", encoding="utf-8")
        db = td_path / "index.db"

        os.environ["VOICE_AGENT_RAG_PLAIN_FALLBACK"] = "1"
        try:
            report = ingest(td_path, db, workers=0)
            assert report.files_total == 1
            assert report.ok == 1
        finally:
            del os.environ["VOICE_AGENT_RAG_PLAIN_FALLBACK"]

        os.environ["VOICE_AGENT_RAG_PLAIN_FALLBACK"] = "0"
        try:
            report2 = ingest(td_path, db, workers=0)
            assert report2.files_total == 0
        finally:
            pass
