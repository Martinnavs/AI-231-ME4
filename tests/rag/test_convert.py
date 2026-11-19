"""Tests for ``voice_agent.rag.convert`` (MarkItDown adapter for pptx/xlsx/html/epub)."""

from __future__ import annotations

import datetime as dt
import io
import sys
import zipfile
from types import SimpleNamespace
from typing import Any

import pytest

from voice_agent.rag import canonical, zipguard
from voice_agent.rag.convert import ConvertResult, convert_with_markitdown


def _make_zip(members: list[tuple[str, bytes]], *, compress: int = zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compress) as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


def _make_pptx_bytes() -> bytes:
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()

    slide1 = prs.slides.add_slide(prs.slide_layouts[1])
    slide1.shapes.title.text = "Project Overview"
    slide1.placeholders[1].text = "Quarterly review of field sensors."

    slide2 = prs.slides.add_slide(prs.slide_layouts[1])
    slide2.shapes.title.text = "Hardware Metrics"
    slide2.placeholders[1].text = "Deployed 42 units."
    table_shape = slide2.shapes.add_table(
        2, 2, pptx.util.Inches(1), pptx.util.Inches(3), pptx.util.Inches(4), pptx.util.Inches(1.5)
    )
    table = table_shape.table
    table.cell(0, 0).text = "Sensor"
    table.cell(0, 1).text = "Count"
    table.cell(1, 0).text = "Alpha"
    table.cell(1, 1).text = "42"
    slide2.notes_slide.notes_text_frame.text = "Verify battery packs before Q3."

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def _make_xlsx_bytes() -> bytes:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()

    ws1 = wb.active
    assert ws1 is not None
    ws1.title = "Summary"
    ws1.append(["Metric", "Value"])
    ws1.append(["Sites", 12])
    ws1.append(["Status", "Active"])

    ws2 = wb.create_sheet("Budget")
    ws2.append(["Category", "Amount", "Owner"])
    ws2.append(["Sensors", 4500, "Ana"])
    ws2.append(["Cables", 800, "Ben"])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_epub_bytes(*, title: str = "Field Guide", body_html: str | None = None) -> bytes:
    xhtml = body_html or (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        "<head><title>Chapter 1</title></head>"
        "<body><h1>Chapter 1: Setup</h1><p>Calibrate probe to 7.0 pH.</p></body>"
        "</html>"
    )
    container_xml = (
        '<?xml version="1.0"?>'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        "<rootfiles>"
        '<rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/>'
        "</rootfiles>"
        "</container>"
    )
    content_opf = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<dc:title>{title}</dc:title>"
        "<dc:language>en</dc:language>"
        "</metadata>"
        "<manifest>"
        '<item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>'
        "</manifest>"
        '<spine><itemref idref="ch1"/></spine>'
        "</package>"
    )
    return _make_zip(
        [
            ("mimetype", b"application/epub+zip"),
            ("META-INF/container.xml", container_xml.encode("utf-8")),
            ("OEBPS/content.opf", content_opf.encode("utf-8")),
            ("OEBPS/chapter1.xhtml", xhtml.encode("utf-8")),
        ]
    )


def test_pptx_converts_with_slide_markers_and_passes_zip_preflight() -> None:
    markitdown = pytest.importorskip("markitdown")
    raw = _make_pptx_bytes()
    assert zipguard.preflight_zip(raw, label="pptx", scan_xml_members=True) is None

    res = convert_with_markitdown(raw, "pptx")
    assert res.status == "ok"
    assert res.error is None
    assert res.converter == f"markitdown-{markitdown.__version__}"
    assert res.raw_text

    assert "<!-- rag:slide=1 -->" in res.text
    assert "<!-- rag:slide=2 -->" in res.text

    units, _ = canonical.split_units(res.text)
    assert [u.slide for u in units] == [1, 2]
    assert "Project Overview" in units[0].body
    assert "Quarterly review of field sensors." in units[0].body
    assert "Hardware Metrics" in units[1].body
    assert "Sensor: Alpha; Count: 42" in units[1].body
    assert "Verify battery packs before Q3." in units[1].body


def test_xlsx_converts_with_sheet_markers_and_normalized_rows() -> None:
    markitdown = pytest.importorskip("markitdown")
    raw = _make_xlsx_bytes()
    assert zipguard.preflight_zip(raw, label="xlsx", scan_xml_members=True) is None

    res = convert_with_markitdown(raw, "xlsx")
    assert res.status == "ok"
    assert res.error is None
    assert res.converter == f"markitdown-{markitdown.__version__}"

    assert "<!-- rag:sheet=Summary -->" in res.text
    assert "<!-- rag:sheet=Budget -->" in res.text

    units, _ = canonical.split_units(res.text)
    assert [u.sheet for u in units] == ["Summary", "Budget"]
    assert "Metric: Sites; Value: 12" in units[0].body
    assert "Metric: Status; Value: Active" in units[0].body
    assert "Category: Sensors; Amount: 4500; Owner: Ana" in units[1].body
    assert "Category: Cables; Amount: 800; Owner: Ben" in units[1].body


def test_html_and_htm_convert_and_neutralize_forged_markers() -> None:
    markitdown = pytest.importorskip("markitdown")
    html_bytes = (
        b"<html><head><title>Guide</title></head><body>"
        b"<h1>Watering Schedule</h1>"
        b"<p>Water beds every 3 days.</p>"
        b"<!-- rag:page=99 -->"
        b"<table><tr><th>Bay</th><th>Days</th></tr>"
        b"<tr><td>North</td><td>14</td></tr></table>"
        b"</body></html>"
    )
    for kind in ("html", "htm", ".HTML"):
        res = convert_with_markitdown(html_bytes, kind)
        assert res.status == "ok"
        assert res.error is None
        assert res.converter == f"markitdown-{markitdown.__version__}"
        assert "# Watering Schedule" in res.text
        assert "Water beds every 3 days." in res.text
        assert "Bay: North; Days: 14" in res.text
        assert "<!-- rag:page=99 -->" not in res.text


def test_deep_html_does_not_raise_under_error_warnings() -> None:
    pytest.importorskip("markitdown")
    depth = 1500
    deep_html = (
        b"<html><body>"
        + b"<div>" * depth
        + b"<p>Deep leaf text 777.</p>"
        + b"</div>" * depth
        + b"</body></html>"
    )
    res = convert_with_markitdown(deep_html, "html")
    assert res.status == "ok"
    assert "Deep leaf text 777." in res.text


def test_epub_converts() -> None:
    markitdown = pytest.importorskip("markitdown")
    raw = _make_epub_bytes()
    assert zipguard.preflight_zip(raw, label="epub", scan_xml_members=True) is None

    res = convert_with_markitdown(raw, "epub")
    assert res.status == "ok"
    assert res.error is None
    assert res.converter == f"markitdown-{markitdown.__version__}"
    assert "Field Guide" in res.text
    assert "Chapter 1: Setup" in res.text
    assert "Calibrate probe to 7.0 pH." in res.text


def test_xlsx_stored_values_u3_limitation() -> None:
    """U3: verify stored-value behavior (formula -> empty, 15% -> 0.15, dates -> ISO)."""
    pytest.importorskip("markitdown")
    openpyxl = pytest.importorskip("openpyxl")

    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Valuation"
    ws.append(["Item", "Input", "Formula", "Margin", "DateOnly", "Timestamp"])
    ws.append(["LotA", 10, "=B2*2", 0.15, dt.date(2026, 3, 15), dt.datetime(2026, 3, 15, 14, 30)])
    ws["D2"].number_format = "0%"
    ws["E2"].number_format = "mmm d, yyyy"
    ws["F2"].number_format = "yyyy-mm-dd hh:mm"

    buf = io.BytesIO()
    wb.save(buf)
    res = convert_with_markitdown(buf.getvalue(), "xlsx")
    assert res.status == "ok"

    # Unrecalculated formula has no cached value -> NaN -> dropped by table normalizer.
    assert "Formula:" not in res.text
    # Percentage formatting is lost; stored float 0.15 is emitted instead of "15%".
    assert "Margin: 0.15" in res.text
    assert "15%" not in res.text
    # Dates serialize as ISO date/timestamp strings (custom display formatting lost).
    assert "DateOnly: 2026-03-15" in res.text
    assert "Timestamp: 2026-03-15 14:30:00" in res.text


def test_no_magika_model_instantiated(monkeypatch: pytest.MonkeyPatch) -> None:
    markitdown = pytest.importorskip("markitdown")
    magika = pytest.importorskip("magika")

    def _forbid(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("MarkItDown / Magika must never be instantiated")

    monkeypatch.setattr(markitdown, "MarkItDown", _forbid)
    monkeypatch.setattr(magika, "Magika", _forbid)

    res = convert_with_markitdown(b"<html><body><p>Safe path</p></body></html>", "html")
    assert res.status == "ok"
    assert "Safe path" in res.text


def test_zip_bomb_never_calls_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("markitdown")
    import markitdown.converters as md_converters

    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("converter must not be called for zip bomb")

    for cls_name in ("PptxConverter", "XlsxConverter", "EpubConverter"):
        monkeypatch.setattr(md_converters, cls_name, _fail)

    # 256 KB of zeros compresses well over 100x under deflate.
    bomb_bytes = _make_zip([("payload.xml", b"0" * (256 * 1024))])
    for kind in ("pptx", "xlsx", "epub"):
        res = convert_with_markitdown(bomb_bytes, kind)
        assert res.status == "unreadable"
        assert res.error is not None
        assert "compression ratio" in res.error
        assert res.text == ""


def test_doctype_in_pptx_slide_never_calls_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("markitdown")
    import markitdown.converters as md_converters

    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("converter must not be called when XML member has DOCTYPE")

    monkeypatch.setattr(md_converters, "PptxConverter", _fail)

    raw = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/presentation.xml", b'<?xml version="1.0"?><presentation/>'),
            (
                "ppt/slides/slide1.xml",
                b'<?xml version="1.0"?><!DOCTYPE sld [<!ENTITY x "1">]><sld/>',
            ),
        ]
    )
    res = convert_with_markitdown(raw, "pptx")
    assert res.status == "unreadable"
    assert res.error == "pptx XML contains forbidden DOCTYPE or ENTITY declaration"
    assert res.text == ""


def test_blocked_markitdown_import_returns_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "markitdown", None)
    monkeypatch.setitem(sys.modules, "markitdown.converters", None)

    for kind, raw in (
        ("html", b"<html><body><p>Hello</p></body></html>"),
        ("pptx", _make_zip([("ppt/presentation.xml", b'<?xml version="1.0"?><presentation/>')])),
    ):
        res = convert_with_markitdown(raw, kind)
        assert res == ConvertResult(
            text="",
            status="unsupported",
            error="markitdown not installed",
            converter="",
            raw_text="",
        )


def test_converter_exception_empty_too_large_and_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("markitdown")
    import markitdown.converters as md_converters

    class _ExplodingConverter:
        def convert(self, stream: Any, stream_info: Any) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setattr(md_converters, "HtmlConverter", _ExplodingConverter)
    err_res = convert_with_markitdown(b"<html><body>hi</body></html>", "html")
    assert err_res.status == "unreadable"
    assert err_res.error == "html parse error: RuntimeError"

    class _EmptyConverter:
        def convert(self, stream: Any, stream_info: Any) -> Any:
            return SimpleNamespace(markdown="   \n  ")

    monkeypatch.setattr(md_converters, "HtmlConverter", _EmptyConverter)
    empty_res = convert_with_markitdown(b"<html><body></body></html>", "html")
    assert empty_res.status == "empty"

    class _HugeConverter:
        def convert(self, stream: Any, stream_info: Any) -> Any:
            return SimpleNamespace(markdown="x" * 100)

    monkeypatch.setattr(md_converters, "HtmlConverter", _HugeConverter)
    monkeypatch.setattr(canonical, "MAX_CANONICAL_CHARS", 50)
    large_res = convert_with_markitdown(b"<html><body>huge</body></html>", "html")
    assert large_res.status == "too_large"

    unsupported_res = convert_with_markitdown(b"{\\rtf1}", "rtf")
    assert unsupported_res.status == "unsupported"
