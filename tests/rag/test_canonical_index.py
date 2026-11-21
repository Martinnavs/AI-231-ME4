"""Tests for DocIndex canonical converter switch, chunking, and per-chunk metadata."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
from test_loaders_binary import (
    _build_docx_bytes,
    _build_encrypted_pdf_bytes,
    _build_pdf_bytes,
    _docx_p,
    _wrap_document_xml,
)

from voice_agent.rag import canonical, convert
from voice_agent.rag.convert import ConvertResult
from voice_agent.rag.docindex import CANONICAL_CONVERTERS, DocIndex

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "rag_corpus"

EXPECTED_SEARCH_KEYS = {
    "source",
    "id",
    "section",
    "page",
    "file_date",
    "text",
    "score",
    "coverage",
}

ALL_MARKITDOWN: dict[str, str] = {
    "md": "identity",
    "txt": "identity",
    "csv": "identity",
    "pdf": "markitdown",
    "docx": "markitdown",
    "pptx": "markitdown",
    "xlsx": "markitdown",
    "html": "markitdown",
    "epub": "markitdown",
}

ALL_PYPDF_IDENTITY: dict[str, str] = {
    "md": "identity",
    "txt": "identity",
    "csv": "identity",
    "pdf": "pypdf",
    "docx": "legacy",
    "pptx": "markitdown",
    "xlsx": "markitdown",
    "html": "markitdown",
    "epub": "markitdown",
}


def _make_pptx_bytes() -> bytes:
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()

    slide1 = prs.slides.add_slide(prs.slide_layouts[1])
    slide1.shapes.title.text = "Project Overview"
    slide1.placeholders[1].text = "Quarterly review of field sensors."

    slide2 = prs.slides.add_slide(prs.slide_layouts[1])
    slide2.shapes.title.text = "Hardware Metrics"
    slide2.placeholders[1].text = "Deployed 42 solar telemetry nodes on 2026-04-15."

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

    ws2 = wb.create_sheet("Budget")
    ws2.append(["Category", "Amount", "Owner"])
    ws2.append(["Sensors", 4500, "Ana"])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_epub_bytes() -> bytes:
    xhtml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        "<head><title>Chapter 1</title></head>"
        "<body><h1>Field Setup</h1><p>Calibrate probe to 7.0 pH on 2026-06-01.</p></body>"
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
        "<dc:title>Guide</dc:title>"
        "<dc:language>en</dc:language>"
        "</metadata>"
        "<manifest>"
        '<item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>'
        "</manifest>"
        '<spine><itemref idref="ch1"/></spine>'
        "</package>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", b"application/epub+zip")
        zf.writestr("META-INF/container.xml", container_xml.encode("utf-8"))
        zf.writestr("OEBPS/content.opf", content_opf.encode("utf-8"))
        zf.writestr("OEBPS/chapter1.xhtml", xhtml.encode("utf-8"))
    return buf.getvalue()


def test_default_mapping_and_supported_set(tmp_path: Path) -> None:
    assert CANONICAL_CONVERTERS == {
        "md": "legacy",
        "txt": "legacy",
        "csv": "legacy",
        "pdf": "legacy",
        "docx": "legacy",
        "pptx": "direct",
        "xlsx": "direct",
        "html": "direct",
        "epub": "direct",
        "rtf": "direct",
        "xls": "direct",
        "doc": "direct",
        "ppt": "direct",
        "tsv": "legacy",
        "log": "legacy",
        "ini": "legacy",
        "toml": "legacy",
        "yaml": "legacy",
        "cfg": "legacy",
        "conf": "legacy",
    }

    legacy_only = DocIndex(tmp_path, converters={})
    assert legacy_only.converters == {
        "md": "legacy",
        "txt": "legacy",
        "csv": "legacy",
        "pdf": "legacy",
        "docx": "legacy",
    }

    (tmp_path / "deck.pptx").write_bytes(b"placeholder")
    report = legacy_only.refresh()
    assert report.failures == [
        ("deck.pptx", "unsupported", "unsupported file extension: .pptx"),
    ]

    with pytest.raises(ValueError, match="unsupported converter mode.*rtf"):
        DocIndex(tmp_path, converters={"rtf": "markitdown"})

    with pytest.raises(ValueError, match="unsupported converter mode"):
        DocIndex(tmp_path, converters={"pptx": "legacy"})


def test_all_arms_index_13_fixtures_with_metadata_and_parity() -> None:
    pytest.importorskip("markitdown")

    legacy_idx = DocIndex(CORPUS_DIR, converters={})
    default_idx = DocIndex(CORPUS_DIR)
    md_idx = DocIndex(CORPUS_DIR, converters=ALL_MARKITDOWN)
    pypdf_idx = DocIndex(CORPUS_DIR, converters=ALL_PYPDF_IDENTITY)

    for idx in (legacy_idx, default_idx, md_idx, pypdf_idx):
        idx.refresh()
        assert idx.failures() == []
        assert len(idx.records()) == 13
        assert idx.chunks
        for chunk in idx.chunks:
            assert chunk.fmt
            assert chunk.converter
            assert "<!-- rag:" not in chunk.text

    for chunk in legacy_idx.chunks:
        assert chunk.converter == "legacy"
        assert chunk.canonical_sha256 == ""

    for chunk in default_idx.chunks:
        assert chunk.converter == "legacy"
        assert chunk.canonical_sha256 == ""

    for chunk in md_idx.chunks:
        assert len(chunk.canonical_sha256) == 64
        if chunk.fmt in ("md", "txt", "csv"):
            assert chunk.converter == "identity"
        else:
            assert chunk.converter.startswith("markitdown-")
            assert f"+pp{canonical.POSTPROC_VERSION}" in chunk.converter

    for chunk in pypdf_idx.chunks:
        if chunk.fmt == "docx":
            assert chunk.converter == "legacy"
            assert chunk.canonical_sha256 == ""
        elif chunk.fmt == "pdf":
            assert len(chunk.canonical_sha256) == 64
            assert chunk.converter.startswith("pypdf-")
            assert f"+pp{canonical.POSTPROC_VERSION}" in chunk.converter
        else:
            assert len(chunk.canonical_sha256) == 64
            assert chunk.converter == "identity"

    # md/txt/csv (id, header, text) identical to legacy across arms.
    def _text_tuples(idx: DocIndex, fmts: set[str]) -> list[tuple[str, str, str]]:
        return [(c.id, c.header, c.text) for c in idx.chunks if c.fmt in fmts]

    legacy_text_chunks = _text_tuples(legacy_idx, {"md", "txt", "csv"})
    assert legacy_text_chunks
    assert _text_tuples(default_idx, {"md", "txt", "csv"}) == legacy_text_chunks
    assert _text_tuples(md_idx, {"md", "txt", "csv"}) == legacy_text_chunks
    assert _text_tuples(pypdf_idx, {"md", "txt", "csv"}) == legacy_text_chunks

    # handbook.pdf pages 1 and 2 unchanged across arms.
    legacy_pdf_pages = [c.page for c in legacy_idx.chunks if c.source == "handbook.pdf"]
    assert sorted(set(legacy_pdf_pages)) == [1, 2]
    assert [c.page for c in md_idx.chunks if c.source == "handbook.pdf"] == legacy_pdf_pages
    assert [c.page for c in pypdf_idx.chunks if c.source == "handbook.pdf"] == legacy_pdf_pages

    # handbook.docx sections unchanged across arms.
    legacy_docx_sections = [c.section for c in legacy_idx.chunks if c.source == "handbook.docx"]
    assert legacy_docx_sections == ["Onboarding checklist", "Equipment return"]
    assert [c.section for c in md_idx.chunks if c.source == "handbook.docx"] == legacy_docx_sections

    # search() result dict keys unchanged across all arms.
    for idx in (legacy_idx, default_idx, md_idx, pypdf_idx):
        hits = idx.search("onboarding checklist", k=3)
        assert hits
        for hit in hits:
            assert set(hit.keys()) == EXPECTED_SEARCH_KEYS


def test_generated_pptx_xlsx_html_htm_epub_locators_and_metadata(tmp_path: Path) -> None:
    pytest.importorskip("markitdown")

    (tmp_path / "deck.pptx").write_bytes(_make_pptx_bytes())
    (tmp_path / "plan.xlsx").write_bytes(_make_xlsx_bytes())
    (tmp_path / "guide.html").write_bytes(
        b"<html><body><h1>Watering</h1><p>Check beds on 2026-05-01.</p>"
        b"<!-- rag:page=99 --></body></html>"
    )
    (tmp_path / "alt.htm").write_bytes(
        b"<html><body><h1>Lighting</h1><p>Inspect grow lamps weekly.</p></body></html>"
    )
    (tmp_path / "manual.epub").write_bytes(_make_epub_bytes())
    (tmp_path / "spoof.md").write_text(
        "# Notes\n\n<!-- rag:page=99 -->\nReal note on 2026-07-04.\n",
        encoding="utf-8",
    )

    idx = DocIndex(tmp_path, converters=ALL_MARKITDOWN)
    report = idx.refresh()
    assert idx.failures() == []
    assert sorted(report.added) == [
        "alt.htm",
        "deck.pptx",
        "guide.html",
        "manual.epub",
        "plan.xlsx",
        "spoof.md",
    ]

    pptx_chunks = [c for c in idx.chunks if c.source == "deck.pptx"]
    assert len(pptx_chunks) == 2
    assert [c.slide for c in pptx_chunks] == [1, 2]
    assert pptx_chunks[0].section.startswith("Slide 1")
    assert pptx_chunks[1].section.startswith("Slide 2")
    assert "Slide 2" in pptx_chunks[1].header
    assert all(c.fmt == "pptx" and c.page is None and c.sheet is None for c in pptx_chunks)

    xlsx_chunks = [c for c in idx.chunks if c.source == "plan.xlsx"]
    assert len(xlsx_chunks) == 2
    assert [c.sheet for c in xlsx_chunks] == ["Summary", "Budget"]
    assert [c.section for c in xlsx_chunks] == ["Sheet Summary", "Sheet Budget"]
    assert xlsx_chunks[1].header == "plan.xlsx > Sheet Budget"
    assert "Category: Sensors; Amount: 4500; Owner: Ana" in xlsx_chunks[1].text
    assert all(c.fmt == "xlsx" and c.page is None and c.slide is None for c in xlsx_chunks)

    htm_rec = next(r for r in idx.records() if r.name == "alt.htm")
    assert htm_rec.kind == "html"
    htm_chunks = [c for c in idx.chunks if c.source == "alt.htm"]
    assert htm_chunks and htm_chunks[0].fmt == "html"

    epub_chunks = [c for c in idx.chunks if c.source == "manual.epub"]
    assert epub_chunks and all(c.fmt == "epub" for c in epub_chunks)
    assert any(
        c.section == "Field Setup" and "Calibrate probe to 7.0 pH" in c.text for c in epub_chunks
    )

    spoof_chunks = [c for c in idx.chunks if c.source == "spoof.md"]
    assert spoof_chunks and spoof_chunks[0].page is None
    for chunk in idx.chunks:
        assert "<!-- rag:" not in chunk.text
        assert chunk.fmt
        assert len(chunk.canonical_sha256) == 64
        assert chunk.converter

    pptx_rec = next(r for r in idx.records() if r.name == "deck.pptx")
    assert pptx_rec.dates_in_text == ["2026-04-15"]

    hits = idx.search("Sensors Ana", k=2)
    assert hits
    assert set(hits[0].keys()) == EXPECTED_SEARCH_KEYS


def test_guard_cases_match_status_and_error_across_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("markitdown")

    monkeypatch.setattr(
        convert,
        "convert_isolated",
        lambda raw_bytes, kind, **kw: (
            convert.convert_with_markitdown(raw_bytes, kind)
            if kind in ("pdf", "docx")
            else convert.convert_isolated(raw_bytes, kind, **kw)
        ),
    )

    pdf_cases = {
        "encrypted.pdf": _build_encrypted_pdf_bytes("Secret payload."),
        "too_many.pdf": _build_pdf_bytes([""] * 501),
        "no_text.pdf": _build_pdf_bytes(["", ""]),
        "corrupt.pdf": b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog\n",
    }
    ratio_bomb = _build_docx_bytes(
        _wrap_document_xml(_docx_p("Normal text.")),
        extra_members=[("word/bomb.txt", b"A" * 50_000)],
        compression=zipfile.ZIP_DEFLATED,
    )
    doctype_docx = _build_docx_bytes(
        _wrap_document_xml(
            _docx_p("Doc body."),
            prefix='<!DOCTYPE foo [<!ENTITY xxe "boom">]>',
        )
    )
    deep_open = "".join(
        f'<w:tbl w:id="t_{i}"><w:tr w:r="r_{i}"><w:tc w:c="c_{i}">' for i in range(120)
    )
    deep_docx = _build_docx_bytes(
        _wrap_document_xml(deep_open + _docx_p("Deep text.") + ("</w:tc></w:tr></w:tbl>" * 120))
    )
    docx_cases = {
        "ratio_bomb.docx": ratio_bomb,
        "doctype.docx": doctype_docx,
        "deep.docx": deep_docx,
        "bad_zip.docx": b"not a zip archive",
        "empty.docx": _build_docx_bytes(_wrap_document_xml("")),
    }
    text_cases = {
        "empty.txt": b"",
        "binary.txt": b"\xff\xfe\x00\x80\x81\x90" * 40,
    }

    for name, data in {**pdf_cases, **docx_cases, **text_cases}.items():
        (tmp_path / name).write_bytes(data)

    legacy_idx = DocIndex(tmp_path, converters={})
    md_idx = DocIndex(tmp_path, converters=ALL_MARKITDOWN)
    pypdf_idx = DocIndex(tmp_path, converters=ALL_PYPDF_IDENTITY)

    legacy_idx.refresh()
    md_idx.refresh()
    pypdf_idx.refresh()

    legacy_failures = {name: (status, err) for name, status, err in legacy_idx.failures()}
    md_failures = {name: (status, err) for name, status, err in md_idx.failures()}
    pypdf_failures = {name: (status, err) for name, status, err in pypdf_idx.failures()}

    assert md_failures == legacy_failures
    assert pypdf_failures == legacy_failures


def test_blocked_markitdown_import_gives_pptx_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "deck.pptx").write_bytes(b"not even parsed when import blocked")
    monkeypatch.setitem(sys.modules, "markitdown", None)
    monkeypatch.setitem(sys.modules, "markitdown.converters", None)

    idx = DocIndex(tmp_path, converters={"pptx": "markitdown"})
    idx.refresh()
    assert idx.failures() == [("deck.pptx", "unsupported", "markitdown not installed")]
    assert idx.chunks == ()


def test_blocked_python_pptx_import_gives_pptx_unsupported_for_default_direct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "deck.pptx").write_bytes(b"not even parsed when import blocked")
    monkeypatch.setitem(sys.modules, "pptx", None)

    idx = DocIndex(tmp_path)
    idx.refresh()
    assert idx.failures() == [("deck.pptx", "unsupported", "python-pptx not installed")]
    assert idx.chunks == ()


def test_fallback_and_timeout_recorded_via_isolated_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "sample.pdf").write_bytes(_build_pdf_bytes(["Fallback page one text."]))
    (tmp_path / "slow.pptx").write_bytes(b"placeholder")

    def _fake_isolated(raw_bytes: bytes, kind: str, **kwargs: Any) -> ConvertResult:
        del raw_bytes, kwargs
        if kind == "pdf":
            return ConvertResult(
                text="<!-- rag:page=1 -->\nFallback page one text.",
                status="ok",
                error=None,
                converter="pypdf-7.1.0",
                raw_text="Fallback page one text.",
                fallback="pypdf-pages",
            )
        return ConvertResult(
            text="",
            status="unreadable",
            error="conversion timed out after 30s",
            converter="markitdown",
        )

    monkeypatch.setattr(convert, "convert_isolated", _fake_isolated)

    idx = DocIndex(tmp_path, converters={"pdf": "markitdown", "pptx": "markitdown"})
    idx.refresh()

    assert idx.failures() == [
        ("slow.pptx", "unreadable", "conversion timed out after 30s"),
    ]
    pdf_chunks = [c for c in idx.chunks if c.source == "sample.pdf"]
    assert len(pdf_chunks) == 1
    assert pdf_chunks[0].converter == (
        f"pypdf-7.1.0+pp{canonical.POSTPROC_VERSION}/fallback=pypdf-pages"
    )
    assert pdf_chunks[0].page == 1
