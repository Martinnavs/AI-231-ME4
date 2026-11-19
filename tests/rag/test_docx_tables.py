"""Offline unit tests for DOCX tables, styles, tracked changes, AlternateContent, and footnotes."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

from test_loaders_binary import (
    CORPUS_DIR,
    _build_docx_bytes,
    _docx_p,
    _wrap_document_xml,
)

from voice_agent.rag import DocIndex, canonical
from voice_agent.rag.docindex import load_docx

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"


def _wrap_styles_xml(inner_styles_xml: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:styles xmlns:w="{_W_NS}">{inner_styles_xml}</w:styles>'
    ).encode()


def _wrap_footnotes_xml(inner_footnotes_xml: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:footnotes xmlns:w="{_W_NS}">{inner_footnotes_xml}</w:footnotes>'
    ).encode()


def _tc(text: str, *, grid_span: int = 1, vmerge: str | None = None) -> str:
    props: list[str] = []
    if grid_span > 1:
        props.append(f'<w:gridSpan w:val="{grid_span}"/>')
    if vmerge == "restart":
        props.append('<w:vMerge w:val="restart"/>')
    elif vmerge == "continue":
        props.append("<w:vMerge/>")
    tc_pr = f"<w:tcPr>{''.join(props)}</w:tcPr>" if props else ""
    body = _docx_p(text) if text else "<w:p/>"
    return f"<w:tc>{tc_pr}{body}</w:tc>"


def _tr(cells_xml: str, *, tbl_header: bool = False, grid_before: int = 0) -> str:
    props: list[str] = []
    if tbl_header:
        props.append("<w:tblHeader/>")
    if grid_before > 0:
        props.append(f'<w:gridBefore w:val="{grid_before}"/>')
    tr_pr = f"<w:trPr>{''.join(props)}</w:trPr>" if props else ""
    return f"<w:tr>{tr_pr}{cells_xml}</w:tr>"


def test_canonical_format_table_row_public_helper() -> None:
    assert canonical.format_table_row(["Name", "Role", "Notes"], ["Alice", "Lead", ""]) == (
        "Name: Alice; Role: Lead"
    )
    assert canonical.format_table_row(["A"], ["1", "NaN", "3"]) == "A: 1; col_3: 3"


def test_docx_simple_table_with_header_row() -> None:
    tbl = (
        "<w:tbl>"
        + _tr(_tc("Date") + _tc("Event") + _tc("Location"))
        + _tr(_tc("2026-06-15") + _tc("Safety Drill") + _tc("East Wing"))
        + _tr(_tc("2026-07-01") + _tc("Audit") + _tc("HQ"))
        + "</w:tbl>"
    )
    raw = _build_docx_bytes(_wrap_document_xml(_docx_p("Schedule") + tbl))
    text, status, err = load_docx(raw)
    assert (status, err) == ("ok", None)
    assert text == (
        "Schedule\n\n"
        "Date: 2026-06-15; Event: Safety Drill; Location: East Wing\n"
        "Date: 2026-07-01; Event: Audit; Location: HQ"
    )


def test_docx_table_without_header_when_first_row_has_empty_cell() -> None:
    tbl = (
        "<w:tbl>"
        + _tr(_tc("") + _tc("Q1") + _tc("Q2"))
        + _tr(_tc("Revenue") + _tc("100") + _tc("120"))
        + "</w:tbl>"
    )
    raw = _build_docx_bytes(_wrap_document_xml(tbl))
    text, status, err = load_docx(raw)
    assert (status, err) == ("ok", None)
    assert text == ("col_2: Q1; col_3: Q2\ncol_1: Revenue; col_2: 100; col_3: 120")


def test_docx_table_grid_before_grid_span_and_vmerge() -> None:
    tbl = (
        "<w:tbl>"
        + _tr(_tc("Region") + _tc("Quarter") + _tc("Units"), tbl_header=True)
        + _tr(_tc("North", vmerge="restart") + _tc("Q1") + _tc("10"))
        + _tr(_tc("", vmerge="continue") + _tc("Q2") + _tc("20"))
        + _tr(_tc("Summary", grid_span=2) + _tc("30"))
        + _tr(_tc("OnlyCol3"), grid_before=2)
        + "</w:tbl>"
    )
    raw = _build_docx_bytes(_wrap_document_xml(tbl))
    text, status, err = load_docx(raw)
    assert (status, err) == ("ok", None)
    assert text == (
        "Region: North; Quarter: Q1; Units: 10\n"
        "Region: North; Quarter: Q2; Units: 20\n"
        "Region: Summary; Units: 30\n"
        "Units: OnlyCol3"
    )


def test_docx_two_row_tbl_header_and_four_row_forward_fill() -> None:
    two_hdr_tbl = (
        "<w:tbl>"
        + _tr(_tc("Dept", vmerge="restart") + _tc("2026", grid_span=2), tbl_header=True)
        + _tr(_tc("", vmerge="continue") + _tc("Rev") + _tc("Cost"), tbl_header=True)
        + _tr(_tc("Ops") + _tc("500") + _tc("200"))
        + "</w:tbl>"
    )
    raw_two = _build_docx_bytes(_wrap_document_xml(two_hdr_tbl))
    text_two, status_two, err_two = load_docx(raw_two)
    assert (status_two, err_two) == ("ok", None)
    assert text_two == "Dept: Ops; 2026 / Rev: 500; 2026 / Cost: 200"

    four_hdr_tbl = (
        "<w:tbl>"
        + _tr(_tc("Annual") + _tc(""), tbl_header=True)
        + _tr(_tc("EMEA") + _tc("APAC"), tbl_header=True)
        + _tr(_tc("H1") + _tc("H2"), tbl_header=True)
        + _tr(_tc("Actual") + _tc("Target"), tbl_header=True)
        + _tr(_tc("42") + _tc("55"))
        + "</w:tbl>"
    )
    raw_four = _build_docx_bytes(_wrap_document_xml(four_hdr_tbl))
    text_four, status_four, err_four = load_docx(raw_four)
    assert (status_four, err_four) == ("ok", None)
    assert text_four == ("Annual / EMEA / H1 / Actual: 42; Annual / APAC / H2 / Target: 55")


def test_docx_1x1_layout_table_unwrapped_in_reading_order() -> None:
    body = (
        _docx_p("Before table.")
        + "<w:tbl>"
        + _tr(_tc("Inside layout cell.") + "")
        + "</w:tbl>"
        + "<w:tbl>"
        + _tr(
            "<w:tc>"
            + _docx_p("Layout Title", style="Heading1")
            + _docx_p("Layout body.")
            + "</w:tc>"
        )
        + "</w:tbl>"
        + _docx_p("After table.")
    )
    raw = _build_docx_bytes(_wrap_document_xml(body))
    text, status, err = load_docx(raw)
    assert (status, err) == ("ok", None)
    assert text == (
        "Before table.\n\nInside layout cell.\n\n# Layout Title\n\nLayout body.\n\nAfter table."
    )


def test_docx_tracked_deletions_skipped_insertions_kept_and_field_code_ignored() -> None:
    body = (
        "<w:p>"
        "<w:r><w:t>Keep </w:t></w:r>"
        "<w:del><w:r><w:delText>OLD_DELETED</w:delText><w:t>ALSO_DELETED</w:t></w:r></w:del>"
        "<w:ins><w:r><w:t>NEW_INSERTED</w:t></w:r></w:ins>"
        "<w:r><w:instrText>MERGEFIELD Secret</w:instrText></w:r>"
        "<w:r><w:fldData>HIDDEN_FLD</w:fldData></w:r>"
        "</w:p>"
        "<w:del>" + _docx_p("ENTIRE_DELETED_PARAGRAPH") + "</w:del>"
        "<w:ins>" + _docx_p("Inserted paragraph.") + "</w:ins>"
    )
    raw = _build_docx_bytes(_wrap_document_xml(body))
    text, status, err = load_docx(raw)
    assert (status, err) == ("ok", None)
    assert "OLD_DELETED" not in text
    assert "ALSO_DELETED" not in text
    assert "ENTIRE_DELETED_PARAGRAPH" not in text
    assert "MERGEFIELD" not in text
    assert "HIDDEN_FLD" not in text
    assert text == "Keep NEW_INSERTED\n\nInserted paragraph."


def test_docx_alternate_content_reads_choice_else_fallback_never_both() -> None:
    both = (
        f'<mc:AlternateContent xmlns:mc="{_MC_NS}">'
        "<mc:Choice>" + _docx_p("Choice text only.") + "</mc:Choice>"
        "<mc:Fallback>" + _docx_p("Fallback must be skipped.") + "</mc:Fallback>"
        "</mc:AlternateContent>"
    )
    fallback_only = (
        f'<mc:AlternateContent xmlns:mc="{_MC_NS}">'
        "<mc:Fallback>" + _docx_p("Fallback when no choice.") + "</mc:Fallback>"
        "</mc:AlternateContent>"
    )
    inline_ac = (
        "<w:p>"
        "<w:r><w:t>Inline </w:t></w:r>"
        f'<mc:AlternateContent xmlns:mc="{_MC_NS}">'
        "<mc:Choice><w:r><w:t>A</w:t></w:r></mc:Choice>"
        "<mc:Fallback><w:r><w:t>B</w:t></w:r></mc:Fallback>"
        "</mc:AlternateContent>"
        "</w:p>"
    )
    raw = _build_docx_bytes(_wrap_document_xml(both + fallback_only + inline_ac))
    text, status, err = load_docx(raw)
    assert (status, err) == ("ok", None)
    assert "Fallback must be skipped." not in text
    assert text == "Choice text only.\n\nFallback when no choice.\n\nInline A"


def test_docx_heading_levels_from_outline_lvl_and_styles_xml() -> None:
    styles_xml = _wrap_styles_xml(
        '<w:style w:type="paragraph" w:styleId="CustomH2">'
        '<w:name w:val="Custom Section"/>'
        '<w:pPr><w:outlineLvl w:val="1"/></w:pPr>'
        "</w:style>"
        '<w:style w:type="paragraph" w:styleId="DocTitle">'
        '<w:name w:val="Title"/>'
        "</w:style>"
        '<w:style w:type="paragraph" w:styleId="DocSubtitle">'
        '<w:name w:val="Subtitle"/>'
        "</w:style>"
        '<w:style w:type="paragraph" w:styleId="InheritedH3">'
        '<w:basedOn w:val="Heading3"/>'
        "</w:style>"
        '<w:style w:type="paragraph" w:styleId="Heading3">'
        '<w:name w:val="heading 3"/>'
        '<w:pPr><w:outlineLvl w:val="2"/></w:pPr>'
        "</w:style>"
    )
    body = (
        _docx_p("Main Title", style="DocTitle")
        + _docx_p("Main Subtitle", style="DocSubtitle")
        + _docx_p("From Style Outline", style="CustomH2")
        + _docx_p("Inherited Heading", style="InheritedH3")
        + '<w:p><w:pPr><w:outlineLvl w:val="7"/></w:pPr><w:r><w:t>Capped Six</w:t></w:r></w:p>'
        + '<w:p><w:pPr><w:pStyle w:val="CustomH2"/><w:outlineLvl w:val="9"/></w:pPr>'
        "<w:r><w:t>Overridden to body</w:t></w:r></w:p>"
    )
    raw = _build_docx_bytes(
        _wrap_document_xml(body),
        extra_members=[("word/styles.xml", styles_xml)],
    )
    text, status, err = load_docx(raw)
    assert (status, err) == ("ok", None)
    assert text == (
        "# Main Title\n\n"
        "## Main Subtitle\n\n"
        "## From Style Outline\n\n"
        "### Inherited Heading\n\n"
        "###### Capped Six\n\n"
        "Overridden to body"
    )


def test_docx_footnotes_reference_and_body() -> None:
    footnotes_xml = _wrap_footnotes_xml(
        '<w:footnote w:type="separator" w:id="-1">' + _docx_p("Separator") + "</w:footnote>"
        '<w:footnote w:type="continuationSeparator" w:id="0">' + _docx_p("Cont") + "</w:footnote>"
        '<w:footnote w:id="1">' + _docx_p("First footnote text.") + "</w:footnote>"
        '<w:footnote w:id="2">'
        "<w:p><w:r><w:t>Second footnote</w:t></w:r>"
        "<w:del><w:r><w:delText> deleted</w:delText></w:r></w:del></w:p>"
        "</w:footnote>"
    )
    body = (
        "<w:p><w:r><w:t>Policy applies immediately.</w:t></w:r>"
        '<w:r><w:footnoteReference w:id="1"/></w:r></w:p>'
        "<w:tbl>"
        + _tr(_tc("Item") + _tc("Rule"))
        + _tr(
            _tc("Leave") + "<w:tc><w:p><w:r><w:t>15 days</w:t></w:r>"
            '<w:r><w:footnoteReference w:id="2"/></w:r></w:p></w:tc>'
        )
        + "</w:tbl>"
    )
    raw = _build_docx_bytes(
        _wrap_document_xml(body),
        extra_members=[("word/footnotes.xml", footnotes_xml)],
    )
    text, status, err = load_docx(raw)
    assert (status, err) == ("ok", None)
    assert text == (
        "Policy applies immediately. [fn 1]\n\n"
        "Item: Leave; Rule: 15 days [fn 2]\n\n"
        "[fn 1] First footnote text.\n"
        "[fn 2] Second footnote"
    )


def test_docx_plain_document_and_handbook_output_unchanged(tmp_path: Path) -> None:
    plain_raw = _build_docx_bytes(
        _wrap_document_xml(
            _docx_p("Section One", style="Heading1")
            + "<w:p><w:r><w:t>First paragraph</w:t><w:tab/><w:t>with tab.</w:t></w:r></w:p>"
            + _docx_p("Sub", style="Heading2")
            + "<w:p><w:r><w:t>Line A</w:t><w:br/><w:t>Line B</w:t></w:r></w:p>"
        )
    )
    plain_text, status, err = load_docx(plain_raw)
    assert (status, err) == ("ok", None)
    assert plain_text == "# Section One\n\nFirst paragraph with tab.\n\n## Sub\n\nLine A\nLine B"

    handbook_bytes = (CORPUS_DIR / "handbook.docx").read_bytes()
    hb_text, hb_status, hb_err = load_docx(handbook_bytes)
    assert (hb_status, hb_err) == ("ok", None)
    assert "# Onboarding checklist" in hb_text
    assert "New hires receive a laptop and an access badge on their first day." in hb_text
    assert "The IT desk is on the 4th floor in room 412." in hb_text
    assert "# Equipment return" in hb_text
    assert "Contact facilities at extension 2207 to arrange the pickup." in hb_text

    (tmp_path / "handbook.docx").write_bytes(handbook_bytes)
    idx = DocIndex(tmp_path)
    idx.refresh()
    hb_chunks = [c for c in idx.chunks if c.source == "handbook.docx"]
    assert [(c.id, c.section) for c in hb_chunks] == [
        ("handbook.docx#0", "Onboarding checklist"),
        ("handbook.docx#1", "Equipment return"),
    ]


def test_docx_styles_and_footnotes_reject_doctype_when_referenced() -> None:
    bad_styles = b'<?xml version="1.0"?><!DOCTYPE styles><w:styles/>'
    raw_styles = _build_docx_bytes(
        _wrap_document_xml(_docx_p("Heading", style="Heading1")),
        extra_members=[("word/styles.xml", bad_styles)],
    )
    assert load_docx(raw_styles) == (
        "",
        "unreadable",
        "docx XML contains forbidden DOCTYPE or ENTITY declaration",
    )

    bad_fn = b'<?xml version="1.0"?><!DOCTYPE footnotes><w:footnotes/>'
    raw_fn = _build_docx_bytes(
        _wrap_document_xml('<w:p><w:r><w:t>Ref</w:t><w:footnoteReference w:id="1"/></w:r></w:p>'),
        extra_members=[("word/footnotes.xml", bad_fn)],
    )
    assert load_docx(raw_fn) == (
        "",
        "unreadable",
        "docx XML contains forbidden DOCTYPE or ENTITY declaration",
    )


def test_docx_30000_cell_table_capped_under_5_seconds() -> None:
    # 300 rows x 100 columns = 30,000 cells, exceeding the 20,000-cell document cap.
    header_cells = "".join(
        f"<w:tc><w:p><w:r><w:t>H{c}</w:t></w:r></w:p></w:tc>" for c in range(100)
    )
    header_row = f"<w:tr>{header_cells}</w:tr>"
    data_cells = "".join(f"<w:tc><w:p><w:r><w:t>{c}</w:t></w:r></w:p></w:tc>" for c in range(100))
    data_row = f"<w:tr>{data_cells}</w:tr>"
    tbl = "<w:tbl>" + header_row + (data_row * 299) + "</w:tbl>"
    raw = _build_docx_bytes(_wrap_document_xml(tbl), compression=zipfile.ZIP_DEFLATED)

    t0 = time.perf_counter()
    text, status, err = load_docx(raw)
    elapsed = time.perf_counter() - t0

    assert (status, err) == ("ok", None)
    assert elapsed < 5.0
    lines = text.splitlines()
    # 20,000 cells / 100 cols = 200 rows (1 header + 199 data rows), truncated cleanly.
    assert len(lines) == 199
    assert lines[0].startswith("H0: 0; H1: 1;")
