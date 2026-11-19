"""Tests for PDF and DOCX conversion in ``voice_agent.rag.convert``."""

from __future__ import annotations

import copy
import io
import json
import subprocess
import sys
import warnings
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pypdf
import pytest
from test_loaders_binary import (
    _build_docx_bytes,
    _build_encrypted_pdf_bytes,
    _build_pdf_bytes,
    _docx_p,
    _wrap_document_xml,
)

from voice_agent.rag import canonical, docindex, zipguard
from voice_agent.rag.convert import (
    ConvertResult,
    convert_isolated,
    convert_pdf,
    convert_with_markitdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "rag_corpus"


def _build_flate_bomb_pdf(uncompressed_bytes: int = 50 * 1024 * 1024) -> bytes:
    """Build a valid PDF <= 20 MB whose FlateDecode content stream expands to many MB."""
    base_ops = b"BT\n/F1 12 Tf\n72 720 Td\n(Flate bomb probe page one.) Tj\nET\n"
    pad_len = max(0, uncompressed_bytes - len(base_ops) - 2)
    payload = base_ops + b"% " + (b"A" * pad_len)
    compressed = zlib.compress(payload, level=9)

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>",
        (
            f"<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode("ascii")
            + compressed
            + b"\nendstream"
        ),
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = [0]
    for obj_id, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        out.extend(body)
        out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    out.extend(trailer.encode("ascii"))
    return bytes(out)


def test_handbook_pdf_and_synthetic_4page_pdf_match_pypdf_page_markers() -> None:
    markitdown = pytest.importorskip("markitdown")

    handbook_bytes = (CORPUS_DIR / "handbook.pdf").read_bytes()
    pypdf_handbook, status, err = docindex.load_pdf_pages(handbook_bytes)
    assert status == "ok" and err is None

    res_handbook = convert_with_markitdown(handbook_bytes, "pdf")
    assert res_handbook.status == "ok"
    assert res_handbook.error is None
    assert res_handbook.fallback is None
    assert res_handbook.converter == f"markitdown-{markitdown.__version__}"
    assert "\f" in res_handbook.raw_text

    units_hb, _ = canonical.split_units(res_handbook.text)
    assert [u.page for u in units_hb] == [p for p, _ in pypdf_handbook]
    assert [u.page for u in units_hb] == [1, 2]

    res_hb_pypdf = convert_pdf(handbook_bytes, mode="pypdf")
    assert res_hb_pypdf.status == "ok"
    assert res_hb_pypdf.fallback is None
    assert res_hb_pypdf.converter == f"pypdf-{pypdf.__version__}"
    units_hb_pypdf, _ = canonical.split_units(res_hb_pypdf.text)
    assert [u.page for u in units_hb_pypdf] == [1, 2]

    synthetic_bytes = _build_pdf_bytes(
        [
            "ACME FIELD MANUAL\nPage one sentence about calibration.",
            "ACME FIELD MANUAL\nPage two sentence about battery checks.",
            "ACME FIELD MANUAL\nPage three sentence about radio range.",
            "ACME FIELD MANUAL\nPage four sentence about shutdown steps.",
        ]
    )
    pypdf_synth, s_status, s_err = docindex.load_pdf_pages(synthetic_bytes)
    assert s_status == "ok" and s_err is None

    res_synth = convert_with_markitdown(synthetic_bytes, "pdf")
    assert res_synth.status == "ok"
    assert res_synth.error is None
    assert res_synth.fallback is None
    assert res_synth.converter == f"markitdown-{markitdown.__version__}"
    assert "\f" in res_synth.raw_text

    units_synth, _ = canonical.split_units(res_synth.text)
    assert [u.page for u in units_synth] == [p for p, _ in pypdf_synth]
    assert [u.page for u in units_synth] == [1, 2, 3, 4]
    assert "ACME FIELD MANUAL" not in res_synth.text

    res_synth_pypdf = convert_pdf(synthetic_bytes, "pypdf")
    assert res_synth_pypdf.status == "ok"
    assert res_synth_pypdf.fallback is None
    assert res_synth_pypdf.converter == f"pypdf-{pypdf.__version__}"
    units_synth_pypdf, _ = canonical.split_units(res_synth_pypdf.text)
    assert [u.page for u in units_synth_pypdf] == [1, 2, 3, 4]


def test_pdf_forced_no_formfeed_falls_back_to_pypdf_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("markitdown")
    import markitdown.converters as md_converters

    class _NoFormFeedPdfConverter:
        def convert(self, stream: Any, stream_info: Any) -> Any:
            return SimpleNamespace(markdown="Joined page one\n\nJoined page two without breaks.")

    monkeypatch.setattr(md_converters, "PdfConverter", _NoFormFeedPdfConverter)

    pdf_bytes = _build_pdf_bytes(
        [
            "Alpha first page unique token 101.",
            "Beta second page unique token 202.",
        ]
    )
    res = convert_with_markitdown(pdf_bytes, "pdf")
    assert res.status == "ok"
    assert res.error is None
    assert res.fallback == "pypdf-pages"
    assert res.converter == f"pypdf-{pypdf.__version__}"

    units, _ = canonical.split_units(res.text)
    assert [u.page for u in units] == [1, 2]
    assert "Alpha first page unique token 101." in units[0].body
    assert "Beta second page unique token 202." in units[1].body
    assert "Joined page one" not in res.text

    # 1-page PDF without \f must also trigger pypdf-pages fallback.
    one_page_bytes = _build_pdf_bytes(["Single page unique token 303."])
    res_one = convert_pdf(one_page_bytes, mode="markitdown")
    assert res_one.status == "ok"
    assert res_one.fallback == "pypdf-pages"
    assert res_one.converter == f"pypdf-{pypdf.__version__}"
    units_one, _ = canonical.split_units(res_one.text)
    assert [u.page for u in units_one] == [1]
    assert "Single page unique token 303." in units_one[0].body


def test_pdf_mismatched_formfeed_count_falls_back_to_pypdf_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("markitdown")
    import markitdown.converters as md_converters

    class _ShortFormFeedPdfConverter:
        def convert(self, stream: Any, stream_info: Any) -> Any:
            return SimpleNamespace(markdown="Only one break\fTwo chunks")

    monkeypatch.setattr(md_converters, "PdfConverter", _ShortFormFeedPdfConverter)

    pdf_bytes = _build_pdf_bytes(["Alpha page one.", "Beta page two.", "Gamma page three."])
    res = convert_with_markitdown(pdf_bytes, "pdf")
    assert res.status == "ok"
    assert res.fallback == "pypdf-pages"
    assert res.converter == f"pypdf-{pypdf.__version__}"
    units, _ = canonical.split_units(res.text)
    assert [u.page for u in units] == [1, 2, 3]


def test_pdf_guard_cases_match_legacy_status_without_calling_markitdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("markitdown")
    import markitdown.converters as md_converters

    def _forbid(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("PdfConverter must not be called on guard failures")

    monkeypatch.setattr(md_converters, "PdfConverter", _forbid)

    cases = [
        ("encrypted", _build_encrypted_pdf_bytes("Confidential text inside encrypted pdf.")),
        ("too_many_pages", _build_pdf_bytes([""] * 501)),
        ("no_text", _build_pdf_bytes(["", ""])),
        ("corrupt", b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n"),
    ]
    for label, raw in cases:
        _, legacy_status, legacy_err = docindex.load_pdf_pages(raw)
        res = convert_with_markitdown(raw, "pdf")
        assert (res.status, res.error) == (legacy_status, legacy_err), label
        assert res.text == ""
        assert res.fallback is None


def test_handbook_docx_headings_match_legacy_sections() -> None:
    markitdown = pytest.importorskip("markitdown")
    handbook_bytes = (CORPUS_DIR / "handbook.docx").read_bytes()

    legacy_text, legacy_status, legacy_err = docindex.load_docx(handbook_bytes)
    assert legacy_status == "ok" and legacy_err is None
    legacy_sections = [
        sec for sec, _ in docindex.split_markdown_sections(legacy_text, max_heading_level=9) if sec
    ]

    res = convert_with_markitdown(handbook_bytes, "docx")
    assert res.status == "ok"
    assert res.error is None
    assert res.fallback is None
    assert res.converter == f"markitdown-{markitdown.__version__}"

    md_headings = [
        line.lstrip("#").strip() for line in res.text.splitlines() if line.startswith("#")
    ]
    assert md_headings == legacy_sections
    assert md_headings == ["Onboarding checklist", "Equipment return"]


def test_docx_guard_cases_match_legacy_without_calling_markitdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("markitdown")
    import markitdown.converters as md_converters

    def _forbid(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("DocxConverter must not be called on guard failures")

    monkeypatch.setattr(md_converters, "DocxConverter", _forbid)

    ratio_bomb = _build_docx_bytes(
        _wrap_document_xml(_docx_p("Normal text paragraph.")),
        extra_members=[("word/bomb.txt", b"A" * 50_000)],
        compression=zipfile.ZIP_DEFLATED,
    )
    many_members = _build_docx_bytes(
        _wrap_document_xml(_docx_p("Too many members.")),
        extra_members=[(f"extra/m_{i:04d}.txt", b"x") for i in range(1000)],
        compression=zipfile.ZIP_STORED,
    )
    doctype_doc = _build_docx_bytes(
        _wrap_document_xml(
            _docx_p("Entity expansion test."),
            prefix='<!DOCTYPE foo [<!ENTITY xxe "boom">]>',
        )
    )
    entity_doc = _build_docx_bytes(
        _wrap_document_xml(
            _docx_p("Standalone entity test."),
            prefix='<!ENTITY xxe "boom">',
        )
    )
    deep_open = "".join(
        f'<w:tbl w:id="t_{i}"><w:tr w:r="r_{i}"><w:tc w:c="c_{i}">' for i in range(120)
    )
    deep_doc = _build_docx_bytes(
        _wrap_document_xml(deep_open + _docx_p("Deep text.") + ("</w:tc></w:tr></w:tbl>" * 120))
    )

    dup_buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(dup_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
            zf.writestr("word/document.xml", _wrap_document_xml(_docx_p("First copy.")))
            zf.writestr("word/document.xml", _wrap_document_xml(_docx_p("Second copy.")))

    overlap_buf = io.BytesIO()
    with zipfile.ZipFile(overlap_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
        zf.writestr("word/document.xml", _wrap_document_xml(_docx_p("Overlap test.")))
        orig_info = zf.getinfo("word/document.xml")
        dup_info = copy.copy(orig_info)
        dup_info.filename = "word/overlap_0.xml"
        zf.filelist.append(dup_info)

    guard_cases = [
        ("ratio_bomb", ratio_bomb),
        ("many_members", many_members),
        ("doctype_doc", doctype_doc),
        ("entity_doc", entity_doc),
        ("deep_doc", deep_doc),
        ("duplicate_member", dup_buf.getvalue()),
        ("overlap_entry", overlap_buf.getvalue()),
        ("not_a_zip", b"plain text disguised as a docx file"),
        ("missing_document_xml", _build_docx_bytes(None)),
        ("empty_doc", _build_docx_bytes(_wrap_document_xml(""))),
    ]

    for label, raw in guard_cases:
        _, legacy_status, legacy_err = docindex.load_docx(raw)
        res = convert_with_markitdown(raw, "docx")
        assert (res.status, res.error) == (legacy_status, legacy_err), label
        assert res.text == ""
        assert res.fallback is None

    # Member-size and total-size cap guards via monkeypatched docindex limits.
    oversize = _build_docx_bytes(
        _wrap_document_xml(_docx_p("Oversized uncompressed member content " + ("x" * 1200))),
        compression=zipfile.ZIP_STORED,
    )
    monkeypatch.setattr(docindex, "MAX_DOCX_MEMBER_BYTES", 500)
    _, ov_status, ov_err = docindex.load_docx(oversize)
    ov_res = convert_with_markitdown(oversize, "docx")
    assert (ov_res.status, ov_res.error) == (ov_status, ov_err)
    assert ov_res.fallback is None

    total_cap = _build_docx_bytes(
        _wrap_document_xml(_docx_p("Short document body.")),
        extra_members=[(f"extra/part_{i}.txt", b"X" * 450) for i in range(5)],
        compression=zipfile.ZIP_STORED,
    )
    monkeypatch.setattr(docindex, "MAX_DOCX_TOTAL_BYTES", 2000)
    _, tc_status, tc_err = docindex.load_docx(total_cap)
    tc_res = convert_with_markitdown(total_cap, "docx")
    assert (tc_res.status, tc_res.error) == (tc_status, tc_err)
    assert tc_res.fallback is None


def test_docx_doctype_in_styles_xml_rejected_without_calling_markitdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("markitdown")
    import markitdown.converters as md_converters

    def _forbid(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("DocxConverter must not be called when styles.xml has DOCTYPE")

    monkeypatch.setattr(md_converters, "DocxConverter", _forbid)

    raw = _build_docx_bytes(
        _wrap_document_xml(_docx_p("Clean main document body.")),
        extra_members=[
            (
                "word/styles.xml",
                b'<?xml version="1.0"?><!DOCTYPE styles [<!ENTITY xxe "boom">]><styles/>',
            )
        ],
    )
    assert docindex.load_docx(raw)[1] == "ok"
    assert zipguard.preflight_zip(raw, label="docx", scan_xml_members=True) is not None

    res = convert_with_markitdown(raw, "docx")
    assert res.status == "unreadable"
    assert res.error == "docx XML contains forbidden DOCTYPE or ENTITY declaration"
    assert res.text == ""
    assert res.fallback is None


def test_converter_exception_falls_back_to_legacy_for_pdf_and_docx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("markitdown")
    import markitdown.converters as md_converters

    class _ExplodingConverter:
        def convert(self, stream: Any, stream_info: Any) -> Any:
            raise RuntimeError("simulated converter crash")

    monkeypatch.setattr(md_converters, "PdfConverter", _ExplodingConverter)
    monkeypatch.setattr(md_converters, "DocxConverter", _ExplodingConverter)

    pdf_bytes = _build_pdf_bytes(["Legacy PDF page one.", "Legacy PDF page two."])
    pdf_res = convert_with_markitdown(pdf_bytes, "pdf")
    assert pdf_res.status == "ok"
    assert pdf_res.error is None
    assert pdf_res.fallback == "legacy"
    assert pdf_res.converter == f"pypdf-{pypdf.__version__}"
    pdf_units, _ = canonical.split_units(pdf_res.text)
    assert [u.page for u in pdf_units] == [1, 2]
    assert "Legacy PDF page one." in pdf_units[0].body
    assert "Legacy PDF page two." in pdf_units[1].body

    docx_bytes = _build_docx_bytes(
        _wrap_document_xml(
            _docx_p("Operations", style="Heading1") + _docx_p("Legacy docx fallback paragraph.")
        )
    )
    docx_res = convert_with_markitdown(docx_bytes, "docx")
    assert docx_res.status == "ok"
    assert docx_res.error is None
    assert docx_res.fallback == "legacy"
    assert docx_res.converter == "legacy"
    assert "# Operations" in docx_res.text
    assert "Legacy docx fallback paragraph." in docx_res.text


def test_convert_pdf_invalid_mode_and_blocked_markitdown_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_bytes = _build_pdf_bytes(["Single page body."])
    with pytest.raises(ValueError, match="unsupported pdf conversion mode"):
        convert_pdf(pdf_bytes, mode="bogus")

    monkeypatch.setitem(sys.modules, "markitdown", None)
    monkeypatch.setitem(sys.modules, "markitdown.converters", None)

    res_pdf = convert_with_markitdown(pdf_bytes, "pdf")
    assert res_pdf == ConvertResult(
        text="",
        status="unsupported",
        error="markitdown not installed",
    )

    docx_bytes = _build_docx_bytes(_wrap_document_xml(_docx_p("Valid docx body.")))
    res_docx = convert_with_markitdown(docx_bytes, "docx")
    assert res_docx == ConvertResult(
        text="",
        status="unsupported",
        error="markitdown not installed",
    )

    # pypdf mode does not need markitdown at all.
    res_pypdf = convert_pdf(pdf_bytes, mode="pypdf")
    assert res_pypdf.status == "ok"
    assert res_pypdf.fallback is None


def test_flate_bomb_pdf_subprocess_within_rss_and_time_limits() -> None:
    pytest.importorskip("markitdown")
    bomb_10mb = _build_flate_bomb_pdf(10 * 1024 * 1024)
    assert len(bomb_10mb) <= docindex.MAX_FILE_BYTES

    script = (
        "import json, resource, sys, time\n"
        "from voice_agent.rag.convert import convert_with_markitdown\n"
        "limit = 4096 * 1024 * 1024\n"
        "soft, hard = resource.getrlimit(resource.RLIMIT_AS)\n"
        "target = limit if hard == resource.RLIM_INFINITY else min(limit, hard)\n"
        "resource.setrlimit(resource.RLIMIT_AS, (target, target))\n"
        "raw = sys.stdin.buffer.read()\n"
        "t0 = time.perf_counter()\n"
        "res = convert_with_markitdown(raw, 'pdf')\n"
        "elapsed = time.perf_counter() - t0\n"
        "peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "print(json.dumps({'status': res.status, 'fallback': res.fallback, "
        "'elapsed_s': round(elapsed, 4), 'peak_rss_mb': round(peak_kb / 1024.0, 2)}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", script],
        input=bomb_10mb,
        capture_output=True,
        timeout=35,
    )
    assert proc.returncode == 0, (
        f"rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    metrics = json.loads(proc.stdout.decode("utf-8"))
    print(f"FLATE_BOMB_10MB_METRICS={metrics} raw_bytes={len(bomb_10mb)}")
    assert metrics["elapsed_s"] < 30.0
    assert metrics["peak_rss_mb"] < 1536.0

    # Larger 50 MB Flate-bomb exceeds 30s in-process; isolated worker enforces timeout.
    bomb_50mb = _build_flate_bomb_pdf(50 * 1024 * 1024)
    assert len(bomb_50mb) <= docindex.MAX_FILE_BYTES
    iso_res = convert_isolated(bomb_50mb, "pdf", timeout_s=2.0, max_address_space_mb=4096)
    assert iso_res.status == "unreadable"
    assert iso_res.error is not None
    assert "timed out" in iso_res.error
