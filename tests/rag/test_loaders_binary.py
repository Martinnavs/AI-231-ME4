"""Offline unit tests for PDF (pypdf) and DOCX (stdlib zip/XML) loaders in DocIndex."""

from __future__ import annotations

import copy
import io
import os
import sys
import warnings
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pypdf
import pytest

from voice_agent.rag import Chunk, DocIndex
from voice_agent.rag import docindex as docindex_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "rag_corpus"

# Pre-T5 baseline ids and headers for the 10 T4 md/txt/csv corpus files.
T4_EXPECTED_IDS_AND_HEADERS: dict[str, list[tuple[str, str]]] = {
    "budget_2026.md": [
        ("budget_2026.md#0", "budget_2026.md > Household Renovation Budget 2026"),
        (
            "budget_2026.md#1",
            "budget_2026.md > Household Renovation Budget 2026 > Approved Total",
        ),
        (
            "budget_2026.md#2",
            "budget_2026.md > Household Renovation Budget 2026 > Category Breakdown",
        ),
        (
            "budget_2026.md#3",
            "budget_2026.md > Household Renovation Budget 2026 > Payment Milestones",
        ),
    ],
    "budget_draft_notes.txt": [
        ("budget_draft_notes.txt#0", "budget_draft_notes.txt"),
    ],
    "clinic_appointments.txt": [
        ("clinic_appointments.txt#0", "clinic_appointments.txt"),
        ("clinic_appointments.txt#1", "clinic_appointments.txt"),
    ],
    "errands_taglish.txt": [
        ("errands_taglish.txt#0", "errands_taglish.txt"),
    ],
    "field_operations_manual.md": [
        (
            "field_operations_manual.md#0",
            "field_operations_manual.md > Field Sensor Operations Manual",
        ),
        (
            "field_operations_manual.md#1",
            "field_operations_manual.md > Field Sensor Operations Manual "
            "> Hardware Enclosure and Power",
        ),
        (
            "field_operations_manual.md#2",
            "field_operations_manual.md > Field Sensor Operations Manual "
            "> Hardware Enclosure and Power",
        ),
        (
            "field_operations_manual.md#3",
            "field_operations_manual.md > Field Sensor Operations Manual > Soil Probe Calibration",
        ),
        (
            "field_operations_manual.md#4",
            "field_operations_manual.md > Field Sensor Operations Manual > Soil Probe Calibration",
        ),
        (
            "field_operations_manual.md#5",
            "field_operations_manual.md > Field Sensor Operations Manual > Radio Mesh Diagnostics",
        ),
        (
            "field_operations_manual.md#6",
            "field_operations_manual.md > Field Sensor Operations Manual > Radio Mesh Diagnostics",
        ),
        (
            "field_operations_manual.md#7",
            "field_operations_manual.md > Field Sensor Operations Manual "
            "> Fault Codes and Recovery",
        ),
        (
            "field_operations_manual.md#8",
            "field_operations_manual.md > Field Sensor Operations Manual "
            "> Fault Codes and Recovery",
        ),
        (
            "field_operations_manual.md#9",
            "field_operations_manual.md > Field Sensor Operations Manual "
            "> Seasonal Preventive Maintenance",
        ),
    ],
    "garden_compost_log.md": [
        ("garden_compost_log.md#0", "garden_compost_log.md > Backyard Garden and Compost Log"),
        (
            "garden_compost_log.md#1",
            "garden_compost_log.md > Backyard Garden and Compost Log > Watering Schedule",
        ),
        (
            "garden_compost_log.md#2",
            "garden_compost_log.md > Backyard Garden and Compost Log > Compost Bay Turnover",
        ),
        (
            "garden_compost_log.md#3",
            "garden_compost_log.md > Backyard Garden and Compost Log > Soil pH and Harvest Notes",
        ),
    ],
    "home_inventory.md": [
        (
            "home_inventory.md#0",
            "home_inventory.md > Home Workshop and Backup Power Inventory",
        ),
        (
            "home_inventory.md#1",
            "home_inventory.md > Home Workshop and Backup Power Inventory "
            "> Uninterruptible Power Supply",
        ),
        (
            "home_inventory.md#2",
            "home_inventory.md > Home Workshop and Backup Power Inventory "
            "> Portable Air Compressor",
        ),
        (
            "home_inventory.md#3",
            "home_inventory.md > Home Workshop and Backup Power Inventory "
            "> Torque Wrench and Calibration",
        ),
    ],
    "router_reset.md": [
        ("router_reset.md#0", "router_reset.md > Router Reset Guide"),
        ("router_reset.md#1", "router_reset.md > Router Reset Guide > Soft Reboot"),
        ("router_reset.md#2", "router_reset.md > Router Reset Guide > Factory Reset"),
        (
            "router_reset.md#3",
            "router_reset.md > Router Reset Guide > Post-Reset Wi-Fi Defaults",
        ),
    ],
    "shift_schedule.csv": [
        ("shift_schedule.csv#0", "shift_schedule.csv"),
        ("shift_schedule.csv#1", "shift_schedule.csv"),
        ("shift_schedule.csv#2", "shift_schedule.csv"),
        ("shift_schedule.csv#3", "shift_schedule.csv"),
    ],
    "untrusted_memo.txt": [
        ("untrusted_memo.txt#0", "untrusted_memo.txt"),
    ],
}

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_pdf_bytes(page_texts: Sequence[str]) -> bytes:
    """Build a minimal valid PDF 1.4 byte string with one content stream per page."""
    objects: list[bytes] = []
    num_pages = len(page_texts)
    page_ids = [4 + 2 * i for i in range(num_pages)]
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>".encode("ascii"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for i, page_text in enumerate(page_texts):
        page_id = 4 + 2 * i
        content_id = page_id + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode("ascii")
        )
        if page_text:
            lines = [ln for ln in page_text.splitlines() if ln]
            ops = ["BT", "/F1 12 Tf", "72 720 Td"]
            for idx, ln in enumerate(lines):
                if idx > 0:
                    ops.append("0 -16 Td")
                ops.append(f"({_pdf_escape(ln)}) Tj")
            ops.append("ET")
            stream = "\n".join(ops).encode("latin-1")
        else:
            stream = b""
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream"
        )

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


def _build_encrypted_pdf_bytes(text: str, password: str = "secret") -> bytes:
    """Create an encrypted single-page PDF using pypdf."""
    base_bytes = _build_pdf_bytes([text])
    reader = pypdf.PdfReader(io.BytesIO(base_bytes))
    writer = pypdf.PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.encrypt(password)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _wrap_document_xml(inner_body_xml: str, *, prefix: str = "") -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f"{prefix}"
        f'<w:document xmlns:w="{_W_NS}">'
        f"<w:body>{inner_body_xml}</w:body>"
        "</w:document>"
    ).encode()


def _docx_p(text: str, *, style: str | None = None) -> str:
    p_pr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{p_pr}<w:r><w:t>{text}</w:t></w:r></w:p>"


def _build_docx_bytes(
    document_xml: bytes | None,
    *,
    extra_members: Sequence[tuple[str, bytes]] = (),
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compression) as zf:
        zf.writestr(
            "[Content_Types].xml",
            b'<?xml version="1.0" encoding="UTF-8"?><Types/>',
        )
        if document_xml is not None:
            zf.writestr("word/document.xml", document_xml)
        for name, data in extra_members:
            zf.writestr(name, data)
    return buf.getvalue()


def test_pdf_multipage_page_numbers_headers_dates_and_search(tmp_path: Path) -> None:
    pdf_bytes = _build_pdf_bytes(
        [
            "Page one introduces general facility rules updated 2026-04-10 for all staff.",
            "Page two states the emergency beacon frequency is 406 MHz on November 18, 2026.",
        ]
    )
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(pdf_bytes)

    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == ["manual.pdf"]
    assert report.failures == []

    rec = idx.records()[0]
    assert rec.kind == "pdf"
    assert rec.status == "ok"
    assert rec.mtime > 0
    assert rec.ingested_at
    assert rec.dates_in_text == ["2026-04-10", "2026-11-18"]

    pdf_chunks = [c for c in idx.chunks if c.source == "manual.pdf"]
    assert [c.id for c in pdf_chunks] == ["manual.pdf#0", "manual.pdf#1"]
    assert [c.page for c in pdf_chunks] == [1, 2]
    assert [c.section for c in pdf_chunks] == ["", ""]
    assert [c.header for c in pdf_chunks] == ["manual.pdf > p. 1", "manual.pdf > p. 2"]

    hits = idx.search("emergency beacon frequency 406 MHz", k=2)
    assert hits
    assert hits[0]["source"] == "manual.pdf"
    assert hits[0]["id"] == "manual.pdf#1"
    assert hits[0]["page"] == 2
    assert hits[0]["section"] == ""
    assert len(hits[0]["file_date"]) == 10
    assert "406 MHz" in hits[0]["text"]


def test_pdf_failure_cases_and_pypdf_behaviour(tmp_path: Path) -> None:
    encrypted_bytes = _build_encrypted_pdf_bytes(
        "Confidential document body long enough to pass minimum chunk size checks."
    )
    corrupt_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R\n"
    no_text_bytes = _build_pdf_bytes(["", ""])
    too_many_pages_bytes = _build_pdf_bytes([""] * 501)

    # Confirm pypdf behaviour as installed.
    enc_reader = pypdf.PdfReader(io.BytesIO(encrypted_bytes))
    assert enc_reader.is_encrypted is True

    with pytest.raises(pypdf.errors.PdfReadError):
        pypdf.PdfReader(io.BytesIO(corrupt_bytes))

    empty_reader = pypdf.PdfReader(io.BytesIO(no_text_bytes))
    assert len(empty_reader.pages) == 2
    assert empty_reader.pages[0].extract_text() == ""

    (tmp_path / "encrypted.pdf").write_bytes(encrypted_bytes)
    (tmp_path / "corrupt.pdf").write_bytes(corrupt_bytes)
    (tmp_path / "scanned.pdf").write_bytes(no_text_bytes)
    (tmp_path / "huge501.pdf").write_bytes(too_many_pages_bytes)

    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == []
    assert idx.chunks == ()

    failures = {name: (status, reason) for name, status, reason in idx.failures()}
    assert set(failures) == {"encrypted.pdf", "corrupt.pdf", "scanned.pdf", "huge501.pdf"}
    assert failures == {name: (status, reason) for name, status, reason in report.failures}

    assert failures["encrypted.pdf"][0] == "unreadable"
    assert "encrypted" in failures["encrypted.pdf"][1].lower()

    assert failures["corrupt.pdf"][0] == "unreadable"
    assert failures["corrupt.pdf"][1]

    assert failures["scanned.pdf"][0] == "unreadable"
    assert "no text layer" in failures["scanned.pdf"][1].lower()

    assert failures["huge501.pdf"][0] == "too_large"
    assert "501" in failures["huge501.pdf"][1]


def test_pdf_missing_pypdf_reports_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "sample.pdf").write_bytes(
        _build_pdf_bytes(["Sample PDF body with enough characters to index cleanly."])
    )
    monkeypatch.setitem(sys.modules, "pypdf", None)
    monkeypatch.setitem(sys.modules, "pypdfium2", None)

    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == []
    assert idx.failures() == [("sample.pdf", "unsupported", "pypdf not installed")]


def test_docx_headings_sections_tables_dates_and_search(tmp_path: Path) -> None:
    body = "".join(
        [
            _docx_p("A", style="Heading1"),
            _docx_p("B", style="Heading2"),
            _docx_p(
                "Solar inverter calibration takes place on 2026-05-12 and June 1, 2026 "
                "in bay three."
            ),
            "<w:tbl><w:tr><w:tc>"
            + _docx_p("Backup generator fuel capacity is 120 liters inside the east shed.")
            + "</w:tc></w:tr></w:tbl>",
            _docx_p("C", style="Heading1"),
            _docx_p(
                "Quarterly safety inspection checklists must be signed by the shift supervisor."
            ),
        ]
    )
    docx_bytes = _build_docx_bytes(_wrap_document_xml(body))
    (tmp_path / "ops.docx").write_bytes(docx_bytes)

    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == ["ops.docx"]
    assert report.failures == []

    rec = idx.records()[0]
    assert rec.kind == "docx"
    assert rec.status == "ok"
    assert rec.dates_in_text == ["2026-05-12", "2026-06-01"]

    chunks = [c for c in idx.chunks if c.source == "ops.docx"]
    assert [c.id for c in chunks] == ["ops.docx#0", "ops.docx#1"]
    assert chunks[0].section == "A > B"
    assert chunks[0].header == "ops.docx > A > B"
    assert chunks[0].page is None
    assert "Backup generator fuel capacity is 120 liters" in chunks[0].text
    assert chunks[1].section == "C"
    assert chunks[1].header == "ops.docx > C"

    hits = idx.search("backup generator fuel capacity 120 liters", k=2)
    assert hits
    assert hits[0]["source"] == "ops.docx"
    assert hits[0]["section"] == "A > B"
    assert hits[0]["page"] is None


def test_docx_untrusted_guards_zip_bomb_ratio_members_entity_oversize_and_bad_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Compression ratio > 100 (50,000 identical bytes compress to ~70 bytes).
    ratio_bomb_docx = _build_docx_bytes(
        _wrap_document_xml(_docx_p("Normal text paragraph.")),
        extra_members=[("word/bomb.txt", b"A" * 50_000)],
        compression=zipfile.ZIP_DEFLATED,
    )
    (tmp_path / "ratio_bomb.docx").write_bytes(ratio_bomb_docx)

    # 2. Oversized member caught by bounded read (stored uncompressed so ratio == 1.0).
    oversize_body = _docx_p("Oversized uncompressed member content " + ("x" * 1200))
    oversize_docx = _build_docx_bytes(
        _wrap_document_xml(oversize_body),
        compression=zipfile.ZIP_STORED,
    )
    (tmp_path / "oversize.docx").write_bytes(oversize_docx)

    # 3. More than 1,000 zip members.
    many_members = [(f"extra/m_{i:04d}.txt", b"x") for i in range(1000)]
    too_many_docx = _build_docx_bytes(
        _wrap_document_xml(_docx_p("Too many zip members in archive.")),
        extra_members=many_members,
        compression=zipfile.ZIP_STORED,
    )
    (tmp_path / "many_members.docx").write_bytes(too_many_docx)

    # 4. Forbidden DOCTYPE / ENTITY in XML.
    doctype_docx = _build_docx_bytes(
        _wrap_document_xml(
            _docx_p("Entity expansion test paragraph."),
            prefix='<!DOCTYPE foo [<!ENTITY xxe "boom">]>',
        )
    )
    (tmp_path / "entity.docx").write_bytes(doctype_docx)

    entity_only_docx = _build_docx_bytes(
        _wrap_document_xml(
            _docx_p("Standalone entity test paragraph."),
            prefix='<!ENTITY xxe "boom">',
        )
    )
    (tmp_path / "entity_only.docx").write_bytes(entity_only_docx)

    # 5. Non-zip renamed .docx.
    (tmp_path / "not_a_zip.docx").write_bytes(b"plain text disguised as a docx file")

    # 6. Valid zip missing word/document.xml.
    (tmp_path / "missing_doc.docx").write_bytes(_build_docx_bytes(None))

    # 7. Valid docx with no text.
    (tmp_path / "empty_doc.docx").write_bytes(_build_docx_bytes(_wrap_document_xml("")))

    monkeypatch.setattr(docindex_mod, "MAX_DOCX_MEMBER_BYTES", 500)

    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == []
    assert idx.chunks == ()

    failures = {name: (status, reason) for name, status, reason in idx.failures()}
    expected_names = {
        "ratio_bomb.docx",
        "oversize.docx",
        "many_members.docx",
        "entity.docx",
        "entity_only.docx",
        "not_a_zip.docx",
        "missing_doc.docx",
        "empty_doc.docx",
    }
    assert set(failures) == expected_names
    assert failures == {name: (status, reason) for name, status, reason in report.failures}

    for name in expected_names:
        expected_status = "empty" if name == "empty_doc.docx" else "unreadable"
        assert failures[name][0] == expected_status
        assert failures[name][1]

    assert "ratio" in failures["ratio_bomb.docx"][1].lower()
    assert "500" in failures["oversize.docx"][1]
    assert "1000" in failures["many_members.docx"][1]
    assert "doctype" in failures["entity.docx"][1].lower()
    assert "entity" in failures["entity_only.docx"][1].lower()
    assert "word/document.xml" in failures["missing_doc.docx"][1]


def test_unchanged_binary_files_not_reparsed_on_refresh_or_touch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "memo.pdf"
    docx_path = tmp_path / "policy.docx"
    pdf_path.write_bytes(
        _build_pdf_bytes(["Monthly maintenance schedule requires checking every valve carefully."])
    )
    docx_path.write_bytes(
        _build_docx_bytes(
            _wrap_document_xml(
                _docx_p("Policy", style="Heading1")
                + _docx_p("All visitors must sign the front lobby logbook before entering.")
            )
        )
    )

    pdf_calls = 0
    docx_calls = 0
    chunk_calls = 0
    orig_pdf = docindex_mod.load_pdf_pages
    orig_docx = docindex_mod.load_docx
    orig_chunk = docindex_mod.chunk_text

    def count_pdf(raw_bytes: bytes) -> tuple[list[tuple[int, str]], str, str | None]:
        nonlocal pdf_calls
        pdf_calls += 1
        return orig_pdf(raw_bytes)

    def count_docx(raw_bytes: bytes) -> tuple[str, str, str | None]:
        nonlocal docx_calls
        docx_calls += 1
        return orig_docx(raw_bytes)

    def count_chunk(*args: object, **kwargs: object) -> list[Chunk]:
        nonlocal chunk_calls
        chunk_calls += 1
        return orig_chunk(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(docindex_mod, "load_pdf_pages", count_pdf)
    monkeypatch.setattr(docindex_mod, "load_docx", count_docx)
    monkeypatch.setattr(docindex_mod, "chunk_text", count_chunk)

    idx = DocIndex(tmp_path)
    r1 = idx.refresh()
    assert sorted(r1.added) == ["memo.pdf", "policy.docx"]
    assert pdf_calls == 1
    assert docx_calls == 1
    assert chunk_calls == 2
    initial_ids = [(c.id, c.header, c.chunk_sha256) for c in idx.chunks]

    # Second refresh without changes: zero re-parses or re-chunks.
    pdf_calls = docx_calls = chunk_calls = 0
    r2 = idx.refresh()
    assert r2.unchanged == 2
    assert r2.added == []
    assert r2.changed == []
    assert pdf_calls == 0
    assert docx_calls == 0
    assert chunk_calls == 0
    assert [(c.id, c.header, c.chunk_sha256) for c in idx.chunks] == initial_ids

    # Touch mtime without changing content: hashes match, zero re-parses or re-chunks.
    for path in (pdf_path, docx_path):
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    pdf_calls = docx_calls = chunk_calls = 0
    r3 = idx.refresh()
    assert r3.unchanged == 2
    assert r3.changed == []
    assert pdf_calls == 0
    assert docx_calls == 0
    assert chunk_calls == 0
    assert [(c.id, c.header, c.chunk_sha256) for c in idx.chunks] == initial_ids


def test_t4_md_txt_csv_chunk_ids_and_headers_unchanged_and_handbook_indexed() -> None:
    idx = DocIndex(CORPUS_DIR)
    idx.refresh()
    assert idx.failures() == []

    by_source: dict[str, list[tuple[str, str]]] = {}
    for chunk in idx.chunks:
        by_source.setdefault(chunk.source, []).append((chunk.id, chunk.header))

    for source, expected in T4_EXPECTED_IDS_AND_HEADERS.items():
        assert by_source.get(source) == expected

    pdf_chunks = [c for c in idx.chunks if c.source == "handbook.pdf"]
    assert [(c.id, c.header, c.page, c.section) for c in pdf_chunks] == [
        ("handbook.pdf#0", "handbook.pdf > p. 1", 1, ""),
        ("handbook.pdf#1", "handbook.pdf > p. 2", 2, ""),
    ]

    docx_chunks = [c for c in idx.chunks if c.source == "handbook.docx"]
    assert [(c.id, c.header, c.page, c.section) for c in docx_chunks] == [
        ("handbook.docx#0", "handbook.docx > Onboarding checklist", None, "Onboarding checklist"),
        ("handbook.docx#1", "handbook.docx > Equipment return", None, "Equipment return"),
    ]


def test_docx_deep_xml_is_unreadable_and_50_levels_loads(tmp_path: Path) -> None:
    # 3000 nested w:tbl/w:tr/w:tc levels with varied attributes so zip ratio stays well under 100.
    deep_open = "".join(
        f'<w:tbl w:id="t_{i}"><w:tr w:r="r_{i}"><w:tc w:c="c_{i}">' for i in range(3000)
    )
    deep_close = "</w:tc></w:tr></w:tbl>" * 3000
    deep_xml = _wrap_document_xml(deep_open + _docx_p("Deep nested content.") + deep_close)
    (tmp_path / "deep.docx").write_bytes(_build_docx_bytes(deep_xml))

    # 50 XML levels: w:document(1) > w:body(2) > 15*(tbl/tr/tc)(45) > p(48) > r(49) > t(50).
    ok50_open = "<w:tbl><w:tr><w:tc>" * 15
    ok50_close = "</w:tc></w:tr></w:tbl>" * 15
    ok50_xml = _wrap_document_xml(
        ok50_open
        + _docx_p("Fifty level nested table text with enough words for retrieval indexing.")
        + ok50_close
    )
    (tmp_path / "ok_50.docx").write_bytes(_build_docx_bytes(ok50_xml))

    normal_xml = _wrap_document_xml(
        _docx_p("Normal sibling document text with enough words to index properly.")
    )
    (tmp_path / "normal.docx").write_bytes(_build_docx_bytes(normal_xml))

    idx = DocIndex(tmp_path)
    report = idx.refresh()

    failures = {name: (status, reason) for name, status, reason in idx.failures()}
    assert set(failures) == {"deep.docx"}
    assert failures["deep.docx"][0] == "unreadable"
    assert "docx XML is nested deeper than 100 levels" in failures["deep.docx"][1]
    assert idx.failures() == report.failures
    assert sorted(report.added) == ["normal.docx", "ok_50.docx"]
    assert any("Fifty level nested table" in c.text for c in idx.chunks)
    assert any("Normal sibling document" in c.text for c in idx.chunks)


def test_docx_duplicate_member_name_is_unreadable(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", b'<?xml version="1.0" encoding="UTF-8"?><Types/>')
            zf.writestr(
                "word/document.xml",
                _wrap_document_xml(_docx_p("First copy of document xml with enough words.")),
            )
            zf.writestr(
                "word/document.xml",
                _wrap_document_xml(_docx_p("Second copy of document xml with enough words.")),
            )
    (tmp_path / "dup.docx").write_bytes(buf.getvalue())

    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == []
    assert idx.chunks == ()
    failures = {name: (status, reason) for name, status, reason in idx.failures()}
    assert set(failures) == {"dup.docx"}
    assert failures["dup.docx"][0] == "unreadable"
    assert "duplicate zip member name" in failures["dup.docx"][1].lower()
    assert idx.failures() == report.failures


def test_docx_overlapped_zip_entries_are_rejected(tmp_path: Path) -> None:
    buf = io.BytesIO()
    doc_xml = _wrap_document_xml(
        _docx_p("Overlapped entry test paragraph with enough words for chunking.")
    )
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", b'<?xml version="1.0" encoding="UTF-8"?><Types/>')
        zf.writestr("word/document.xml", doc_xml)
        orig_info = zf.getinfo("word/document.xml")
        for i in range(2):
            dup_info = copy.copy(orig_info)
            dup_info.filename = f"word/overlap_{i}.xml"
            zf.filelist.append(dup_info)
    (tmp_path / "overlap.docx").write_bytes(buf.getvalue())

    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == []
    assert idx.chunks == ()
    failures = {name: (status, reason) for name, status, reason in idx.failures()}
    assert set(failures) == {"overlap.docx"}
    assert failures["overlap.docx"][0] == "unreadable"
    assert "docx has overlapping zip entries" in failures["overlap.docx"][1]
    assert idx.failures() == report.failures


def test_docx_total_uncompressed_bytes_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docindex_mod, "MAX_DOCX_TOTAL_BYTES", 2000)
    extra = [(f"extra/part_{i}.txt", b"X" * 450) for i in range(5)]
    docx_bytes = _build_docx_bytes(
        _wrap_document_xml(_docx_p("Short document xml body with enough words.")),
        extra_members=extra,
        compression=zipfile.ZIP_STORED,
    )
    (tmp_path / "total_cap.docx").write_bytes(docx_bytes)

    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == []
    failures = {name: (status, reason) for name, status, reason in idx.failures()}
    assert set(failures) == {"total_cap.docx"}
    assert failures["total_cap.docx"][0] == "unreadable"
    assert "2000" in failures["total_cap.docx"][1]
    assert idx.failures() == report.failures


def test_docx_only_inflates_document_xml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docx_bytes = _build_docx_bytes(
        _wrap_document_xml(
            _docx_p("Only document xml should be inflated when loading this valid archive.")
        ),
        extra_members=[
            ("word/styles.xml", b"<w:styles/>"),
            ("word/media/sample.bin", b"A" * 200),
        ],
        compression=zipfile.ZIP_STORED,
    )
    (tmp_path / "clean.docx").write_bytes(docx_bytes)

    opened: list[str] = []
    orig_open = zipfile.ZipFile.open

    def spy_open(  # type: ignore[no-untyped-def]
        self: zipfile.ZipFile, name: str | zipfile.ZipInfo, *args: object, **kwargs: object
    ):
        member_name = name.filename if isinstance(name, zipfile.ZipInfo) else name
        opened.append(member_name)
        return orig_open(self, name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(zipfile.ZipFile, "open", spy_open)

    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == ["clean.docx"]
    assert idx.failures() == []
    assert opened == ["word/document.xml"]
