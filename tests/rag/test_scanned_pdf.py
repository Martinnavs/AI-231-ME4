"""Offline unit tests for scanned PDF detection (ticket 61)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_agent.rag import DocIndex


def _build_pdf_bytes(page_texts: list[str]) -> bytes:
    """Build a minimal valid PDF 1.4 with one content stream per page."""

    def _pdf_escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

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
            ops = ["BT", "/F1 12 Tf", "72 720 Td", f"({_pdf_escape(page_text)}) Tj", "ET"]
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


def test_all_scanned_pdf_is_unreadable(tmp_path: Path) -> None:
    pdf_bytes = _build_pdf_bytes(["", ""])
    (tmp_path / "scanned.pdf").write_bytes(pdf_bytes)
    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == ["scanned.pdf"]
    rec = idx.records()[0]
    assert rec.status == "ok"
    assert "scanned" in (rec.error or "").lower()


def test_mixed_pdf_indexes_text_pages_with_note(tmp_path: Path) -> None:
    long_text = (
        "This page has enough characters to clear the "
        "twenty character threshold for scanned detection."
    )
    short_text = "Hi"  # <20 chars
    pdf_bytes = _build_pdf_bytes([long_text, short_text, long_text])
    (tmp_path / "mixed.pdf").write_bytes(pdf_bytes)
    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == ["mixed.pdf"]
    rec = idx.records()[0]
    assert rec.status == "ok"
    assert len(idx.chunks) == 2


def test_all_text_pdf_has_no_error(tmp_path: Path) -> None:
    long = (
        "A page of text that is clearly long enough to pass all scanned PDF detection thresholds."
    )
    pdf_bytes = _build_pdf_bytes([long, long])
    (tmp_path / "full.pdf").write_bytes(pdf_bytes)
    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == ["full.pdf"]
    rec = idx.records()[0]
    assert rec.status == "ok"
    assert rec.error is None


@pytest.mark.parametrize(
    "pages,expect_scanned",
    [
        (["x"] * 1, True),  # 1/1: no page has text at all
        (["x", "long text here to pass"] * 1, False),  # 1/2 sparse: the text page survives
    ],
)
def test_scanned_threshold_boundary(pages: list[str], expect_scanned: bool, tmp_path: Path) -> None:
    pdf_bytes = _build_pdf_bytes(pages)
    (tmp_path / "threshold.pdf").write_bytes(pdf_bytes)
    idx = DocIndex(tmp_path)
    report = idx.refresh()
    if expect_scanned:
        assert report.added == ["threshold.pdf"]
        rec = idx.records()[0]
        assert rec.status == "ok"
        assert "scanned" in (rec.error or "").lower()
    else:
        assert report.added == ["threshold.pdf"]


def test_majority_sparse_pdf_is_indexed_not_scanned(tmp_path: Path) -> None:
    # Ticket 251: more than half the pages sparse no longer means scanned --
    # the text pages are indexed and the sparse page numbers go in the note.
    long_text = "long text here to pass"
    pages = ["x"] * 3 + [long_text] * 2
    pdf_bytes = _build_pdf_bytes(pages)
    (tmp_path / "mixed.pdf").write_bytes(pdf_bytes)
    idx = DocIndex(tmp_path)
    report = idx.refresh()
    assert report.added == ["mixed.pdf"]
    rec = idx.records()[0]
    assert rec.status == "ok"
    assert len(idx.chunks) == 2
    assert "scanned" not in (rec.error or "").lower()
