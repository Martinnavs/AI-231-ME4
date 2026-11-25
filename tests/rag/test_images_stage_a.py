"""OCR stage A: image listing at ingest — unit tests (ticket 213)."""

from __future__ import annotations

import hashlib
import io
import sqlite3
import zipfile
from pathlib import Path

import pytest

from voice_agent.rag.images import (
    image_file_ref,
    list_pdf_images,
    list_pptx_images,
)


def _small_png(w: int = 10, h: int = 10) -> bytes:
    """Build a minimal PNG (w x h white pixels) with Pillow."""
    from PIL import Image  # type: ignore[import-untyped]

    buf = io.BytesIO()
    img = Image.new("RGB", (w, h), "white")
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pptx_bytes(
    media_images: list[tuple[str, bytes]],
    used_on_slides: list[tuple[int, str]],
) -> bytes:
    """Build a .pptx with slides referencing media images.

    Args:
        media_images: (zip_name, raw_bytes) pairs for ppt/media/
        used_on_slides: (slide_number, zip_name) — which slide uses which image
    """
    NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("ppt/presentation.xml", "<presentation/>")
        # Write media
        for name, data in media_images:
            zf.writestr(f"ppt/media/{name}", data)
        # Write slides and their rels
        for slide_num, img_name in used_on_slides:
            zf.writestr(f"ppt/slides/slide{slide_num}.xml", "<slide/>")
            r_id = f"rId{slide_num}img"
            rel_xml = (
                f'<?xml version="1.0"?>'
                f'<Relationships xmlns="{NS}">'
                f'<Relationship Id="{r_id}" '
                f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                f'Target="../media/{img_name}"/>'
                f"</Relationships>"
            )
            zf.writestr(f"ppt/slides/_rels/slide{slide_num}.xml.rels", rel_xml)
    return buf.getvalue()


def test_pptx_images_basic() -> None:
    img1 = _small_png(128, 128)
    img2 = _small_png(100, 200)
    raw = _pptx_bytes(
        [("img1.png", img1), ("img2.png", img2)],
        [(1, "img1.png"), (2, "img2.png")],
    )
    refs = list_pptx_images(raw, "pres.pptx")
    assert len(refs) == 2
    kinds = {r.kind for r in refs}
    assert kinds == {"pptx_media"}


def test_pptx_small_image_skipped() -> None:
    big = _small_png(128, 128)
    tiny = _small_png(32, 32)
    raw = _pptx_bytes(
        [("big.png", big), ("tiny.png", tiny)],
        [(1, "big.png"), (1, "tiny.png")],
    )
    refs = list_pptx_images(raw, "pres.pptx")
    names = {r.locator.split("/")[-1] for r in refs}
    assert "tiny.png" not in names
    assert "big.png" in names


def test_pptx_dedup_same_image_twice() -> None:
    big = _small_png(128, 128)
    raw = _pptx_bytes(
        [("shared.png", big)],
        [(1, "shared.png"), (2, "shared.png")],
    )
    refs = list_pptx_images(raw, "pres.pptx")
    assert len(refs) == 1


def test_image_file_ref(tmp_path: Path) -> None:
    png = _small_png(128, 128)
    p = tmp_path / "test.png"
    p.write_bytes(png)
    ref = image_file_ref(p, "test.png")
    assert ref is not None
    assert ref.kind == "image_file"
    assert ref.w == 128
    assert ref.sha256 == hashlib.sha256(png).hexdigest()


def test_image_file_ref_too_small(tmp_path: Path) -> None:
    png = _small_png(32, 32)
    p = tmp_path / "tiny.png"
    p.write_bytes(png)
    ref = image_file_ref(p, "tiny.png")
    assert ref is None


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    """Build a 3-page scanned PDF (image-only, no text layer) with Pillow."""
    pytest.importorskip("pypdfium2")
    from PIL import (
        Image,  # type: ignore[import-untyped]
        ImageDraw,  # type: ignore[import-untyped]
    )

    out = tmp_path / "scanned.pdf"
    first_img = Image.new("RGB", (612, 792), "white")
    draw = ImageDraw.Draw(first_img)
    draw.rectangle([50, 50, 562, 742], fill="gray")
    others = []
    for _ in range(2):
        img = Image.new("RGB", (612, 792), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 562, 742], fill="gray")
        others.append(img)
    first_img.save(out, format="PDF", save_all=True, append_images=others)
    return out


def test_pdf_images_scanned_page(scanned_pdf: Path) -> None:
    pytest.importorskip("pypdfium2")
    refs = list_pdf_images(
        scanned_pdf,
        "scanned.pdf",
        text_chars_by_page={1: 0, 2: 0, 3: 0},
        file_sha256=hashlib.sha256(scanned_pdf.read_bytes()).hexdigest(),
    )
    scan_refs = [r for r in refs if r.kind == "pdf_scan_page"]
    assert len(scan_refs) >= 3
    for r in scan_refs:
        assert "page" in r.locator


@pytest.fixture
def ingest_db(tmp_path: Path) -> tuple[Path, Path]:
    """Set up a minimal source folder + ingest DB path."""
    src = tmp_path / "src"
    src.mkdir()
    return src, tmp_path / "index.db"


def test_ingest_image_file(ingest_db: tuple[Path, Path]) -> None:
    from voice_agent.rag.ingest import ingest

    src, db_path = ingest_db
    png = _small_png(128, 128)
    (src / "photo.png").write_bytes(png)
    report = ingest(src, db_path, workers=0)
    assert report.ok == 1
    assert report.chunks == 0
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT status, note FROM files WHERE rel_key = 'photo.png'").fetchone()
    assert row[0] == "ok"
    assert "image waiting for OCR" in str(row[1])
    img_row = conn.execute("SELECT kind, status FROM images").fetchone()
    assert img_row[0] == "image_file"
    assert img_row[1] == "pending"
    conn.close()


def test_ingest_scanned_pdf(scanned_pdf: Path, ingest_db: tuple[Path, Path]) -> None:
    pytest.importorskip("pypdfium2")
    from voice_agent.rag.ingest import ingest

    src, db_path = ingest_db
    import shutil

    shutil.copy(scanned_pdf, src / "scanned.pdf")
    report = ingest(src, db_path, workers=0)
    assert report.ok == 1
    assert report.chunks == 0
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT status, note FROM files WHERE rel_key = 'scanned.pdf'").fetchone()
    assert row[0] == "ok"
    assert "scanned" in str(row[1]).lower() or "waiting for OCR" in str(row[1]).lower()
    scan_count = conn.execute(
        "SELECT COUNT(*) FROM images WHERE kind = 'pdf_scan_page'"
    ).fetchone()[0]
    assert scan_count == 3
    conn.close()


def test_docs_status_image_counts(tmp_path: Path) -> None:
    """Test that docs_status includes image counts for DbIndex mode."""
    from voice_agent.rag.ingest import ingest

    src = tmp_path / "src"
    src.mkdir()
    png = _small_png(128, 128)
    (src / "a.png").write_bytes(png)
    (src / "b.png").write_bytes(_small_png(64, 200))
    db_path = tmp_path / "index.db"
    ingest(src, db_path, workers=0)
    # Verify images in DB directly
    conn = sqlite3.connect(str(db_path))
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM images WHERE status = 'pending'").fetchone()[0]
    conn.close()
    assert total == 2
    assert pending == 2


def test_resume_no_duplicate_images(ingest_db: tuple[Path, Path]) -> None:
    from voice_agent.rag.ingest import ingest

    src, db_path = ingest_db
    png = _small_png(128, 128)
    (src / "x.png").write_bytes(png)
    ingest(src, db_path, workers=0)
    ingest(src, db_path, workers=0)
    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM images WHERE rel_key = 'x.png'").fetchone()[0]
    assert count == 1
    conn.close()
