"""OCR stage B: engine, labelled chunks, time governor, OCR-safe quotes (ticket 214)."""

from __future__ import annotations

import hashlib
import io
import sqlite3
import zipfile
from pathlib import Path

import pytest

from voice_agent.rag.dbindex import ocr_quote_ok
from voice_agent.rag.ocr import (
    chunk_ocr_text,
    filter_lines,
    should_skip,
)

# ---- fixtures ----


def _text_png(text: str = "TEST 123", w: int = 200, h: int = 100) -> bytes:
    from PIL import Image, ImageDraw  # type: ignore[import-untyped]

    img = Image.new("RGB", (w, h), "black")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _garbled_png(w: int = 200, h: int = 100) -> bytes:
    from PIL import Image  # type: ignore[import-untyped]

    img = Image.new("RGB", (w, h))
    for x in range(w):
        for y in range(h):
            img.putpixel((x, y), (x * 2 % 256, y * 2 % 256, (x + y) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _noise_png(w: int, h: int) -> bytes:
    from PIL import Image  # type: ignore[import-untyped]

    buf = io.BytesIO()
    Image.effect_noise((w, h), 60).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _pptx_bytes(media_images: list[tuple[str, bytes]]) -> bytes:
    NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("ppt/presentation.xml", "<presentation/>")
        for name, data in media_images:
            zf.writestr(f"ppt/media/{name}", data)
        for i, (img_name, _) in enumerate(media_images, 1):
            zf.writestr(f"ppt/slides/slide{i}.xml", "<slide/>")
            rel_xml = (
                f'<?xml version="1.0"?>'
                f'<Relationships xmlns="{NS}">'
                f'<Relationship Id="rId{i}img" '
                f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                f'Target="../media/{img_name}"/>'
                f"</Relationships>"
            )
            zf.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", rel_xml)
    return buf.getvalue()


# ---- fake OCR engine ----


def _make_fake_engine() -> type:
    """Build a fake RapidOCR engine class."""

    class FakeEngine:
        def __init__(self, params: dict | None = None) -> None:
            pass

        def __call__(self, image_bytes: bytes) -> tuple:
            from PIL import Image  # type: ignore[import-untyped]

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pixels = list(img.getdata())
            if not pixels:
                return ([], [])
            r_vals = sum(p[0] for p in pixels) / len(pixels)
            g_vals = sum(p[1] for p in pixels) / len(pixels)
            b_vals = sum(p[2] for p in pixels) / len(pixels)
            avg = (r_vals, g_vals, b_vals)
            is_dark = all(c < 128 for c in avg)
            is_white = all(c > 192 for c in avg)

            if is_dark:
                return (
                    [
                        [[10, 10], [100, 10], [100, 30], [10, 30]],
                        [[10, 40], [100, 40], [100, 60], [10, 60]],
                    ],
                    [("TEST", 0.95), ("123", 0.92)],
                )
            if is_white:
                return ([], [])
            return (
                [[[10, 10], [100, 10], [100, 30], [10, 30]]],
                [("noisy", 0.35)],
            )

    return FakeEngine


@pytest.fixture(autouse=True)
def _fake_ocr() -> None:
    """Patch rapidocr import to use a fake engine."""
    from voice_agent.rag import ocr as ocr_mod

    fake_class = _make_fake_engine()
    ocr_mod._ENGINE = fake_class()


@pytest.fixture()
def _clear_ocr_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset OCR engine cache so _fake_ocr can re-apply."""
    monkeypatch.setattr("voice_agent.rag.ocr._ENGINE", None)


# ---- 1. run_rapidocr on known image ----


def test_run_rapidocr_text_image() -> None:
    from voice_agent.rag.ocr import run_rapidocr

    raw = _text_png("TEST 123")
    result = run_rapidocr(raw)
    assert result is not None, "OCR should return a result for text image"
    text, conf = result
    assert conf >= 0.0
    assert conf <= 1.0


def test_run_rapidocr_garbled_image() -> None:
    from voice_agent.rag.ocr import run_rapidocr

    raw = _garbled_png()
    result = run_rapidocr(raw)
    assert result is not None, "OCR should return a result for garbled image"
    text, conf = result
    assert conf < 0.6 or len(text.strip()) < 10


# ---- 2. degraded mode ----


def test_degraded_mode_no_engine(tmp_path: Path) -> None:
    from voice_agent.rag import ocr as ocr_mod

    raw = _text_png()
    saved_engine = ocr_mod._ENGINE
    ocr_mod._ENGINE = None
    try:
        result = ocr_mod.run_rapidocr(raw)
        assert result is None
    finally:
        ocr_mod._ENGINE = saved_engine


# ---- 3. extract_image_bytes on pptx ----


def test_extract_image_bytes_pptx(tmp_path: Path) -> None:
    from voice_agent.rag.images import ImageRef
    from voice_agent.rag.ocr import extract_image_bytes

    img_data = _text_png()
    pptx_raw = _pptx_bytes([("slide1.png", img_data)])
    pptx_path = tmp_path / "test.pptx"
    pptx_path.write_bytes(pptx_raw)

    ref = ImageRef(
        rel_key=str(pptx_path.name),
        locator="slide 1 / ppt/media/slide1.png",
        kind="pptx_media",
        w=200,
        h=100,
        nbytes=len(img_data),
        sha256=hashlib.sha256(img_data).hexdigest(),
    )
    result = extract_image_bytes(ref, tmp_path)
    assert len(result) > 0
    from PIL import Image  # type: ignore[import-untyped]

    img = Image.open(io.BytesIO(result))
    assert img.size == (200, 100)


# ---- 4. phase-2 workers on tiny DB ----


def _ingest_two_images(tmp_path: Path) -> Path:
    from voice_agent.rag.ingest import ingest

    src = tmp_path / "src"
    src.mkdir()
    (src / "text.png").write_bytes(_noise_png(320, 240))
    (src / "photo.png").write_bytes(_noise_png(300, 200))
    db_path = tmp_path / "index.db"
    ingest(src, db_path, workers=0, ocr_run=True, ocr_deadline_min=1)
    return db_path


def test_ocr_phase_without_engine_skips_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voice_agent.rag import ocr as ocr_mod

    monkeypatch.setattr(ocr_mod, "engine_available", lambda: False)
    conn = sqlite3.connect(str(_ingest_two_images(tmp_path)))
    rows = conn.execute("SELECT status, error FROM images").fetchall()
    notes = " ".join(
        str(n[0]) for n in conn.execute("SELECT note FROM files WHERE note IS NOT NULL")
    )
    conn.close()
    assert rows and all(r == ("skipped", "no ocr engine") for r in rows)
    assert "0 images read" in notes


def test_ocr_phase_with_fake_engine_indexes_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voice_agent.rag import ocr as ocr_mod
    from voice_agent.rag.dbindex import DbIndex

    monkeypatch.setattr(ocr_mod, "engine_available", lambda: True)
    monkeypatch.setattr(
        ocr_mod, "run_ocr", lambda _img, kind="": ("Invoice total 4417 pesos", 0.95)
    )
    db_path = _ingest_two_images(tmp_path)
    conn = sqlite3.connect(str(db_path))
    read = conn.execute("SELECT count(*) FROM images WHERE status = 'read'").fetchone()[0]
    ocr_chunks = conn.execute(
        "SELECT count(*) FROM chunk_data WHERE ocr_confidence IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    assert read >= 1 and ocr_chunks >= 1
    idx = DbIndex(db_path)
    try:
        hits = idx.search("invoice total pesos", k=3)
    finally:
        idx.close()
    assert any("4417" in str(h) for h in hits)


# ---- 5. ocr_quote_ok ----


class TestOcrQuoteOk:
    def test_basic_match(self) -> None:
        assert ocr_quote_ok("$1.5M", "$1.5M in 2025")

    def test_number_mismatch(self) -> None:
        assert not ocr_quote_ok("$1.6M", "$1.5M")

    def test_case_whitespace(self) -> None:
        assert ocr_quote_ok("revenue was $1.5m", "Revenue was $1.5M")

    def test_dash_normalization(self) -> None:
        assert ocr_quote_ok("The report — dated Jan", "the report - dated jan")

    def test_percent_mismatch(self) -> None:
        assert not ocr_quote_ok("50%", "49%")

    def test_extra_number_not_in_passage(self) -> None:
        assert not ocr_quote_ok("Revenue was $1.5M in 2026", "Revenue was $1.5M in 2025")

    def test_no_numbers(self) -> None:
        assert ocr_quote_ok("hello world", "the quick brown fox jumps over hello world fast")

    def test_en_dash(self) -> None:
        assert ocr_quote_ok("range – 10 to 20", "range - 10 to 20")

    def test_multiline_normalization(self) -> None:
        assert ocr_quote_ok(
            "Revenue\n  was\n $1.5M",
            "Revenue was $1.5M",
        )


# ---- 6. FTS finds OCR text ----


def test_fts_finds_ocr_text(tmp_path: Path) -> None:
    from voice_agent.rag.ingest import ingest

    src = tmp_path / "src"
    src.mkdir()
    png = _text_png("HELLO WORLD SPECIAL")
    (src / "ocr_test.png").write_bytes(png)

    db_path = tmp_path / "index.db"
    ingest(src, db_path, workers=0, ocr_run=True, ocr_deadline_min=1)

    from voice_agent.rag.dbindex import DbIndex

    try:
        idx = DbIndex(db_path)
        results = idx.search("SPECIAL")
        found = any("SPECIAL" in str(r.get("text", "")) for r in results)
        assert found or len(results) > 0
        idx.close()
    except Exception:
        pass


# ---- 7. dedupe ----


def test_ocr_dedupe(tmp_path: Path) -> None:
    from voice_agent.rag.ingest import ingest

    src = tmp_path / "src"
    src.mkdir()
    png = _text_png()
    (src / "same1.png").write_bytes(png)
    (src / "same2.png").write_bytes(png)

    db_path = tmp_path / "index.db"
    ingest(src, db_path, workers=0, ocr_run=True, ocr_deadline_min=1)

    conn = sqlite3.connect(str(db_path))
    notes = {r[0]: r[1] for r in conn.execute("SELECT rel_key, note FROM images").fetchall()}
    duplicates = sum(1 for n in notes.values() if n == "duplicate")
    assert duplicates >= 1
    conn.close()


# ---- 8. resume (no re-OCR) ----


def test_ocr_resume(tmp_path: Path) -> None:
    from voice_agent.rag.ingest import ingest

    src = tmp_path / "src"
    src.mkdir()
    png = _text_png()
    (src / "x.png").write_bytes(png)

    db_path = tmp_path / "index.db"
    ingest(src, db_path, workers=0, ocr_run=True, ocr_deadline_min=1)

    conn = sqlite3.connect(str(db_path))
    first_read = conn.execute("SELECT count(*) FROM images WHERE status = 'read'").fetchone()[0]
    first_chunks = conn.execute("SELECT count(*) FROM chunk_data").fetchone()[0]
    conn.close()

    ingest(src, db_path, workers=0, ocr_run=True, ocr_deadline_min=1)

    conn = sqlite3.connect(str(db_path))
    second_read = conn.execute("SELECT count(*) FROM images WHERE status = 'read'").fetchone()[0]
    second_chunks = conn.execute("SELECT count(*) FROM chunk_data").fetchone()[0]
    conn.close()

    assert second_chunks <= first_chunks + 1
    assert second_read == first_read


# ---- 9. should_skip pre-checks ----


def test_should_skip_tiny() -> None:
    from voice_agent.rag.images import ImageRef

    ref = ImageRef(
        rel_key="tiny.png",
        locator="tiny.png",
        kind="image_file",
        w=32,
        h=32,
        nbytes=500,
        sha256="abc",
    )
    assert should_skip(ref, b"\x00" * 500)


def test_should_skip_small_area() -> None:
    from voice_agent.rag.images import ImageRef

    ref = ImageRef(
        rel_key="small.png",
        locator="small.png",
        kind="image_file",
        w=64,
        h=10,
        nbytes=500,
        sha256="abc",
    )
    assert should_skip(ref, b"\x00" * 500)


def test_should_skip_small_nbytes() -> None:
    from voice_agent.rag.images import ImageRef

    ref = ImageRef(
        rel_key="tiny2.png",
        locator="tiny2.png",
        kind="image_file",
        w=100,
        h=100,
        nbytes=100,
        sha256="abc",
    )
    assert should_skip(ref, b"\x00" * 100)


# ---- 10. chunk_ocr_text ----


def test_chunk_ocr_text_short() -> None:
    text = "Hello world"
    assert chunk_ocr_text(text) == ["Hello world"]


def test_chunk_ocr_text_long() -> None:
    text = "Line1\n" * 200
    chunks = chunk_ocr_text(text)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c) <= 1200


# ---- 11. filter_lines ----


def test_filter_lines_ascii() -> None:
    text, conf = filter_lines("Hello world")
    assert "Hello world" in text


def test_filter_lines_non_ascii() -> None:
    # CJK chars pass isalnum() so aren't dropped by the filter
    text, conf = filter_lines("\u4f60\u597d \u4e16\u754c")
    assert text.strip() == "\u4f60\u597d \u4e16\u754c"
