"""Zip input writes `images` rows like a folder does, and is not capped at upload sizes."""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

from PIL import Image

from voice_agent.rag import ingest


def _zip_with_image(tmp_path: Path) -> Path:
    png = tmp_path / "pic.png"
    Image.effect_noise((200, 120), 60).convert("RGB").save(png)
    zpath = tmp_path / "docs.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("notes.md", "# Notes\n\nThe launch code is 4417.\n")
        zf.write(png, "figs/pic.png")
    return zpath


def test_zip_ingest_lists_images_and_keeps_text(tmp_path: Path) -> None:
    zpath = _zip_with_image(tmp_path)
    out = tmp_path / "docs.db"
    assert ingest.main([str(zpath), "--out", str(out)]) == 0
    conn = sqlite3.connect(str(out))
    try:
        kinds = conn.execute("SELECT rel_key, kind FROM images").fetchall()
        note = conn.execute("SELECT note FROM files WHERE rel_key = 'figs/pic.png'").fetchone()
        chunks = conn.execute("SELECT count(*) FROM chunk_data").fetchone()[0]
    finally:
        conn.close()
    assert kinds == [("figs/pic.png", "image_file")]
    assert note is not None and "1 images waiting for OCR" in note[0]
    assert chunks >= 1


def test_zip_preflight_accepts_more_than_upload_limits(tmp_path: Path) -> None:
    zpath = tmp_path / "many.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for i in range(1100):  # the upload default allows 1000 members
            zf.writestr(f"f{i}.md", f"# Doc {i}\n\nfact number {i}\n")
    out = tmp_path / "many.db"
    assert ingest.main([str(zpath), "--out", str(out)]) == 0
    conn = sqlite3.connect(str(out))
    try:
        ok = conn.execute("SELECT count(*) FROM files WHERE status = 'ok'").fetchone()[0]
    finally:
        conn.close()
    assert ok == 1100
