"""Tests for DocIndex integration with the canonical Markdown disk cache."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_loaders_binary import _build_docx_bytes, _build_pdf_bytes, _docx_p, _wrap_document_xml

from voice_agent.rag import canonical, canonical_cache, convert
from voice_agent.rag import docindex as docindex_mod
from voice_agent.rag.convert import ConvertResult
from voice_agent.rag.docindex import DocIndex

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


def _make_pptx_bytes(title: str = "Quarterly Update", body: str = "Installed 18 sensors.") -> bytes:
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    slide.placeholders[1].text = body
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def _make_xlsx_bytes() -> bytes:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Budget"
    ws.append(["Item", "Cost"])
    ws.append(["Batteries", 1200])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _block_all_converters(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("converter should not be called on cache hit")

    monkeypatch.setattr(docindex_mod, "load_pdf_pages", _raise)
    monkeypatch.setattr(docindex_mod, "load_docx", _raise)
    monkeypatch.setattr(docindex_mod, "decode_and_load", _raise)
    monkeypatch.setattr(convert, "convert_isolated", _raise)
    monkeypatch.setattr(convert, "convert_with_markitdown", _raise)
    monkeypatch.setattr(convert, "convert_pdf", _raise)


def test_second_docindex_hits_cache_with_converters_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("markitdown")

    (tmp_path / "spoof.md").write_text(
        "# Notes\n\n<!-- rag:page=99 -->\n<!-- rag:converter=evil -->\nReal markdown text.\n",
        encoding="utf-8",
    )
    (tmp_path / "memo.pdf").write_bytes(_build_pdf_bytes(["Page one telemetry check."]))
    (tmp_path / "policy.docx").write_bytes(
        _build_docx_bytes(
            _wrap_document_xml(_docx_p("Safety", style="Heading1") + _docx_p("Wear gloves."))
        )
    )
    (tmp_path / "deck.pptx").write_bytes(_make_pptx_bytes())
    (tmp_path / "plan.xlsx").write_bytes(_make_xlsx_bytes())
    (tmp_path / "guide.html").write_bytes(
        b"<html><body><h1>Setup</h1><!-- rag:page=88 --><p>Connect power.</p></body></html>"
    )

    first = DocIndex(tmp_path, converters=ALL_MARKITDOWN)
    first_report = first.refresh()
    assert first.failures() == []
    assert len(first_report.added) == 6

    cache_dir = tmp_path / ".rag_cache" / "canonical"
    cached_files = sorted(cache_dir.glob("*.md"))
    assert len(cached_files) == 6
    for path in cached_files:
        first_line = path.read_text(encoding="utf-8").partition("\n")[0]
        assert first_line.startswith("<!-- rag:converter=")
        assert first_line.endswith(" -->")

    # Block every converter and loader; a second DocIndex must satisfy everything from disk cache.
    _block_all_converters(monkeypatch)

    second = DocIndex(tmp_path, converters=ALL_MARKITDOWN)
    second_report = second.refresh()
    assert second.failures() == []
    assert sorted(second_report.added) == sorted(first_report.added)
    assert second.chunks == first.chunks

    spoof_chunks = [c for c in second.chunks if c.source in ("spoof.md", "guide.html")]
    assert len(spoof_chunks) == 2
    assert all(c.page is None for c in spoof_chunks)
    assert all(c.converter != "evil" for c in spoof_chunks)
    assert all("<!-- rag:" not in c.text for c in second.chunks)


def test_second_docindex_pypdf_and_fallback_label_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_dir = tmp_path / "pypdf_docs"
    pdf_dir.mkdir()
    (pdf_dir / "report.pdf").write_bytes(_build_pdf_bytes(["First page.", "Second page."]))

    idx1 = DocIndex(pdf_dir, converters={"pdf": "pypdf"})
    idx1.refresh()
    assert idx1.failures() == []
    assert len(idx1.chunks) == 2

    _block_all_converters(monkeypatch)
    idx2 = DocIndex(pdf_dir, converters={"pdf": "pypdf"})
    idx2.refresh()
    assert idx2.failures() == []
    assert idx2.chunks == idx1.chunks

    # Verify fallback converter labels are persisted inside the cache file and restored on hit.
    fb_dir = tmp_path / "fallback_docs"
    fb_dir.mkdir()
    (fb_dir / "form.pdf").write_bytes(_build_pdf_bytes(["Form page one."]))

    def _fake_isolated(raw_bytes: bytes, kind: str, **kwargs: Any) -> ConvertResult:
        del raw_bytes, kind, kwargs
        return ConvertResult(
            text="<!-- rag:page=1 -->\nForm page one.",
            status="ok",
            error=None,
            converter="pypdf-7.1.0",
            raw_text="Form page one.",
            fallback="pypdf-pages",
        )

    monkeypatch.setattr(convert, "convert_isolated", _fake_isolated)
    fb1 = DocIndex(fb_dir, converters={"pdf": "markitdown"})
    fb1.refresh()
    expected_label = f"pypdf-7.1.0+pp{canonical.POSTPROC_VERSION}/fallback=pypdf-pages"
    assert fb1.chunks[0].converter == expected_label

    _block_all_converters(monkeypatch)
    fb2 = DocIndex(fb_dir, converters={"pdf": "markitdown"})
    fb2.refresh()
    assert fb2.failures() == []
    assert fb2.chunks == fb1.chunks
    assert fb2.chunks[0].converter == expected_label


def test_none_cache_dir_and_legacy_mode_do_not_write_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "note.md").write_text("# Title\n\nBody text.\n", encoding="utf-8")
    (tmp_path / "memo.pdf").write_bytes(_build_pdf_bytes(["Legacy pdf text."]))

    no_cache = DocIndex(tmp_path, converters=ALL_MARKITDOWN, canonical_cache_dir=None)
    no_cache.refresh()
    assert no_cache.failures() == []
    assert no_cache.canonical_cache.enabled is False
    assert not (tmp_path / ".rag_cache").exists()

    legacy_idx = DocIndex(tmp_path, converters={})
    legacy_idx.refresh()
    assert legacy_idx.failures() == []
    assert not (tmp_path / ".rag_cache").exists()

    # Version change in converter invalidates cache key and forces re-conversion.
    ver_dir = tmp_path / "ver_docs"
    ver_dir.mkdir()
    (ver_dir / "deck.pptx").write_bytes(b"pptx-bytes")

    calls = 0

    def _counting_isolated(raw_bytes: bytes, kind: str, **kwargs: Any) -> ConvertResult:
        nonlocal calls
        del raw_bytes, kind, kwargs
        calls += 1
        return ConvertResult(
            text=f"<!-- rag:slide=1 -->\nConverted v{calls}",
            status="ok",
            converter="markitdown-0.1.9",
        )

    monkeypatch.setattr(convert, "convert_isolated", _counting_isolated)
    monkeypatch.setitem(sys.modules, "markitdown", SimpleNamespace(__version__="0.1.9"))
    monkeypatch.setitem(sys.modules, "markitdown.converters", SimpleNamespace())

    v1_idx = DocIndex(ver_dir, converters={"pptx": "markitdown"})
    v1_idx.refresh()
    assert calls == 1

    monkeypatch.setitem(sys.modules, "markitdown", SimpleNamespace(__version__="0.2.0"))
    v2_idx = DocIndex(ver_dir, converters={"pptx": "markitdown"})
    v2_idx.refresh()
    assert calls == 2


def test_symlinked_cache_dir_disables_caching_and_index_ok(tmp_path: Path) -> None:
    pytest.importorskip("markitdown")

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("# Heading\n\nContent.\n", encoding="utf-8")
    (docs / "deck.pptx").write_bytes(_make_pptx_bytes())

    outside = tmp_path / "outside_target"
    outside.mkdir()
    (docs / ".rag_cache").symlink_to(outside)

    idx = DocIndex(docs, converters=ALL_MARKITDOWN)
    assert idx.canonical_cache.enabled is False

    report = idx.refresh()
    assert idx.failures() == []
    assert sorted(report.added) == ["deck.pptx", "note.md"]
    assert list(outside.iterdir()) == []

    # Direct symlink on canonical subdirectory also disables caching while keeping index ok.
    docs2 = tmp_path / "docs2"
    docs2.mkdir()
    (docs2 / "note.md").write_text("# Heading\n\nContent.\n", encoding="utf-8")
    target2 = docs2 / "real_target"
    target2.mkdir()
    (docs2 / ".rag_cache").mkdir()
    (docs2 / ".rag_cache" / "canonical").symlink_to(target2)

    idx2 = DocIndex(docs2, converters=ALL_MARKITDOWN)
    assert idx2.canonical_cache.enabled is False
    idx2.refresh()
    assert idx2.failures() == []
    assert len(idx2.chunks) == 1
    assert list(target2.iterdir()) == []


def test_readonly_folder_indexes_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("markitdown")

    docs = tmp_path / "ro_docs"
    docs.mkdir()
    (docs / "note.md").write_text("# Plan\n\nRead-only folder test.\n", encoding="utf-8")
    (docs / "deck.pptx").write_bytes(_make_pptx_bytes())

    real_open = os.open

    def _deny_write_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)) and (
            dir_fd is not None or str(docs) in os.fsdecode(path)
        ):
            raise PermissionError("read-only folder")
        if dir_fd is not None:
            return real_open(path, flags, mode, dir_fd=dir_fd)
        return real_open(path, flags, mode)

    docs.chmod(0o555)
    try:
        monkeypatch.setattr(os, "open", _deny_write_open)
        idx = DocIndex(docs, converters=ALL_MARKITDOWN)
        report = idx.refresh()
        assert idx.failures() == []
        assert sorted(report.added) == ["deck.pptx", "note.md"]
        assert len(idx.chunks) == 2
        assert not (docs / ".rag_cache").exists()
    finally:
        docs.chmod(0o755)


def test_prune_removes_stale_entries_and_keeps_foreign_files_and_failures_uncached(
    tmp_path: Path,
) -> None:
    pytest.importorskip("markitdown")

    note_path = tmp_path / "note.md"
    deck_path = tmp_path / "deck.pptx"
    bad_pdf = tmp_path / "corrupt.pdf"
    note_path.write_text("# Alpha\n\nKeep this note.\n", encoding="utf-8")
    deck_path.write_bytes(_make_pptx_bytes())
    bad_pdf.write_bytes(b"%PDF-1.4\nbroken")

    idx = DocIndex(tmp_path, converters=ALL_MARKITDOWN)
    idx.refresh()
    assert [f[0] for f in idx.failures()] == ["corrupt.pdf"]

    cache_dir = tmp_path / ".rag_cache" / "canonical"
    initial_cache_files = sorted(p.name for p in cache_dir.glob("*.md"))
    assert len(initial_cache_files) == 2  # Failures are never cached.

    stale_hex = "a" * 64 + ".md"
    (cache_dir / stale_hex).write_text("stale cache entry", encoding="utf-8")
    foreign_files = [
        cache_dir / "readme.txt",
        cache_dir / "notes.md",
        cache_dir / ("b" * 63 + ".md"),
        cache_dir / ".keep.tmp",
    ]
    for path in foreign_files:
        path.write_text("foreign", encoding="utf-8")
    foreign_dir = cache_dir / ("c" * 64 + ".md")
    foreign_dir.mkdir()

    # Clean refresh does not prune yet.
    clean_report = idx.refresh()
    assert clean_report.unchanged == 2
    assert (cache_dir / stale_hex).is_file()

    # Touch note.md without changing content and delete deck.pptx -> dirty refresh prunes stale.
    st = note_path.stat()
    os.utime(note_path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    deck_path.unlink()

    dirty_report = idx.refresh()
    assert dirty_report.removed == ["deck.pptx"]
    assert dirty_report.unchanged == 1

    remaining_hex = sorted(p.name for p in cache_dir.iterdir() if p.is_file() and len(p.name) == 67)
    assert len(remaining_hex) == 1
    assert not (cache_dir / stale_hex).exists()
    for path in foreign_files:
        assert path.is_file(), f"foreign file deleted: {path.name}"
    assert foreign_dir.is_dir()

    # Modifying note.md replaces its key and prunes the previous key.
    old_note_cache = remaining_hex[0]
    note_path.write_text("# Alpha\n\nUpdated note content.\n", encoding="utf-8")
    idx.refresh()
    new_hex = sorted(p.name for p in cache_dir.iterdir() if p.is_file() and len(p.name) == 67)
    assert len(new_hex) == 1
    assert new_hex[0] != old_note_cache
    assert not (cache_dir / old_note_cache).exists()


def test_tampered_cache_files_are_reconverted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_path = tmp_path / "note.md"
    pdf_path = tmp_path / "memo.pdf"
    note_path.write_text("# Note\n\nOriginal note body.\n", encoding="utf-8")
    pdf_path.write_bytes(_build_pdf_bytes(["Original page one text."]))

    converters = {"md": "identity", "pdf": "pypdf"}
    pdf_calls = 0
    orig_convert_pdf = convert.convert_pdf

    def _counting_pdf(raw_bytes: bytes, mode: str = "markitdown") -> ConvertResult:
        nonlocal pdf_calls
        pdf_calls += 1
        return orig_convert_pdf(raw_bytes, mode=mode)

    monkeypatch.setattr(convert, "convert_pdf", _counting_pdf)

    idx = DocIndex(tmp_path, converters=converters)
    idx.refresh()
    assert pdf_calls == 1
    expected_md_chunks = [c for c in idx.chunks if c.source == "note.md"]
    expected_pdf_chunks = [c for c in idx.chunks if c.source == "memo.pdf"]

    cache_dir = tmp_path / ".rag_cache" / "canonical"
    note_key = docindex_mod._canonical_cache_key_for(
        docindex_mod.hash_bytes(note_path.read_bytes()),
        "md",
        "identity",
    )
    pdf_key = docindex_mod._canonical_cache_key_for(
        docindex_mod.hash_bytes(pdf_path.read_bytes()),
        "pdf",
        "pypdf",
    )
    note_cache_file = cache_dir / f"{note_key}.md"
    pdf_cache_file = cache_dir / f"{pdf_key}.md"
    assert note_cache_file.is_file()
    assert pdf_cache_file.is_file()

    valid_pdf_header = pdf_cache_file.read_text(encoding="utf-8").partition("\n")[0]

    # Tamper cases on note.md (identity) and memo.pdf (pypdf).
    md_tamper_payloads: list[bytes] = [
        b"missing converter header\n",
        b"<!-- rag:converter=spoofed -->\nTampered text.\n",
        b"<!-- rag:converter= -->\nTampered text.\n",
        b"<!-- rag:converter=identity -->\n<!-- rag:page=99 -->\nSpoofed page.\n",
        b"<!-- rag:converter=identity -->\n<!-- rag:converter=evil -->\nSpoofed conv.\n",
        b"\xff\xfe\xfd invalid utf-8",
        b"<!-- rag:converter=identity -->\n" + (b"x" * (canonical.MAX_CANONICAL_CHARS + 1)),
    ]
    pdf_tamper_payloads: list[bytes] = [
        b"no header at all",
        b"<!-- rag:converter=evil -->\n<!-- rag:page=1 -->\nTampered.\n",
        f"{valid_pdf_header}\n<!-- rag:page=1 -->\n".encode(),  # Marker-only, empty body.
        f"{valid_pdf_header}\n<!-- rag:slide=99 -->\nTampered slide.\n".encode(),
        f"{valid_pdf_header}\n<!-- rag:converter=evil -->\nTampered conv.\n".encode(),
        f"{valid_pdf_header}\n   <!-- rag:page=1 -->\nIndented marker.\n".encode(),
        f"{valid_pdf_header}\n<!-- rag:bogus=1 -->\nUnknown marker.\n".encode(),
        b"x" * (canonical_cache.MAX_CANONICAL_BYTES + 1),
    ]

    for payload in md_tamper_payloads:
        note_cache_file.write_bytes(payload)
        reloaded = DocIndex(tmp_path, converters=converters)
        reloaded.refresh()
        assert reloaded.failures() == []
        assert [c for c in reloaded.chunks if c.source == "note.md"] == expected_md_chunks
        restored = note_cache_file.read_text(encoding="utf-8")
        assert restored.startswith("<!-- rag:converter=identity -->\n")

    for payload in pdf_tamper_payloads:
        pdf_calls = 0
        pdf_cache_file.write_bytes(payload)
        reloaded = DocIndex(tmp_path, converters=converters)
        reloaded.refresh()
        assert pdf_calls == 1
        assert reloaded.failures() == []
        assert [c for c in reloaded.chunks if c.source == "memo.pdf"] == expected_pdf_chunks
        restored = pdf_cache_file.read_text(encoding="utf-8")
        assert restored.startswith(f"{valid_pdf_header}\n")


def test_markitdown_unchanged_and_touch_skips_reconvert_and_rechunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "memo.pdf"
    docx_path = tmp_path / "policy.docx"
    pdf_path.write_bytes(_build_pdf_bytes(["Valve maintenance schedule."]))
    docx_path.write_bytes(
        _build_docx_bytes(
            _wrap_document_xml(_docx_p("Policy", style="Heading1") + _docx_p("Sign the logbook."))
        )
    )

    conv_calls = 0
    chunk_calls = 0
    orig_chunk = docindex_mod.chunk_canonical

    def _count_isolated(raw_bytes: bytes, kind: str, **kwargs: Any) -> ConvertResult:
        nonlocal conv_calls
        del raw_bytes, kwargs
        conv_calls += 1
        if kind == "pdf":
            return ConvertResult(
                text="<!-- rag:page=1 -->\nValve maintenance schedule.",
                status="ok",
                converter="markitdown-0.1.9",
            )
        return ConvertResult(
            text="# Policy\n\nSign the logbook.",
            status="ok",
            converter="markitdown-0.1.9",
        )

    def _count_chunk(*args: Any, **kwargs: Any) -> list[docindex_mod.Chunk]:
        nonlocal chunk_calls
        chunk_calls += 1
        return orig_chunk(*args, **kwargs)

    monkeypatch.setattr(convert, "convert_isolated", _count_isolated)
    monkeypatch.setattr(docindex_mod, "chunk_canonical", _count_chunk)

    idx = DocIndex(tmp_path, converters=ALL_MARKITDOWN)
    r1 = idx.refresh()
    assert sorted(r1.added) == ["memo.pdf", "policy.docx"]
    assert conv_calls == 2
    assert chunk_calls == 2

    conv_calls = chunk_calls = 0
    r2 = idx.refresh()
    assert r2.unchanged == 2
    assert conv_calls == 0
    assert chunk_calls == 0

    for path in (pdf_path, docx_path):
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))

    conv_calls = chunk_calls = 0
    r3 = idx.refresh()
    assert r3.unchanged == 2
    assert conv_calls == 0
    assert chunk_calls == 0
