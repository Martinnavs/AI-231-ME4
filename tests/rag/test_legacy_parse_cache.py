from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_loaders_binary import (
    _build_docx_bytes,
    _build_pdf_bytes,
    _docx_p,
    _wrap_document_xml,
)

from voice_agent.rag import docindex as docindex_mod
from voice_agent.rag.docindex import DocIndex


def _make_folder(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.pdf").write_bytes(_build_pdf_bytes(["Alpha page one 2024-01-02.", "Beta page two."]))
    (root / "b.docx").write_bytes(
        _build_docx_bytes(
            _wrap_document_xml(_docx_p("Docx paragraph one.") + _docx_p("Docx paragraph two."))
        )
    )
    (root / "c.csv").write_text("name,qty\nbolt,4\nnut,9\n", encoding="utf-8")
    (root / "d.md").write_text("# Title\n\nMarkdown body text.\n", encoding="utf-8")
    (root / "e.txt").write_text("Plain text body here.\n", encoding="utf-8")
    return root


def _sig(idx: DocIndex) -> list[tuple[str, str, Any, Any]]:
    return [(c.id, c.text, c.page, c.section) for c in idx.chunks]


def _cache_files(root: Path) -> list[Path]:
    return sorted((root / ".rag_cache" / "canonical").glob("*.md"))


def _forbid_parsers(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("parser called on cache hit")

    monkeypatch.setattr(docindex_mod, "load_pdf_pages", _boom)
    monkeypatch.setattr(docindex_mod, "load_docx", _boom)
    monkeypatch.setattr(docindex_mod, "decode_and_load", _boom)


def test_restart_uses_cache_without_parsers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_folder(tmp_path / "docs")
    first = DocIndex(root, converters={})
    first.refresh()
    assert first.failures() == []
    assert len(_cache_files(root)) == 5
    expected = _sig(first)
    assert expected

    _forbid_parsers(monkeypatch)
    second = DocIndex(root, converters={})
    second.refresh()
    assert second.failures() == []
    assert _sig(second) == expected


def test_changed_bytes_reparse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_folder(tmp_path / "docs")
    DocIndex(root, converters={}).refresh()
    (root / "d.md").write_text("# Title\n\nChanged markdown.\n", encoding="utf-8")
    idx = DocIndex(root, converters={})
    idx.refresh()
    assert any("Changed markdown" in c.text for c in idx.chunks)
    assert not any("Markdown body text" in c.text for c in idx.chunks)


def test_pdf_engine_switch_reparses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if docindex_mod._pdfium_module() is None:
        pytest.skip("pypdfium2 not installed")
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.pdf").write_bytes(_build_pdf_bytes(["Engine page."]))
    monkeypatch.setenv(docindex_mod.PDF_ENGINE_ENV, "pypdf")
    DocIndex(root, converters={}).refresh()
    (pypdf_file,) = _cache_files(root)
    monkeypatch.setenv(docindex_mod.PDF_ENGINE_ENV, "pdfium")
    calls: list[int] = []
    real = docindex_mod.load_pdf_pages
    monkeypatch.setattr(docindex_mod, "load_pdf_pages", lambda b: (calls.append(1), real(b))[1])
    DocIndex(root, converters={}).refresh()
    assert calls == [1]
    (pdfium_file,) = _cache_files(root)
    assert pdfium_file.name != pypdf_file.name


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '{"v":1}',
        '{"v":2,"pages":[[1,"x"]]}',
        '{"v":1,"pages":[]}',
        '{"v":1,"pages":[[0,"x"]]}',
        '{"v":1,"pages":[["1","x"]]}',
        '{"v":1,"pages":[[true,"x"]]}',
        '{"v":1,"pages":"x"}',
        '{"v":1,"text":"x"}',
        json.dumps({"v": 1, "pages": [[i + 1, "x"] for i in range(501)]}),
    ],
)
def test_tampered_pdf_payload_falls_back(tmp_path: Path, payload: str) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.pdf").write_bytes(_build_pdf_bytes(["Genuine page text."]))
    first = DocIndex(root, converters={})
    first.refresh()
    expected = _sig(first)
    (path,) = _cache_files(root)
    path.write_text(payload, encoding="utf-8")
    second = DocIndex(root, converters={})
    second.refresh()
    assert second.failures() == []
    assert _sig(second) == expected


@pytest.mark.parametrize(
    "payload", ["{", '{"v":1,"text":""}', '{"v":1,"text":5}', '{"v":1,"pages":[]}']
)
def test_tampered_text_payload_falls_back(tmp_path: Path, payload: str) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "d.md").write_text("Real text.\n", encoding="utf-8")
    first = DocIndex(root, converters={})
    first.refresh()
    expected = _sig(first)
    (path,) = _cache_files(root)
    path.write_text(payload, encoding="utf-8")
    second = DocIndex(root, converters={})
    second.refresh()
    assert _sig(second) == expected


def test_failures_not_cached(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "bad.pdf").write_bytes(b"%PDF-1.4 garbage")
    (root / "empty.md").write_text("   \n", encoding="utf-8")
    idx = DocIndex(root, converters={})
    idx.refresh()
    assert len(idx.failures()) == 2
    assert _cache_files(root) == []


def test_none_cache_dir_writes_nothing(tmp_path: Path) -> None:
    root = _make_folder(tmp_path / "docs")
    idx = DocIndex(root, converters={}, canonical_cache_dir=None)
    idx.refresh()
    assert idx.failures() == []
    assert not (root / ".rag_cache").exists()


def test_prune_removes_deleted_source_cache(tmp_path: Path) -> None:
    root = _make_folder(tmp_path / "docs")
    idx = DocIndex(root, converters={})
    idx.refresh()
    assert len(_cache_files(root)) == 5
    (root / "e.txt").unlink()
    idx.refresh()
    assert len(_cache_files(root)) == 4


def test_rename_keeps_new_source_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_folder(tmp_path / "docs")
    DocIndex(root, converters={}).refresh()
    (root / "e.txt").rename(root / "renamed.txt")
    _forbid_parsers(monkeypatch)
    idx = DocIndex(root, converters={})
    idx.refresh()
    sources = {c.source for c in idx.chunks}
    assert "renamed.txt" in sources
    assert "e.txt" not in sources
