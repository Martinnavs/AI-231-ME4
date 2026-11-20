"""Tests for the ``VOICE_AGENT_RAG_PDF_ENGINE`` switch in ``docindex.load_pdf_pages``."""

from __future__ import annotations

import builtins
from typing import Any

import pytest
from test_loaders_binary import _build_encrypted_pdf_bytes, _build_pdf_bytes

from voice_agent.rag import docindex

pytest.importorskip("pypdfium2")

ENV = "VOICE_AGENT_RAG_PDF_ENGINE"
ENGINES = ["pypdf", "pdfium"]


def _load(monkeypatch: pytest.MonkeyPatch, name: str, data: bytes) -> Any:
    monkeypatch.setenv(ENV, name)
    return docindex.load_pdf_pages(data)


def test_same_pages_and_text(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _build_pdf_bytes(["First page sentence.", "", "Third page sentence."])
    a = _load(monkeypatch, "pypdf", data)
    b = _load(monkeypatch, "pdfium", data)
    assert a[1:] == b[1:] == ("ok", None)
    assert [n for n, _ in a[0]] == [n for n, _ in b[0]] == [1, 3]
    for (_, ta), (_, tb) in zip(a[0], b[0], strict=True):
        assert ta.split() == tb.split()


@pytest.mark.parametrize("name", ENGINES)
def test_encrypted(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    got = _load(monkeypatch, name, _build_encrypted_pdf_bytes("secret text"))
    assert got == ([], "unreadable", "pdf is encrypted")


@pytest.mark.parametrize("name", ENGINES)
def test_corrupt(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    pages, status, err = _load(monkeypatch, name, b"not a pdf at all")
    assert pages == []
    assert status == "unreadable"
    assert err and err.startswith("pdf parse error")


@pytest.mark.parametrize("name", ENGINES)
def test_no_text_layer(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    got = _load(monkeypatch, name, _build_pdf_bytes(["", ""]))
    assert got == ([], "unreadable", "pdf has no text layer")


@pytest.mark.parametrize("name", ENGINES)
def test_page_limit(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    got = _load(monkeypatch, name, _build_pdf_bytes([""] * 501))
    assert got[1] == "too_large"
    assert got[2] == "pdf has 501 pages (exceeds limit of 500)"


def test_page_limit_checked_before_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    class Doc:
        def __len__(self) -> int:
            return 501

        def __getitem__(self, i: int) -> Any:
            raise AssertionError("page accessed before limit check")

        def close(self) -> None:
            pass

    fake = type("M", (), {"PdfDocument": staticmethod(lambda _b: Doc())})
    monkeypatch.setattr(docindex, "_pdfium_module", lambda: fake)
    got = _load(monkeypatch, "pdfium", b"x")
    assert got[1] == "too_large"


def test_pdfium_missing_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *a: Any, **k: Any) -> Any:
        if name.startswith("pypdfium2"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for name in ("pdfium", "auto"):
        pages, status, _ = _load(monkeypatch, name, _build_pdf_bytes(["Hello fallback."]))
        assert status == "ok"
        assert pages == [(1, "Hello fallback.")]


def test_auto_picks_pdfium(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(
        docindex,
        "_load_pdf_pages_pdfium",
        lambda _m, _b: (called.append(True), ([(1, "x")], "ok", None))[1],
    )
    _load(monkeypatch, "AUTO", _build_pdf_bytes(["Hi"]))
    assert called


def test_unknown_value_uses_pypdf(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any) -> Any:
        raise AssertionError("pdfium used")

    monkeypatch.setattr(docindex, "_load_pdf_pages_pdfium", boom)
    assert _load(monkeypatch, "bogus", _build_pdf_bytes(["Hi there"]))[1] == "ok"


def test_default_is_auto_and_uses_pdfium(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(
        docindex,
        "_load_pdf_pages_pdfium",
        lambda _m, _b: (called.append(True), ([(1, "x")], "ok", None))[1],
    )
    monkeypatch.delenv(ENV, raising=False)
    assert docindex.load_pdf_pages(_build_pdf_bytes(["Hi"]))[1] == "ok"
    assert called


def test_crlf_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    pages, status, _ = _load(monkeypatch, "pdfium", _build_pdf_bytes(["line one\nline two"]))
    assert status == "ok"
    assert "\r" not in pages[0][1]
    assert pages[0][1] == "line one\nline two"
