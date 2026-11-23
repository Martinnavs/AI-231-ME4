"""Tests for in-process conversion in ``voice_agent.rag.convert``."""

from __future__ import annotations

import io
from unittest import mock

import pytest

from voice_agent.rag import convert, docindex


def _make_pptx_bytes() -> bytes:
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "In-process Overview"
    slide.placeholders[1].text = "In-process test."
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def _make_xlsx_bytes() -> bytes:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Test"
    ws.append(["A", 1])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@pytest.mark.parametrize("kind", ["pptx", "xlsx", "html"])
def test_inprocess_equals_isolated(kind: str) -> None:
    pytest.importorskip("pptx")
    pytest.importorskip("openpyxl")
    pytest.importorskip("lxml")
    if kind == "pptx":
        raw = _make_pptx_bytes()
    elif kind == "xlsx":
        raw = _make_xlsx_bytes()
    else:
        raw = b"<html><body><p>Equivalence test</p></body></html>"
    isolated = convert.convert_isolated(raw, kind, engine="direct")
    inprocess = convert.convert_inprocess(raw, kind)
    assert isolated == inprocess


def test_corrupt_pptx_unreadable() -> None:
    pytest.importorskip("pptx")
    result = convert.convert_inprocess(b"\x00\x01\xff\xfe", "pptx")
    assert result.status == "unreadable"
    assert "Traceback" not in str(result)


def test_env_always_forces_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pptx")
    monkeypatch.setenv("VOICE_AGENT_RAG_CONVERT_ISOLATION", "always")
    raw = _make_pptx_bytes()
    with mock.patch.object(convert, "convert_isolated", wraps=convert.convert_isolated) as spy:
        docindex._convert_canonical(raw, "pptx", "direct")  # type: ignore[private]
        spy.assert_called_once()


def test_default_uses_inprocess_small_isolated_epub_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pptx")
    monkeypatch.delenv("VOICE_AGENT_RAG_CONVERT_ISOLATION", raising=False)
    raw = _make_pptx_bytes()

    with mock.patch.object(
        convert, "convert_inprocess", return_value=convert.ConvertResult(text="ok", status="ok")
    ) as inprocess_spy:
        docindex._convert_canonical(raw, "pptx", "direct")  # type: ignore[private]
        inprocess_spy.assert_called_once()

    docindex._convert_canonical(raw, "epub", "direct")  # type: ignore[private]

    monkeypatch.setattr(docindex, "INPROCESS_MAX_BYTES", 0)
    with mock.patch.object(
        convert, "convert_isolated", return_value=convert.ConvertResult(text="ok", status="ok")
    ) as isolated_spy:
        docindex._convert_canonical(raw, "pptx", "direct")  # type: ignore[private]
        isolated_spy.assert_called_once()
