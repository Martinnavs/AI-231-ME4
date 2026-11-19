"""Tests for isolated subprocess conversion in ``voice_agent.rag.convert``."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from typing import Any

import pytest

from voice_agent.rag import convert
from voice_agent.rag.convert import ConvertResult, convert_isolated, convert_with_markitdown


def _make_zip(members: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


def _make_pptx_bytes() -> bytes:
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Isolated Overview"
    slide.placeholders[1].text = "Subprocess check passed."
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
    ws.append(["Sensor", 120])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_epub_bytes() -> bytes:
    xhtml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        "<head><title>Chapter 1</title></head>"
        "<body><h1>Chapter 1</h1><p>Probe reading 7.0 pH.</p></body>"
        "</html>"
    )
    container_xml = (
        '<?xml version="1.0"?>'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        "<rootfiles>"
        '<rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/>'
        "</rootfiles>"
        "</container>"
    )
    content_opf = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<dc:title>Guide</dc:title>"
        "<dc:language>en</dc:language>"
        "</metadata>"
        "<manifest>"
        '<item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>'
        "</manifest>"
        '<spine><itemref idref="ch1"/></spine>'
        "</package>"
    )
    return _make_zip(
        [
            ("mimetype", b"application/epub+zip"),
            ("META-INF/container.xml", container_xml.encode("utf-8")),
            ("OEBPS/content.opf", content_opf.encode("utf-8")),
            ("OEBPS/chapter1.xhtml", xhtml.encode("utf-8")),
        ]
    )


def test_isolated_matches_in_process_for_pptx_xlsx_html_epub() -> None:
    pytest.importorskip("markitdown")
    html_bytes = (
        b"<html><body><h1>Watering</h1><p>Check every 3 days.</p>"
        b"<table><tr><th>Bay</th><th>Days</th></tr>"
        b"<tr><td>North</td><td>14</td></tr></table></body></html>"
    )
    cases: list[tuple[bytes, str]] = [
        (_make_pptx_bytes(), "pptx"),
        (_make_xlsx_bytes(), "xlsx"),
        (html_bytes, "html"),
        (html_bytes, "htm"),
        (_make_epub_bytes(), "epub"),
    ]
    for raw_bytes, kind in cases:
        expected = convert_with_markitdown(raw_bytes, kind)
        actual = convert_isolated(raw_bytes, kind)
        assert actual.status == "ok", f"unexpected failure for {kind}: {actual}"
        assert actual == expected


def test_unsupported_kind_passed_through_unchanged() -> None:
    raw = b"%PDF-1.4\n"
    assert convert_isolated(raw, "pdf") == convert_with_markitdown(raw, "pdf")


def test_preflight_runs_in_child_for_doctype_and_zip_bomb() -> None:
    doctype_pptx = _make_zip(
        [
            ("[Content_Types].xml", b'<?xml version="1.0"?><Types/>'),
            ("ppt/presentation.xml", b'<?xml version="1.0"?><presentation/>'),
            (
                "ppt/slides/slide1.xml",
                b'<?xml version="1.0"?><!DOCTYPE sld [<!ENTITY x "1">]><sld/>',
            ),
        ]
    )
    expected_doctype = convert_with_markitdown(doctype_pptx, "pptx")
    actual_doctype = convert_isolated(doctype_pptx, "pptx")
    assert actual_doctype.status == "unreadable"
    assert actual_doctype == expected_doctype

    bomb_bytes = _make_zip([("payload.xml", b"0" * (256 * 1024))])
    for kind in ("pptx", "xlsx", "epub"):
        expected_bomb = convert_with_markitdown(bomb_bytes, kind)
        actual_bomb = convert_isolated(bomb_bytes, kind)
        assert actual_bomb.status == "unreadable"
        assert actual_bomb == expected_bomb


def test_timeout_kills_and_reaps_child(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("markitdown")
    xlsx_bytes = _make_xlsx_bytes()
    real_popen = subprocess.Popen
    spawned: list[subprocess.Popen[bytes]] = []

    def _spy_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        proc: subprocess.Popen[bytes] = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(convert.subprocess, "Popen", _spy_popen)

    res = convert_isolated(xlsx_bytes, "xlsx", timeout_s=0.05)
    assert res.status == "unreadable"
    assert res.text == ""
    assert res.error is not None
    assert "timed out" in res.error
    assert res.converter == "markitdown"

    assert len(spawned) == 1
    proc = spawned[0]
    assert proc.poll() is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(proc.pid, os.WNOHANG)


def test_tiny_address_space_limit_returns_unreadable() -> None:
    pytest.importorskip("markitdown")
    xlsx_bytes = _make_xlsx_bytes()
    res = convert_isolated(xlsx_bytes, "xlsx", max_address_space_mb=8)
    assert res.status == "unreadable"
    assert res.text == ""
    assert res.error is not None


def _patch_child_script(monkeypatch: pytest.MonkeyPatch, script: str) -> None:
    real_popen = subprocess.Popen

    def _fake_popen(
        cmd: list[str],
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.Popen[bytes]:
        return real_popen([sys.executable, "-I", "-c", script], *args, **kwargs)

    monkeypatch.setattr(convert.subprocess, "Popen", _fake_popen)


def test_nonzero_exit_returns_unreadable_without_echoing_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_child_script(
        monkeypatch,
        "import sys; sys.stderr.write('SECRET_STDERR_MARKER\\n'); sys.exit(7)",
    )
    res = convert_isolated(b"<html><body>hi</body></html>", "html")
    assert res.status == "unreadable"
    assert res.text == ""
    assert res.error is not None
    assert "SECRET_STDERR_MARKER" not in res.error
    captured = capsys.readouterr()
    assert "SECRET_STDERR_MARKER" not in captured.err
    assert "SECRET_STDERR_MARKER" not in captured.out


@pytest.mark.parametrize(
    "script",
    [
        "import sys; sys.stdout.buffer.write(b'not valid json\\n')",
        "import sys; sys.stdout.buffer.write(b'\\xff\\xfe\\xfd')",
        'import sys; sys.stdout.write(\'{"text": "hi", "status": "ok"}\')',
        (
            'import sys; sys.stdout.write(\'{"text": "hi", "status": "ok", '
            '"error": null, "converter": "", "raw_text": "", "extra": 1}\')'
        ),
        (
            'import sys; sys.stdout.write(\'{"text": 123, "status": "ok", '
            '"error": null, "converter": "", "raw_text": ""}\')'
        ),
        (
            'import sys; sys.stdout.write(\'{"text": "hi", "status": "bogus", '
            '"error": null, "converter": "", "raw_text": ""}\')'
        ),
    ],
)
def test_malformed_child_stdout_returns_unreadable(
    monkeypatch: pytest.MonkeyPatch,
    script: str,
) -> None:
    _patch_child_script(monkeypatch, script)
    res = convert_isolated(b"<html><body>hi</body></html>", "html")
    assert res.status == "unreadable"
    assert res.text == ""
    assert res.error is not None


def test_oversize_child_stdout_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_child_script(
        monkeypatch,
        "import sys; sys.stdout.buffer.write(b'x' * (33 * 1024 * 1024))",
    )
    res = convert_isolated(b"<html><body>hi</body></html>", "html")
    assert res.status == "unreadable"
    assert res.text == ""
    assert res.error is not None


def test_spawn_value_error_does_not_escape() -> None:
    res = convert_isolated(b"<html><body>hi</body></html>", "html\x00bad")
    assert res == ConvertResult(
        text="",
        status="unreadable",
        error="isolated converter spawn failed: ValueError",
        converter="markitdown",
        raw_text="",
    )


def test_child_env_is_bare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/tmp/home")
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai-key")
    monkeypatch.setenv("PYTHONPATH", "/tmp/evil")

    env = convert._child_env()
    assert set(env) <= {"PATH", "HOME", "LANG", "LC_ALL"}
    assert env == {
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp/home",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    assert "GEMINI_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert "PYTHONPATH" not in env

    # Verify via a real isolated child process that secrets do not leak into os.environ.
    proc = subprocess.run(
        [sys.executable, "-I", "-c", "import json, os; print(json.dumps(sorted(os.environ)))"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    child_keys = set(json.loads(proc.stdout))
    assert child_keys <= {"PATH", "HOME", "LANG", "LC_ALL"}
    assert "GEMINI_API_KEY" not in child_keys
    assert "OPENAI_API_KEY" not in child_keys
    assert "PYTHONPATH" not in child_keys
