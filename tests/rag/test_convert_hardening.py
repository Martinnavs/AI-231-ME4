"""Security hardening tests for isolated conversion and parent-side validation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from voice_agent.rag import canonical, convert
from voice_agent.rag.convert import ConvertResult, convert_isolated
from voice_agent.rag.docindex import DocIndex


def _patch_child_script(
    monkeypatch: pytest.MonkeyPatch,
    script: str,
) -> list[subprocess.Popen[bytes]]:
    real_popen = subprocess.Popen
    spawned: list[subprocess.Popen[bytes]] = []

    def _fake_popen(
        cmd: list[str],
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.Popen[bytes]:
        proc = real_popen([sys.executable, "-I", "-c", script], *args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(convert.subprocess, "Popen", _fake_popen)
    return spawned


def test_chunked_stdout_kills_child_at_cap_and_bounds_parent_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Child attempts to stream 200 MB (far over the 32 MB cap).
    flood_script = (
        "import sys\n"
        "chunk = b'A' * (1024 * 1024)\n"
        "for _ in range(200):\n"
        "    sys.stdout.buffer.write(chunk)\n"
        "    sys.stdout.buffer.flush()\n"
    )
    spawned = _patch_child_script(monkeypatch, flood_script)

    tracemalloc.start()
    try:
        res = convert_isolated(b"<html><body>hi</body></html>", "html", timeout_s=10.0)
        _current, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert res.status == "unreadable"
    assert res.text == ""
    assert res.error == "isolated converter output exceeded 32 MB"
    assert len(spawned) == 1
    assert spawned[0].poll() is not None
    # Parent-side buffering must stay well bounded near the 32 MB cap.
    assert peak_bytes < 48 * 1024 * 1024


def test_cjk_2_9m_char_html_converts_via_convert_isolated() -> None:
    pytest.importorskip("markitdown")
    unit = "中文測試資料"  # 6 CJK chars (18 UTF-8 bytes)
    repeats = 485_000  # 2,910,000 CJK chars (~8.73 MB UTF-8)
    body = unit * repeats
    assert len(body) == 2_910_000
    assert len(body) < canonical.MAX_CANONICAL_CHARS

    html_bytes = f"<html><body><p>{body}</p></body></html>".encode()
    res = convert_isolated(html_bytes, "html", timeout_s=30.0)
    assert res.status == "ok", f"unexpected failure: {res}"
    assert res.error is None
    assert len(res.text) >= 2_900_000
    assert unit in res.text
    assert res.raw_text == ""


def test_process_group_killed_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_file = tmp_path / "grandchild.pid"
    pgid_file = tmp_path / "child.pgid"
    child_script = (
        "import os, subprocess, sys, time\n"
        f"pgid_path = {str(pgid_file)!r}\n"
        f"pid_path = {str(pid_file)!r}\n"
        "with open(pgid_path, 'w', encoding='utf-8') as f:\n"
        "    f.write(f'{os.getpid()},{os.getpgid(0)}')\n"
        "gc = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(30)'])\n"
        "with open(pid_path, 'w', encoding='utf-8') as f:\n"
        "    f.write(str(gc.pid))\n"
        "time.sleep(30)\n"
    )
    spawned = _patch_child_script(monkeypatch, child_script)

    res = convert_isolated(b"<html></html>", "html", timeout_s=0.4)
    assert res.status == "unreadable"
    assert res.text == ""
    assert res.error is not None
    assert "timed out" in res.error

    assert len(spawned) == 1
    assert spawned[0].poll() is not None

    if pgid_file.exists():
        child_pid_str, child_pgid_str = pgid_file.read_text(encoding="utf-8").split(",")
        assert int(child_pid_str) == int(child_pgid_str)
        assert int(child_pgid_str) != os.getpgid(0)

    if pid_file.exists():
        grandchild_pid = int(pid_file.read_text(encoding="utf-8"))
        # Wait briefly for SIGKILL delivery to reap/terminate the grandchild.
        deadline = time.monotonic() + 1.0
        alive = True
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                alive = False
                break
            time.sleep(0.02)
        assert not alive, f"grandchild {grandchild_pid} survived process-group kill"


def test_rlimit_core_zero_set_in_worker_child() -> None:
    script = (
        "import json, resource, sys\n"
        "from voice_agent.rag import convert\n"
        "def _fake_conv(raw, kind):\n"
        "    soft, hard = resource.getrlimit(resource.RLIMIT_CORE)\n"
        "    return convert.ConvertResult(\n"
        "        text=json.dumps([soft, hard]), status='ok', converter='markitdown-0.1.9'\n"
        "    )\n"
        "convert.convert_with_markitdown = _fake_conv\n"
        "raise SystemExit(convert._worker_main(['html']))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", script],
        input=b"<html><body>hi</body></html>",
        capture_output=True,
        timeout=15,
        check=True,
    )
    payload = json.loads(proc.stdout.decode("utf-8"))
    assert payload["status"] == "ok"
    assert json.loads(payload["text"]) == [0, 0]


@pytest.mark.parametrize(
    ("filename", "raw_bytes", "bad_result"),
    [
        (
            "overlen.html",
            b"<html><body>hi</body></html>",
            ConvertResult(
                text="x" * (canonical.MAX_CANONICAL_CHARS + 1),
                status="ok",
                converter="markitdown-0.1.9",
            ),
        ),
        (
            "spoof_marker.html",
            b"<html><body>hi</body></html>",
            ConvertResult(
                text="<!-- rag:page=1 -->\nInjected page marker.",
                status="ok",
                converter="markitdown-0.1.9",
            ),
        ),
        (
            "wrong_marker.pptx",
            b"placeholder",
            ConvertResult(
                text="<!-- rag:page=1 -->\nPage marker inside pptx.",
                status="ok",
                converter="markitdown-0.1.9",
            ),
        ),
        (
            "indented_marker.pdf",
            b"%PDF-1.4\n",
            ConvertResult(
                text="   <!-- rag:page=1 -->\nIndented marker.",
                status="ok",
                converter="markitdown-0.1.9",
            ),
        ),
        (
            "converter_marker.pdf",
            b"%PDF-1.4\n",
            ConvertResult(
                text="<!-- rag:converter=evil -->\nSpoofed header.",
                status="ok",
                converter="markitdown-0.1.9",
            ),
        ),
        (
            "empty_label.html",
            b"<html><body>hi</body></html>",
            ConvertResult(
                text="Normal body text.",
                status="ok",
                converter="",
            ),
        ),
        (
            "bad_label.html",
            b"<html><body>hi</body></html>",
            ConvertResult(
                text="Normal body text.",
                status="ok",
                converter="evil",
            ),
        ),
        (
            "inject_label.html",
            b"<html><body>hi</body></html>",
            ConvertResult(
                text="Normal body text.",
                status="ok",
                converter="markitdown-0.1.9\n-->",
            ),
        ),
        (
            "long_label.html",
            b"<html><body>hi</body></html>",
            ConvertResult(
                text="Normal body text.",
                status="ok",
                converter="markitdown-" + ("a" * 130),
            ),
        ),
        (
            "bare_legacy.html",
            b"<html><body>hi</body></html>",
            ConvertResult(
                text="Normal body text.",
                status="ok",
                converter="legacy",
            ),
        ),
    ],
)
def test_parent_side_validation_rejects_invalid_child_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    raw_bytes: bytes,
    bad_result: ConvertResult,
) -> None:
    (tmp_path / filename).write_bytes(raw_bytes)
    monkeypatch.setattr(convert, "convert_isolated", lambda _raw, _kind, **kw: bad_result)
    monkeypatch.setitem(sys.modules, "markitdown", sys.modules.get("markitdown") or object())
    monkeypatch.setitem(
        sys.modules, "markitdown.converters", sys.modules.get("markitdown.converters") or object()
    )

    idx = DocIndex(
        tmp_path,
        converters={"html": "markitdown", "pptx": "markitdown", "pdf": "markitdown"},
    )
    idx.refresh()

    failures = idx.failures()
    assert len(failures) == 1
    assert failures[0][0] == filename
    assert failures[0][1] == "unreadable"
    assert idx.chunks == ()
    assert not (tmp_path / ".rag_cache" / "canonical").exists()


def test_parse_child_result_rejects_over_max_canonical_chars() -> None:
    payload = json.dumps(
        {
            "text": "a" * (canonical.MAX_CANONICAL_CHARS + 1),
            "status": "ok",
            "error": None,
            "converter": "markitdown-0.1.9",
            "raw_text": "",
            "fallback": None,
        }
    ).encode("utf-8")
    res = convert._parse_child_result(payload)
    assert res.status == "unreadable"
    assert res.text == ""
