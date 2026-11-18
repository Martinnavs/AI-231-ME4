"""Offline unit tests for voice_agent.rag.embed_client against fake_sidecar.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from voice_agent.rag.embed_client import SidecarClient, SidecarError

FAKE_SCRIPT = Path(__file__).resolve().with_name("fake_sidecar.py")


def _make_client(
    tmp_path: Path,
    *,
    python: str | None = None,
    backend: str = "ok",
    start_timeout: float = 2.0,
    request_timeout: float = 1.0,
) -> SidecarClient:
    return SidecarClient(
        python=python or sys.executable,
        script=FAKE_SCRIPT,
        cache_dir=tmp_path / "cache",
        model="fake/model",
        model_dir=tmp_path / "models",
        threads=1,
        backend=backend,
        start_timeout=start_timeout,
        request_timeout=request_timeout,
    )


def _set_behavior(tmp_path: Path, behavior: dict[str, str]) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "fake_behavior.json").write_text(json.dumps(behavior), encoding="utf-8")


def test_missing_python_raises_sidecar_error_and_check_reports_reason(tmp_path: Path) -> None:
    missing_py = str(tmp_path / "does-not-exist-python")
    client = _make_client(tmp_path, python=missing_py)
    with pytest.raises(SidecarError):
        client.start()
    err = client.check()
    assert isinstance(err, str) and err


def test_check_ok_and_failure(tmp_path: Path) -> None:
    ok_client = _make_client(tmp_path, backend="ok")
    assert ok_client.check() is None

    bad_client = _make_client(tmp_path, backend="fail_check")
    err = bad_client.check()
    assert err is not None and "fake check failed" in err


def test_never_started_and_double_close(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    assert not client.alive
    client.close()
    client.close()

    with pytest.raises(SidecarError):
        client.ping()

    client.start()
    assert client.alive
    proc = client._proc
    assert proc is not None
    client.close()
    assert not client.alive
    assert proc.poll() is not None
    client.close()


def test_hang_timeout_kills_child_and_allows_clean_restart(tmp_path: Path) -> None:
    client = _make_client(tmp_path, backend="hang_ignore_sigterm", request_timeout=0.3)
    try:
        client.start()
        proc = client._proc
        assert proc is not None

        with pytest.raises(SidecarError, match="timed out"):
            client.score_query("hello", ["k1"])

        assert not client.alive
        assert proc.poll() is not None

        # Switch behavior and restart cleanly on the same client instance.
        _set_behavior(tmp_path, {"default": "ok"})
        client.start()
        assert client.alive
        assert client.score_query("hello", ["k1", "k2"]) == [(0, 2.0), (1, 1.0)]
    finally:
        client.close()


def test_hang_on_ping_raises_sidecar_error_and_reaps_child(tmp_path: Path) -> None:
    client = _make_client(tmp_path, backend="hang_ping", start_timeout=0.3)
    with pytest.raises(SidecarError, match="timed out"):
        client.start()
    assert not client.alive
    assert client._proc is None


@pytest.mark.parametrize(
    "mode",
    ["eof", "garbage", "wrong_id"],
)
def test_protocol_violations_kill_child_and_raise_sidecar_error(tmp_path: Path, mode: str) -> None:
    _set_behavior(tmp_path, {"embed_passages": mode})
    client = _make_client(tmp_path, request_timeout=1.5)
    try:
        client.start()
        proc = client._proc
        assert proc is not None

        with pytest.raises(SidecarError):
            client.embed_passages(["k1"], ["hello"])

        assert not client.alive
        assert proc.poll() is not None
    finally:
        client.close()


def test_ok_false_raises_sidecar_error(tmp_path: Path) -> None:
    _set_behavior(tmp_path, {"embed_passages": "ok_false"})
    client = _make_client(tmp_path, request_timeout=1.5)
    try:
        client.start()
        with pytest.raises(SidecarError, match="fake failure"):
            client.embed_passages(["k1"], ["hello"])
    finally:
        client.close()


def test_large_5mb_response_line_succeeds(tmp_path: Path) -> None:
    _set_behavior(tmp_path, {"embed_passages": "large_5mb"})
    client = _make_client(tmp_path, request_timeout=2.0)
    try:
        client.start()
        done = client.embed_passages(["k1", "k2"], ["alpha", "beta"])
        assert done == ["k1", "k2"]
    finally:
        client.close()
