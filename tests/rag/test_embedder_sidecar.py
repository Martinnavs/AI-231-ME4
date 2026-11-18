"""Offline tests for voice_agent.rag.embedder_sidecar and its protocol."""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from voice_agent.rag import embed_cache, embedder_sidecar
from voice_agent.rag.embed_cache import EmbedCache, embed_key, model_slug
from voice_agent.rag.embed_client import SidecarClient

SIDECAR_SCRIPT = Path(embedder_sidecar.__file__).resolve()
_REAL_PYTHON = Path.home() / ".venvs" / "rag-embed" / "bin" / "python"


def _real_model_skip_reason() -> str | None:
    if os.environ.get("VOICE_AGENT_RAG_REAL_MODEL") != "1":
        return "VOICE_AGENT_RAG_REAL_MODEL=1 not set (offline default)"
    if not _REAL_PYTHON.is_file():
        return f"thin env python missing at {_REAL_PYTHON}"
    err = SidecarClient(
        python=str(_REAL_PYTHON),
        script=SIDECAR_SCRIPT,
        cache_dir=Path.home() / ".cache" / "voice_agent" / "fastembed",
        model=embedder_sidecar.DEFAULT_MODEL,
        model_dir=Path.home() / ".cache" / "voice_agent" / "fastembed",
        backend="fastembed",
    ).check()
    if err is not None:
        return f"real model not on disk (--check: {err})"
    return None


_REAL_SKIP_REASON = _real_model_skip_reason()


def test_layout_constants_agree_with_embed_cache() -> None:
    assert embedder_sidecar.CACHE_PATH_FORMAT == embed_cache.CACHE_PATH_FORMAT
    assert embedder_sidecar.CACHE_VECTOR_DTYPE == embed_cache.CACHE_VECTOR_DTYPE
    assert embedder_sidecar._model_slug("answerdotai/answerai-colbert-small-v1") == model_slug(
        "answerdotai/answerai-colbert-small-v1"
    )


def test_sidecar_script_has_no_voice_agent_or_eager_fastembed_imports() -> None:
    tree = ast.parse(SIDECAR_SCRIPT.read_text(encoding="utf-8"))
    top_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".")[0])
    assert "voice_agent" not in top_imports
    assert "fastembed" not in top_imports


def test_sidecar_end_to_end_with_hash_backend_and_cache_interop(tmp_path: Path) -> None:
    model = "test/hash-colbert"
    cache_dir = tmp_path / "cache"
    model_dir = tmp_path / "models"
    client = SidecarClient(
        python=sys.executable,
        script=SIDECAR_SCRIPT,
        cache_dir=cache_dir,
        model=model,
        model_dir=model_dir,
        threads=1,
        backend="hash",
        start_timeout=2.0,
        request_timeout=2.0,
    )
    try:
        assert client.check() is None
        client.start()

        pong = client.ping()
        assert pong["ok"] is True
        assert pong["model"] == model
        assert pong["dim"] == 16
        assert pong["backend"] == "hash"
        assert isinstance(pong["pid"], int)

        k_hit = embed_key("doc.md > Reset", "Hold the power button for twelve seconds to reset.")
        k_miss_text = embed_key("doc.md > Colors", "The indicator light turns green when ready.")
        k_tie_a = embed_key("doc.md > TieA", "Alpha beta gamma.")
        k_tie_b = embed_key("doc.md > TieB", "Alpha beta gamma.")
        k_missing_file = embed_key("doc.md > Missing", "Never embedded.")

        done = client.embed_passages(
            [k_hit, k_miss_text, k_tie_a, k_tie_b],
            [
                "Hold the power button for twelve seconds to reset.",
                "The indicator light turns green when ready.",
                "Alpha beta gamma.",
                "Alpha beta gamma.",
            ],
        )
        assert done == [k_hit, k_miss_text, k_tie_a, k_tie_b]

        # Interoperability check: EmbedCache loads the exact files written by the sidecar.
        cache = EmbedCache(cache_dir, model_slug(model))
        assert cache.missing([k_hit, k_miss_text, k_tie_a, k_tie_b, k_missing_file]) == [
            k_missing_file
        ]
        loaded = cache.load(k_hit)
        assert loaded is not None
        assert loaded.dtype == np.float16
        assert loaded.ndim == 2
        assert loaded.shape[1] == 16

        keys_in_order = [k_miss_text, k_tie_a, k_hit, k_missing_file, k_tie_b]
        ranking = client.score_query("twelve seconds reset button", keys_in_order)
        assert len(ranking) == len(keys_in_order)
        assert ranking[0][0] == 2  # k_hit contains all query tokens
        assert ranking[-1][0] == 3  # k_missing_file has no cache file
        assert math.isinf(ranking[-1][1]) and ranking[-1][1] < 0

        # Raw protocol check for "missing" list and tie-breaking by chunk position.
        raw_resp = client.request(
            "score_query",
            {"query": "alpha beta", "keys": [k_tie_a, k_tie_b, k_missing_file]},
        )
        assert raw_resp["missing"] == [k_missing_file]
        tie_ranking = raw_resp["ranking"]
        assert [pos for pos, _ in tie_ranking] == [0, 1, 2]
        assert tie_ranking[0][1] == pytest.approx(tie_ranking[1][1])
    finally:
        client.close()


def test_sidecar_keeps_running_on_bad_requests_isolates_stdout_and_exits_on_eof(
    tmp_path: Path,
) -> None:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "VOICE_AGENT_SIDECAR_TEST_NOISE": "1",
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-I",
            str(SIDECAR_SCRIPT),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--model",
            "test/hash",
            "--model-dir",
            str(tmp_path / "models"),
            "--threads",
            "1",
            "--backend",
            "hash",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    try:
        proc.stdin.write("{not valid json\n")
        proc.stdin.write(json.dumps({"id": 2, "op": "bogus_op"}) + "\n")
        proc.stdin.write(json.dumps({"id": 3, "op": "ping"}) + "\n")
        proc.stdin.write(
            json.dumps({"id": 4, "op": "embed_passages", "keys": ["k1"], "texts": ["hi"]}) + "\n"
        )
        proc.stdin.close()

        stdout_text = proc.stdout.read()
        stderr_text = proc.stderr.read()
        rc = proc.wait(timeout=2.0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)

    assert rc == 0
    lines = [json.loads(line) for line in stdout_text.strip().splitlines()]
    assert len(lines) == 4
    assert lines[0]["ok"] is False and "malformed_json" in lines[0]["error"]
    assert lines[1]["id"] == 2 and lines[1]["ok"] is False and "unknown_op" in lines[1]["error"]
    assert lines[2]["id"] == 3 and lines[2]["ok"] is True and lines[2]["dim"] == 16
    assert lines[3]["id"] == 4 and lines[3]["ok"] is True and lines[3]["done"] == ["k1"]
    assert "test-only hash backend init stdout noise" in stderr_text
    assert "test-only hash backend init fd1 noise" in stderr_text


def test_cli_arg_parsing_check_and_download_hash_backend(tmp_path: Path) -> None:
    parser = embedder_sidecar.build_parser()
    parsed = parser.parse_args([])
    assert parsed.model == embedder_sidecar.DEFAULT_MODEL
    assert str(parsed.model_dir) == embedder_sidecar.DEFAULT_MODEL_DIR
    assert parsed.threads == 4
    assert parsed.backend == "fastembed"
    assert parsed.check is False
    assert parsed.download is False

    check_proc = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SIDECAR_SCRIPT),
            "--backend",
            "hash",
            "--model",
            "test/hash",
            "--model-dir",
            str(tmp_path / "models"),
            "--check",
        ],
        capture_output=True,
        text=True,
        timeout=2.0,
        check=False,
    )
    assert check_proc.returncode == 0
    check_data = json.loads(check_proc.stdout.strip())
    assert check_data == {"ok": True, "model": "test/hash", "dim": 16}

    dl_dir = tmp_path / "dl_models"
    dl_proc = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SIDECAR_SCRIPT),
            "--backend",
            "hash",
            "--model",
            "test/hash",
            "--model-dir",
            str(dl_dir),
            "--download",
        ],
        capture_output=True,
        text=True,
        timeout=2.0,
        check=False,
    )
    assert dl_proc.returncode == 0
    dl_data = json.loads(dl_proc.stdout.strip())
    assert dl_data["ok"] is True
    assert dl_data["model"] == "test/hash"
    assert dl_data["model_dir"] == str(dl_dir)
    assert isinstance(dl_data["size_bytes"], int)


@pytest.mark.skipif(_REAL_SKIP_REASON is not None, reason=_REAL_SKIP_REASON or "")
def test_real_fastembed_model_maxsim_ordering(tmp_path: Path) -> None:
    model = embedder_sidecar.DEFAULT_MODEL
    model_dir = Path.home() / ".cache" / "voice_agent" / "fastembed"
    cache_dir = tmp_path / "cache"
    client = SidecarClient(
        python=str(_REAL_PYTHON),
        script=SIDECAR_SCRIPT,
        cache_dir=cache_dir,
        model=model,
        model_dir=model_dir,
        threads=4,
        backend="fastembed",
        start_timeout=60.0,
        request_timeout=30.0,
    )
    try:
        client.start()
        k_related = embed_key(
            "doc.md > Guide", "Restart the Raspberry Pi audio service using systemctl."
        )
        k_unrelated = embed_key("doc.md > Recipe", "Simmer tomatoes with garlic and olive oil.")
        client.embed_passages(
            [k_related, k_unrelated],
            [
                "Restart the Raspberry Pi audio service using systemctl.",
                "Simmer tomatoes with garlic and olive oil.",
            ],
        )
        ranking = client.score_query(
            "How do I reboot the Pi sound daemon?", [k_unrelated, k_related]
        )
        assert ranking[0][0] == 1
        assert ranking[0][1] > ranking[1][1]
    finally:
        client.close()
