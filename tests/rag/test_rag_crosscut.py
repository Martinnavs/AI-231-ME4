import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from voice_agent.rag import DocIndex
from voice_agent.rag.embed_client import _reduced_env

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "rag_corpus"
GOLDEN = REPO_ROOT / "config" / "eval" / "rag_golden.jsonl"
COMMITTED_SNAPSHOT = REPO_ROOT / "pipeline" / "tests" / "fixtures" / "rag_passages.json"
FLOOR_TYPES = ("lexical", "numeric", "identifier")


def _metrics(*args: str, embed_python: str | None = None) -> subprocess.CompletedProcess[str]:
    env = {**os.environ}
    env.pop("VOICE_AGENT_EMBED_PYTHON", None)
    if embed_python is not None:
        env["VOICE_AGENT_EMBED_PYTHON"] = embed_python
    return subprocess.run(
        [sys.executable, "-m", "voice_agent.rag.metrics", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _copy_corpus(tmp_path: Path) -> Path:
    folder = tmp_path / "rag_corpus"
    shutil.copytree(FIXTURES, folder, ignore=shutil.ignore_patterns(".rag_cache"))
    return folder


def _table_rows(stdout: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        if " | " not in line or line.startswith("type"):
            continue
        cells = [cell.strip() for cell in line.split("|")]
        rows[cells[0]] = cells
    return rows


def test_metrics_cli_dump_reproduces_the_committed_snapshot_byte_for_byte(tmp_path: Path) -> None:
    out = tmp_path / "snapshot.json"
    result = _metrics("--retriever", "bm25", "--dump-passages", str(out))
    assert result.returncode == 0, result.stderr
    assert out.read_bytes() == COMMITTED_SNAPSHOT.read_bytes()


def test_metrics_cli_reports_the_golden_types_and_their_counts() -> None:
    result = _metrics("--retriever", "bm25")
    assert result.returncode == 0, result.stderr
    assert "retriever: bm25" in result.stdout
    rows = _table_rows(result.stdout)
    assert rows["overall"][1] == "29"
    assert rows["lexical"][1] == "10"
    assert rows["numeric"][1] == "7"
    assert rows["identifier"][1] == "3"

    pooled_hits = 0.0
    pooled_n = 0
    for q_type in FLOOR_TYPES:
        n = int(rows[q_type][1])
        pooled_hits += n * float(rows[q_type][3])
        pooled_n += n
    assert pooled_hits / pooled_n >= 0.9 - 0.002


def test_metrics_cli_auto_without_a_sidecar_prints_the_same_table_as_bm25(tmp_path: Path) -> None:
    corpus = _copy_corpus(tmp_path)
    cache_dir = tmp_path / "cache"
    missing = str(tmp_path / "no-such-python")

    bm25 = _metrics("--folder", str(corpus), "--golden", str(GOLDEN), "--retriever", "bm25")
    auto = _metrics(
        "--folder",
        str(corpus),
        "--golden",
        str(GOLDEN),
        "--retriever",
        "auto",
        "--cache-dir",
        str(cache_dir),
        embed_python=missing,
    )

    assert bm25.returncode == 0, bm25.stderr
    assert auto.returncode == 0, auto.stderr
    assert "hybrid unavailable" in auto.stderr
    assert "retriever: bm25" in auto.stdout
    assert auto.stdout.split("\n", 1)[1] == bm25.stdout.split("\n", 1)[1]
    assert list(corpus.rglob("*.npy")) == []
    assert not (corpus / ".rag_cache").exists()
    assert list(cache_dir.rglob("*.npy")) == []
    assert not cache_dir.exists()


def _docx_with_members(path: Path, members: int) -> None:
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Member limit paragraph with enough words so the chunk "
        "clears the minimum size easily.</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("word/document.xml", body)
        for i in range(members - 1):
            zf.writestr(f"extra/m_{i:04d}.txt", "x")


def test_docx_member_limit_accepts_exactly_1000_members_and_names_1001(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    _docx_with_members(folder / "at_limit.docx", 1000)
    _docx_with_members(folder / "over_limit.docx", 1001)

    index = DocIndex(folder)
    index.refresh()

    failures = {name: reason for name, _status, reason in index.failures()}
    assert set(failures) == {"over_limit.docx"}
    assert "1001 > 1000" in failures["over_limit.docx"]
    assert any(chunk.source == "at_limit.docx" for chunk in index.chunks)


def test_sidecar_child_env_is_offline_and_inherits_only_path_and_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOICE_AGENT_TEST_ENV_PROBE", "must-not-leak")
    env = _reduced_env()
    assert env["HF_HUB_OFFLINE"] == "1"
    assert "VOICE_AGENT_TEST_ENV_PROBE" not in env
    assert set(env) <= {"HF_HUB_OFFLINE", "PATH", "HOME"}
