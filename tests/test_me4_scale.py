"""Tests for me4_scale_corpus.py and me4_scale_probe.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from voice_agent.rag.backends import make_retriever_factory
from voice_agent.rag.docindex import DocIndex

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CORPUS_SCRIPT = _SCRIPTS / "me4_scale_corpus.py"
PROBE_SCRIPT = _SCRIPTS / "me4_scale_probe.py"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict]]:
    """Generate a small synthetic corpus (3 MB) once for all tests."""
    out = tmp_path_factory.mktemp("data") / "corpus"
    result = subprocess.run(
        [sys.executable, str(CORPUS_SCRIPT), "--out", str(out), "--mb", "3"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"corpus gen failed: {result.stderr}"
    needles_path = out.parent / "needles.json"
    needles = json.loads(needles_path.read_text())
    return out, needles


def test_corpus_size(corpus: tuple[Path, list[dict]]) -> None:
    out, _ = corpus
    files = [f for f in out.glob("*") if f.is_file()]
    assert len(files) >= 5, "expected multiple corpus files"
    total = sum(f.stat().st_size for f in files)
    assert total >= 2_000_000, f"corpus too small: {total}"


def test_all_types_present(corpus: tuple[Path, list[dict]]) -> None:
    out, _ = corpus
    for ext in (".pdf", ".csv", ".docx", ".pptx", ".xlsx", ".md", ".txt"):
        assert any(f.suffix == ext for f in out.glob("*")), f"missing type: {ext}"


def test_needles_json(corpus: tuple[Path, list[dict]]) -> None:
    out, needles = corpus
    files = [f for f in out.glob("*") if f.is_file()]
    assert len(needles) == len(files), f"needles ({len(needles)}) != files ({len(files)})"
    for n in needles:
        assert "file" in n and "query" in n and "answer" in n
        assert n["file"] in {f.name for f in files}


def test_docx_readable(corpus: tuple[Path, list[dict]]) -> None:
    out, needles = corpus
    docx_files = [f for f in out.glob("*.docx")]
    assert len(docx_files) >= 1, "no docx files"
    # Verify DocIndex can read at least one docx
    cache_dir = out / ".rag_cache"
    factory = make_retriever_factory(mode="fts5", cache_dir=cache_dir)
    ix = DocIndex(out, factory)
    ix.refresh()
    docx_needle = next(n for n in needles if n["file"].endswith(".docx"))
    results = ix.search(docx_needle["query"], 4)
    texts = " ".join(r.get("text", "") for r in results)
    assert docx_needle["answer"] in texts or docx_needle["query"] in texts


def test_probe_recall(corpus: tuple[Path, list[dict]]) -> None:
    out, needles = corpus
    result = subprocess.run(
        [
            sys.executable,
            str(PROBE_SCRIPT),
            "--corpus",
            str(out),
            "--needles",
            str(out.parent / "needles.json"),
            "--sample-s",
            "1.0",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"probe failed: {result.stderr}"
    data = json.loads(result.stdout)
    for key in (
        "files",
        "ready_s",
        "first_searchable_s",
        "needle_recall",
        "search_p50_ms",
        "search_p95_ms",
        "reopen_s",
        "chunks",
        "peak_rss_mb",
        "cache_mb",
        "failures",
        "samples",
        "cache",
    ):
        assert key in data, f"missing key: {key}"
    assert data["cache"] == "cold"
    hit, total = data["needle_recall"].split("/")
    assert int(hit) == int(total), f"recall not 100%: {data['needle_recall']}"
    assert data["first_searchable_s"] <= data["ready_s"] + 1.0
