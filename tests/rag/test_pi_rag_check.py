"""Tests for scripts/pi_rag_check.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from voice_agent.rag import fts5 as fts5_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "pi_rag_check.py"
FIXTURES_CORPUS = REPO_ROOT / "tests" / "fixtures" / "rag_corpus"

_SPEC = importlib.util.spec_from_file_location("pi_rag_check", SCRIPT_PATH)
assert _SPEC and _SPEC.loader
pi_rag_check = importlib.util.module_from_spec(_SPEC)
sys.modules["pi_rag_check"] = pi_rag_check
_SPEC.loader.exec_module(pi_rag_check)


def _snapshot_tree(folder: Path) -> tuple[int, dict[str, tuple[int, int, str]]]:
    root_mtime_ns = folder.stat().st_mtime_ns
    entries: dict[str, tuple[int, int, str]] = {}
    for entry in sorted(folder.rglob("*")):
        rel = entry.relative_to(folder).as_posix()
        st = entry.lstat()
        digest = hashlib.sha256(entry.read_bytes()).hexdigest() if entry.is_file() else "<dir>"
        entries[rel] = (st.st_mtime_ns, st.st_size, digest)
    return root_mtime_ns, entries


def _parse_report(output: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        fields[key.strip()] = val.strip()
    return fields


def test_main_runs_on_fixtures_copy_and_leaves_dir_unmodified(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURES_CORPUS, corpus)
    before = _snapshot_tree(corpus)

    rc = pi_rag_check.main(["--folder", str(corpus)])
    captured = capsys.readouterr()

    assert rc == 0, captured.err
    assert _snapshot_tree(corpus) == before

    fields = _parse_report(captured.out)
    expected_keys = {
        "python",
        "platform",
        "sqlite_version",
        "sqlite_compile_options",
        "import_time_ms",
        "rss_after_import",
        "heavy_modules_loaded",
        "tokenizer",
        "fts5_probe",
        "retriever_status",
        "fallback",
        "folder",
        "docindex_refresh",
        "fts5_build_s",
        "db_location",
        "db_size_bytes",
        "reopen_to_first_query_ms",
        "queries",
        "query_p50_ms",
        "query_p95_ms",
        "nonempty_queries",
        "peak_rss",
    }
    assert expected_keys <= set(fields)
    assert "ENABLE_FTS5=" in fields["sqlite_compile_options"]
    assert fields["fts5_probe"] == "ok"
    assert fields["retriever_status"] == "fts5:file"
    assert fields["fallback"] == "none"
    assert "files=13" in fields["docindex_refresh"]
    assert "chunks=" in fields["docindex_refresh"]
    assert "seconds=" in fields["docindex_refresh"]
    assert fields["db_location"].startswith("temp (")
    db_bytes = int(fields["db_size_bytes"].split()[0])
    assert db_bytes > 0
    assert "inserted=0, deleted=0" in fields["reopen_to_first_query_ms"]
    assert "source=builtin" in fields["queries"]
    assert float(fields["query_p50_ms"]) >= 0.0
    assert float(fields["query_p95_ms"]) >= float(fields["query_p50_ms"])
    hits_num = int(fields["nonempty_queries"].split("/")[0])
    assert hits_num > 0


def test_main_with_custom_queries_k_and_compare_bm25(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURES_CORPUS, corpus)
    before = _snapshot_tree(corpus)

    queries_file = tmp_path / "queries.txt"
    queries_file.write_text(
        "\n".join(
            [
                "# comment line",
                "daily meal allowance expense claims",
                json.dumps({"id": "q2", "q": "factory reset pinhole twelve seconds"}),
                json.dumps({"query": "facilities extension equipment return pickup"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = pi_rag_check.main(
        [
            "--folder",
            str(corpus),
            "--queries",
            str(queries_file),
            "--k",
            "3",
            "--compare-bm25",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0, captured.err
    assert _snapshot_tree(corpus) == before

    fields = _parse_report(captured.out)
    assert fields["queries"] == f"count=3, k=3, source={queries_file}"
    assert fields["nonempty_queries"] == "3/3"
    assert fields["bm25_compare"] == "ok"
    assert float(fields["bm25_build_s"]) >= 0.0
    assert float(fields["bm25_first_query_ms"]) >= 0.0
    assert float(fields["bm25_p50_ms"]) >= 0.0
    assert float(fields["bm25_p95_ms"]) >= float(fields["bm25_p50_ms"])
    assert fields["bm25_nonempty_queries"] == "3/3"
    assert "MiB" in fields["bm25_rss"]


def test_main_in_place_writes_db_under_folder(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURES_CORPUS, corpus, ignore=shutil.ignore_patterns(".rag_cache"))

    rc = pi_rag_check.main(["--folder", str(corpus), "--in-place"])
    captured = capsys.readouterr()

    assert rc == 0, captured.err
    fields = _parse_report(captured.out)
    expected_db = corpus.resolve() / ".rag_cache" / "fts5" / "index.sqlite3"
    assert expected_db.is_file()
    assert fields["db_location"] == f"in-place ({expected_db})"


def test_main_probe_failure_falls_back_to_rank_bm25_and_exits_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURES_CORPUS, corpus)
    before = _snapshot_tree(corpus)

    monkeypatch.setattr(
        fts5_mod,
        "fts5_available",
        lambda _tok="": (False, "compiled without FTS5"),
    )

    rc = pi_rag_check.main(["--folder", str(corpus)])
    captured = capsys.readouterr()

    assert rc == 0, captured.err
    assert _snapshot_tree(corpus) == before
    assert "fallback: rank-bm25" in captured.out

    fields = _parse_report(captured.out)
    assert fields["fallback"] == "rank-bm25"
    assert fields["fts5_probe"] == "failed (compiled without FTS5)"
    assert fields["retriever_status"] == "bm25 (fts5 unavailable: compiled without FTS5)"
    assert fields["db_size_bytes"].startswith("0 ")
    assert int(fields["nonempty_queries"].split("/")[0]) > 0


def test_compare_bm25_skipped_when_rank_bm25_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    shutil.copy(FIXTURES_CORPUS / "router_reset.md", corpus / "router_reset.md")

    orig_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str, package: str | None = None) -> object:
        if name == "rank_bm25":
            return None
        return orig_find_spec(name, package)

    monkeypatch.setattr(pi_rag_check.importlib.util, "find_spec", _fake_find_spec)

    rc = pi_rag_check.main(["--folder", str(corpus), "--compare-bm25"])
    captured = capsys.readouterr()

    assert rc == 0, captured.err
    fields = _parse_report(captured.out)
    assert fields["bm25_compare"] == "skipped (rank-bm25 not installed)"


def test_subprocess_lean_import_path_leaves_heavy_modules_unloaded(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURES_CORPUS, corpus)
    before = _snapshot_tree(corpus)

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--folder", str(corpus)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"
    assert _snapshot_tree(corpus) == before

    fields = _parse_report(proc.stdout)
    assert fields["heavy_modules_loaded"] == "none"
    assert fields["fts5_probe"] == "ok"
    assert fields["fallback"] == "none"


def test_invalid_inputs_return_nonzero_without_creating_folder(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_dir = tmp_path / "does_not_exist"
    assert pi_rag_check.main(["--folder", str(missing_dir)]) == 2
    assert not missing_dir.exists()
    assert "does not exist" in capsys.readouterr().err

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    assert pi_rag_check.main(["--folder", str(corpus), "--k", "0"]) == 2
    assert "--k must be >= 1" in capsys.readouterr().err

    empty_queries = tmp_path / "empty.txt"
    empty_queries.write_text("# only comments\n\n", encoding="utf-8")
    assert pi_rag_check.main(["--folder", str(corpus), "--queries", str(empty_queries)]) == 2
    assert "no queries" in capsys.readouterr().err
