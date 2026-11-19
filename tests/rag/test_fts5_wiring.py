"""Offline tests for F04 wiring: docs.py (--retriever fts5, docs_status) and metrics.py."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from test_loaders_binary import _build_pdf_bytes

from voice_agent.mcp_servers.docs import build_server
from voice_agent.rag import fts5 as fts5_mod
from voice_agent.rag import metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_CORPUS = REPO_ROOT / "tests" / "fixtures" / "rag_corpus"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def _folder_snapshot(folder: Path) -> dict[str, tuple[int, str]]:
    snap: dict[str, tuple[int, str]] = {}
    for entry in sorted(folder.rglob("*")):
        rel = entry.relative_to(folder).as_posix()
        st = entry.stat()
        digest = hashlib.sha256(entry.read_bytes()).hexdigest() if entry.is_file() else "<dir>"
        snap[rel] = (st.st_mtime_ns, digest)
    return snap


async def test_docs_server_fts5_search_over_fixtures(tmp_path: Path) -> None:
    corpus = tmp_path / "fixtures_copy"
    shutil.copytree(FIXTURES_CORPUS, corpus)

    server = build_server(corpus, retriever_mode="fts5")
    async with Client(server) as client:
        status = (await client.call_tool("docs_status", {})).structured_content
        assert status["retriever"] == "fts5:file"
        assert status["failures"] == []
        assert (corpus / ".rag_cache" / "fts5" / "index.sqlite3").is_file()

        meal_hits = (
            await client.call_tool("search_docs", {"query": "daily meal allowance expense claims"})
        ).structured_content["result"]
        assert meal_hits
        assert meal_hits[0]["source"] == "handbook.pdf"
        assert "350 pesos" in meal_hits[0]["text"]
        assert {"source", "id", "section", "page", "file_date", "text", "score", "coverage"} <= set(
            meal_hits[0]
        )

        reset_hits = (
            await client.call_tool("search_docs", {"query": "factory reset pinhole twelve seconds"})
        ).structured_content["result"]
        assert reset_hits
        assert reset_hits[0]["source"] == "router_reset.md"


async def test_docs_server_stdio_cli_accepts_fts5_retriever(tmp_path: Path) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    shutil.copy(FIXTURES_CORPUS / "router_reset.md", corpus / "router_reset.md")

    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "voice_agent.mcp_servers.docs", "--corpus", str(corpus), "--retriever", "fts5"],
        cwd=str(REPO_ROOT),
    )
    async with Client(transport) as client:
        status = (await client.call_tool("docs_status", {})).structured_content
        assert status["retriever"] == "fts5:file"
        hits = (
            await client.call_tool("search_docs", {"query": "factory reset pinhole twelve seconds"})
        ).structured_content["result"]
        assert hits
        assert hits[0]["source"] == "router_reset.md"


async def test_docs_status_fallback_when_fts5_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    shutil.copy(FIXTURES_CORPUS / "router_reset.md", corpus / "router_reset.md")

    monkeypatch.setattr(
        fts5_mod,
        "fts5_available",
        lambda tokenizer=fts5_mod.PRETOK_TOKENIZER: (False, "compiled without FTS5"),
    )

    async with Client(build_server(corpus, retriever_mode="fts5")) as client:
        status = (await client.call_tool("docs_status", {})).structured_content
        assert status["retriever"] == "bm25 (fts5 unavailable: compiled without FTS5)"
        hits = (
            await client.call_tool("search_docs", {"query": "factory reset pinhole twelve seconds"})
        ).structured_content["result"]
        assert hits
        assert hits[0]["source"] == "router_reset.md"

    assert not (corpus / ".rag_cache" / "fts5").exists()


def test_metrics_fts5_in_memory_and_cache_dir_never_writes_into_folder(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "readonly_docs"
    corpus.mkdir()
    (corpus / "guide.md").write_text(
        "# Inverter Guide\n\nSolar inverters shut down at 60 degrees Celsius.\n",
        encoding="utf-8",
    )
    golden = tmp_path / "golden.jsonl"
    _write_jsonl(
        golden,
        [
            {
                "id": "q1",
                "q": "At what temperature do solar inverters shut down?",
                "type": "numeric",
                "relevant": [{"file": "guide.md", "contains": "60 degrees Celsius."}],
            }
        ],
    )

    before_snap = _folder_snapshot(corpus)
    os.chmod(corpus, 0o555)
    try:
        rc_mem = metrics.main(
            [
                "--folder",
                str(corpus),
                "--golden",
                str(golden),
                "--retriever",
                "fts5",
            ]
        )
        out_mem = capsys.readouterr().out
        assert rc_mem == 0
        assert "retriever: fts5:memory" in out_mem
        assert not (corpus / ".rag_cache").exists()

        cache_dir = tmp_path / "metrics_cache"
        rc_file = metrics.main(
            [
                "--folder",
                str(corpus),
                "--golden",
                str(golden),
                "--retriever",
                "fts5",
                "--fts5-tokenizer",
                "porter",
                "--cache-dir",
                str(cache_dir),
            ]
        )
        out_file = capsys.readouterr().out
        assert rc_file == 0
        assert "retriever: fts5:file" in out_file
        assert (cache_dir / "fts5" / "index.sqlite3").is_file()
        assert not (corpus / ".rag_cache").exists()
    finally:
        os.chmod(corpus, 0o755)

    assert _folder_snapshot(corpus) == before_snap


def test_metrics_fts5_exits_1_when_unavailable_and_never_writes_into_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "guide.md").write_text(
        "# Inverter Guide\n\nSolar inverters shut down at 60 degrees Celsius.\n",
        encoding="utf-8",
    )
    golden = tmp_path / "golden.jsonl"
    _write_jsonl(
        golden,
        [
            {
                "id": "q1",
                "q": "At what temperature do solar inverters shut down?",
                "type": "numeric",
                "relevant": [{"file": "guide.md", "contains": "60 degrees Celsius."}],
            }
        ],
    )

    monkeypatch.setattr(
        fts5_mod,
        "fts5_available",
        lambda tokenizer=fts5_mod.PRETOK_TOKENIZER: (False, "fts5 disabled for test"),
    )

    before_snap = _folder_snapshot(corpus)
    cache_dir = tmp_path / "unused_cache"
    report_path = tmp_path / "report.json"

    os.chmod(corpus, 0o555)
    try:
        rc = metrics.main(
            [
                "--folder",
                str(corpus),
                "--golden",
                str(golden),
                "--retriever",
                "fts5",
                "--cache-dir",
                str(cache_dir),
                "--report-json",
                str(report_path),
            ]
        )
    finally:
        os.chmod(corpus, 0o755)

    captured = capsys.readouterr()
    assert rc == 1
    assert "fts5 unavailable: fts5 disabled for test" in captured.err
    assert not report_path.exists()
    assert not (corpus / ".rag_cache").exists()
    assert _folder_snapshot(corpus) == before_snap


def test_metrics_group_by_format_difficulty_and_compare_arms(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "guide.md").write_text(
        "# Water Pump Guide\n\nPrime the intake line for 30 seconds before startup.\n",
        encoding="utf-8",
    )
    (corpus / "schedule.csv").write_text(
        "shift,lead,budget_php\nmorning,Elena,12500\n",
        encoding="utf-8",
    )
    (corpus / "manual.pdf").write_bytes(_build_pdf_bytes(["Backup turbine spins at 1800 rpm."]))

    golden = tmp_path / "golden.jsonl"
    _write_jsonl(
        golden,
        [
            {
                "id": "q-md",
                "q": "How long should the intake line be primed?",
                "type": "lexical",
                "format": "markdown_guide",
                "difficulty": ["easy", "multi_hop"],
                "relevant": [
                    {
                        "file": "guide.md",
                        "contains": "Prime the intake line for 30 seconds before startup.",
                    }
                ],
            },
            {
                "id": "q-csv",
                "q": "What is the morning shift budget?",
                "type": "table_cell",
                "difficulty": ["multi_hop", "numeric_unit"],
                "relevant": [
                    {
                        "file": "schedule.csv",
                        "contains": "shift morning lead elena budget php 12,500",
                    }
                ],
            },
            {
                "id": "q-pdf",
                "q": "What speed does the backup turbine spin at?",
                "type": "numeric",
                "format": "",
                "difficulty": [],
                "relevant": [
                    {
                        "file": "manual.pdf",
                        "contains": "Backup turbine spins at 1800 rpm.",
                    }
                ],
            },
            {
                "id": "q-unans",
                "q": "Unanswerable question",
                "type": "unanswerable",
                "format": "md",
                "difficulty": ["excluded_tag"],
                "relevant": [],
            },
        ],
    )

    before_snap = _folder_snapshot(corpus)
    single_report_path = tmp_path / "single_grouped.json"
    rc_single = metrics.main(
        [
            "--folder",
            str(corpus),
            "--golden",
            str(golden),
            "--retriever",
            "fts5",
            "--fts5-tokenizer",
            "pretok",
            "--group-by",
            "type,format,difficulty",
            "--match",
            "normalized",
            "--report-json",
            str(single_report_path),
        ]
    )
    out_single = capsys.readouterr().out
    assert rc_single == 0
    assert "retriever: fts5:memory" in out_single
    assert "markdown_guide" in out_single
    assert "csv" in out_single
    assert "multi_hop" in out_single
    assert "excluded_tag" not in out_single

    single_payload = json.loads(single_report_path.read_text(encoding="utf-8"))
    single_metrics = single_payload["arms"][0]["metrics"]
    assert single_metrics["overall"]["n"] == 3
    assert set(single_metrics["by_type"]) == {"lexical", "table_cell", "numeric"}
    assert set(single_metrics["by_format"]) == {"markdown_guide", "csv", "pdf"}
    assert single_metrics["by_format"]["markdown_guide"]["n"] == 1
    assert single_metrics["by_format"]["csv"]["n"] == 1
    assert single_metrics["by_format"]["pdf"]["n"] == 1
    assert set(single_metrics["by_difficulty"]) == {"easy", "multi_hop", "numeric_unit"}
    assert single_metrics["by_difficulty"]["multi_hop"]["n"] == 2
    assert single_metrics["by_difficulty"]["easy"]["n"] == 1
    assert single_metrics["by_difficulty"]["numeric_unit"]["n"] == 1

    compare_report_path = tmp_path / "compare_grouped.json"
    rc_compare = metrics.main(
        [
            "--folder",
            str(corpus),
            "--golden",
            str(golden),
            "--retriever",
            "fts5",
            "--fts5-tokenizer",
            "porter",
            "--compare-arms",
            "legacy,pypdf",
            "--group-by",
            "type,format,difficulty",
            "--match",
            "normalized",
            "--report-json",
            str(compare_report_path),
        ]
    )
    out_compare = capsys.readouterr().out
    assert rc_compare == 0
    assert "=== arm: legacy (retriever: fts5:memory, match: normalized) ===" in out_compare
    assert "=== arm: pypdf (retriever: fts5:memory, match: normalized) ===" in out_compare
    assert "multi_hop" in out_compare
    assert "markdown_guide" in out_compare

    compare_payload = json.loads(compare_report_path.read_text(encoding="utf-8"))
    assert [arm["arm"] for arm in compare_payload["arms"]] == ["legacy", "pypdf"]
    for arm in compare_payload["arms"]:
        assert arm["retriever"] == "fts5:memory"
        arm_m = arm["metrics"]
        assert arm_m["overall"]["n"] == 3
        assert set(arm_m["by_format"]) == {"markdown_guide", "csv", "pdf"}
        assert arm_m["by_difficulty"]["multi_hop"]["n"] == 2
        assert "excluded_tag" not in arm_m["by_difficulty"]

    rc_bad_group = metrics.main(
        [
            "--folder",
            str(corpus),
            "--golden",
            str(golden),
            "--group-by",
            "bogus",
        ]
    )
    err_bad = capsys.readouterr().err
    assert rc_bad_group == 2
    assert "error:" in err_bad

    assert not (corpus / ".rag_cache").exists()
    assert _folder_snapshot(corpus) == before_snap
