"""Offline tests for metrics retriever routing via backends and kreuzberg removal."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_agent.rag import metrics
from voice_agent.rag.fts5 import Fts5Retriever

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_CORPUS = REPO_ROOT / "tests" / "fixtures" / "rag_corpus"
GOLDEN_PATH = REPO_ROOT / "config" / "eval" / "rag_golden.jsonl"


def test_metrics_auto_without_thin_env_uses_fts5_and_matches_fts5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    index, retriever, label = metrics._build_index_for_eval(
        FIXTURES_CORPUS,
        retriever_mode="auto",
        cache_dir=None,
        converters=None,
    )
    try:
        assert isinstance(retriever, Fts5Retriever)
        assert label.startswith("fts5")
        assert index.chunks
    finally:
        if retriever is not None:
            retriever.close()

    auto_report = metrics.evaluate_arm(
        "default",
        folder=FIXTURES_CORPUS,
        golden_path=GOLDEN_PATH,
        retriever_mode="auto",
        cache_dir=None,
    )
    fts5_report = metrics.evaluate_arm(
        "default",
        folder=FIXTURES_CORPUS,
        golden_path=GOLDEN_PATH,
        retriever_mode="fts5",
        cache_dir=None,
    )
    assert auto_report["retriever"].startswith("fts5")
    assert fts5_report["retriever"].startswith("fts5")
    assert auto_report["metrics"] == fts5_report["metrics"]


def test_metrics_cli_auto_without_thin_env_and_cache_dir_unset_matches_fts5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    auto_rc = metrics.main(
        [
            "--folder",
            str(FIXTURES_CORPUS),
            "--golden",
            str(GOLDEN_PATH),
            "--retriever",
            "auto",
        ]
    )
    auto_captured = capsys.readouterr()

    fts5_rc = metrics.main(
        [
            "--folder",
            str(FIXTURES_CORPUS),
            "--golden",
            str(GOLDEN_PATH),
            "--retriever",
            "fts5",
        ]
    )
    fts5_captured = capsys.readouterr()

    assert auto_rc == 0, auto_captured.err
    assert fts5_rc == 0, fts5_captured.err
    assert "retriever: fts5" in auto_captured.out
    assert "retriever: fts5" in fts5_captured.out
    assert auto_captured.out.split("\n", 1)[1] == fts5_captured.out.split("\n", 1)[1]


def test_metrics_rejects_the_removed_hybrid_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        metrics.main(
            [
                "--folder",
                str(FIXTURES_CORPUS),
                "--golden",
                str(GOLDEN_PATH),
                "--retriever",
                "hybrid",
                "--cache-dir",
                str(tmp_path / "cli_cache"),
            ]
        )
    assert excinfo.value.code == 2
    assert "invalid choice: 'hybrid'" in capsys.readouterr().err


def test_resolve_converters_arg_rejects_kreuzberg() -> None:
    with pytest.raises(ValueError):
        metrics.resolve_converters_arg("kreuzberg")
    with pytest.raises(ValueError):
        metrics.resolve_converters_arg("pdf=kreuzberg")
