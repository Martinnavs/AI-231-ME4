"""Offline tests for golden set schema, retrieval metrics, floors, and snapshot drift."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from doc_chat.corpus import tokenize
from voice_agent.rag.docindex import DocIndex
from voice_agent.rag.metrics import (
    dump_passages,
    evaluate,
    format_metrics_table,
    is_relevant,
    load_benchmark_user_turns,
    load_golden,
    mrr,
    recall_at_k,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "rag_corpus"
GOLDEN_PATH = REPO_ROOT / "config" / "eval" / "rag_golden.jsonl"
BENCHMARK_PATH = REPO_ROOT / "pipeline" / "config" / "eval" / "benchmark_v1.jsonl"
SNAPSHOT_PATH = REPO_ROOT / "pipeline" / "tests" / "fixtures" / "rag_passages.json"

FLOOR_TYPES = frozenset({"lexical", "numeric", "identifier"})
EXPECTED_TYPE_COUNTS = {
    "lexical": 10,
    "numeric": 7,
    "identifier": 3,
    "paraphrase": 6,
    "taglish": 3,
    "unanswerable": 4,
    "off_topic": 3,
}


def test_recall_at_k_and_mrr_toy_rankings() -> None:
    assert recall_at_k([True, False, False], 1) == 1.0
    assert recall_at_k([False, True, False], 1) == 0.0
    assert recall_at_k([False, True, False], 2) == 1.0
    assert recall_at_k([False, False, False], 5) == 0.0
    assert recall_at_k([True], 0) == 0.0
    assert recall_at_k([], 3) == 0.0

    assert mrr([True, False, False]) == pytest.approx(1.0)
    assert mrr([False, True, False]) == pytest.approx(0.5)
    assert mrr([False, False, True]) == pytest.approx(1.0 / 3.0)
    assert mrr([False, False, False]) == pytest.approx(0.0)
    assert mrr([]) == pytest.approx(0.0)


def test_is_relevant_whitespace_normalization_and_source_match() -> None:
    relevant = [{"file": "guide.md", "contains": "hold the button for twelve seconds."}]
    assert is_relevant(
        {"source": "guide.md", "text": "Please   hold the\nbutton\tfor twelve   seconds. Now."},
        relevant,
    )
    assert not is_relevant(
        {"source": "other.md", "text": "hold the button for twelve seconds."},
        relevant,
    )
    assert not is_relevant(
        {"source": "guide.md", "text": "hold the button for ten seconds."},
        relevant,
    )


def test_evaluate_toy_rankings_and_excludes_non_answerable() -> None:
    golden = [
        {
            "id": "q1",
            "q": "alpha question",
            "type": "lexical",
            "relevant": [{"file": "a.md", "contains": "alpha fact"}],
        },
        {
            "id": "q2",
            "q": "beta question",
            "type": "lexical",
            "relevant": [{"file": "b.md", "contains": "beta fact"}],
        },
        {
            "id": "q3",
            "q": "gamma question",
            "type": "numeric",
            "relevant": [{"file": "c.md", "contains": "42 widgets"}],
        },
        {
            "id": "u1",
            "q": "unanswerable question",
            "type": "unanswerable",
            "relevant": [],
        },
        {
            "id": "o1",
            "q": "off topic question",
            "type": "off_topic",
            "relevant": [],
        },
    ]

    rankings: dict[str, list[dict[str, Any]]] = {
        "alpha question": [
            {"source": "a.md", "text": "This states the alpha fact clearly."},
            {"source": "x.md", "text": "Distractor."},
        ],
        "beta question": [
            {"source": "x.md", "text": "Miss 1."},
            {"source": "y.md", "text": "Miss 2."},
            {"source": "b.md", "text": "Here is the beta fact."},
        ],
        "gamma question": [
            {"source": "x.md", "text": "Nothing relevant here."},
            {"source": "y.md", "text": "Also nothing."},
        ],
    }

    def fake_search(query: str, k: int) -> list[dict[str, Any]]:
        assert query in rankings  # unanswerable/off_topic should not be scored
        return rankings[query][:k]

    report = evaluate(fake_search, golden, ks=(1, 3, 5))
    overall = report["overall"]
    assert overall["n"] == 3
    assert overall["recall"][1] == pytest.approx(1.0 / 3.0)
    assert overall["recall"][3] == pytest.approx(2.0 / 3.0)
    assert overall["recall"][5] == pytest.approx(2.0 / 3.0)
    assert overall["mrr"] == pytest.approx((1.0 + 1.0 / 3.0 + 0.0) / 3.0)

    by_type = report["by_type"]
    assert set(by_type) == {"lexical", "numeric"}
    assert by_type["lexical"]["n"] == 2
    assert by_type["lexical"]["recall"][1] == pytest.approx(0.5)
    assert by_type["lexical"]["recall"][3] == pytest.approx(1.0)
    assert by_type["lexical"]["mrr"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    assert by_type["numeric"]["n"] == 1
    assert by_type["numeric"]["recall"][3] == pytest.approx(0.0)
    assert by_type["numeric"]["mrr"] == pytest.approx(0.0)

    table = format_metrics_table(report)
    assert "overall" in table
    assert "lexical" in table
    assert "numeric" in table


def test_golden_set_schema_counts_anchors_and_paraphrase_shape() -> None:
    golden = load_golden(GOLDEN_PATH)
    counts: dict[str, int] = {}
    ids: list[str] = []

    index = DocIndex(CORPUS_DIR)
    index.refresh()
    assert index.failures() == []
    assert len(index.records()) == 13
    assert len(index.chunks) >= 20

    chunks_by_source: dict[str, list[str]] = {}
    for chunk in index.chunks:
        chunks_by_source.setdefault(chunk.source, []).append(" ".join(chunk.text.split()))

    for row in golden:
        row_id = str(row["id"])
        q_type = str(row["type"])
        question = str(row["q"])
        relevant = row["relevant"]
        ids.append(row_id)
        counts[q_type] = counts.get(q_type, 0) + 1

        if q_type in {"unanswerable", "off_topic"}:
            assert relevant == [], f"{row_id} should have empty relevant"
            continue

        assert relevant, f"{row_id} should have at least one relevant anchor"
        for rel in relevant:
            file_name = str(rel["file"])
            contains = str(rel["contains"])
            assert len(contains) < 150, f"{row_id} anchor exceeds 150 chars"
            norm_contains = " ".join(contains.split())
            file_chunks = chunks_by_source.get(file_name, [])
            assert any(norm_contains in text for text in file_chunks), (
                f"{row_id} anchor not found intact in any chunk of {file_name}"
            )
            if q_type == "paraphrase":
                q_tokens = set(tokenize(question))
                anchor_tokens = set(tokenize(contains))
                overlap = q_tokens & anchor_tokens
                assert len(overlap) <= 1, (
                    f"{row_id} shares {len(overlap)} content words with anchor: {overlap}"
                )

    assert len(ids) == len(set(ids))
    assert counts == EXPECTED_TYPE_COUNTS


def test_failure_files_named_without_changing_metrics(tmp_path: Path) -> None:
    golden = load_golden(GOLDEN_PATH)
    baseline_idx = DocIndex(CORPUS_DIR)
    baseline_idx.refresh()
    baseline_report = evaluate(baseline_idx.search, golden, ks=(1, 3, 5))

    tmp_corpus = tmp_path / "rag_corpus"
    shutil.copytree(CORPUS_DIR, tmp_corpus)
    (tmp_corpus / "empty_notes.txt").write_bytes(b"")
    (tmp_corpus / "garbage_binary.txt").write_bytes(b"\xff\xfe\x00\x80\x81\x90" * 40)

    noisy_idx = DocIndex(tmp_corpus)
    noisy_idx.refresh()
    failures = {name: status for name, status, _ in noisy_idx.failures()}
    assert failures == {"empty_notes.txt": "empty", "garbage_binary.txt": "unreadable"}

    noisy_report = evaluate(noisy_idx.search, golden, ks=(1, 3, 5))
    assert noisy_report == baseline_report


def test_retrieval_floors_on_lexical_numeric_identifier() -> None:
    golden = load_golden(GOLDEN_PATH)
    index = DocIndex(CORPUS_DIR)
    index.refresh()

    full_report = evaluate(index.search, golden, ks=(1, 3, 5))
    print("\n" + format_metrics_table(full_report, ks=(1, 3, 5)))

    floor_rows = [row for row in golden if str(row["type"]) in FLOOR_TYPES]
    floor_report = evaluate(index.search, floor_rows, ks=(1, 3, 5))
    overall = floor_report["overall"]
    assert overall["recall"][3] >= 0.9, f"Recall@3 floor missed: {overall}"
    assert overall["mrr"] >= 0.75, f"MRR floor missed: {overall}"


def test_passage_snapshot_matches_committed_and_is_deterministic(tmp_path: Path) -> None:
    out_a = dump_passages(
        tmp_path / "snap_a.json",
        folder=CORPUS_DIR,
        golden_path=GOLDEN_PATH,
        k=4,
        benchmark_path=BENCHMARK_PATH,
    )
    out_b = dump_passages(
        tmp_path / "snap_b.json",
        folder=CORPUS_DIR,
        golden_path=GOLDEN_PATH,
        k=4,
        benchmark_path=BENCHMARK_PATH,
    )

    bytes_a = out_a.read_bytes()
    bytes_b = out_b.read_bytes()
    assert bytes_a == bytes_b
    assert SNAPSHOT_PATH.is_file(), f"Missing committed snapshot: {SNAPSHOT_PATH}"
    assert bytes_a == SNAPSHOT_PATH.read_bytes()

    payload = json.loads(bytes_a.decode("utf-8"))
    assert payload["corpus"] == "tests/fixtures/rag_corpus"
    assert payload["k"] == 4
    expected_questions = set(str(row["q"]) for row in load_golden(GOLDEN_PATH)) | set(
        load_benchmark_user_turns(BENCHMARK_PATH)
    )
    assert set(payload["questions"]) == expected_questions
    for hits in payload["questions"].values():
        for hit in hits:
            assert hit["file_date"] == "2026-01-15"
