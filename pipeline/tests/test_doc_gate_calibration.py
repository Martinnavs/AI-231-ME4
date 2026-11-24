"""Offline calibration and false-trigger checks for the implicit docs gate over the T4 snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voice_pipeline import doc_route

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = Path(__file__).resolve().parent / "fixtures" / "rag_passages.json"
GOLDEN_PATH = REPO_ROOT / "config" / "eval" / "rag_golden.jsonl"
BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "config" / "eval" / "benchmark_v1.jsonl"


def _load_snapshot() -> dict[str, Any]:
    data = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _load_golden() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_non_rag_benchmark_turns() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for line in BENCHMARK_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        cid = str(case.get("id") or "")
        if cid.startswith("rag-"):
            continue
        for turn in case.get("turns", []):
            if turn.get("role") == "user" and str(turn.get("content", "")).strip():
                out.append((cid, str(turn["content"])))
    return out


def test_implicit_gate_calibration_and_zero_false_triggers() -> None:
    snap = _load_snapshot()["questions"]
    golden = _load_golden()
    bench = _load_non_rag_benchmark_turns()
    assert len(bench) >= 33

    by_type: dict[str, list[bool]] = {}
    for row in golden:
        q = str(row["q"])
        passages = snap[q]
        triggered = not doc_route.refers_to_documents(q) and doc_route.implicit_gate_passes(
            q, passages
        )
        by_type.setdefault(str(row["type"]), []).append(triggered)

    bench_triggers = [
        (cid, q)
        for cid, q in bench
        if not doc_route.refers_to_documents(q) and doc_route.implicit_gate_passes(q, snap[q])
    ]

    summary = {
        q_type: f"{sum(flags)}/{len(flags)} ({sum(flags) / len(flags):.3f})"
        for q_type, flags in by_type.items()
    }
    print(
        f"\n[doc_gate theta={doc_route.IMPLICIT_COVERAGE_THRESHOLD}] "
        f"golden={summary} benchmark_v1={len(bench_triggers)}/{len(bench)}"
    )

    # Zero false triggers on off_topic golden rows and non-rag benchmark_v1 user turns.
    assert sum(by_type["off_topic"]) == 0, f"off_topic false triggers: {by_type['off_topic']}"
    assert bench_triggers == [], f"benchmark_v1 false triggers: {bench_triggers}"

    # Measured at best(top3) >= 0.34: 25/29 answerable (86.2%), 0 off_topic, 0/34 benchmark_v1.
    # Assert >= measured - 0.10 per ticket rule; paraphrase and taglish are logged only.
    for key in ("lexical", "numeric", "identifier"):
        flags = by_type[key]
        rate = sum(flags) / len(flags)
        assert rate >= 0.90, f"{key} trigger rate {rate:.3f} below floor 0.90"
