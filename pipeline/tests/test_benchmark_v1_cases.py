"""Verification tests for the full ``pipeline/config/eval/benchmark_v1.jsonl`` suite."""

from __future__ import annotations

import asyncio
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from voice_pipeline import bench_fakes, benchmark, benchmark_cases

REPO_ROOT = Path(__file__).resolve().parents[2]
FRESHQA_CSV = REPO_ROOT / "tests/fixtures/freshqa_sample.csv"


def _load_freshqa_rows() -> list[dict[str, str]]:
    with FRESHQA_CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_report(out_dir: Path) -> dict[str, Any]:
    reports = sorted(out_dir.glob("*.json"))
    assert len(reports) == 1
    data: dict[str, Any] = json.loads(reports[0].read_text(encoding="utf-8"))
    return data


def test_benchmark_v1_strata_counts_hard_share_and_unique_ids() -> None:
    cases = benchmark_cases.load_cases(benchmark.DEFAULT_CASES)

    assert len(cases) == 39
    assert len({case.id for case in cases}) == len(cases)

    counts = Counter(case.stratum for case in cases)
    assert counts == {
        "regression": 13,
        "false_premise": 8,
        "hard": 6,
        "happy": 12,
    }

    hard_count = sum(1 for case in cases if benchmark_cases.hard(case))
    assert hard_count == 27
    assert hard_count * 2 >= len(cases)


def test_benchmark_v1_freshqa_selection_matches_csv() -> None:
    cases = benchmark_cases.load_cases(benchmark.DEFAULT_CASES)
    rows = _load_freshqa_rows()
    by_fid = {row["id"]: row for row in rows}

    # Fixed selection rule: sort by integer FreshQA id ascending and take the first N in each class.
    expected_fp_rows = sorted(
        (row for row in rows if row["false_premise"] == "TRUE"),
        key=lambda row: int(row["id"]),
    )[:8]
    expected_hard_rows = sorted(
        (
            row
            for row in rows
            if row["false_premise"] == "FALSE"
            and row["fact_type"] in {"fast-changing", "slow-changing"}
        ),
        key=lambda row: int(row["id"]),
    )[:6]

    fp_cases = [case for case in cases if case.stratum == "false_premise"]
    assert len(fp_cases) == 8
    fp_fids: list[str] = []
    for case in fp_cases:
        assert case.offline is None
        assert case.expect.false_premise is True
        assert case.note == "premise correction: judge pending"
        assert case.provenance.startswith("freshqa:")
        fid = case.provenance.removeprefix("freshqa:")
        assert fid in by_fid
        assert len(case.turns) == 1
        assert case.turns[0].content == by_fid[fid]["question"].strip()
        fp_fids.append(fid)
    assert fp_fids == [row["id"] for row in expected_fp_rows]

    hard_cases = [case for case in cases if case.stratum == "hard"]
    assert len(hard_cases) == 6
    hard_fids: list[str] = []
    for case in hard_cases:
        assert case.offline is None
        assert case.expect.changeable is True
        assert case.expect.required_tools == ("web_answer",)
        assert case.provenance.startswith("freshqa:")
        fid = case.provenance.removeprefix("freshqa:")
        assert fid in by_fid
        assert len(case.turns) == 1
        assert case.turns[0].content == by_fid[fid]["question"].strip()
        hard_fids.append(fid)
    assert hard_fids == [row["id"] for row in expected_hard_rows]


def test_benchmark_v1_full_file_offline_run_and_happy_cases(tmp_path: Path) -> None:
    cases = benchmark_cases.load_cases(benchmark.DEFAULT_CASES)
    out_dir = tmp_path / "out"

    exit_code = benchmark.main(["--cases", str(benchmark.DEFAULT_CASES), "--out", str(out_dir)])
    assert exit_code == 0

    report = _load_report(out_dir)
    assert report["known_failing_ok"] is True

    expected_skipped = [case.id for case in cases if case.offline is None]
    assert len(expected_skipped) == 20
    assert report["skipped_live_only"] == expected_skipped

    expected_offline_ids = {case.id for case in cases if case.offline is not None}
    assert len(expected_offline_ids) == 19
    assert {item["id"] for item in report["cases"]} == expected_offline_ids

    happy_results = [item for item in report["cases"] if item["stratum"] == "happy"]
    assert len(happy_results) == 12
    for item in happy_results:
        assert item["known_failing"]["declared"] == []
        assert item["known_failing"]["ok"] is True
        failed = [v for v in item["verdicts"] if v["outcome"] == "fail"]
        assert failed == [], f"{item['id']} had failing verdict(s): {failed}"


def test_benchmark_v1_happy_cases_exercise_expected_routes(tmp_path: Path) -> None:
    cases_by_id = {case.id: case for case in benchmark_cases.load_cases(benchmark.DEFAULT_CASES)}

    async def run_happy() -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for case_id, case in cases_by_id.items():
            if case.stratum == "happy":
                rows[case_id] = await bench_fakes.run_offline_case(case, tmp_path / case_id)
        return rows

    rows = asyncio.run(run_happy())
    assert set(rows) == {
        "happy-weather-routed",
        "happy-clock-fast",
        "happy-settled-fact-gemini",
        "happy-lights-fast",
        "happy-joke-gemini",
        "happy-backchannel",
        "rag-cited-answer",
        "rag-silent-file",
        "rag-hostile-file",
        "rag-failure-named",
        "rag-changed-file",
        "rag-implicit-question",
    }

    assert rows["happy-weather-routed"]["route"] == "routed"
    assert rows["happy-weather-routed"]["routed"] is True
    assert [c["tool"] for c in rows["happy-weather-routed"]["calls"]] == ["weather"]

    assert rows["happy-clock-fast"]["route"] == "fast"
    assert [c["tool"] for c in rows["happy-clock-fast"]["calls"]] == ["clock"]

    assert rows["happy-settled-fact-gemini"]["route"] == "routed"
    assert not rows["happy-settled-fact-gemini"].get("routed")
    assert rows["happy-settled-fact-gemini"]["calls"] == []
    assert "Paris" in rows["happy-settled-fact-gemini"]["text"]

    assert rows["happy-lights-fast"]["route"] == "fast"
    assert [c["tool"] for c in rows["happy-lights-fast"]["calls"]] == ["smart_home"]

    assert rows["happy-joke-gemini"]["route"] == "routed"
    assert not rows["happy-joke-gemini"].get("routed")
    assert rows["happy-joke-gemini"]["calls"] == []
    assert rows["happy-joke-gemini"]["text"]

    assert rows["happy-backchannel"]["route"] == "backchannel"
    assert rows["happy-backchannel"]["calls"] == []
    assert rows["happy-backchannel"]["text"] == ""
