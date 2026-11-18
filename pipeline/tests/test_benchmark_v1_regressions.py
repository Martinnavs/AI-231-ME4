"""Verification tests for the 13 regression cases in ``pipeline/config/eval/benchmark_v1.jsonl``."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from voice_pipeline import benchmark, benchmark_cases


def _load_report(out_dir: Path) -> dict[str, Any]:
    reports = sorted(out_dir.glob("*.json"))
    assert len(reports) == 1
    data: dict[str, Any] = json.loads(reports[0].read_text(encoding="utf-8"))
    return data


def test_benchmark_v1_schema_and_live_notes() -> None:
    cases = [
        case
        for case in benchmark_cases.load_cases(benchmark.DEFAULT_CASES)
        if case.stratum == "regression"
    ]
    assert len(cases) == 13
    assert all(case.stratum == "regression" for case in cases)
    assert all(
        "docs/AGENT-ASSESSMENT.md" in case.provenance or "docs/SEARCH-SPEC.md" in case.provenance
        for case in cases
    )

    live_only = [case for case in cases if case.offline is None]
    assert len(live_only) == 6
    for case in live_only:
        assert case.note is not None and case.note.strip()


def test_benchmark_v1_regressions_run_and_match_declared_known_failing(tmp_path: Path) -> None:
    regression_ids = [
        case.id
        for case in benchmark_cases.load_cases(benchmark.DEFAULT_CASES)
        if case.stratum == "regression"
    ]
    out_dir = tmp_path / "out"
    exit_code = benchmark.main(
        ["--cases", str(benchmark.DEFAULT_CASES), "--out", str(out_dir), "--only", *regression_ids]
    )
    assert exit_code == 0

    report = _load_report(out_dir)
    assert report["known_failing_ok"] is True

    by_id = {item["id"]: item for item in report["cases"]}
    assert set(by_id) == {
        "reg-f2-hedged-no-note",
        "reg-f3-duplicate-lookups",
        "reg-f14-joined-failure-from-memory",
        "reg-f14-control-sequential",
        "reg-f19-polite-prefix-note",
        "reg-f17-sources-substring-backstop",
        "reg-control-hedged-router-wins",
    }

    expected_verdicts: dict[str, dict[str, str]] = {
        "reg-f2-hedged-no-note": {
            "lookup_required": "pass",
            "forbidden_absent": "pass",
            "searched": "pass",
            "duplicate_lookups": "pass",
            "note_reached": "pass",
            "support": "pass",
            "source_coverage": "pass",
            "abstention": "pass",
            "doc_citation": "n/a",
        },
        "reg-f3-duplicate-lookups": {
            "lookup_required": "pass",
            "forbidden_absent": "pass",
            "searched": "n/a",
            "duplicate_lookups": "fail",
            "note_reached": "n/a",
            "support": "pass",
            "source_coverage": "pass",
            "abstention": "pass",
            "doc_citation": "n/a",
        },
        "reg-f14-joined-failure-from-memory": {
            "lookup_required": "pass",
            "forbidden_absent": "n/a",
            "searched": "n/a",
            "duplicate_lookups": "pass",
            "note_reached": "n/a",
            "support": "fail",
            "source_coverage": "n/a",
            "abstention": "n/a",
            "doc_citation": "n/a",
        },
        "reg-f14-control-sequential": {
            "lookup_required": "pass",
            "forbidden_absent": "n/a",
            "searched": "n/a",
            "duplicate_lookups": "pass",
            "note_reached": "n/a",
            "support": "pass",
            "source_coverage": "n/a",
            "abstention": "n/a",
            "doc_citation": "n/a",
        },
        "reg-f19-polite-prefix-note": {
            "lookup_required": "pass",
            "forbidden_absent": "n/a",
            "searched": "pass",
            "duplicate_lookups": "pass",
            "note_reached": "fail",
            "support": "pass",
            "source_coverage": "pass",
            "abstention": "pass",
            "doc_citation": "n/a",
        },
        "reg-f17-sources-substring-backstop": {
            "lookup_required": "pass",
            "forbidden_absent": "n/a",
            "searched": "pass",
            "duplicate_lookups": "pass",
            "note_reached": "pass",
            "support": "pass",
            "source_coverage": "fail",
            "abstention": "pass",
            "doc_citation": "n/a",
        },
        "reg-control-hedged-router-wins": {
            "lookup_required": "pass",
            "forbidden_absent": "pass",
            "searched": "pass",
            "duplicate_lookups": "pass",
            "note_reached": "n/a",
            "support": "pass",
            "source_coverage": "pass",
            "abstention": "pass",
            "doc_citation": "n/a",
        },
    }
    for case_id, want in expected_verdicts.items():
        got = {v["name"]: v["outcome"] for v in by_id[case_id]["verdicts"]}
        assert got == want, f"{case_id}: {by_id[case_id]['verdicts']}"
        assert by_id[case_id]["known_failing"]["ok"] is True

    assert by_id["reg-f2-hedged-no-note"]["known_failing"]["declared"] == []
    assert by_id["reg-f3-duplicate-lookups"]["known_failing"]["declared"] == [
        "F3:duplicate_lookups"
    ]
    assert by_id["reg-f14-joined-failure-from-memory"]["known_failing"]["declared"] == [
        "F14:support"
    ]

    assert report["skipped_live_only"] == [
        "reg-f20-tavily-closest-result",
        "reg-s6-microsoft-ceo-zero-search",
        "reg-s3-intel-ceo-memory",
        "reg-s6-nvidia-ceo-relevance",
        "reg-f15-who-is-was-cache",
        "reg-f6-error-in-news",
    ]


def test_benchmark_v1_regressions_are_deterministic(tmp_path: Path) -> None:
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    assert benchmark.main(["--cases", str(benchmark.DEFAULT_CASES), "--out", str(out1)]) == 0
    assert benchmark.main(["--cases", str(benchmark.DEFAULT_CASES), "--out", str(out2)]) == 0

    r1 = _load_report(out1)
    r2 = _load_report(out2)
    assert r1["evaluators"] == r2["evaluators"]
    assert r1["skipped_live_only"] == r2["skipped_live_only"]
    assert [(c["id"], c["verdicts"], c["known_failing"]) for c in r1["cases"]] == [
        (c["id"], c["verdicts"], c["known_failing"]) for c in r2["cases"]
    ]


def _f14_support(tmp_path: Path, name: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    cases_path = tmp_path / f"{name}.jsonl"
    cases_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out_dir = tmp_path / f"{name}-out"
    assert benchmark.main(["--cases", str(cases_path), "--out", str(out_dir)]) == 0
    result: dict[str, str] = {}
    for item in _load_report(out_dir)["cases"]:
        verdicts = {v["name"]: v["outcome"] for v in item["verdicts"]}
        result[item["id"]] = verdicts["support"]
    return result


def test_f14_case_discriminates_join_from_authored_text(tmp_path: Path) -> None:
    rows = {
        json.loads(line)["id"]: json.loads(line)
        for line in benchmark.DEFAULT_CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    parallel = rows["reg-f14-joined-failure-from-memory"]
    control = rows["reg-f14-control-sequential"]
    assert parallel["known_failing"] == ["F14:support"]
    assert "known_failing" not in control
    assert parallel["offline"]["model"]["parallel"] is True
    assert not control["offline"]["model"].get("parallel")
    assert parallel["offline"]["model"]["text"] == control["offline"]["model"]["text"]

    got = _f14_support(tmp_path, "both", [parallel, control])
    assert got == {
        "reg-f14-joined-failure-from-memory": "fail",
        "reg-f14-control-sequential": "pass",
    }

    unjoined = copy.deepcopy(parallel)
    unjoined["id"] = "f14-unjoined"
    unjoined["offline"]["model"]["parallel"] = False
    unjoined.pop("known_failing")
    assert _f14_support(tmp_path, "unjoined", [unjoined]) == {"f14-unjoined": "pass"}
