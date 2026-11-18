"""Tests for the offline benchmark runner CLI ``voice_pipeline.benchmark``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import LOOKED_UP, LOOKUP, USER
from voice_pipeline import agent_server, bench_fakes, benchmark, evaluators


def _write_cases(path: Path, cases: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    return path


def _happy_case(case_id: str = "happy-nvidia") -> dict[str, Any]:
    return {
        "id": case_id,
        "stratum": "happy",
        "provenance": "test",
        "turns": [{"role": "user", "content": USER}],
        "expect": {
            "required_tools": ["web_answer"],
            "forbidden_tools": ["search_docs"],
            "changeable": True,
            "abstain": False,
        },
        "offline": {
            "hedged": False,
            "stamp": True,
            "router": {"reason": "escalate for test"},
            "model": {
                "calls": [{"tool": "web_answer", "args": {"query": "ceo of nvidia"}}],
                "text": LOOKED_UP,
            },
            "tools": {
                "web_answer": {
                    "result": LOOKED_UP,
                    "ledger": [{"month": "2026-11", "searches": 1, "by": "model"}],
                }
            },
        },
    }


def _known_failing_f3_case(
    case_id: str = "reg-f3-dup",
    *,
    duplicate: bool = True,
    known_failing: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "stratum": "regression",
        "provenance": "F3",
        "turns": [{"role": "user", "content": "Tell me about Nvidia's CEO."}],
        "expect": {
            "required_tools": ["web_answer"],
            "changeable": False,
            "abstain": False,
        },
        "known_failing": (known_failing if known_failing is not None else ["F3:duplicate_lookups"]),
        "offline": {
            "hedged": duplicate,
            "stamp": True,
            "router": {"lines": LOOKUP} if duplicate else {"reason": "direct"},
            "model": {
                "calls": [{"tool": "web_answer", "args": {"query": "ceo of nvidia"}}],
                "text": LOOKED_UP,
            },
            "tools": {
                "web_answer": {
                    "result": LOOKED_UP,
                    "ledger": [{"month": "2026-11", "searches": 1}],
                    "by_router": {"delay_s": 30.0, "result": LOOKED_UP},
                }
            },
        },
    }


def _live_only_case(case_id: str = "live-fp-1") -> dict[str, Any]:
    return {
        "id": case_id,
        "stratum": "false_premise",
        "provenance": "freshqa:test",
        "turns": [{"role": "user", "content": "When did Mars become a US state?"}],
        "expect": {"false_premise": True},
        "note": "Needs live model behaviour.",
    }


def _load_single_report(out_dir: Path) -> dict[str, Any]:
    reports = sorted(out_dir.glob("*.json"))
    assert len(reports) == 1
    data: dict[str, Any] = json.loads(reports[0].read_text(encoding="utf-8"))
    return data


def test_offline_run_report_shape_header_printed_rates_and_live_skip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_probe(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("OnlineProbe must not be used offline")

    monkeypatch.setattr(agent_server, "OnlineProbe", fail_probe)

    cases_file = _write_cases(
        tmp_path / "cases.jsonl",
        [_happy_case(), _known_failing_f3_case(), _live_only_case("live-skip-1")],
    )
    out_dir = tmp_path / "out"

    code = benchmark.main(["--cases", str(cases_file), "--out", str(out_dir)])
    assert code == 0

    captured = capsys.readouterr()
    assert "offline: numbers test server code, not model behaviour" in captured.out
    assert "live-skip-1" in captured.out
    assert "2 offline, 1 skipped (live-only)" in captured.out

    report = _load_single_report(out_dir)
    assert report["header"] == "offline: numbers test server code, not model behaviour"
    assert report["mode"] == "offline"
    assert report["skipped_live_only"] == ["live-skip-1"]
    assert report["known_failing_ok"] is True

    assert [c["id"] for c in report["cases"]] == ["happy-nvidia", "reg-f3-dup"]
    for item in report["cases"]:
        assert isinstance(item["turn"], str) and item["turn"].startswith(bench_fakes.RUN_STEM)
        verdicts = item["verdicts"]
        assert [v["name"] for v in verdicts] == list(evaluators.EVALUATORS)
        for v in verdicts:
            assert set(v) == {"name", "outcome", "reason"}
            assert v["outcome"] in ("pass", "fail", "n/a")
        assert set(item["known_failing"]) == {
            "declared",
            "expected_evaluators",
            "failed_evaluators",
            "missing",
            "unexpected",
            "ok",
        }
        assert item["known_failing"]["ok"] is True

    f3_kf = report["cases"][1]["known_failing"]
    assert f3_kf["declared"] == ["F3:duplicate_lookups"]
    assert f3_kf["failed_evaluators"] == ["duplicate_lookups"]

    eval_stats = report["evaluators"]
    assert list(eval_stats) == list(evaluators.EVALUATORS)

    dup = eval_stats["duplicate_lookups"]
    assert dup["overall"] == {
        "pass": 1,
        "fail": 1,
        "n/a": 0,
        "total": 2,
        "applicable": 2,
        "pass_rate": 0.5,
    }
    assert dup["by_stratum"]["happy"]["pass_rate"] == 1.0
    assert dup["by_stratum"]["regression"]["pass_rate"] == 0.0
    assert "false_premise" not in dup["by_stratum"]

    forb = eval_stats["forbidden_absent"]
    assert forb["overall"] == {
        "pass": 1,
        "fail": 0,
        "n/a": 1,
        "total": 2,
        "applicable": 1,
        "pass_rate": 1.0,
    }
    assert forb["by_stratum"]["regression"]["pass_rate"] is None

    # Printed output includes the same per-evaluator pass rates overall and by stratum.
    assert "duplicate_lookups" in captured.out
    assert "50.0%" in captured.out
    assert "100.0%" in captured.out
    assert "0.0%" in captured.out


def test_determinism_two_runs_produce_identical_rates_and_verdicts(tmp_path: Path) -> None:
    cases_file = _write_cases(
        tmp_path / "cases.jsonl",
        [_happy_case(), _known_failing_f3_case(), _live_only_case()],
    )
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    assert benchmark.main(["--cases", str(cases_file), "--out", str(out1)]) == 0
    assert benchmark.main(["--cases", str(cases_file), "--out", str(out2)]) == 0

    r1 = _load_single_report(out1)
    r2 = _load_single_report(out2)
    assert r1["evaluators"] == r2["evaluators"]
    assert r1["skipped_live_only"] == r2["skipped_live_only"]
    assert [(c["id"], c["verdicts"], c["known_failing"]) for c in r1["cases"]] == [
        (c["id"], c["verdicts"], c["known_failing"]) for c in r2["cases"]
    ]


def test_exit_code_1_when_known_failing_passes_or_fails_wrong_evaluator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Case 1: declared F3:duplicate_lookups actually passes -> exit 1.
    passes_file = _write_cases(
        tmp_path / "passes.jsonl",
        [_happy_case(), _known_failing_f3_case("fixed-f3", duplicate=False)],
    )
    code1 = benchmark.main(["--cases", str(passes_file), "--out", str(tmp_path / "out1")])
    assert code1 == 1
    out_text = capsys.readouterr().out
    assert "known_failing mismatch for fixed-f3" in out_text
    r1 = _load_single_report(tmp_path / "out1")
    assert r1["known_failing_ok"] is False
    assert r1["cases"][1]["known_failing"]["missing"] == ["duplicate_lookups"]

    # Case 2: declared evaluator does not fail, a different evaluator fails -> exit 1.
    wrong_eval = _known_failing_f3_case(
        "wrong-eval", duplicate=False, known_failing=["F3:duplicate_lookups"]
    )
    wrong_eval["expect"]["forbidden_tools"] = ["web_answer"]
    wrong_file = _write_cases(tmp_path / "wrong.jsonl", [wrong_eval])
    code2 = benchmark.main(["--cases", str(wrong_file), "--out", str(tmp_path / "out2")])
    assert code2 == 1
    r2 = _load_single_report(tmp_path / "out2")
    assert r2["cases"][0]["known_failing"]["missing"] == ["duplicate_lookups"]
    assert r2["cases"][0]["known_failing"]["unexpected"] == ["forbidden_absent"]

    # Case 3: declared evaluator fails PLUS an unexpected evaluator fails -> exit 1.
    extra_fail = _known_failing_f3_case(
        "extra-fail", duplicate=True, known_failing=["F3:duplicate_lookups"]
    )
    extra_fail["expect"]["forbidden_tools"] = ["web_answer"]
    extra_file = _write_cases(tmp_path / "extra.jsonl", [extra_fail])
    code3 = benchmark.main(["--cases", str(extra_file), "--out", str(tmp_path / "out3")])
    assert code3 == 1
    r3 = _load_single_report(tmp_path / "out3")
    assert r3["cases"][0]["known_failing"]["missing"] == []
    assert r3["cases"][0]["known_failing"]["unexpected"] == ["forbidden_absent"]


def test_exit_code_2_on_harness_errors_and_0_when_default_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Missing default case file -> prints message and exits 0.
    monkeypatch.setattr(benchmark, "DEFAULT_CASES", tmp_path / "missing_default.jsonl")
    assert benchmark.main(["--out", str(tmp_path / "out0")]) == 0
    assert "no default case file" in capsys.readouterr().out

    # Explicit --cases pointing to missing file -> exit 2.
    assert benchmark.main(["--cases", str(tmp_path / "nope.jsonl")]) == 2
    assert "error:" in capsys.readouterr().out

    # Bad JSONL case file -> exit 2.
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("{not valid json\n", encoding="utf-8")
    assert benchmark.main(["--cases", str(bad_file), "--out", str(tmp_path / "out_bad")]) == 2
    assert "error loading cases" in capsys.readouterr().out

    # Exception inside run_offline_case -> exit 2.
    good_file = _write_cases(tmp_path / "good.jsonl", [_happy_case()])

    async def boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("simulated harness crash")

    monkeypatch.setattr(bench_fakes, "run_offline_case", boom)
    assert benchmark.main(["--cases", str(good_file), "--out", str(tmp_path / "out_err")]) == 2
    assert "simulated harness crash" in capsys.readouterr().out


def test_only_filters_cases_and_rejects_unknown_ids(tmp_path: Path) -> None:
    cases_file = _write_cases(
        tmp_path / "cases.jsonl",
        [_happy_case("c1"), _known_failing_f3_case("c2"), _live_only_case("c3")],
    )
    out_dir = tmp_path / "out_only"

    code = benchmark.main(["--cases", str(cases_file), "--out", str(out_dir), "--only", "c1", "c3"])
    assert code == 0
    report = _load_single_report(out_dir)
    assert [c["id"] for c in report["cases"]] == ["c1"]
    assert report["skipped_live_only"] == ["c3"]

    assert (
        benchmark.main(
            ["--cases", str(cases_file), "--out", str(tmp_path / "out_missing"), "--only", "nope"]
        )
        == 2
    )


def test_registry_lists_all_evaluators_and_returns_no_aggregate_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cases_file = _write_cases(tmp_path / "cases.jsonl", [_happy_case()])
    out_dir = tmp_path / "out_no_agg"

    assert benchmark.main(["--cases", str(cases_file), "--out", str(out_dir)]) == 0
    out_text = capsys.readouterr().out
    report = _load_single_report(out_dir)

    assert list(report["evaluators"]) == list(evaluators.EVALUATORS)
    forbidden_keys = {"aggregate", "composite", "score", "overall_score", "total_score"}
    assert not (forbidden_keys & set(report))
    assert not (forbidden_keys & set(report["evaluators"]))
    lowered = out_text.lower()
    assert "composite" not in lowered
    assert "aggregate" not in lowered
