"""Tests for benchmark case schema, strict JSONL loader, and helpers."""

from __future__ import annotations

import copy
import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest

from voice_pipeline import evaluators
from voice_pipeline.benchmark_cases import (
    Case,
    CaseError,
    Expect,
    ModelCall,
    ModelScript,
    OfflineFixture,
    RouterScript,
    ToolOverride,
    ToolScript,
    Turn,
    expect_mapping,
    hard,
    load_cases,
)

DOCS = Path(__file__).resolve().parents[2] / "docs/AGENT-BENCHMARK.md"


def _valid_case(case_id: str = "c1", **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": case_id,
        "stratum": "hard",
        "provenance": "F2",
        "turns": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Who is the CEO of Intel?"},
        ],
        "expect": {
            "required_tools": ["web_answer"],
            "forbidden_tools": ["search_docs"],
            "changeable": True,
            "abstain": False,
            "passages": ["Lip-Bu Tan is CEO of Intel."],
            "must": ["Lip-Bu Tan"],
            "false_premise": False,
        },
        "known_failing": ["F2:note_reached"],
        "offline": {
            "hedged": False,
            "stamp": False,
            "router": {"reason": "need search", "delay_s": 0.1},
            "model": {
                "calls": [{"tool": "web_answer", "args": {"query": "CEO of Intel"}}],
                "text": "Lip-Bu Tan is the CEO of Intel.",
                "delay_s": 0.0,
            },
            "tools": {
                "web_answer": {
                    "result": {"answer": "Lip-Bu Tan"},
                    "delay_s": 0.05,
                    "by_router": {"result": ["fallback"], "delay_s": 0.2, "ledger": [{"x": 1}]},
                    "ledger": [{"month": "2026-11", "searches": 1}],
                }
            },
        },
        "note": "multi-turn sample",
    }
    base.update(overrides)
    return base


def _write_jsonl(tmp_path: Path, *rows: dict[str, Any] | str) -> Path:
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_cases_parses_typed_frozen_objects(tmp_path: Path) -> None:
    live_only = _valid_case("c2", stratum="happy", expect={}, note="live")
    del live_only["known_failing"]
    del live_only["offline"]
    path = _write_jsonl(tmp_path, _valid_case("c1"), "", live_only)

    cases = load_cases(path)
    assert len(cases) == 2

    c1, c2 = cases
    assert isinstance(c1, Case)
    assert c1.id == "c1"
    assert c1.stratum == "hard"
    assert c1.provenance == "F2"
    assert c1.turns == (
        Turn(role="user", content="Hi"),
        Turn(role="assistant", content="Hello."),
        Turn(role="user", content="Who is the CEO of Intel?"),
    )
    assert c1.expect == Expect(
        required_tools=("web_answer",),
        forbidden_tools=("search_docs",),
        changeable=True,
        abstain=False,
        passages=("Lip-Bu Tan is CEO of Intel.",),
        must=("Lip-Bu Tan",),
        false_premise=False,
    )
    assert c1.known_failing == ("F2:note_reached",)
    assert c1.note == "multi-turn sample"
    assert c1.offline == OfflineFixture(
        hedged=False,
        stamp=False,
        router=RouterScript(lines=None, reason="need search", delay_s=0.1),
        model=ModelScript(
            calls=(ModelCall(tool="web_answer", args={"query": "CEO of Intel"}),),
            text="Lip-Bu Tan is the CEO of Intel.",
            delay_s=0.0,
        ),
        tools={
            "web_answer": ToolScript(
                result={"answer": "Lip-Bu Tan"},
                delay_s=0.05,
                by_router=ToolOverride(
                    result=["fallback"],
                    delay_s=0.2,
                    ledger=({"x": 1},),
                ),
                ledger=({"month": "2026-11", "searches": 1},),
            )
        },
    )

    assert c2.offline is None
    assert c2.known_failing == ()
    assert c2.expect == Expect()

    with pytest.raises(dataclasses.FrozenInstanceError):
        c1.id = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        c1.expect.changeable = False  # type: ignore[misc]
    assert c1.offline is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        c1.offline.hedged = True  # type: ignore[misc]


def test_offline_defaults_hedged_and_stamp_to_true(tmp_path: Path) -> None:
    row = _valid_case(
        "c-defaults",
        offline={"model": {"text": "Hello."}},
    )
    [case] = load_cases(_write_jsonl(tmp_path, row))
    assert case.offline == OfflineFixture(
        model=ModelScript(calls=(), text="Hello.", delay_s=0.0),
        router=None,
        tools={},
        hedged=True,
        stamp=True,
    )


def test_hard_and_expect_mapping_helpers(tmp_path: Path) -> None:
    rows = [
        _valid_case("r", stratum="regression"),
        _valid_case("h", stratum="hard"),
        _valid_case("fp", stratum="false_premise"),
        _valid_case(
            "ok",
            stratum="happy",
            expect={"changeable": False, "abstain": False, "must": ["42"], "false_premise": True},
        ),
    ]
    cases = {c.id: c for c in load_cases(_write_jsonl(tmp_path, *rows))}

    assert hard(cases["r"]) is True
    assert hard(cases["h"]) is True
    assert hard(cases["fp"]) is True
    assert hard(cases["ok"]) is False

    mapped = expect_mapping(cases["ok"])
    assert mapped == {"changeable": False, "abstain": False}
    assert "must" not in mapped
    assert "false_premise" not in mapped

    verdicts = evaluators.evaluate_all(
        {"question": "What is 6 times 7?", "text": "42.", "calls": [], "ledger": {"rows": []}},
        expect_mapping(cases["ok"]),
    )
    assert len(verdicts) == len(evaluators.EVALUATORS)


def test_doc_example_case_loads(tmp_path: Path) -> None:
    text = DOCS.read_text(encoding="utf-8")
    block = re.search(r"<!-- example-case -->\s*```json\n(.*?)\n```", text, re.S)
    assert block is not None
    one_line = json.dumps(json.loads(block.group(1)))
    path = _write_jsonl(tmp_path, one_line)

    [case] = load_cases(path)
    assert case.id == "reg-f3-duplicate-nvidia"
    assert case.stratum == "regression"
    assert hard(case) is True
    assert case.known_failing == ("F3:duplicate_lookups",)
    assert case.offline is not None
    assert case.offline.hedged is True
    assert case.offline.stamp is True


# --- Invalid input classes: each asserts CaseError naming line and id ---


def test_invalid_json_syntax_raises_case_error(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, _valid_case("c1"), "{not json")
    with pytest.raises(CaseError, match=r"line 2"):
        load_cases(path)


def test_non_object_line_raises_case_error(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, "[1, 2, 3]")
    with pytest.raises(CaseError, match=r"line 1.*<unknown>"):
        load_cases(path)


@pytest.mark.parametrize("missing_key", ["id", "stratum", "provenance", "turns", "expect"])
def test_missing_required_case_key_raises_case_error(tmp_path: Path, missing_key: str) -> None:
    row = _valid_case("missing-top")
    del row[missing_key]
    path = _write_jsonl(tmp_path, _valid_case("ok-line"), row)
    expected_id = "<unknown>" if missing_key == "id" else "missing-top"
    with pytest.raises(CaseError) as exc:
        load_cases(path)
    assert "line 2" in str(exc.value)
    assert expected_id in str(exc.value)


def test_unknown_case_key_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("bad-extra", extra_field=123)
    path = _write_jsonl(tmp_path, row)
    with pytest.raises(CaseError, match=r"line 1.*bad-extra.*unknown"):
        load_cases(path)


def test_wrong_type_case_field_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("bad-note-type", note=42)
    path = _write_jsonl(tmp_path, row)
    with pytest.raises(CaseError, match=r"line 1.*bad-note-type.*note"):
        load_cases(path)


def test_empty_id_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("")
    path = _write_jsonl(tmp_path, row)
    with pytest.raises(CaseError, match=r"line 1.*id="):
        load_cases(path)


def test_duplicate_id_raises_case_error(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path, _valid_case("dup-1"), "", _valid_case("dup-1"))
    with pytest.raises(CaseError, match=r"line 3.*dup-1.*duplicate"):
        load_cases(path)


def test_unknown_stratum_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("bad-stratum", stratum="medium")
    path = _write_jsonl(tmp_path, row)
    with pytest.raises(CaseError, match=r"line 1.*bad-stratum.*stratum"):
        load_cases(path)


def test_empty_turns_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("empty-turns", turns=[])
    path = _write_jsonl(tmp_path, row)
    with pytest.raises(CaseError, match=r"line 1.*empty-turns.*turns"):
        load_cases(path)


def test_turns_unknown_role_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("bad-role", turns=[{"role": "system", "content": "hi"}])
    path = _write_jsonl(tmp_path, row)
    with pytest.raises(CaseError, match=r"line 1.*bad-role.*role"):
        load_cases(path)


def test_turns_ending_with_assistant_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case(
        "assistant-last",
        turns=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    )
    path = _write_jsonl(tmp_path, row)
    with pytest.raises(CaseError, match=r"line 1.*assistant-last.*user"):
        load_cases(path)


def test_turns_unknown_key_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("turn-key", turns=[{"role": "user", "content": "hi", "extra": 1}])
    path = _write_jsonl(tmp_path, row)
    with pytest.raises(CaseError, match=r"line 1.*turn-key.*unknown"):
        load_cases(path)


def test_expect_unknown_key_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("bad-expect-key", expect={"required": ["web_answer"]})
    path = _write_jsonl(tmp_path, row)
    with pytest.raises(CaseError, match=r"line 1.*bad-expect-key.*unknown"):
        load_cases(path)


def test_expect_wrong_bool_type_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("bad-expect-bool", expect={"changeable": "yes"})
    path = _write_jsonl(tmp_path, row)
    with pytest.raises(CaseError, match=r"line 1.*bad-expect-bool.*changeable"):
        load_cases(path)


def test_known_failing_bad_format_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("bad-kf-fmt", known_failing=["F2"])
    path = _write_jsonl(tmp_path, row)
    with pytest.raises(CaseError, match=r"line 1.*bad-kf-fmt.*known_failing"):
        load_cases(path)


def test_known_failing_unknown_evaluator_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("bad-kf-eval", known_failing=["F2:does_not_exist"])
    path = _write_jsonl(tmp_path, row)
    with pytest.raises(CaseError, match=r"line 1.*bad-kf-eval.*does_not_exist"):
        load_cases(path)


def test_offline_unknown_key_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("off-unknown")
    offline = copy.deepcopy(row["offline"])
    offline["bogus"] = True
    row["offline"] = offline
    with pytest.raises(CaseError, match=r"line 1.*off-unknown.*unknown"):
        load_cases(_write_jsonl(tmp_path, row))


def test_offline_missing_model_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("off-no-model", offline={"hedged": True})
    with pytest.raises(CaseError, match=r"line 1.*off-no-model.*model"):
        load_cases(_write_jsonl(tmp_path, row))


def test_offline_bool_instead_of_delay_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("off-bool-delay", offline={"model": {"text": "ok", "delay_s": True}})
    with pytest.raises(CaseError, match=r"line 1.*off-bool-delay.*delay_s"):
        load_cases(_write_jsonl(tmp_path, row))


def test_offline_negative_delay_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("off-neg-delay", offline={"model": {"text": "ok", "delay_s": -0.1}})
    with pytest.raises(CaseError, match=r"line 1.*off-neg-delay.*delay_s"):
        load_cases(_write_jsonl(tmp_path, row))


def test_offline_router_both_lines_and_reason_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case(
        "off-router-both",
        offline={"router": {"lines": "", "reason": "x"}, "model": {"text": "ok"}},
    )
    with pytest.raises(CaseError, match=r"line 1.*off-router-both.*lines.*reason"):
        load_cases(_write_jsonl(tmp_path, row))


def test_offline_router_neither_lines_nor_reason_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case(
        "off-router-neither",
        offline={"router": {"delay_s": 0.1}, "model": {"text": "ok"}},
    )
    with pytest.raises(CaseError, match=r"line 1.*off-router-neither.*lines.*reason"):
        load_cases(_write_jsonl(tmp_path, row))


def test_offline_model_missing_text_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case("off-model-no-text", offline={"model": {"calls": []}})
    with pytest.raises(CaseError, match=r"line 1.*off-model-no-text.*text"):
        load_cases(_write_jsonl(tmp_path, row))


def test_offline_model_call_wrong_args_type_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case(
        "off-call-args",
        offline={"model": {"calls": [{"tool": "web_answer", "args": ["bad"]}], "text": "ok"}},
    )
    with pytest.raises(CaseError, match=r"line 1.*off-call-args.*args"):
        load_cases(_write_jsonl(tmp_path, row))


def test_offline_tool_missing_result_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case(
        "off-tool-no-res",
        offline={"model": {"text": "ok"}, "tools": {"web_answer": {"delay_s": 0.0}}},
    )
    with pytest.raises(CaseError, match=r"line 1.*off-tool-no-res.*result"):
        load_cases(_write_jsonl(tmp_path, row))


def test_offline_tool_invalid_result_type_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case(
        "off-tool-bad-res",
        offline={"model": {"text": "ok"}, "tools": {"web_answer": {"result": 123}}},
    )
    with pytest.raises(CaseError, match=r"line 1.*off-tool-bad-res.*result"):
        load_cases(_write_jsonl(tmp_path, row))


def test_offline_tool_override_unknown_key_raises_case_error(tmp_path: Path) -> None:
    row = _valid_case(
        "off-override-key",
        offline={
            "model": {"text": "ok"},
            "tools": {"web_answer": {"result": "ok", "by_router": {"by_router": {}}}},
        },
    )
    with pytest.raises(CaseError, match=r"line 1.*off-override-key.*unknown"):
        load_cases(_write_jsonl(tmp_path, row))


def test_offline_model_parallel_loads_true_and_defaults_false(tmp_path: Path) -> None:
    row_true = _valid_case("off-par-true", offline={"model": {"text": "ok", "parallel": True}})
    row_default = _valid_case("off-par-default", offline={"model": {"text": "ok"}})
    [case_true, case_default] = load_cases(_write_jsonl(tmp_path, row_true, row_default))

    assert case_true.offline is not None
    assert case_true.offline.model.parallel is True
    assert case_default.offline is not None
    assert case_default.offline.model.parallel is False


@pytest.mark.parametrize("bad_parallel", ["yes", 1])
def test_offline_model_non_bool_parallel_raises_case_error(
    tmp_path: Path, bad_parallel: Any
) -> None:
    row = _valid_case(
        "off-bad-parallel",
        offline={"model": {"text": "ok", "parallel": bad_parallel}},
    )
    with pytest.raises(CaseError, match=r"line 1.*off-bad-parallel.*parallel.*bool"):
        load_cases(_write_jsonl(tmp_path, row))
