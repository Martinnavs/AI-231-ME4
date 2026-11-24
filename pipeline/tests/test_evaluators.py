"""Tests for single-purpose code evaluators over real T1 `AgentServer.handle` turn rows."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from conftest import LOOKED_UP
from voice_pipeline import agent_tools, bench_fakes, doc_route, evaluators
from voice_pipeline.agent_server import AgentServer
from voice_pipeline.router import Routed, spec

ROOT_GROUNDING = (
    Path(__file__).resolve().parents[2] / "src/voice_agent/mcp_servers/gemini_grounding.py"
)


class _FakeRouter:
    def __init__(self, lines: str = "", reason: str | None = None) -> None:
        self.lines, self.reason = lines, reason

    async def route(self, history: Any, user: str, names: Any) -> Routed:
        if self.reason is not None:
            return Routed(reason=self.reason)
        return Routed(spec.parse_output(self.lines), self.lines, 0.1, None, user)


class _ScriptedAgent:
    """Fake non-hedged agent calling optional tools via `_recorded` and speaking `reply`."""

    def __init__(
        self,
        tools: ToolsSchema,
        calls: list[tuple[str, dict[str, Any]]] | None = None,
        reply: str = "",
    ) -> None:
        self.handlers = {t.name: t.handler for t in tools.standard_tools}
        self.calls = calls or []
        self.reply = reply

    async def ask(self, messages: Any, *, on_text: Any, **_: Any) -> dict[str, Any]:
        got: list[Any] = []

        async def keep(result: Any, **_kw: Any) -> None:
            got.append(result)

        for idx, (tool_name, args) in enumerate(self.calls, start=1):
            params = SimpleNamespace(
                arguments=dict(args),
                result_callback=keep,
                tool_call_id=f"model-{idx}",
                function_name=tool_name,
            )
            handler = self.handlers[tool_name]
            assert handler is not None
            await handler(params)
        text = self.reply if self.reply else (str(got[0]) if got else "")
        if text:
            await on_text(text)
        return {"status": "ok", "text": text}

    async def cancel(self) -> None:
        return None


async def _serve_custom(
    serve_turn: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    question: str,
    lookup_result: str = LOOKED_UP,
    reply: str = "",
    router_lines: str = "",
    router_reason: str | None = "test: escalate",
    model_calls: list[tuple[str, dict[str, Any]]] | None = None,
    ledger_rows: list[dict[str, Any]] | None = None,
    tool_name: str = "web_answer",
    stamp: bool = True,
) -> dict[str, Any]:
    """Run one turn through the real `AgentServer.handle` and return its logged T1 row."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent_tools, "CHAIN", None)
    monkeypatch.setattr(agent_tools, "SCRATCHPAD", None)
    ledger = tmp_path / "ledger.jsonl"
    record: list[dict[str, Any]] = []

    async def handler(params: Any) -> None:
        for row in ledger_rows or []:
            with ledger.open("a") as sink:
                sink.write(json.dumps(row) + "\n")
        await params.result_callback(lookup_result)

    schema = FunctionSchema(
        name=tool_name, description="d", properties={}, required=[], handler=handler
    )
    tools = ToolsSchema(standard_tools=[agent_tools._recorded(schema, record)])
    agent = _ScriptedAgent(tools, calls=model_calls, reply=reply)
    router = _FakeRouter(lines=router_lines, reason=router_reason)
    server = AgentServer(agent, record, tmp_path / "20261118-120000.jsonl", tools, router)
    server.ledger = ledger
    if stamp:
        server.stamp = lambda: "[Now: test]"
    _, row = await serve_turn(server, question)
    return row


def test_hedge_string_is_pinned_against_gemini_grounding() -> None:
    """`gemini_grounding.py` cannot be imported here: pin HEDGE via AST."""
    tree = ast.parse(ROOT_GROUNDING.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "HEDGE":
            assert ast.literal_eval(node.value) == evaluators.HEDGE
            return
    raise AssertionError("HEDGE not found in gemini_grounding.py")


async def test_registry_lists_all_evaluators_and_returns_no_aggregate_score(
    make_world: Any, serve_turn: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    assert list(evaluators.EVALUATORS) == [
        "lookup_required",
        "forbidden_absent",
        "searched",
        "duplicate_lookups",
        "note_reached",
        "support",
        "source_coverage",
        "abstention",
        "doc_citation",
    ]
    world = make_world(model_wins=False)
    _, row = await serve_turn(world.server)
    verdicts = evaluators.evaluate_all(row, {})
    assert isinstance(verdicts, list)
    assert all(isinstance(v, evaluators.Verdict) for v in verdicts)
    assert [v.name for v in verdicts] == list(evaluators.EVALUATORS)
    assert all(v.outcome in ("pass", "fail", "n/a") for v in verdicts)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


# --- lookup_required ---------------------------------------------------------------------------


async def test_lookup_required_pass_fail_na(make_world: Any, serve_turn: Any) -> None:
    world_pass = make_world(model_wins=False)
    _, row_pass = await serve_turn(world_pass.server)
    assert (
        evaluators.evaluate_lookup_required(row_pass, {"required_tools": ["web_answer"]}).outcome
        == "pass"
    )

    # Hedged model answers from memory without calling web_answer; router's web_answer is cancelled.
    world_fail = make_world(model_wins=True)
    world_fail.agents[1].tool = None
    _, row_fail = await serve_turn(world_fail.server)
    verdict_fail = evaluators.evaluate_lookup_required(row_fail, {"required_tools": ["web_answer"]})
    assert verdict_fail.outcome == "fail"
    assert "web_answer" in verdict_fail.reason

    assert evaluators.evaluate_lookup_required(row_pass, {}).outcome == "n/a"


# --- forbidden_absent --------------------------------------------------------------------------


async def test_forbidden_absent_pass_fail_na(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_world: Any, serve_turn: Any
) -> None:
    fast_row = await _serve_custom(
        serve_turn, tmp_path / "p", monkeypatch, question="What time is it?"
    )
    assert fast_row["route"] == "fast"
    assert (
        evaluators.evaluate_forbidden_absent(fast_row, {"forbidden_tools": ["web_answer"]}).outcome
        == "pass"
    )

    world = make_world(model_wins=False)
    _, lookup_row = await serve_turn(world.server)
    verdict_fail = evaluators.evaluate_forbidden_absent(
        lookup_row, {"forbidden_tools": ["web_answer"]}
    )
    assert verdict_fail.outcome == "fail"
    assert "web_answer" in verdict_fail.reason

    assert evaluators.evaluate_forbidden_absent(lookup_row, {}).outcome == "n/a"


# --- searched ----------------------------------------------------------------------------------


async def test_searched_pass_fail_check_excluded_and_na(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_world: Any, serve_turn: Any
) -> None:
    world_pass = make_world(model_wins=False)
    _, row_pass = await serve_turn(world_pass.server)
    assert evaluators.evaluate_searched(row_pass, {"changeable": True}).outcome == "pass"

    # Zero-search ledger row fails.
    (tmp_path / "z").mkdir()
    row_zero = await _serve_custom(
        serve_turn,
        tmp_path / "z",
        monkeypatch,
        question="Who is the CEO of Microsoft?",
        model_calls=[("web_answer", {"query": "CEO of Microsoft"})],
        reply="Satya Nadella is the CEO of Microsoft.",
        ledger_rows=[{"month": "2026-11", "searches": 0}],
    )
    assert evaluators.evaluate_searched(row_zero, {"changeable": True}).outcome == "fail"

    # Background second-look ledger row carrying `check` must be excluded by `searched`.
    (tmp_path / "c").mkdir()
    row_check_only = await _serve_custom(
        serve_turn,
        tmp_path / "c",
        monkeypatch,
        question="Who is the CEO of Microsoft?",
        model_calls=[("web_answer", {"query": "CEO of Microsoft"})],
        reply="Satya Nadella is the CEO of Microsoft.",
        ledger_rows=[{"month": "2026-11", "searches": 1, "check": "SUPPORTED"}],
    )
    assert evaluators.evaluate_searched(row_check_only, {"changeable": True}).outcome == "fail"

    # Mixed slice: `check` row (searches=1) + own zero-search row still fails.
    row_check_and_zero = await _serve_custom(
        serve_turn,
        tmp_path / "c2",
        monkeypatch,
        question="Who is the CEO of Microsoft?",
        model_calls=[("web_answer", {"query": "CEO of Microsoft"})],
        reply="Satya Nadella is the CEO of Microsoft.",
        ledger_rows=[
            {"month": "2026-11", "searches": 1, "check": "SUPPORTED"},
            {"month": "2026-11", "searches": 0},
        ],
    )
    assert evaluators.evaluate_searched(row_check_and_zero, {"changeable": True}).outcome == "fail"

    # Non-changeable and fast-path turns are n/a.
    (tmp_path / "na").mkdir()
    row_settled = await _serve_custom(
        serve_turn,
        tmp_path / "na",
        monkeypatch,
        question="What is the capital of France?",
        reply="Paris.",
    )
    assert evaluators.evaluate_searched(row_settled, {}).outcome == "n/a"
    row_clock = await _serve_custom(
        serve_turn, tmp_path / "na", monkeypatch, question="What time is it?"
    )
    assert evaluators.evaluate_searched(row_clock, {}).outcome == "n/a"


# --- duplicate_lookups -------------------------------------------------------------------------


async def test_duplicate_lookups_pass_joined_fail_f3_and_na(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_world: Any, serve_turn: Any
) -> None:
    world_pass = make_world(model_wins=False)
    _, row_single = await serve_turn(world_pass.server)
    assert evaluators.evaluate_duplicate_lookups(row_single, {}).outcome == "pass"

    # Two calls in one turn where the second is joined (cached=True) is still 1 execution -> pass.
    (tmp_path / "j").mkdir()
    row_joined = await _serve_custom(
        serve_turn,
        tmp_path / "j",
        monkeypatch,
        question="Who is the CEO of Nvidia?",
        model_calls=[
            ("web_answer", {"query": "ceo of nvidia"}),
            ("web_answer", {"query": "nvidia ceo"}),
        ],
    )
    assert [c["status"] for c in row_joined["calls"]] == ["ok", "joined"]
    assert evaluators.evaluate_duplicate_lookups(row_joined, {}).outcome == "pass"

    # F3: router started web_answer (cancelled) AND model ran web_answer (ok) -> 2 -> fail.
    world_f3 = make_world(model_wins=True)
    _, row_f3 = await serve_turn(world_f3.server)
    verdict_f3 = evaluators.evaluate_duplicate_lookups(row_f3, {})
    assert verdict_f3.outcome == "fail"
    assert "2" in verdict_f3.reason

    # Fast-path turn with no lookup calls -> n/a.
    (tmp_path / "na").mkdir()
    row_fast = await _serve_custom(
        serve_turn, tmp_path / "na", monkeypatch, question="What time is it?"
    )
    assert evaluators.evaluate_duplicate_lookups(row_fast, {}).outcome == "n/a"


async def test_duplicate_lookups_failure_reason_breaks_down_by_and_status(
    make_world: Any, serve_turn: Any
) -> None:
    world = make_world(model_wins=True)
    _, row = await serve_turn(world.server)
    verdict = evaluators.evaluate_duplicate_lookups(row, {})
    assert verdict.outcome == "fail"
    assert "router: 1 cancelled, model: 1 ok" in verdict.reason


# --- note_reached ------------------------------------------------------------------------------


async def test_note_reached_pass_fail_f2_f19_and_na(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_world: Any, serve_turn: Any
) -> None:
    (tmp_path / "p").mkdir()
    row_pass = await _serve_custom(
        serve_turn,
        tmp_path / "p",
        monkeypatch,
        question="Who is the CEO of Nvidia?",
        model_calls=[("web_answer", {"query": "ceo of nvidia"})],
    )
    assert "note" in row_pass["tail"]
    assert evaluators.evaluate_note_reached(row_pass, {}).outcome == "pass"

    # F2 (fixed by F2a): a hedged turn won by Gemini now carries `note` before `now`.
    world_f2 = make_world(model_wins=True)
    _, row_f2 = await serve_turn(world_f2.server)
    assert row_f2["tail"] == ["note", "now"]
    verdict_f2 = evaluators.evaluate_note_reached(row_f2, {})
    assert verdict_f2.outcome == "pass"

    # F19: polite prefix bypasses changeable_question regex -> `note` missing even non-hedged.
    (tmp_path / "f19").mkdir()
    row_f19 = await _serve_custom(
        serve_turn,
        tmp_path / "f19",
        monkeypatch,
        question="Hello, can you tell me who the CEO of Nvidia is?",
        model_calls=[("web_answer", {"query": "ceo of nvidia"})],
    )
    assert "note" not in row_f19["tail"]
    assert evaluators.evaluate_note_reached(row_f19, {"changeable": True}).outcome == "fail"

    # Router won (did not go to Gemini) and non-changeable question are both n/a.
    world_router = make_world(model_wins=False)
    _, row_router = await serve_turn(world_router.server)
    assert evaluators.evaluate_note_reached(row_router, {}).outcome == "n/a"

    (tmp_path / "na").mkdir()
    row_settled = await _serve_custom(
        serve_turn,
        tmp_path / "na",
        monkeypatch,
        question="What is the capital of France?",
        reply="Paris.",
    )
    assert evaluators.evaluate_note_reached(row_settled, {}).outcome == "n/a"


# --- support & tokenizer rules -----------------------------------------------------------------


async def test_support_cnn_jensen_huang_sourced_reply_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serve_turn: Any
) -> None:
    row = await _serve_custom(
        serve_turn,
        tmp_path,
        monkeypatch,
        question="Who is the CEO of Nvidia?",
        lookup_result="As per CNN and FT, Jensen Huang is the CEO of Nvidia.",
        model_calls=[("web_answer", {"query": "ceo of nvidia"})],
        reply="As per CNN, Jensen Huang is the CEO of Nvidia.",
    )
    assert evaluators.evaluate_support(row, {}).outcome == "pass"


async def test_support_fail_lists_offending_tokens_and_na(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serve_turn: Any
) -> None:
    (tmp_path / "f").mkdir()
    row_fail = await _serve_custom(
        serve_turn,
        tmp_path / "f",
        monkeypatch,
        question="Who is the CEO of Nvidia?",
        lookup_result="As per CNN and FT, Jensen Huang is the CEO of Nvidia.",
        model_calls=[("web_answer", {"query": "ceo of nvidia"})],
        reply="As per CNN, Pat Gelsinger became CEO in 2021.",
    )
    verdict = evaluators.evaluate_support(row_fail, {})
    assert verdict.outcome == "fail"
    assert "2021" in verdict.reason
    assert "Pat" in verdict.reason and "Gelsinger" in verdict.reason

    (tmp_path / "na").mkdir()
    row_fast = await _serve_custom(
        serve_turn, tmp_path / "na", monkeypatch, question="What time is it?"
    )
    assert evaluators.evaluate_support(row_fast, {}).outcome == "n/a"

    row_bc = await _serve_custom(serve_turn, tmp_path / "na", monkeypatch, question="mm-hmm")
    assert row_bc["route"] == "backchannel"
    assert evaluators.evaluate_support(row_bc, {}).outcome == "n/a"


async def test_support_tokenizer_comma_numbers_and_no_digit_substring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serve_turn: Any
) -> None:
    # 158,000 == 158000 (both directions)
    (tmp_path / "a").mkdir()
    row_comma = await _serve_custom(
        serve_turn,
        tmp_path / "a",
        monkeypatch,
        question="How many employees?",
        lookup_result="The company has 158000 employees.",
        model_calls=[("web_answer", {"query": "employees"})],
        reply="It has 158,000 employees.",
    )
    assert evaluators.evaluate_support(row_comma, {}).outcome == "pass"

    (tmp_path / "b").mkdir()
    row_plain = await _serve_custom(
        serve_turn,
        tmp_path / "b",
        monkeypatch,
        question="How many employees?",
        lookup_result="The company has 158,000 employees.",
        model_calls=[("web_answer", {"query": "employees"})],
        reply="It has 158000 employees.",
    )
    assert evaluators.evaluate_support(row_plain, {}).outcome == "pass"

    # "5" must NOT substring-match "450"
    (tmp_path / "c").mkdir()
    row_sub = await _serve_custom(
        serve_turn,
        tmp_path / "c",
        monkeypatch,
        question="How many seats?",
        lookup_result="The hall holds 450 seats.",
        model_calls=[("web_answer", {"query": "seats"})],
        reply="The hall holds 5 seats.",
    )
    verdict_sub = evaluators.evaluate_support(row_sub, {})
    assert verdict_sub.outcome == "fail"
    assert "5" in verdict_sub.reason


async def test_support_tokenizer_sentence_initial_words_and_proper_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serve_turn: Any
) -> None:
    # Ordinary sentence-initial words ("The", "He", "It's") and possessives are handled.
    row_ok = await _serve_custom(
        serve_turn,
        tmp_path / "ok",
        monkeypatch,
        question="Who is the CEO of Nvidia?",
        lookup_result="Jensen Huang founded Nvidia in 1993. Sources: CNN.",
        model_calls=[("web_answer", {"query": "nvidia"})],
        reply="The CEO is Jensen Huang. He founded Nvidia's GPU business in 1993. Sources: CNN.",
    )
    # "GPU" is not in evidence -> make sure evidence includes GPU or keep reply to supported terms.
    assert evaluators.evaluate_support(row_ok, {}).outcome == "fail"
    assert evaluators.evaluate_support(row_ok, {}).reason == "unsupported: GPU"

    row_clean = await _serve_custom(
        serve_turn,
        tmp_path / "ok2",
        monkeypatch,
        question="Who is the CEO of Nvidia?",
        lookup_result="Jensen Huang founded Nvidia in 1993. Sources: CNN.",
        model_calls=[("web_answer", {"query": "nvidia"})],
        reply="The CEO is Jensen Huang. He founded Nvidia's core in 1993. It's from CNN.",
    )
    assert evaluators.evaluate_support(row_clean, {}).outcome == "pass"

    # Hyphenated name ("Lip-Bu Tan") passes when supported, fails when not.
    row_hyphen = await _serve_custom(
        serve_turn,
        tmp_path / "hyp",
        monkeypatch,
        question="Who is the CEO of Intel?",
        lookup_result="Lip-Bu Tan is the CEO of Intel. Sources: Intel and CNBC.",
        model_calls=[("web_answer", {"query": "ceo of intel"})],
        reply="According to CNBC, Lip-Bu Tan is the CEO of Intel.",
    )
    assert evaluators.evaluate_support(row_hyphen, {}).outcome == "pass"

    # Unsupported proper noun at sentence start is still caught.
    row_bad = await _serve_custom(
        serve_turn,
        tmp_path / "bad",
        monkeypatch,
        question="Who makes the chip?",
        lookup_result="Nvidia makes the chip. Sources: CNN.",
        model_calls=[("web_answer", {"query": "chip"})],
        reply="Apple makes the chip. Sources: CNN.",
    )
    verdict = evaluators.evaluate_support(row_bad, {})
    assert verdict.outcome == "fail"
    assert "Apple" in verdict.reason


async def test_support_tokenizer_source_names_and_dates_and_passages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serve_turn: Any
) -> None:
    # Hallucinated source name fails.
    (tmp_path / "src").mkdir()
    row_bad_src = await _serve_custom(
        serve_turn,
        tmp_path / "src",
        monkeypatch,
        question="Who is the CEO of Nvidia?",
        lookup_result="As per CNN and FT, Jensen Huang is the CEO of Nvidia.",
        model_calls=[("web_answer", {"query": "ceo of nvidia"})],
        reply="As per Reuters, Jensen Huang is the CEO of Nvidia.",
    )
    verdict_src = evaluators.evaluate_support(row_bad_src, {})
    assert verdict_src.outcome == "fail"
    assert "Reuters" in verdict_src.reason

    # Month name mismatch fails even at sentence start.
    (tmp_path / "date").mkdir()
    row_bad_date = await _serve_custom(
        serve_turn,
        tmp_path / "date",
        monkeypatch,
        question="When was the release?",
        lookup_result="Released on December 18, 2026.",
        model_calls=[("web_answer", {"query": "release"})],
        reply="November 18, 2026 was the release.",
    )
    verdict_date = evaluators.evaluate_support(row_bad_date, {})
    assert verdict_date.outcome == "fail"
    assert "November" in verdict_date.reason

    # Document passages via search_docs JSON result.
    (tmp_path / "doc").mkdir()
    doc_json = json.dumps([{"source": "policy.md", "text": "Alice Smith leads Acme in 2026."}])
    row_doc = await _serve_custom(
        serve_turn,
        tmp_path / "doc",
        monkeypatch,
        question="Who leads Acme?",
        tool_name="search_docs",
        lookup_result=doc_json,
        model_calls=[("search_docs", {"query": "Acme"})],
        reply="According to policy.md, Alice Smith leads Acme in 2026.",
    )
    assert evaluators.evaluate_support(row_doc, {}).outcome == "pass"


async def test_support_non_ascii_names_fold_for_match_and_preserve_originals_on_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serve_turn: Any
) -> None:
    # Accented reply matches unaccented evidence.
    (tmp_path / "ascii_ev").mkdir()
    row_ascii_ev = await _serve_custom(
        serve_turn,
        tmp_path / "ascii_ev",
        monkeypatch,
        question="Where was Jose Rizal born?",
        lookup_result="As per CNN, Jose Rizal was born in Calamba.",
        model_calls=[("web_answer", {"query": "jose rizal birthplace"})],
        reply="As per CNN, José Rizal was born in Calamba.",
    )
    assert evaluators.evaluate_support(row_ascii_ev, {}).outcome == "pass"

    # Accented reply matches accented evidence.
    (tmp_path / "acc_ev").mkdir()
    row_acc_ev = await _serve_custom(
        serve_turn,
        tmp_path / "acc_ev",
        monkeypatch,
        question="Where was José Rizal born?",
        lookup_result="As per CNN, José Rizal was born in Calamba.",
        model_calls=[("web_answer", {"query": "jose rizal birthplace"})],
        reply="As per CNN, José Rizal was born in Calamba.",
    )
    assert evaluators.evaluate_support(row_acc_ev, {}).outcome == "pass"

    # Non-ASCII possessive ("Parañaque's") matches folded ASCII evidence ("Paranaque").
    (tmp_path / "poss").mkdir()
    row_poss = await _serve_custom(
        serve_turn,
        tmp_path / "poss",
        monkeypatch,
        question="Who is the mayor?",
        lookup_result="As per GMA, Eric Olivarez is the mayor of Paranaque.",
        model_calls=[("web_answer", {"query": "mayor of paranaque"})],
        reply="Parañaque's mayor is Eric Olivarez, per GMA.",
    )
    assert evaluators.evaluate_support(row_poss, {}).outcome == "pass"

    # Absent non-ASCII name still fails and preserves original spelling in reason.
    (tmp_path / "absent").mkdir()
    row_absent = await _serve_custom(
        serve_turn,
        tmp_path / "absent",
        monkeypatch,
        question="Who was born in Calamba?",
        lookup_result="As per CNN, Andres Bonifacio was born in Calamba.",
        model_calls=[("web_answer", {"query": "born in calamba"})],
        reply="As per CNN, Rizál was born in Calamba.",
    )
    verdict = evaluators.evaluate_support(row_absent, {})
    assert verdict.outcome == "fail"
    assert verdict.reason == "unsupported: Rizál"


# --- source_coverage ---------------------------------------------------------------------------


async def test_source_coverage_pass_cited_by_server_fail_and_na(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serve_turn: Any
) -> None:
    # Model names a source directly -> pass.
    (tmp_path / "p1").mkdir()
    row_model = await _serve_custom(
        serve_turn,
        tmp_path / "p1",
        monkeypatch,
        question="Who is the CEO of Nvidia?",
        lookup_result="As per CNN and FT, Jensen Huang is the CEO of Nvidia.",
        model_calls=[("web_answer", {"query": "ceo of nvidia"})],
        reply="As per CNN, Jensen Huang is the CEO of Nvidia.",
    )
    assert evaluators.evaluate_source_coverage(row_model, {}).outcome == "pass"

    # Model drops source, server backstop appends it (`cited_by_server=True`) -> pass.
    (tmp_path / "p2").mkdir()
    row_server = await _serve_custom(
        serve_turn,
        tmp_path / "p2",
        monkeypatch,
        question="Who is the CEO of Nvidia?",
        lookup_result="As per CNN and FT, Jensen Huang is the CEO of Nvidia.",
        model_calls=[("web_answer", {"query": "ceo of nvidia"})],
        reply="The CEO of Nvidia is Jensen Huang.",
    )
    assert row_server.get("cited_by_server") is True
    assert evaluators.evaluate_source_coverage(row_server, {}).outcome == "pass"

    # F17: reply contains word "sources", suppressing server backstop without naming CNN or FT.
    (tmp_path / "f").mkdir()
    row_f17 = await _serve_custom(
        serve_turn,
        tmp_path / "f",
        monkeypatch,
        question="Who is the CEO of Nvidia?",
        lookup_result="As per CNN and FT, Jensen Huang is the CEO of Nvidia.",
        model_calls=[("web_answer", {"query": "ceo of nvidia"})],
        reply="Various sources say Jensen Huang is the CEO of Nvidia.",
    )
    assert not row_f17.get("cited_by_server")
    verdict_fail = evaluators.evaluate_source_coverage(row_f17, {})
    assert verdict_fail.outcome == "fail"
    assert "CNN" in verdict_fail.reason

    # Question-word collision: "Nvidia" in subject is not a citation unless attributed.
    row_collision = await _serve_custom(
        serve_turn,
        tmp_path / "col",
        monkeypatch,
        question="Who is the CEO of Nvidia?",
        lookup_result="Jensen Huang is the CEO. Sources: Nvidia and SEC.",
        model_calls=[("web_answer", {"query": "ceo of nvidia"})],
        reply="Other sources say the CEO of Nvidia is Jensen Huang.",
    )
    assert evaluators.evaluate_source_coverage(row_collision, {}).outcome == "fail"

    row_attributed = await _serve_custom(
        serve_turn,
        tmp_path / "attr",
        monkeypatch,
        question="Who is the CEO of Nvidia?",
        lookup_result="Jensen Huang is the CEO. Sources: Nvidia and SEC.",
        model_calls=[("web_answer", {"query": "ceo of nvidia"})],
        reply="Nvidia says Jensen Huang is the CEO of Nvidia.",
    )
    assert evaluators.evaluate_source_coverage(row_attributed, {}).outcome == "pass"

    # Lookup without sources -> n/a.
    (tmp_path / "na").mkdir()
    row_na = await _serve_custom(
        serve_turn,
        tmp_path / "na",
        monkeypatch,
        question="What is the capital of France?",
        lookup_result="Paris is the capital of France.",
        model_calls=[("web_answer", {"query": "capital of france"})],
        reply="Paris.",
    )
    assert evaluators.evaluate_source_coverage(row_na, {}).outcome == "n/a"


# --- abstention --------------------------------------------------------------------------------


async def test_abstention_pass_fail_f20_and_na(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serve_turn: Any
) -> None:
    (tmp_path / "p1").mkdir()
    row_hedge = await _serve_custom(
        serve_turn,
        tmp_path / "p1",
        monkeypatch,
        question="Who won yesterday's match?",
        lookup_result="I couldn't find that.",
        model_calls=[("web_answer", {"query": "match"})],
        reply=evaluators.HEDGE,
    )
    assert evaluators.evaluate_abstention(row_hedge, {"abstain": True}).outcome == "pass"
    assert evaluators.evaluate_abstention(row_hedge, {"abstain": False}).outcome == "fail"

    (tmp_path / "p2").mkdir()
    row_docs = await _serve_custom(
        serve_turn,
        tmp_path / "p2",
        monkeypatch,
        question="What is the secret code?",
        tool_name="search_docs",
        lookup_result="[]",
        model_calls=[("search_docs", {"query": "secret code"})],
        reply="The documents don't say what the secret code is.",
    )
    assert evaluators.evaluate_abstention(row_docs, {"abstain": True}).outcome == "pass"

    (tmp_path / "a").mkdir()
    row_answer = await _serve_custom(
        serve_turn,
        tmp_path / "a",
        monkeypatch,
        question="Who is the CEO of Nvidia?",
        lookup_result="As per CNN, Jensen Huang is the CEO of Nvidia.",
        model_calls=[("web_answer", {"query": "ceo of nvidia"})],
        reply="As per CNN, Jensen Huang is the CEO of Nvidia.",
    )
    assert evaluators.evaluate_abstention(row_answer, {"abstain": False}).outcome == "pass"
    assert evaluators.evaluate_abstention(row_answer, {"abstain": True}).outcome == "fail"
    assert evaluators.evaluate_abstention(row_answer, {}).outcome == "n/a"

    # F20: Tavily-style "I couldn't confirm X ... The closest result says: ..." still guesses.
    (tmp_path / "f20").mkdir()
    row_f20 = await _serve_custom(
        serve_turn,
        tmp_path / "f20",
        monkeypatch,
        question="Who is the CEO of Acme?",
        lookup_result="I couldn't confirm that in what I found. The closest result says: Jane Doe.",
        model_calls=[("web_answer", {"query": "ceo of acme"})],
        reply="I couldn't confirm that in what I found. The closest result says: Jane Doe.",
    )
    assert evaluators.evaluate_abstention(row_f20, {"abstain": True}).outcome == "fail"


async def _serve_docs_turn(
    serve_turn: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    question: str,
    passages: list[dict[str, Any]],
    doc_reply: str,
) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent_tools, "CHAIN", None)
    monkeypatch.setattr(agent_tools, "SCRATCHPAD", None)
    record: list[dict[str, Any]] = []

    async def handler(params: Any) -> None:
        await params.result_callback(passages)

    schema = FunctionSchema(
        name="search_docs", description="d", properties={}, required=[], handler=handler
    )
    tools = ToolsSchema(standard_tools=[agent_tools._recorded(schema, record)])
    agent = _ScriptedAgent(tools, reply="")
    server = AgentServer(
        agent,
        record,
        tmp_path / "20261118-120000.jsonl",
        tools,
        doc_request=bench_fakes.fake_doc_request(doc_reply),
    )
    _, row = await serve_turn(server, question)
    return row


async def test_support_docs_citation_and_not_found_offer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serve_turn: Any
) -> None:
    # Section + file_date in evidence -> support passes.
    row_sec = await _serve_docs_turn(
        serve_turn,
        tmp_path / "sec",
        monkeypatch,
        question="According to my documents, where does Project Zephyr launch?",
        passages=[
            bench_fakes.search_docs_passage(
                "notes.md",
                "Project Zephyr launches in Cebu with 42 solar panels.",
                section="Launch Plan",
                file_date="2026-11-18",
            )
        ],
        doc_reply=(
            "ANSWER: Project Zephyr launches in Cebu.\nQUOTE: Project Zephyr launches in Cebu"
        ),
    )
    assert row_sec["route"] == "docs"
    assert evaluators.evaluate_support(row_sec, {}).outcome == "pass"

    # Page + file_date in evidence -> support passes.
    row_page = await _serve_docs_turn(
        serve_turn,
        tmp_path / "page",
        monkeypatch,
        question="What do my documents say about paid leave?",
        passages=[
            bench_fakes.search_docs_passage(
                "handbook.pdf",
                "Employees receive 15 days of paid leave per year.",
                page=4,
                file_date="2026-11-17",
            )
        ],
        doc_reply="ANSWER: Employees get 15 days of paid leave.\nQUOTE: 15 days of paid leave",
    )
    assert evaluators.evaluate_support(row_page, {}).outcome == "pass"

    # NOT_FOUND_OFFER ("... Want me to look it up on the web?") -> support passes.
    row_offer = await _serve_docs_turn(
        serve_turn,
        tmp_path / "offer",
        monkeypatch,
        question="What do my documents say about the secret code?",
        passages=[],
        doc_reply="NONE",
    )
    assert row_offer["text"] == doc_route.NOT_FOUND_OFFER
    assert evaluators.evaluate_support(row_offer, {}).outcome == "pass"

    # Invented proper name or number on a docs turn still fails support.
    bad_name = {**row_sec, "text": f"{row_sec['text']} Per Alice."}
    assert evaluators.evaluate_support(bad_name, {}).outcome == "fail"
    bad_num = {**row_sec, "text": f"{row_sec['text']} (999 items)."}
    assert evaluators.evaluate_support(bad_num, {}).outcome == "fail"


# --- doc_citation ------------------------------------------------------------------------------


async def test_doc_citation_pass_fail_folded_and_na(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serve_turn: Any
) -> None:
    passage = bench_fakes.search_docs_passage(
        "notes.md",
        "Project Zephyr launches in Cebu.",
        section="Launch Plan",
        file_date="2026-11-18",
    )
    row_pass = await _serve_docs_turn(
        serve_turn,
        tmp_path / "pass",
        monkeypatch,
        question="According to my documents, where does Project Zephyr launch?",
        passages=[passage],
        doc_reply=(
            "ANSWER: Project Zephyr launches in Cebu.\nQUOTE: Project Zephyr launches in Cebu"
        ),
    )
    assert row_pass["route"] == "docs"
    assert evaluators.evaluate_doc_citation(row_pass, {}).outcome == "pass"

    # Folded-name match: accented source vs ASCII reply (and vice versa).
    folded_row = {
        "route": "docs",
        "question": "What does my file say?",
        "text": "As per resume.md: Alice works here.",
        "citation_metadata": {
            "file": "resume.md",
            "date": "2026-11-18",
            "section": "",
            "quote": "Alice works here.",
        },
        "calls": [
            {
                "tool": "search_docs",
                "status": "ok",
                "result_text": json.dumps(
                    [
                        bench_fakes.search_docs_passage(
                            "résumé.md",
                            "Alice works here.",
                            file_date="2026-11-18",
                        )
                    ]
                ),
            }
        ],
    }
    assert evaluators.evaluate_doc_citation(folded_row, {}).outcome == "pass"

    # Wrong file fails.
    row_wrong_file = {
        **row_pass,
        "text": "As per other.md: Project Zephyr launches in Cebu.",
        "citation_metadata": {
            "file": "other.md",
            "date": "2026-11-18",
            "section": "Launch Plan",
            "quote": "Project Zephyr launches in Cebu",
        },
    }
    assert evaluators.evaluate_doc_citation(row_wrong_file, {}).outcome == "fail"

    # Wrong date fails.
    row_wrong_date = {
        **row_pass,
        "text": "As per notes.md: Project Zephyr launches in Cebu.",
        "citation_metadata": {
            "file": "notes.md",
            "date": "2026-11-01",
            "section": "Launch Plan",
            "quote": "Project Zephyr launches in Cebu",
        },
    }
    assert evaluators.evaluate_doc_citation(row_wrong_date, {}).outcome == "fail"

    # Neither file nor date fails.
    row_neither = {**row_pass, "text": "Project Zephyr launches in Cebu."}
    row_neither.pop("citation_metadata", None)
    assert evaluators.evaluate_doc_citation(row_neither, {}).outcome == "fail"

    # Route not docs -> n/a.
    row_routed = await _serve_custom(
        serve_turn,
        tmp_path / "routed",
        monkeypatch,
        question="Who is the CEO of Nvidia?",
        model_calls=[("web_answer", {"query": "ceo of nvidia"})],
        reply="As per CNN, Jensen Huang is the CEO of Nvidia.",
    )
    assert row_routed["route"] != "docs"
    assert evaluators.evaluate_doc_citation(row_routed, {}).outcome == "n/a"

    # Abstention reply on docs route -> n/a.
    row_abstain = await _serve_docs_turn(
        serve_turn,
        tmp_path / "abstain",
        monkeypatch,
        question="What do my documents say about the WiFi password?",
        passages=[passage],
        doc_reply="NONE",
    )
    assert row_abstain["route"] == "docs"
    assert row_abstain["text"] == doc_route.NOT_FOUND_OFFER
    assert evaluators.evaluate_doc_citation(row_abstain, {}).outcome == "n/a"
