"""scripts/sufficiency_eval.py: the offline sufficiency harness.

All through fakes: a fake transport (no network, no relay, no spend), a scripted document
lane, a scripted verdict. The harness's own logic is under test - the shared Qwen request,
the gold-answer substitution, the label scoring, the loud failure, the stability and judge
wiring. The real lane and verdict live in the frozen pipeline and are covered by its suite.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "sufficiency_eval", Path(__file__).resolve().parents[1] / "scripts" / "sufficiency_eval.py"
)
assert _SPEC and _SPEC.loader
se = importlib.util.module_from_spec(_SPEC)
sys.modules["sufficiency_eval"] = se  # so @dataclass can resolve the module
_SPEC.loader.exec_module(se)

FAKE_KEY = "«redacted:sk-…»"
DOC_SYSTEM = "the frozen document lane's system prompt"
VERDICT_SYSTEM = "the frozen verdict system prompt"


class FakeTransport:
    """In-memory relay: returns canned responses in order, records every payload."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if not isinstance(item, dict):
            raise TypeError("fake transport: expected a canned response dict or an exception")
        return item


def qwen_reply(text: str, *, prompt_tokens: int = 10, completion_tokens: int = 5) -> dict[str, Any]:
    """A minimal relay response with a message content and a usage block."""
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


class FakeDoc:
    """A DocResult-shaped object (the frozen lane's result, as the harness sees it)."""

    def __init__(
        self, status: str, spoken: str | None = None, file: str | None = None, calls: int = 1
    ) -> None:
        self.status = status
        self.spoken = spoken
        self.file = file
        self.calls = calls


class CallingLane:
    """A scripted document lane that really uses its request (like the real one does): a
    transport failure therefore surfaces as an error status, exactly like the frozen lane's."""

    def __init__(self, by_question: dict[str, FakeDoc]) -> None:
        self.by_question = by_question
        self.questions: list[str] = []

    async def __call__(
        self,
        question: str,
        *,
        context: Any,
        request: Any,
        file_names: Any = (),
        budget_s: float = 0.0,
        allow_fallback: bool = True,
    ) -> FakeDoc:
        self.questions.append(question)
        try:
            await request(DOC_SYSTEM, f"question: {question}")
        except Exception:
            return FakeDoc("error")
        return self.by_question.get(question, FakeDoc("none"))


class ScriptedVerdict:
    """A scripted verdict with the frozen verdict's call shape (request, question, answer,
    source_year, today). An unknown answer is no verdict (a failed call)."""

    def __init__(self, by_answer: dict[str, str]) -> None:
        self.by_answer = dict(by_answer)
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, request: Any, question: str, answer: str, source_year: str | None, today: str
    ) -> Any:
        self.calls.append(
            {"question": question, "answer": answer, "source_year": source_year, "today": today}
        )
        verdict = self.by_answer.get(answer)
        if verdict is None:
            return None
        return SimpleNamespace(verdict=verdict, web_question="", reason=f"why {verdict}")


class CallingVerdict(ScriptedVerdict):
    """Like ScriptedVerdict, but really uses its request (like the frozen verdict does)."""

    async def __call__(
        self, request: Any, question: str, answer: str, source_year: str | None, today: str
    ) -> Any:
        await super().__call__(request, question, answer, source_year, today)
        try:
            await request(VERDICT_SYSTEM, f"answer: {answer}")
        except Exception:
            return None
        verdict = self.by_answer.get(answer)
        if verdict is None:
            return None
        return SimpleNamespace(verdict=verdict, web_question="", reason=f"why {verdict}")


def make_harness(
    transport: FakeTransport,
    lane: Any,
    verdict_fn: Any,
    *,
    gold_answer: bool = False,
    cards: dict[str, dict[str, Any]] | None = None,
) -> se.Harness:
    async def context_fn(question: str) -> list[dict[str, Any]]:
        return [{"source": "a/one.txt", "text": "passage text", "part": "whole"}]

    return se.Harness(
        transport=transport,
        model="fake/model",
        context_fn=context_fn,
        ask_documents_fn=lane,
        verdict_fn=verdict_fn,
        file_names=["a/one.txt"],
        cards=cards or {},
        today="2026-11-27",
        gold_answer=gold_answer,
    )


def row(
    id: str = "imp-001",
    kind: str = "needs_web_stale",
    question: str = "What's the rate right now?",
    web_needed: bool = True,
    ideal: str = "The old rate is 209; verify live.",
    doc_file: str = "a/one.txt",
) -> dict[str, Any]:
    return {
        "id": id,
        "kind": kind,
        "question": question,
        "web_needed": web_needed,
        "ideal_answer": ideal,
        "doc_file": doc_file,
    }


def _rec(
    id: str,
    kind: str,
    web_needed: bool,
    verdict: str,
    passed: bool,
    *,
    doc_status: str = "accepted",
    source: str = "lane",
    doc_spoken: str | None = "The rate is 209.",
    doc_s: float = 0.5,
    verdict_s: float = 0.2,
) -> dict[str, Any]:
    return {
        "id": id,
        "kind": kind,
        "question": "q?",
        "web_needed": web_needed,
        "doc_status": doc_status,
        "doc_spoken": doc_spoken,
        "doc_file_cited": None,
        "doc_calls": 1,
        "doc_s": doc_s,
        "verdict_answer": "a",
        "answer_source": source,
        "source_year": None,
        "verdict": verdict,
        "web_question": "",
        "reason": "the reason",
        "verdict_s": verdict_s,
        "passed": passed,
    }


# --- the shared Qwen request ---


def test_relay_payload_is_the_shared_qwen_json_request() -> None:
    payload = se.relay_payload("fake/model", "sys", "usr")
    assert payload["model"] == "fake/model"
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == se.MAX_TOKENS
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert payload["messages"][0]["content"] == "sys"
    assert payload["messages"][1]["content"] == "usr"


async def test_both_requests_share_the_qwen_setup_and_tag_tokens() -> None:
    tokens = se.TokenCounter()
    failures: list[str] = []
    fake = FakeTransport(
        [
            qwen_reply("doc-out", prompt_tokens=11, completion_tokens=2),
            qwen_reply("verdict-out", prompt_tokens=21, completion_tokens=7),
        ]
    )
    doc_request, verdict_request = se.make_requests(fake, "fake/model", tokens, failures)
    d, v = await doc_request(DOC_SYSTEM, "u1"), await verdict_request(VERDICT_SYSTEM, "u2")
    assert (d, v) == ("doc-out", "verdict-out")
    # the document lane keeps the working example's JSON-only nudge; the verdict's own
    # prompt already asks for JSON-only and is sent as the frozen release sends it
    assert fake.payloads[0]["messages"][0]["content"] == DOC_SYSTEM + se.JSON_ONLY_SUFFIX
    assert fake.payloads[1]["messages"][0]["content"] == VERDICT_SYSTEM
    assert fake.payloads[0]["model"] == fake.payloads[1]["model"] == "fake/model"
    assert (
        fake.payloads[0]["response_format"]
        == fake.payloads[1]["response_format"]
        == {"type": "json_object"}
    )
    assert (tokens.doc_prompt, tokens.doc_completion) == (11, 2)
    assert (tokens.verdict_prompt, tokens.verdict_completion) == (21, 7)
    assert failures == []


# --- the relay: one in flight, one retry, loud, key only in the header ---


class _FakeResp:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_relay_retries_once_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    relay = se.Relay("http://127.0.0.1:18080/v1", FAKE_KEY, "fake/model")
    calls = {"n": 0}

    def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeResp:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(str(req), 500, "boom", {}, None)
        return _FakeResp(json.dumps(qwen_reply("ok")).encode())

    monkeypatch.setattr(se.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(se.time, "sleep", lambda _s: None)
    data = relay.post(se.relay_payload("fake/model", "s", "u"))
    assert calls["n"] == 2 and data["choices"][0]["message"]["content"] == "ok"


def test_relay_fails_loudly_after_one_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    relay = se.Relay("http://127.0.0.1:18080/v1", FAKE_KEY, "fake/model")
    calls = {"n": 0}

    def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeResp:
        calls["n"] += 1
        raise urllib.error.HTTPError(str(req), 503, "busy", {}, None)

    monkeypatch.setattr(se.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(se.time, "sleep", lambda _s: None)
    with pytest.raises(se.RelayError, match="HTTP 503"):
        relay.post(se.relay_payload("fake/model", "s", "u"))
    assert calls["n"] == 2  # one attempt + one retry, then loud


def test_relay_key_travels_only_in_the_header() -> None:
    relay = se.Relay("http://127.0.0.1:18080/v1", FAKE_KEY, "fake/model")
    req = relay.request({"model": "fake/model"})
    assert req.get_header("Authorization") == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in req.full_url
    assert FAKE_KEY not in req.data.decode("utf-8")


# --- the row flow: the lane answer, the gold fallback, the modes ---


async def test_accepted_row_scores_the_lane_spoken_text() -> None:
    transport = FakeTransport([qwen_reply("{}")])
    spoken = "Per the sheet: 209 a night."
    lane = CallingLane({"What's the rate right now?": FakeDoc("accepted", spoken=spoken)})
    verdict_fn = ScriptedVerdict({spoken: "stale"})
    h = make_harness(transport, lane, verdict_fn, cards={"a/one.txt": {"date_max": "2024-07-01"}})
    records, _ = await se.run_round(h, [row()])
    rec = records[0]
    assert rec["doc_status"] == "accepted" and rec["answer_source"] == "lane"
    assert rec["verdict"] == "stale" and rec["passed"] is True
    call = verdict_fn.calls[0]
    assert call["answer"] == spoken  # the verdict sees what the lane spoke
    assert call["source_year"] == "2024"  # from the file card's date
    assert call["today"] == "2026-11-27"


async def test_rejected_row_is_scored_on_the_gold_answer() -> None:
    transport = FakeTransport([qwen_reply("{}")])
    lane = CallingLane({})  # no scripted reply: the lane does not accept
    verdict_fn = ScriptedVerdict({"The old rate is 209; verify live.": "missing_half"})
    h = make_harness(transport, lane, verdict_fn)
    records, _ = await se.run_round(h, [row()])
    rec = records[0]
    assert rec["doc_status"] == "none" and rec["answer_source"] == "gold"
    assert rec["verdict"] == "missing_half" and rec["passed"] is True
    assert verdict_fn.calls[0]["answer"] == "The old rate is 209; verify live."


async def test_gold_answer_mode_scores_the_gold_answer_even_when_accepted() -> None:
    spoken = "Per the sheet: 209 a night."
    transport = FakeTransport([qwen_reply("{}")])
    lane = CallingLane({"What's the rate right now?": FakeDoc("accepted", spoken=spoken)})
    verdict_fn = ScriptedVerdict({"The old rate is 209; verify live.": "stale"})
    h = make_harness(transport, lane, verdict_fn, gold_answer=True)
    records, _ = await se.run_round(h, [row()])
    assert records[0]["doc_status"] == "accepted"
    assert records[0]["answer_source"] == "gold"
    assert verdict_fn.calls[0]["answer"] == "The old rate is 209; verify live."


def test_source_year_from_card_date_else_none() -> None:
    assert se.source_year_from_card({"date_max": "2024-07-01"}) == "2024"
    assert se.source_year_from_card({"date_max": None}) is None
    assert se.source_year_from_card({"date_max": "unknown"}) is None
    assert se.source_year_from_card(None) is None


def test_harness_source_year_is_the_row_file_card_else_none() -> None:
    h = make_harness(FakeTransport([]), None, None, cards={"a/one.txt": {"date_max": "2025-10-01"}})
    assert h.source_year_for("a/one.txt") == "2025"
    assert h.source_year_for("b/other.txt") is None


def test_gold_doc_answer_drops_the_web_hedge_sentences() -> None:
    # the verdict must see only what the documents state, never the web fact the gold
    # ideal answer carries (that would make a needs-web row read as complete)
    ideal = (
        "Your FY2026 schedule lists Zorblat lodging at 209 a night for November through "
        "February. That schedule ended in September, so for the current rate verify live."
    )
    assert se.gold_doc_answer(ideal) == (
        "Your FY2026 schedule lists Zorblat lodging at 209 a night for November through February."
    )
    ideal2 = "The campus is in Virel City, which is in the Eastern time zone (web)."
    # every sentence is a hedge: the whole answer is kept, a row is never scored on nothing
    assert se.gold_doc_answer(ideal2) == ideal2
    docs_only = "The Zorblat has 264 kilobytes of memory."
    assert se.gold_doc_answer(docs_only) == docs_only
    assert se.gold_doc_answer("") == ""


async def test_rejected_row_sees_the_document_half_not_the_web_hedge() -> None:
    ideal = "The sheet lists 209 a night. For the current rate verify live."
    transport = FakeTransport([qwen_reply("{}")])
    lane = CallingLane({})
    verdict_fn = ScriptedVerdict({"The sheet lists 209 a night.": "stale"})
    h = make_harness(transport, lane, verdict_fn)
    records, _ = await se.run_round(h, [row(ideal=ideal)])
    assert records[0]["verdict_answer"] == "The sheet lists 209 a night."
    assert records[0]["verdict"] == "stale" and records[0]["passed"] is True


async def test_label_rules_per_verdict() -> None:
    # wrong_question still counts as a trigger; a failed verdict call is no pass anywhere
    spoken_a, spoken_b = "sa", "sb"
    transport = FakeTransport([qwen_reply("{}"), qwen_reply("{}")])
    lane = CallingLane(
        {"qa": FakeDoc("accepted", spoken=spoken_a), "qb": FakeDoc("accepted", spoken=spoken_b)}
    )
    verdict_fn = ScriptedVerdict({spoken_a: "wrong_question", spoken_b: "garbage"})
    h = make_harness(transport, lane, verdict_fn)
    records, _ = await se.run_round(
        h,
        [
            row(id="r1", question="qa", web_needed=True),
            row(id="r2", question="qb", kind="docs_only_control", web_needed=False),
        ],
    )
    assert records[0]["verdict"] == "wrong_question" and records[0]["passed"] is True
    assert records[1]["verdict"] == "none" and records[1]["passed"] is False


# --- loud failure and call order ---


async def test_a_dead_relay_fails_loudly() -> None:
    # the row layer swallows the transport failure (lane: error status), the harness must not
    transport = FakeTransport([se.RelayError("the relay is unreachable (ConnectionRefusedError)")])
    lane = CallingLane({})
    h = make_harness(transport, lane, ScriptedVerdict({}))
    with pytest.raises(se.RunAborted, match="relay failed"):
        await se.run_round(h, [row()])


async def test_calls_run_one_at_a_time_in_row_order() -> None:
    transport = FakeTransport([qwen_reply("{}")] * 4)
    lane = CallingLane(
        {"qa": FakeDoc("accepted", spoken="sa"), "qb": FakeDoc("accepted", spoken="sb")}
    )
    verdict_fn = CallingVerdict({"sa": "stale", "sb": "sufficient"})
    h = make_harness(transport, lane, verdict_fn)
    records, _ = await se.run_round(
        h,
        [
            row(id="r1", question="qa", web_needed=True),
            row(id="r2", question="qb", kind="docs_only_control", web_needed=False),
        ],
    )
    # strictly sequential, per row: the lane's call, then the verdict's call
    systems = [p["messages"][0]["content"] for p in transport.payloads]
    assert systems == [
        DOC_SYSTEM + se.JSON_ONLY_SUFFIX,
        VERDICT_SYSTEM,
        DOC_SYSTEM + se.JSON_ONLY_SUFFIX,
        VERDICT_SYSTEM,
    ]
    assert records[0]["passed"] is True and records[1]["passed"] is True


# --- scoring ---


def test_score_rows_confusion_recall_over_trigger_and_failures() -> None:
    records = [
        _rec("r1", "k_a", True, "stale", True),
        _rec("r2", "k_a", True, "sufficient", False),
        _rec("r3", "k_b", True, "missing_half", True),
        _rec("r4", "k_b", True, "none", False),
        _rec("r5", "ctrl", False, "sufficient", True),
        _rec("r6", "ctrl", False, "stale", False),
    ]
    s = se.score_rows(records)
    assert s["confusion_by_kind"]["k_a"] == {
        "sufficient": 1,
        "stale": 1,
        "missing_half": 0,
        "wrong_question": 0,
        "none": 0,
    }
    assert s["recall_needs_web_by_kind"]["k_a"] == {"n": 2, "passed": 1, "recall": 0.5}
    assert s["recall_needs_web_by_kind"]["k_b"] == {"n": 2, "passed": 1, "recall": 0.5}
    assert s["recall_needs_web"] == {"n": 4, "passed": 2, "recall": 0.5}
    assert s["over_trigger_controls"] == {"n": 2, "over_triggered": 1, "rate": 0.5}
    assert s["agreement"] == {"n": 6, "passed": 3, "rate": 0.5}
    assert [f["id"] for f in s["failures"]] == ["r2", "r4", "r6"]
    assert all({"id", "kind", "verdict", "reason"} <= set(f) for f in s["failures"])


def test_percentile_nearest_rank() -> None:
    assert se.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0
    assert se.percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0
    assert se.percentile([], 50) is None


# --- the report: stability, tokens, latency, rendering ---


def test_build_report_two_rounds_reports_stability() -> None:
    r1 = [_rec("r1", "k_a", True, "stale", True), _rec("r2", "k_a", True, "stale", True)]
    r2 = [_rec("r1", "k_a", True, "sufficient", False), _rec("r2", "k_a", True, "stale", True)]
    report = se.build_report(
        mode="lane",
        today="2026-11-27",
        model="fake/model",
        db=Path("d"),
        jsonl_path=Path("j.jsonl"),
        repeat=2,
        rounds=[r1, r2],
        round_tokens=[se.TokenCounter(doc_prompt=5, doc_completion=1), se.TokenCounter()],
        judge_result=None,
    )
    assert report["mode"] == "lane" and report["n_rows"] == 2
    assert report["stability"]["stable"] == 1 and report["stability"]["rate"] == 0.5
    assert report["stability"]["flips"] == [
        {"id": "r1", "kind": "k_a", "verdicts": ["stale", "sufficient"]}
    ]
    assert report["rows"][0]["verdicts"] == ["stale", "sufficient"]
    assert report["tokens"]["doc"] == {"prompt": 5, "completion": 1, "total": 6}
    assert report["latency_s"]["doc"]["p50"] == 0.5
    assert len(report["rounds"]) == 2 and report["rounds"][1]["verdict_counts"]["sufficient"] == 1


async def test_repeat_two_rounds_end_to_end_with_a_flipping_verdict() -> None:
    class FlippingVerdict(ScriptedVerdict):
        def __init__(self) -> None:
            super().__init__({})
            self.seen: list[str] = []

        async def __call__(
            self, request: Any, question: str, answer: str, source_year: str | None, today: str
        ) -> Any:
            self.seen.append(answer)
            verdict = "stale" if len(self.seen) % 2 == 1 else "sufficient"
            return SimpleNamespace(verdict=verdict, web_question="", reason="flip")

    transport = FakeTransport([qwen_reply("{}")] * 2)
    lane = CallingLane({"What's the rate right now?": FakeDoc("accepted", spoken="s")})
    verdict_fn = FlippingVerdict()
    h = make_harness(transport, lane, verdict_fn)
    r1, t1 = await se.run_round(h, [row()])
    r2, t2 = await se.run_round(h, [row()])
    assert [r1[0]["verdict"], r2[0]["verdict"]] == ["stale", "sufficient"]
    report = se.build_report(
        mode="lane",
        today="2026-11-27",
        model="fake/model",
        db=Path("d"),
        jsonl_path=Path("j.jsonl"),
        repeat=2,
        rounds=[r1, r2],
        round_tokens=[t1, t2],
        judge_result=None,
    )
    assert report["stability"]["stable"] == 0 and report["stability"]["rate"] == 0.0


def test_render_table_carries_the_required_sections() -> None:
    r1 = [
        _rec("r1", "k_a", True, "stale", True),
        _rec("r2", "ctrl", False, "stale", False),
    ]
    report = se.build_report(
        mode="lane",
        today="2026-11-27",
        model="fake/model",
        db=Path("d"),
        jsonl_path=Path("j.jsonl"),
        repeat=1,
        rounds=[r1],
        round_tokens=[se.TokenCounter()],
        judge_result=None,
    )
    text = se.render_table(report)
    for needle in (
        "mode=lane",
        "today=2026-11-27",
        "confusion by kind",
        "needs-web recall by kind",
        "over-trigger on controls",
        "verdict agreement",
        "latency s",
        "tokens:",
        "failures (1)",
        "r2",
    ):
        assert needle in text, needle


# --- the optional judge: directness on the spoken lane answer ---


def _judge_letter(letter: str) -> dict[str, Any]:
    """A minimal first-token logprob response for the judge's directness letters A/B/C."""
    probs = {"A": -0.01, "B": -0.02, "C": -6.0}
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": letter},
                "logprobs": {
                    "content": [
                        {
                            "token": letter,
                            "logprob": probs[letter],
                            "top_logprobs": [
                                {"token": t, "logprob": lp} for t, lp in sorted(probs.items())
                            ],
                        }
                    ]
                },
            }
        ]
    }


def test_judge_directness_grades_only_the_spoken_rows() -> None:
    judge = se._load_judge()
    records = [
        _rec("r1", "k_a", True, "stale", True, doc_status="accepted"),
        _rec("r2", "k_b", True, "none", False, doc_status="none", doc_spoken=None),
    ]
    transport = FakeTransport([_judge_letter("A")])
    result = se.grade_directness(records, transport, "fake/model", judge)
    assert result is not None and result["n"] == 1
    assert result["counts"] == {"A": 1}
    assert result["labels"] == {"A": "answers"}
    assert records[0]["judge_directness"]["choice"] == "A"
    assert records[1]["judge_directness"] is None
    # the judge prompt carried the question and the spoken answer, and nothing else
    prompt = transport.payloads[0]["messages"][1]["content"]
    assert "q?" in prompt and "The rate is 209." in prompt


def test_judge_result_none_when_nothing_was_spoken() -> None:
    judge = se._load_judge()
    records = [_rec("r1", "k_a", True, "stale", True, doc_status="none", doc_spoken=None)]
    transport = FakeTransport([])
    assert se.grade_directness(records, transport, "fake/model", judge) is None
    assert transport.payloads == []


# --- the data file loading ---


def test_load_rows_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "x", "kind": "k", "question": "q?"}\n{"kind": "k"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing 'id'"):
        se.load_rows(path)


def test_load_rows_accepts_the_labelled_set(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"id": "x", "kind": "k", "question": "q?", "web_needed": True}) + "\n",
        encoding="utf-8",
    )
    rows = se.load_rows(path)
    assert len(rows) == 1 and rows[0]["id"] == "x"


# --- the ticket-260 set/split selection ---


def _split_row(rid: str, split: str, question: str = "q?") -> dict[str, Any]:
    return {
        "id": rid,
        "kind": "needs_web_stale",
        "question": question,
        "web_needed": True,
        "split": split,
    }


def test_resolve_set_paths_prefers_the_jsonl_override(tmp_path: Path) -> None:
    custom = tmp_path / "custom.jsonl"
    assert se.resolve_set_paths("both", custom) == [custom]
    assert se.resolve_set_paths("v1", None) == [se.DEFAULT_JSONL]
    assert se.resolve_set_paths("v2", None) == [se.V2_JSONL]
    assert se.resolve_set_paths("both", None) == [se.DEFAULT_JSONL, se.V2_JSONL]


def test_filter_split_selects_rows_and_rejects_missing_field() -> None:
    rows = [_split_row("a-001", "dev"), _split_row("a-002", "test"), _split_row("a-003", "dev")]
    assert [r["id"] for r in se.filter_split(rows, "all")] == ["a-001", "a-002", "a-003"]
    assert [r["id"] for r in se.filter_split(rows, "dev")] == ["a-001", "a-003"]
    assert [r["id"] for r in se.filter_split(rows, "test")] == ["a-002"]
    bare = {"id": "a-004", "kind": "k", "question": "q?"}
    with pytest.raises(RuntimeError, match="has no 'split' field"):
        se.filter_split([bare], "dev")


async def test_split_selection_runs_only_the_selected_rows(tmp_path: Path) -> None:
    # load a set through load_rows, pick the dev split, run the round over the fake
    # transport: only the selected rows reach the lane
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(_split_row(f"x-{i:03d}", "dev" if i % 2 == 0 else "test", question=f"q{i}?"))
            for i in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    rows = se.filter_split(se.load_rows(path), "dev")
    assert [r["id"] for r in rows] == ["x-000", "x-002"]
    transport = FakeTransport([qwen_reply("{}")] * 2)
    lane = CallingLane({f"q{i}?": FakeDoc("accepted", spoken=f"s{i}") for i in (0, 2)})
    verdict_fn = ScriptedVerdict({f"s{i}": "stale" for i in (0, 2)})
    h = make_harness(transport, lane, verdict_fn)
    records, _ = await se.run_round(h, rows)
    assert [rec["id"] for rec in records] == ["x-000", "x-002"]
    assert lane.questions == ["q0?", "q2?"]
    assert all(rec["passed"] for rec in records)


def test_report_and_table_carry_the_selection() -> None:
    r1 = [_rec("r1", "k_a", True, "stale", True)]
    selection = {"set": "both", "split": "dev", "files": ["a.jsonl", "b.jsonl"]}
    report = se.build_report(
        mode="lane",
        today="2026-11-27",
        model="fake/model",
        db=Path("d"),
        jsonl_path=Path("a.jsonl"),
        repeat=1,
        rounds=[r1],
        round_tokens=[se.TokenCounter()],
        judge_result=None,
        selection=selection,
    )
    assert report["selection"] == selection
    text = se.render_table(report)
    for needle in ("set=both", "split=dev", "a.jsonl", "b.jsonl"):
        assert needle in text, needle
    # no selection: the table keeps the old shape
    bare = se.build_report(
        mode="lane",
        today="2026-11-27",
        model="fake/model",
        db=Path("d"),
        jsonl_path=Path("a.jsonl"),
        repeat=1,
        rounds=[r1],
        round_tokens=[se.TokenCounter()],
        judge_result=None,
    )
    assert bare["selection"] is None
    assert "set=" not in se.render_table(bare)
