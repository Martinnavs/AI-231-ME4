"""Pre-deploy 250: a sufficiency check after an accepted document answer.

An accepted document answer was spoken as is, with no check of whether it really
answers the question. Now, behind the flag ``SUFFICIENCY`` (off by default), ONE
call to the document lane's own cheap JSON model asks one of four verdicts about
the answer that was just spoken: sufficient / stale / missing_half /
wrong_question. The verdict starts as a task BEFORE the document answer is spoken
(no first-audio penalty); the document answer always stands as spoken today, and
the check can only add ONE supplement sentence after it (worded as an addition,
never a retraction) or log a wrong_question.

Fail-open: the cheap gate limits the call to time-sensitive questions, sources
older than the current period, and short answers (never pleasantries, never
non-document routes); it runs at most once per turn; and any error, timeout or
unparsable reply speaks nothing extra (the old behaviour, byte-for-byte).

The tests use a scripted fake request and a scripted web lookup, with synthetic
names only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voice_pipeline import agent_server, evaluators, sufficiency
from voice_pipeline.agent_server import AgentServer
from voice_pipeline.ask_documents import DocResult

# --- the scripted verdicts (the patterns are what is under test, not the words) ---
V_SUFFICIENT = json.dumps(
    {"verdict": "sufficient", "web_question": "", "reason": "the answer states what was asked"}
)
V_STALE = json.dumps(
    {
        "verdict": "stale",
        "web_question": "What is the current Zorblat index?",
        "reason": "the source is three years old",
    }
)
V_MISSING_HALF = json.dumps(
    {
        "verdict": "missing_half",
        "web_question": "Where is the office of the Zorblat bureau?",
        "reason": "the office location is outside the file",
    }
)
V_WRONG = json.dumps(
    {
        "verdict": "wrong_question",
        "web_question": "How deep is the Quenth river now?",
        "reason": "the answer is about the bureau, not the river",
    }
)
V_GARBAGE = "not json at all"  # an unparsable reply: fail-open

WEB_NEWER = "The current Zorblat index is 14.2."
WEB_MISSING = "The office is on Virel Street."
WEB_WRONG = "The Quenth river is 40 meters deep."
# the web lookup's own honest line (evaluators.HEDGE); a reply this way did not check
HEDGE = "I couldn't confirm that with a search, so I won't guess."

NOW = datetime(2026, 11, 27)  # fixed for the gate unit tests (the server level uses the real clock)

SETTLED_Q = "What was the Zorblat index in the sheet?"  # no present/next cue
NOW_Q = "What is the Zorblat index right now?"  # time-sensitive ("now")
LONG_ANSWER = "As per the sheet: the Zorblat index is 12.5 and it is unchanged."
DOC_ANSWER = "As per the sheet: the Zorblat index is 12.5."
FILE_NOW = f"sheet_{NOW.year}.csv"  # a current source
FILE_OLD = f"sheet_{NOW.year - 3}.csv"  # a source from an earlier period


class _Rec:
    def __init__(self) -> None:
        self.verdict_calls: list[str] = []
        self.web_calls: list[str] = []


def make_server(
    tmp_path: Path,
    *,
    verdict_reply: str,
    web_reply: str,
    verdict_sleep_s: float = 0.0,
    verdict_error: bool = False,
) -> tuple[AgentServer, _Rec]:
    """An AgentServer with a scripted verdict model (its own cheap JSON request) and a
    scripted web lookup (the ledger/hedge/directness live in the real web server)."""
    rec = _Rec()

    async def wa(params: Any) -> None:
        rec.web_calls.append(params.arguments.get("query", ""))
        await params.result_callback(web_reply)

    async def fake_json(system: str, content: str) -> str:
        assert system == sufficiency.SYSTEM  # only the verdict call reaches this fake
        rec.verdict_calls.append(content)
        if verdict_error:
            raise RuntimeError("the verdict model is down")
        if verdict_sleep_s:
            await asyncio.sleep(verdict_sleep_s)
        return verdict_reply

    tools = SimpleNamespace(standard_tools=[SimpleNamespace(name="web_answer", handler=wa)])
    return (
        AgentServer(object(), [], tmp_path / "250.jsonl", tools=tools, doc_json_request=fake_json),
        rec,
    )


def doc_result(file_name: str, spoken: str) -> DocResult:
    """An accepted document answer, as the checked document lane hands it over."""
    return DocResult(
        "accepted",
        spoken=spoken,
        answer=spoken,
        file=file_name,
        sources=[file_name],
        calls=1,
    )


def make_on_text() -> tuple[list[tuple[str, float]], Any]:
    events: list[tuple[str, float]] = []

    async def on_text(text: str) -> None:
        events.append((text, time.perf_counter()))

    return events, on_text


def turn(server: AgentServer, said: str, doc_result_: DocResult, on_text: Any) -> Any:
    """One accepted-docs answer through the real hook, the way both call sites build it.
    Returns the (unawaited) hook coroutine; the caller awaits it."""
    return server._maybe_supplement(
        said,
        doc_result_,
        server.spoken(
            doc_result_.spoken or "",
            on_text,
            docs="accepted",
            doc_s=0.1,
            route="docs",
            doc_calls=doc_result_.calls,
            fallback=None,
            sources=doc_result_.sources,
        ),
        on_text,
    )


# --- the flag: off by default, the old behaviour byte-for-byte ---


async def test_off_by_default_speaks_nothing_extra(tmp_path: Path) -> None:
    server, rec = make_server(tmp_path, verdict_reply=V_STALE, web_reply=WEB_NEWER)
    events, on_text = make_on_text()
    res = await turn(server, NOW_Q, doc_result(FILE_NOW, DOC_ANSWER), on_text)
    assert rec.verdict_calls == []  # the check is not paid for
    assert [t for t, _ in events] == [DOC_ANSWER]
    assert "sufficiency" not in res


# --- sufficient: the answer stands, nothing extra ---


async def test_sufficient_speaks_nothing_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    server, rec = make_server(tmp_path, verdict_reply=V_SUFFICIENT, web_reply=WEB_NEWER)
    events, on_text = make_on_text()
    res = await turn(server, NOW_Q, doc_result(FILE_NOW, DOC_ANSWER), on_text)
    assert len(rec.verdict_calls) == 1
    assert rec.web_calls == []  # the web is not paid for
    assert [t for t, _ in events] == [DOC_ANSWER]
    assert res["sufficiency"] == "sufficient"
    assert res["sufficiency_web"] is False
    assert res["sufficiency_s"] >= 0


# --- stale: ONE supplement sentence after the document answer, worded as an addition ---


async def test_stale_speaks_the_newer_figure_after_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    server, rec = make_server(tmp_path, verdict_reply=V_STALE, web_reply=WEB_NEWER)
    events, on_text = make_on_text()
    res = await turn(server, NOW_Q, doc_result(FILE_NOW, DOC_ANSWER), on_text)
    lead = "The web has a newer figure: "
    assert [t for t, _ in events] == [DOC_ANSWER, lead + WEB_NEWER]  # doc answer, then ONE add
    assert res["text"] == f"{DOC_ANSWER} {lead}{WEB_NEWER}"  # the log keeps what was spoken
    assert res["sufficiency"] == "stale"
    assert res["sufficiency_web"] is True
    assert rec.web_calls == ["What is the current Zorblat index?"]  # the lookup ran once


# --- missing_half: the outside fact is spoken as an addition ---


async def test_missing_half_speaks_the_missing_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    server, rec = make_server(tmp_path, verdict_reply=V_MISSING_HALF, web_reply=WEB_MISSING)
    events, on_text = make_on_text()
    # a short answer (5 words) pays for the check through the gate's short-answer leg
    doc = "The file says Keira Moss."
    res = await turn(
        server,
        "Who runs the Zorblat bureau, and where is its office?",
        doc_result(f"notes_{NOW.year}.md", doc),
        on_text,
    )
    lead = "From the web: "
    assert [t for t, _ in events] == [doc, lead + WEB_MISSING]
    assert res["sufficiency"] == "missing_half"
    assert res["sufficiency_web"] is True
    assert rec.web_calls == ["Where is the office of the Zorblat bureau?"]


# --- wrong_question: the web is checked, but nothing extra is spoken ---


async def test_wrong_question_is_logged_not_spoken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    server, rec = make_server(tmp_path, verdict_reply=V_WRONG, web_reply=WEB_WRONG)
    events, on_text = make_on_text()
    doc = "As per the file: Keira Moss runs the bureau."
    # a source from an earlier period pays for the check through the gate's source-year leg
    with caplog.at_level(logging.INFO, logger="voice_pipeline.agent_server"):
        res = await turn(
            server, "How deep is the Quenth river?", doc_result(FILE_OLD, doc), on_text
        )
    assert [t for t, _ in events] == [doc]  # nothing extra is spoken
    assert rec.web_calls == ["How deep is the Quenth river now?"]  # the web was checked
    assert res["sufficiency"] == "wrong_question"
    assert res["sufficiency_web"] is False
    assert any("wrong_question" in m for m in caplog.messages)  # logged, not spoken


# --- a hedged web lookup did not actually check the web: nothing extra ---


async def test_a_hedged_web_speaks_nothing_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    server, rec = make_server(tmp_path, verdict_reply=V_STALE, web_reply=HEDGE)
    events, on_text = make_on_text()
    res = await turn(server, NOW_Q, doc_result(FILE_NOW, DOC_ANSWER), on_text)
    assert [t for t, _ in events] == [DOC_ANSWER]
    assert rec.web_calls == ["What is the current Zorblat index?"]  # the lookup ran (and hedged)
    assert res["sufficiency"] == "stale"
    assert res["sufficiency_web"] is False


# --- a failed verdict (garbage, timeout, error): nothing extra, the old behaviour ---


@pytest.mark.parametrize(
    ("verdict_reply", "verdict_sleep_s", "verdict_error"),
    [
        pytest.param(V_GARBAGE, 0.0, False, id="garbage"),
        pytest.param(V_STALE, 2.0, False, id="timeout"),  # past the 1.5 s verdict cap
        pytest.param(V_STALE, 0.0, True, id="error"),
    ],
)
async def test_a_failed_verdict_speaks_nothing_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verdict_reply: str,
    verdict_sleep_s: float,
    verdict_error: bool,
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    server, rec = make_server(
        tmp_path,
        verdict_reply=verdict_reply,
        web_reply=WEB_NEWER,
        verdict_sleep_s=verdict_sleep_s,
        verdict_error=verdict_error,
    )
    events, on_text = make_on_text()
    res = await turn(server, NOW_Q, doc_result(FILE_NOW, DOC_ANSWER), on_text)
    assert len(rec.verdict_calls) == 1  # the check was paid for once, then it failed
    assert rec.web_calls == []
    assert [t for t, _ in events] == [DOC_ANSWER]  # nothing extra is spoken
    assert res["sufficiency"] == "none"
    assert res["sufficiency_web"] is False

    # byte-for-byte the old behaviour: the same turn with the flag off logs the same row
    # (the sufficiency fields are the only additions)
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", False)
    ctrl_server, ctrl_rec = make_server(tmp_path, verdict_reply=verdict_reply, web_reply=WEB_NEWER)
    ctrl_events, ctrl_on_text = make_on_text()
    res2 = await turn(ctrl_server, NOW_Q, doc_result(FILE_NOW, DOC_ANSWER), ctrl_on_text)
    assert ctrl_rec.verdict_calls == []
    assert [t for t, _ in ctrl_events] == [t for t, _ in events]
    assert {k: v for k, v in res.items() if not k.startswith("sufficiency")} == res2


# --- the cheap gate ---


def test_the_gate_skips_a_settled_question_with_a_current_source_and_a_long_answer() -> None:
    assert sufficiency.should_check(SETTLED_Q, str(NOW.year), LONG_ANSWER, now=NOW) is False


def test_the_gate_fires_on_a_time_sensitive_question() -> None:
    assert sufficiency.time_sensitive("What is the Zorblat index right now?")
    assert sufficiency.time_sensitive("What is the latest exchange rate?")
    assert sufficiency.time_sensitive("Is the note still right?")
    assert not sufficiency.time_sensitive("What was the Zorblat index in the sheet?")


def test_the_gate_fires_on_a_source_older_than_the_current_period() -> None:
    assert sufficiency.source_earlier(str(NOW.year - 3), NOW)
    assert not sufficiency.source_earlier(str(NOW.year), NOW)
    assert sufficiency.source_earlier(None, NOW) is False
    # fiscal periods: the fiscal year starts in July (NOW is November, fiscal year FY27)
    assert sufficiency.source_earlier("FY25", NOW)
    assert sufficiency.source_earlier(f"FY{NOW.year % 100:02d}", NOW)  # FY26 < FY27
    next_fy = f"FY{(NOW.year + 1) % 100:02d}"  # FY27: the current fiscal year
    assert not sufficiency.source_earlier(next_fy, NOW)
    assert not sufficiency.source_earlier(next_fy, datetime(2026, 3, 2))  # ...or the next one


def test_the_gate_fires_on_a_short_answer() -> None:
    assert sufficiency.should_check(SETTLED_Q, None, "It is 12.5.", now=NOW)


def test_pleasantries_never_fire_the_gate() -> None:
    long_reply = "You're welcome, happy to help with that."
    assert sufficiency.should_check("Thanks a lot!", None, long_reply, now=NOW) is False
    assert sufficiency.time_sensitive("Goodbye!") is False
    assert sufficiency.time_sensitive("Okay, got it.") is False


async def test_a_settled_turn_with_a_current_source_and_a_long_answer_makes_no_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    server, rec = make_server(tmp_path, verdict_reply=V_STALE, web_reply=WEB_NEWER)
    events, on_text = make_on_text()
    res = await turn(server, SETTLED_Q, doc_result(FILE_NOW, LONG_ANSWER), on_text)
    assert rec.verdict_calls == []  # the call is not paid for
    assert "sufficiency" not in res
    assert [t for t, _ in events] == [LONG_ANSWER]


async def test_a_settled_turn_with_an_older_source_makes_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    server, rec = make_server(tmp_path, verdict_reply=V_SUFFICIENT, web_reply=WEB_NEWER)
    events, on_text = make_on_text()
    res = await turn(server, SETTLED_Q, doc_result(FILE_OLD, LONG_ANSWER), on_text)
    assert len(rec.verdict_calls) == 1  # the older source pays for the check
    assert res["sufficiency"] == "sufficient"
    assert [t for t, _ in events] == [LONG_ANSWER]


def test_time_sensitive_uses_the_root_classifier_when_it_is_importable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def changeable(q: str) -> bool:
        seen.append(q)
        return "virel" in q.lower()

    monkeypatch.setattr(sufficiency, "_grounding", SimpleNamespace(changeable=changeable))
    assert sufficiency.time_sensitive("Who is the head of Virel House?")
    assert not sufficiency.time_sensitive("What was the Zorblat index in the sheet?")
    assert len(seen) == 2  # the root classifier answered, not the local regex


def test_time_sensitive_falls_back_to_the_regex_when_the_classifier_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(q: str) -> bool:
        raise RuntimeError("no model installed")

    monkeypatch.setattr(sufficiency, "_grounding", SimpleNamespace(changeable=broken))
    assert sufficiency.time_sensitive("What is the Zorblat index right now?")
    assert not sufficiency.time_sensitive("What was the Zorblat index in the sheet?")


# --- at most one check per turn (the per-turn flag, reset by handle) ---


async def test_the_check_runs_at_most_once_per_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    server, rec = make_server(tmp_path, verdict_reply=V_STALE, web_reply=WEB_NEWER)
    _, on_text = make_on_text()
    res1 = await turn(server, NOW_Q, doc_result(FILE_NOW, DOC_ANSWER), on_text)
    assert res1["sufficiency"] == "stale"
    assert len(rec.verdict_calls) == 1
    # the flag is still set (the turn is not over): a second accepted answer pays nothing
    _, on_text2 = make_on_text()
    res2 = await turn(server, NOW_Q, doc_result(FILE_NOW, DOC_ANSWER), on_text2)
    assert len(rec.verdict_calls) == 1
    assert "sufficiency" not in res2
    # the next turn resets the flag (handle() does this per turn)
    server._sufficiency_attempted = False
    _, on_text3 = make_on_text()
    res3 = await turn(server, NOW_Q, doc_result(FILE_NOW, DOC_ANSWER), on_text3)
    assert len(rec.verdict_calls) == 2
    assert res3["sufficiency"] == "stale"


# --- no first-audio penalty: the document answer speaks before the verdict completes ---


async def test_the_first_audio_is_not_delayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    server, rec = make_server(
        tmp_path, verdict_reply=V_STALE, web_reply=WEB_NEWER, verdict_sleep_s=1.0
    )
    events, on_text = make_on_text()
    t0 = time.perf_counter()
    res = await turn(server, NOW_Q, doc_result(FILE_NOW, DOC_ANSWER), on_text)
    assert res["sufficiency"] == "stale"
    assert len(events) == 2
    first_text, first_t = events[0]
    second_text, second_t = events[1]
    assert first_text == DOC_ANSWER  # the document answer is the first thing spoken
    assert first_t - t0 < 0.5  # ...well before the verdict (1.0 s) could have completed
    assert second_text.startswith("The web has a newer figure: ")
    assert second_t - t0 > 0.9  # the supplement comes only after the verdict completed


# --- the wiring: both accepted-docs speaks go through the hook, and the hedge copy is pinned ---


def test_both_accepted_docs_speaks_go_through_the_hook() -> None:
    src = Path(agent_server.__file__).read_text()
    assert src.count("self._maybe_supplement(") == 2  # the two accepted-docs call sites


def test_the_hedge_copy_is_pinned_to_evaluators() -> None:
    assert sufficiency.HEDGE_TAIL == evaluators.HEDGE
    assert sufficiency.HEDGE_TAIL == HEDGE
