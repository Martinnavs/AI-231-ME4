"""Tests for pipeline/src/voice_pipeline/ask_documents.py (ticket 209)."""

from __future__ import annotations

import asyncio
import json

import pytest

from voice_pipeline.ask_documents import (
    _eval_calc,
    _sources_of,
    ask_documents,
)


def _p(source: str, text: str, part: str = "chunk") -> dict:
    return {
        "source": source,
        "id": f"{source}#0",
        "section": "",
        "page": None,
        "file_date": "2026-01-01",
        "text": text,
        "score": 1.0,
        "coverage": 1.0,
        "part": part,
    }


class _FakeRequest:
    """Scripted request that returns JSON text."""

    def __init__(self, replies: list[str | Exception]):
        self.replies = list(replies)
        self.idx = 0

    async def __call__(self, system: str, content: str) -> str:
        if self.idx >= len(self.replies):
            raise RuntimeError("no more scripted replies")
        reply = self.replies[self.idx]
        self.idx += 1
        if isinstance(reply, Exception):
            raise reply
        return reply

    @property
    def call_count(self) -> int:
        return self.idx


def _json_reply(status: str, **kwargs: object) -> str:
    d: dict = {"status": status}
    for k, v in kwargs.items():
        d[k] = v
    return json.dumps(d)


# ---- Test 1: accepted single-file answer ----


@pytest.mark.asyncio
async def test_accepted_answer() -> None:
    passages = [
        _p(
            "gsa-tts/security_incidents.md",
            "It is critical that you notify GSA IT within 1 hour of suspected incident.",
        )
    ]
    req = _FakeRequest(
        [
            _json_reply(
                "answer",
                answer="Within 1 hour.",
                file="gsa-tts/security_incidents.md",
                quotes=["within 1 hour of suspected"],
            )
        ]
    )

    async def ctx(q: str):
        return passages

    r = await ask_documents(
        "How quickly notify GSA IT about a suspected incident?", context=ctx, request=req
    )
    assert r.status == "accepted"
    assert r.spoken == "As per security_incidents.md: Within 1 hour."
    assert r.calls == 1


# ---- Test 2: file not among sources ----


@pytest.mark.asyncio
async def test_file_not_in_sources() -> None:
    passages = [_p("other.md", "Some text here.")]
    req = _FakeRequest([_json_reply("answer", answer="X", file="wrong.md", quotes=["text here"])])

    async def ctx(q: str):
        return passages

    r = await ask_documents("Q?", context=ctx, request=req)
    assert r.status == "check_failed"
    assert r.rejected is not None
    assert r.rejected["failed"] == "file_known"


# ---- Test 3: quote not verbatim ----


@pytest.mark.asyncio
async def test_quote_not_verbatim() -> None:
    passages = [_p("file.md", "The actual text is here and now.")]
    req = _FakeRequest(
        [
            _json_reply(
                "answer", answer="Something", file="file.md", quotes=["totally made up phrase"]
            )
        ]
    )

    async def ctx(q: str):
        return passages

    r = await ask_documents("Q?", context=ctx, request=req)
    assert r.status == "check_failed"
    assert r.rejected is not None
    assert r.rejected["failed"] == "quote"


# ---- Test 4: CALC ----


@pytest.mark.asyncio
async def test_calc_accepted() -> None:
    passages = [_p("stats.md", "70% (7 out of 10) passed, and 80% (8 out of 10) passed.")]
    req = _FakeRequest(
        [
            _json_reply(
                "answer",
                answer="10 percentage points",
                file="stats.md",
                quotes=[
                    "70% (7 out of 10) passed",
                    "80% (8 out of 10) passed",
                ],
                calc="80 - 70",
            )
        ]
    )

    async def ctx(q: str):
        return passages

    r = await ask_documents("What is the percentage point difference?", context=ctx, request=req)
    assert r.status == "accepted"


@pytest.mark.asyncio
async def test_calc_bad_operand() -> None:
    passages = [_p("stats.md", "70% (7 out of 10) passed and 80% (8 out of 10) passed.")]
    req = _FakeRequest(
        [
            _json_reply(
                "answer",
                answer="5 points",
                file="stats.md",
                quotes=["70% (7 out of 10) passed"],
                calc="80 - 75",
            )
        ]
    )

    async def ctx(q: str):
        return passages

    r = await ask_documents("Diff?", context=ctx, request=req)
    assert r.status == "check_failed"
    assert r.rejected is not None
    assert r.rejected["failed"] == "calc"


@pytest.mark.asyncio
async def test_calc_injection() -> None:
    passages = [_p("file.md", "Some text.")]
    req = _FakeRequest(
        [
            _json_reply(
                "answer",
                answer="hacked",
                file="file.md",
                quotes=["Some text"],
                calc="__import__('os').system('echo pwned')",
            )
        ]
    )

    async def ctx(q: str):
        return passages

    r = await ask_documents("Q?", context=ctx, request=req)
    assert r.status == "check_failed"
    assert r.rejected is not None
    assert r.rejected["failed"] == "calc"


# ---- Test 5: first/last guard ----


@pytest.mark.asyncio
async def test_ordinal_first_grep_accepted() -> None:
    grep_text = (
        "L3: First invalid user attempt from 10.0.0.1\n"
        "L9: Second invalid user attempt from 10.0.0.2\n"
        "L40: Third invalid user attempt from 10.0.0.3"
    )
    passages = [_p("log/app.log", grep_text, part="grep")]
    req = _FakeRequest(
        [
            _json_reply(
                "answer",
                answer="10.0.0.1",
                file="log/app.log",
                quotes=["First invalid user attempt from 10.0.0.1"],
            )
        ]
    )

    async def ctx(q: str):
        return passages

    r = await ask_documents("What is the first invalid user?", context=ctx, request=req)
    assert r.status == "accepted"


@pytest.mark.asyncio
async def test_ordinal_middle_grep_rejected() -> None:
    grep_text = (
        "L3: First invalid user attempt from 10.0.0.1\n"
        "L9: Second invalid user attempt from 10.0.0.2\n"
        "L40: Third invalid user attempt from 10.0.0.3"
    )
    passages = [_p("log/app.log", grep_text, part="grep")]
    req = _FakeRequest(
        [
            _json_reply(
                "answer",
                answer="10.0.0.2",
                file="log/app.log",
                quotes=["Second invalid user attempt from 10.0.0.2"],
            )
        ]
    )

    async def ctx(q: str):
        return passages

    r = await ask_documents("What is the first invalid user?", context=ctx, request=req)
    assert r.status == "check_failed"
    assert r.rejected is not None
    assert r.rejected["failed"] == "ordinal"


@pytest.mark.asyncio
async def test_ordinal_last_grep_accepted() -> None:
    grep_text = (
        "L3: First invalid user attempt from 10.0.0.1\n"
        "L9: Second invalid user attempt from 10.0.0.2\n"
        "L40: Third invalid user attempt from 10.0.0.3"
    )
    passages = [_p("log/app.log", grep_text, part="grep")]
    req = _FakeRequest(
        [
            _json_reply(
                "answer",
                answer="10.0.0.3",
                file="log/app.log",
                quotes=["Third invalid user attempt from 10.0.0.3"],
            )
        ]
    )

    async def ctx(q: str):
        return passages

    r = await ask_documents("What is the last invalid user?", context=ctx, request=req)
    assert r.status == "accepted"


@pytest.mark.asyncio
async def test_ordinal_whole_passage_accepted() -> None:
    passages = [_p("log/app.log", "L3: First entry\nL9: Middle\nL40: Last", part="whole")]
    req = _FakeRequest(
        [_json_reply("answer", answer="First entry", file="log/app.log", quotes=["First entry"])]
    )

    async def ctx(q: str):
        return passages

    r = await ask_documents("What is the first entry?", context=ctx, request=req)
    assert r.status == "accepted"


@pytest.mark.asyncio
async def test_ordinal_chunk_log_rejected() -> None:
    passages = [_p("log/app.log", "Some entry in the log.", part="chunk")]
    req = _FakeRequest(
        [
            _json_reply(
                "answer", answer="Some entry", file="log/app.log", quotes=["Some entry in the"]
            )
        ]
    )

    async def ctx(q: str):
        return passages

    r = await ask_documents("What is the first entry?", context=ctx, request=req)
    assert r.status == "check_failed"
    assert r.rejected is not None
    assert r.rejected["failed"] == "ordinal"


# ---- Test 6: WEB composition ----


@pytest.mark.asyncio
async def test_web_composition() -> None:
    passages = [_p("pico/pico_datasheet.pdf", "The pico has a Cortex-M0+ processor.")]
    req = _FakeRequest(
        [
            _json_reply(
                "web",
                answer="Cortex-M0+ processor",
                file="pico/pico_datasheet.pdf",
                quotes=["Cortex-M0+ processor"],
                web_question="what is the clock speed of Cortex-M0+?",
            )
        ]
    )

    async def ctx(q: str):
        return passages

    async def web(q: str):
        return "As per example.org: 133 MHz clock speed."

    r = await ask_documents("What processor and speed?", context=ctx, request=req, web=web)
    assert r.status == "accepted"
    assert r.spoken is not None
    assert r.spoken.startswith("As per pico_datasheet.pdf:")
    assert "example.org" in r.spoken


@pytest.mark.asyncio
async def test_web_none() -> None:
    passages = [_p("pico/pico_datasheet.pdf", "The pico has a Cortex-M0+ processor.")]
    req = _FakeRequest(
        [
            _json_reply(
                "web",
                answer="Cortex-M0+ processor",
                file="pico/pico_datasheet.pdf",
                quotes=["Cortex-M0+ processor"],
                web_question="speed?",
            )
        ]
    )

    async def ctx(q: str):
        return passages

    async def web_none(q: str):
        return None

    r = await ask_documents(
        "What processor is in the pico datasheet?", context=ctx, request=req, web=web_none
    )
    assert r.status == "accepted"
    assert r.spoken is not None
    assert r.spoken.endswith("I couldn't confirm the rest on the web.")


# ---- Test 7: Fallback ----


@pytest.mark.asyncio
async def test_fallback_file() -> None:
    async def ctx(q: str):
        if "schedule" in q.lower():
            return [_p("gsa-tts/work_schedules.md", "Meetings are at 9am every Monday.")]
        return [_p("other.md", "Nothing relevant here.")]

    req = _FakeRequest(
        [
            _json_reply("none"),
            _json_reply("file", file="gsa-tts/work_schedules.md"),
            _json_reply(
                "answer",
                answer="9am every Monday",
                file="gsa-tts/work_schedules.md",
                quotes=["Meetings are at 9am every"],
            ),
        ]
    )

    async def catalog():
        return ["gsa-tts/work_schedules.md", "other.md", "third.md"]

    r = await ask_documents("When are meetings?", context=ctx, request=req, catalog=catalog)
    assert r.status == "accepted"
    assert r.calls == 3
    assert r.fallback == "file"


@pytest.mark.asyncio
async def test_fallback_no_double_request() -> None:
    async def ctx(q: str):
        return [_p("other.md", "Nothing relevant.")]

    req = _FakeRequest(
        [
            _json_reply("none"),
            _json_reply("file", file="other.md"),
            _json_reply("none"),
        ]
    )

    async def catalog():
        return ["other.md"]

    r = await ask_documents("Q?", context=ctx, request=req, catalog=catalog)
    assert r.status == "none"
    assert r.calls == 3  # initial + fallback request + second verdict (no triple fallback)


# ---- Test 8: Timeout ----


@pytest.mark.asyncio
async def test_timeout() -> None:
    async def slow_req(system: str, content: str) -> str:
        await asyncio.sleep(0.2)
        return _json_reply("answer", answer="X")

    async def ctx(q: str):
        return [_p("f.md", "text")]

    r = await ask_documents("Q?", context=ctx, request=slow_req, budget_s=0.05)
    assert r.status == "timeout"
    assert r.calls == 1


# ---- Test 9: Non-JSON reply ----


@pytest.mark.asyncio
async def test_non_json() -> None:
    async def bad_req(system: str, content: str) -> str:
        return "This is not JSON at all"

    async def ctx(q: str):
        return [_p("f.md", "text")]

    r = await ask_documents("Q?", context=ctx, request=bad_req)
    assert r.status == "error"
    assert r.rejected is not None
    assert r.rejected["failed"] == "json"


# ---- Test 10: named-file guard ----


@pytest.mark.asyncio
async def test_named_file_guard() -> None:
    """Question names openssh file, answer comes from linux file -> named_file."""
    passages = [_p("loghub/linux_2k.log", "CPU model is Intel Pentium III (Coppermine)")]
    req = _FakeRequest(
        [
            _json_reply(
                "answer",
                answer="CPU model is Pentium III",
                file="loghub/linux_2k.log",
                quotes=["CPU model is Intel Pentium III"],
            )
        ]
    )

    async def ctx(q: str):
        return passages

    r = await ask_documents(
        "What CPU model does the machine in the OpenSSH log have?",
        context=ctx,
        request=req,
        file_names=["loghub/openssh_2k.log", "loghub/linux_2k.log"],
    )
    assert r.status == "check_failed"
    assert r.rejected is not None
    assert r.rejected["failed"] == "named_file"


# ---- Helper tests ----


def test_eval_calc_safe() -> None:
    val, nums, safe = _eval_calc("80 - 70")
    assert val == 10.0 and nums == ["80", "70"] and safe


def test_eval_calc_unsafe() -> None:
    val, nums, safe = _eval_calc("__import__('os')")
    assert safe is False


def test_sources_of() -> None:
    p1 = _p("a.md", "x")
    p2 = _p("b.md", "y")
    p3 = _p("a.md", "z")
    assert _sources_of([p1, p2, p3]) == ["a.md", "b.md"]
