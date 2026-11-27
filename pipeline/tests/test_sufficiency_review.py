"""Adversarial review tests for sufficiency check (ticket 256).

Tests marked ``xfail(strict=True)`` document R1/R2 bugs found in ``sufficiency.py`` and
``agent_server.py`` without modifying production files on this frozen-release branch.
When those bugs are fixed, remove the ``xfail`` markers and keep these as regression tests.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from voice_pipeline import agent_tools, ask_documents, compound, doc_route, sufficiency, temporal
from voice_pipeline.agent_server import AgentServer
from voice_pipeline.ask_documents import DocResult
from voice_pipeline.payload_guard import PayloadFilter, SourceFilter
from voice_pipeline.spoken_filter import SentenceFilter

V_STALE = json.dumps(
    {
        "verdict": "stale",
        "web_question": "What is the current Zorblat index?",
        "reason": "the source is old",
    }
)
V_MISSING_HALF = json.dumps(
    {
        "verdict": "missing_half",
        "web_question": "Where is the Zorblat bureau office?",
        "reason": "office is outside the file",
    }
)
V_WRONG = json.dumps(
    {
        "verdict": "wrong_question",
        "web_question": "How deep is the Quenth river now?",
        "reason": "answer is about the bureau",
    }
)
NOW = datetime(2026, 11, 27)


def _recorded_web_tools(
    record: list[dict[str, Any]],
    web_calls: list[str],
    web_reply: str,
) -> ToolsSchema:
    async def wa(params: Any) -> None:
        web_calls.append(params.arguments.get("query", ""))
        await params.result_callback(web_reply)

    schema = FunctionSchema(
        name="web_answer",
        description="web lookup",
        properties={"query": {"type": "string"}},
        required=["query"],
        handler=wa,
    )
    return ToolsSchema(standard_tools=[agent_tools._recorded(schema, record)])


# --- R1-1: REPLY_SCHEMA breaks production verdict() ---


@pytest.mark.xfail(
    strict=True,
    reason="R1-1: ask_documents.gemini_json_request hardcodes REPLY_SCHEMA which lacks 'verdict'",
)
async def test_gemini_json_request_schema_allows_sufficiency_verdict() -> None:
    props = ask_documents.REPLY_SCHEMA.get("properties", {})
    assert "verdict" in props
    assert "reason" in props

    # Even if Gemini obeys REPLY_SCHEMA perfectly, sufficiency.verdict() drops it as None.
    async def schema_constrained_json(_system: str, _user: str) -> str:
        return json.dumps({"status": "answer", "answer": "stale", "web_question": "q?"})

    got = await sufficiency.verdict(
        schema_constrained_json,
        "What is the index right now?",
        "As per sheet.csv: 12.5.",
        "2023",
        "2026-11-27",
    )
    assert got is not None


# --- R1-2: ask_documents(status="web") double lookup + REPEAT_PREFIX spoken aloud ---


@pytest.mark.xfail(
    strict=True,
    reason="R1-2: _maybe_supplement re-runs web_answer after status='web' and speaks REPEAT_PREFIX",
)
async def test_doc_web_result_does_not_double_lookup_or_speak_repeat_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    monkeypatch.setattr(agent_tools, "CHAIN", None)
    monkeypatch.setattr(agent_tools, "SCRATCHPAD", None)
    agent_tools.reset_turn_lookups()

    record: list[dict[str, Any]] = []
    web_calls: list[str] = []
    verdict_calls: list[str] = []
    web_reply = "As per Directory, the office is on Virel Street."

    async def fake_json(_system: str, content: str) -> str:
        verdict_calls.append(content)
        return V_MISSING_HALF

    tools = _recorded_web_tools(record, web_calls, web_reply)
    server = AgentServer(
        object(), record, tmp_path / "r1_2.jsonl", tools=tools, doc_json_request=fake_json
    )

    # Simulate ask_documents(status="web") having already called web_answer in this turn.
    first_web = await server.lookup("web_answer", {"query": "Where is the Zorblat office?"})
    assert first_web == web_reply
    doc = DocResult(
        "accepted",
        spoken=f"As per notes_2026.md: Keira Moss. {web_reply}",
        answer="Keira Moss",
        file="notes_2026.md",
        web_question="Where is the Zorblat office?",
        web_text=web_reply,
        sources=["notes_2026.md"],
        calls=1,
    )
    spoken_events: list[str] = []

    async def on_text(text: str) -> None:
        spoken_events.append(text)

    await server._maybe_supplement(
        "Who runs the Zorblat bureau right now, and where is it?",
        doc,
        server.spoken(doc.spoken or "", on_text, docs="accepted", route="docs"),
        on_text,
    )
    assert len(web_calls) == 1
    assert not any(agent_tools.REPEAT_PREFIX in s for s in spoken_events)
    assert len(spoken_events) == 1


# --- R1-3: wrong_question pays for web_answer and poisons turn_source_sentence / turn_claim ---


@pytest.mark.xfail(
    strict=True,
    reason="R1-3: wrong_question calls web_answer and leaks into turn_source_sentence/turn_claim",
)
async def test_wrong_question_does_not_call_web_or_leak_source_and_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    monkeypatch.setattr(agent_tools, "CHAIN", None)
    monkeypatch.setattr(agent_tools, "SCRATCHPAD", None)
    agent_tools.reset_turn_lookups()

    record: list[dict[str, Any]] = []
    web_calls: list[str] = []

    async def fake_json(_system: str, _content: str) -> str:
        return V_WRONG

    tools = _recorded_web_tools(
        record, web_calls, "As per BBC, the Quenth river is 40 meters deep."
    )
    server = AgentServer(
        object(), record, tmp_path / "r1_3.jsonl", tools=tools, doc_json_request=fake_json
    )
    doc = DocResult(
        "accepted",
        spoken="As per sheet_2023.csv: Keira Moss runs the bureau.",
        answer="Keira Moss runs the bureau.",
        file="sheet_2023.csv",
        sources=["sheet_2023.csv"],
        calls=1,
    )
    events: list[str] = []

    async def on_text(text: str) -> None:
        events.append(text)

    res = await server._maybe_supplement(
        "How deep is the Quenth river?",
        doc,
        server.spoken(doc.spoken or "", on_text, docs="accepted", route="docs"),
        on_text,
    )
    assert web_calls == []
    assert agent_tools.turn_source_sentence(res["text"]) == ""
    assert agent_tools.turn_claim() == ""


# --- R1-4: cancelled speak leaks verdict task; WS disconnect holds self.lock ---


@pytest.mark.xfail(
    strict=True,
    reason="R1-4: cancelling speak leaves the background sufficiency.verdict task running",
)
async def test_cancelled_speak_cancels_pending_verdict_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    started = asyncio.Event()
    tasks: list[asyncio.Task[Any]] = []

    async def slow_json(_system: str, _content: str) -> str:
        t = asyncio.current_task()
        if t is not None:
            tasks.append(t)
        started.set()
        await asyncio.sleep(5.0)
        return V_STALE

    server = AgentServer(object(), [], tmp_path / "r1_4a.jsonl", doc_json_request=slow_json)
    doc = DocResult(
        "accepted",
        spoken="As per sheet_2026.csv: 12.5.",
        answer="12.5.",
        file="sheet_2026.csv",
        sources=["sheet_2026.csv"],
        calls=1,
    )

    async def cancelled_speak() -> dict[str, Any]:
        await started.wait()
        raise asyncio.CancelledError

    async def on_text(_text: str) -> None:
        pass

    with pytest.raises(asyncio.CancelledError):
        await server._maybe_supplement(
            "What is the Zorblat index right now?",
            doc,
            cancelled_speak(),
            on_text,
        )
    await asyncio.sleep(0)
    try:
        assert tasks and (tasks[0].cancelled() or tasks[0].done())
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.xfail(
    strict=True,
    reason="R1-4: ws.wait_closed() does not cancel _maybe_supplement and stalls self.lock",
)
async def test_ws_disconnect_stops_maybe_supplement_promptly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    monkeypatch.setattr(agent_tools, "CHAIN", None)
    monkeypatch.setattr(agent_tools, "SCRATCHPAD", None)

    async def fc(params: Any) -> None:
        await params.result_callback(
            [{"source": "sheet_2026.csv", "text": "Zorblat index: 12.5", "part": "whole"}]
        )

    async def wa(params: Any) -> None:
        await asyncio.sleep(0.5)
        await params.result_callback("The current Zorblat index is 14.2.")

    async def fake_json(system: str, _content: str) -> str:
        if system == sufficiency.SYSTEM:
            await asyncio.sleep(0.8)
            return V_STALE
        return json.dumps(
            {
                "status": "answer",
                "answer": "12.5",
                "file": "sheet_2026.csv",
                "quotes": ["Zorblat index: 12.5"],
            }
        )

    tools = SimpleNamespace(
        standard_tools=[
            SimpleNamespace(name="file_context", handler=fc),
            SimpleNamespace(name="web_answer", handler=wa),
        ]
    )
    fake_agent = SimpleNamespace(
        tap=SimpleNamespace(usage={"requests": 0, "prompt": 0, "completion": 0, "cached": 0}),
        cancel=lambda: asyncio.sleep(0),
        hide=lambda _n: None,
    )
    server = AgentServer(
        fake_agent,
        [],
        tmp_path / "r1_4b.jsonl",
        tools=tools,
        doc_json_request=fake_json,
        docs_first=True,
    )

    first_text_sent = asyncio.Event()

    class _FakeWS:
        async def recv(self) -> str:
            return json.dumps(
                {"messages": [{"role": "user", "content": "What is the Zorblat index right now?"}]}
            )

        async def send(self, payload: str) -> None:
            if "text" in json.loads(payload):
                first_text_sent.set()

        async def wait_closed(self) -> None:
            await first_text_sent.wait()

    t0 = asyncio.get_running_loop().time()
    await server.handle(_FakeWS())  # type: ignore[arg-type]
    elapsed = asyncio.get_running_loop().time() - t0
    assert elapsed < 0.4


# --- R2-1: SentenceFilter spacing and single-sentence "The ..." hold ---


@pytest.mark.xfail(
    strict=True,
    reason="R2-1: SentenceFilter emits missing space '.The' and holds 'The web...' supplements",
)
async def test_sentence_filter_keeps_space_and_flushes_stale_supplement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    payloads = PayloadFilter()
    sources = SourceFilter()
    sentences = SentenceFilter(lambda: False)
    emitted: list[str] = []

    async def on_text(text: str) -> None:
        if out := sentences.feed(sources.feed(payloads.feed(text))):
            emitted.append(out)

    async def wa(params: Any) -> None:
        await params.result_callback("14.2.")

    async def fake_json(_system: str, _content: str) -> str:
        return V_STALE

    tools = SimpleNamespace(standard_tools=[SimpleNamespace(name="web_answer", handler=wa)])
    server = AgentServer(
        object(), [], tmp_path / "r2_1.jsonl", tools=tools, doc_json_request=fake_json
    )
    doc = DocResult(
        "accepted",
        spoken="As per sheet_2026.csv: 12.5.",
        answer="12.5.",
        file="sheet_2026.csv",
        sources=["sheet_2026.csv"],
        calls=1,
    )
    await server._maybe_supplement(
        "What is the Zorblat index right now?",
        doc,
        server.spoken(doc.spoken or "", on_text, docs="accepted", route="docs"),
        on_text,
    )
    joined = "".join(emitted) + sentences.finish()
    assert "12.5. The web" in joined

    # Also after a 2-sentence doc answer, passing=False holds "The web..." until finish():
    sentences2 = SentenceFilter(lambda: False)
    out1 = sentences2.feed("As per sheet_2026.csv: First sentence. Second sentence.")
    out2 = sentences2.feed("The web has a newer figure: 14.2.")
    assert out1
    assert "The web has a newer figure" in out2


# --- R2-2: temporal.source_year strips FY before source_earlier ---


@pytest.mark.xfail(
    strict=True,
    reason="R2-2: temporal.source_year strips 'FY' so source_earlier misses FY2026 in FY27",
)
def test_temporal_source_year_preserves_fiscal_year_for_source_earlier() -> None:
    file_name = "gsa/fy2026_per_diem_master_rates.xlsx"
    sy = temporal.source_year(None, file_name, file_name)
    assert sufficiency.source_earlier(sy, NOW) is True


# --- R2-3: format_citation prefix pushes answer length past SHORT_ANSWER_WORDS ---


@pytest.mark.xfail(
    strict=True,
    reason="R2-3: format_citation spoken text bypasses SHORT_ANSWER_WORDS=6 in should_check",
)
async def test_short_answer_gate_uses_answer_not_citation_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    verdict_calls: list[str] = []

    async def fake_json(_system: str, content: str) -> str:
        verdict_calls.append(content)
        return V_MISSING_HALF

    server = AgentServer(object(), [], tmp_path / "r2_3.jsonl", doc_json_request=fake_json)
    spoken = doc_route.format_citation(
        "Keira Moss runs it.", "Keira Moss runs it", {"file": "notes_2026.md"}
    )
    doc = DocResult(
        "accepted",
        spoken=spoken,
        answer="Keira Moss runs it.",
        file="notes_2026.md",
        sources=["notes_2026.md"],
        calls=1,
    )

    async def on_text(_text: str) -> None:
        pass

    await server._maybe_supplement(
        "Who runs the bureau, and where is its office?",
        doc,
        server.spoken(doc.spoken or "", on_text, docs="accepted", route="docs"),
        on_text,
    )
    assert len(verdict_calls) == 1


# --- R2-4: offline mode still calls web_answer ---


@pytest.mark.xfail(
    strict=True,
    reason="R2-4: _maybe_supplement does not check server.offline() before calling web_answer",
)
async def test_offline_server_skips_sufficiency_web_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    web_calls: list[str] = []

    async def wa(params: Any) -> None:
        web_calls.append(params.arguments.get("query", ""))
        await params.result_callback("14.2.")

    async def fake_json(_system: str, _content: str) -> str:
        return V_STALE

    tools = SimpleNamespace(standard_tools=[SimpleNamespace(name="web_answer", handler=wa)])
    server = AgentServer(
        object(), [], tmp_path / "r2_4.jsonl", tools=tools, doc_json_request=fake_json
    )
    server.probe = SimpleNamespace(online=False)  # type: ignore[assignment]
    doc = DocResult(
        "accepted",
        spoken="As per sheet_2026.csv: 12.5.",
        answer="12.5.",
        file="sheet_2026.csv",
        sources=["sheet_2026.csv"],
        calls=1,
    )

    async def on_text(_text: str) -> None:
        pass

    res = await server._maybe_supplement(
        "What is the Zorblat index right now?",
        doc,
        server.spoken(doc.spoken or "", on_text, docs="accepted", route="docs"),
        on_text,
    )
    assert web_calls == []
    assert res["sufficiency_web"] is False


# --- Non-regression invariants (currently passing) ---


async def test_compound_247_and_sufficiency_250_do_not_stack_on_docs_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    monkeypatch.setattr(compound, "COMPOUND", True)
    verdict_calls: list[str] = []

    async def fc(params: Any) -> None:
        await params.result_callback(
            [{"source": "a/sheet_2026.csv", "text": "Zorblat index: 12.5", "part": "whole"}]
        )

    async def wa(params: Any) -> None:
        await params.result_callback("The krona trades at 0.11 dollars.")

    async def fake_json(system: str, content: str) -> str:
        if system == sufficiency.SYSTEM:
            verdict_calls.append(content)
            return V_STALE
        if system == compound.SYSTEM:
            return json.dumps(
                {
                    "kind": "doc_and_web",
                    "doc_question": "What is the Zorblat index in my sheet?",
                    "web_question": "What is the krona exchange rate right now?",
                }
            )
        return json.dumps(
            {
                "status": "answer",
                "answer": "12.5",
                "file": "a/sheet_2026.csv",
                "quotes": ["Zorblat index: 12.5"],
            }
        )

    tools = SimpleNamespace(
        standard_tools=[
            SimpleNamespace(name="file_context", handler=fc),
            SimpleNamespace(name="web_answer", handler=wa),
        ]
    )
    server = AgentServer(
        SimpleNamespace(hide=lambda _n: None),
        [],
        tmp_path / "inv_247.jsonl",
        tools=tools,
        doc_json_request=fake_json,
        docs_first=True,
    )
    doc_route.set_cached_files(server, ["a/sheet_2026.csv"])

    async def on_text(_text: str) -> None:
        pass

    res = await server._docs_miss_turn(
        DocResult("none"),
        0.1,
        "What is the Zorblat index in my sheet, and what's the krona rate right now?",
        [{"role": "user", "content": "q"}],
        on_text,
        None,
        None,
    )
    assert res["compound"] is True
    assert verdict_calls == []
    assert "sufficiency" not in res


async def test_me3_fast_path_unaffected_when_sufficiency_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    verdict_calls: list[str] = []

    async def fake_json(_system: str, content: str) -> str:
        verdict_calls.append(content)
        return V_STALE

    server = AgentServer(
        object(), [], tmp_path / "me3_fast.jsonl", doc_json_request=fake_json, docs_first=True
    )
    frames: list[dict[str, Any]] = []

    class _FakeWS:
        async def recv(self) -> str:
            return json.dumps({"messages": [{"role": "user", "content": "What time is it?"}]})

        async def send(self, payload: str) -> None:
            frames.append(json.loads(payload))

        async def wait_closed(self) -> None:
            await asyncio.sleep(5.0)

    await server.handle(_FakeWS())  # type: ignore[arg-type]
    assert verdict_calls == []
    assert any("done" in f for f in frames)
