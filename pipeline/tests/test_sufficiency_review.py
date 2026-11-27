"""Adversarial review tests for the sufficiency check (tickets 256/258).

The R1-1..R1-4 and R2-4 findings of the 2026-11-27 review were fixed by ticket 258; those
tests were ``xfail(strict=True)`` bug documents and are now plain regression tests. The R2-1,
R2-2 and R2-3 tests stay ``xfail(strict=True)`` (ticket 259).

Every fake JSON transport ENFORCES the schema it is given (a reply carrying a key the schema
lacks raises, like the production structured output): a schema-ignoring fake is what masked
the hardcoded REPLY_SCHEMA behind the verdict call (R1-1).
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


def _enforce_schema(reply: str, schema: dict[str, Any] | None) -> str:
    """Like the production transport's structured output: a reply that carries a key the
    schema it was given lacks cannot happen, so the scripted reply is checked the same way
    (R1-1: a schema-ignoring fake is what masked the hardcoded REPLY_SCHEMA)."""
    if schema is None:
        return reply
    try:
        data = json.loads(reply)
    except ValueError:
        return reply  # unparsable: not a schema violation; the caller's fail-open case
    if isinstance(data, dict) and set(data) - set(schema["properties"]):
        raise AssertionError("the scripted reply carries keys outside the schema it was given")
    return reply


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


# --- R1-1: the verdict (and the split) pass their own schema to the transport ---


def test_the_verdict_and_split_schemas_carry_their_keys() -> None:
    props = sufficiency.SUFFICIENCY_SCHEMA["properties"]
    assert set(props) == {"verdict", "web_question", "reason"}
    assert set(props["verdict"]["enum"]) == sufficiency.VERDICTS
    assert set(sufficiency.SUFFICIENCY_SCHEMA["required"]) == {
        "verdict",
        "web_question",
        "reason",
    }
    props = compound.SPLIT_SCHEMA["properties"]
    assert set(props) == {"kind", "doc_question", "web_question"}
    assert set(props["kind"]["enum"]) == {
        "single",
        "doc_then_web",
        "doc_and_web",
        "doc_vs_web",
    }
    # the document lane's own schema is unchanged: every existing caller is unchanged
    assert set(ask_documents.REPLY_SCHEMA["properties"]) == {
        "status",
        "answer",
        "file",
        "quotes",
        "calc",
        "web_question",
    }
    assert "verdict" not in ask_documents.REPLY_SCHEMA["properties"]


async def test_verdict_passes_its_own_schema_and_parses_the_reply() -> None:
    """R1-1 regression: a transport that enforces the schema it is given (the production
    structured output) can emit the verdict only because verdict() asks for
    SUFFICIENCY_SCHEMA — the reply is checked against the schema the call passed, not
    against a hardcoded REPLY_SCHEMA."""
    seen: list[dict[str, Any] | None] = []
    stale = {
        "verdict": "stale",
        "web_question": "What is the current Zorblat index?",
        "reason": "the source is old",
    }

    async def enforcing(
        system: str, content: str, schema: dict[str, Any] | None = None
    ) -> str:
        seen.append(schema)
        return _enforce_schema(json.dumps(stale), schema)

    got = await sufficiency.verdict(
        enforcing,
        "What is the index right now?",
        "As per sheet.csv: 12.5.",
        "2023",
        "2026-11-27",
    )
    assert got is not None
    assert got.verdict == "stale"
    assert got.web_question == "What is the current Zorblat index?"
    assert seen and seen[0] is sufficiency.SUFFICIENCY_SCHEMA


async def test_a_reply_confined_to_reply_schema_still_fails_open() -> None:
    """If the transport were confined to the document lane's REPLY_SCHEMA it could only emit
    status/answer/...: that reply has no verdict key and verdict() drops it as None (fail-
    open, nothing extra said) — the fix is the schema the call passes, not a looser parser."""
    reply = {"status": "answer", "answer": "stale", "web_question": "q?"}

    async def reply_schema_only(
        system: str, content: str, schema: dict[str, Any] | None = None
    ) -> str:
        # a transport confined to REPLY_SCHEMA could only emit those keys
        assert set(reply) <= set(ask_documents.REPLY_SCHEMA["properties"])
        return json.dumps(reply)

    got = await sufficiency.verdict(
        reply_schema_only,
        "What is the index right now?",
        "As per sheet.csv: 12.5.",
        "2023",
        "2026-11-27",
    )
    assert got is None


async def test_split_passes_its_own_schema_and_parses_the_reply() -> None:
    """R1-1 regression (compound): the split call asks for SPLIT_SCHEMA, so a schema-
    enforcing transport can emit kind/doc_question/web_question and split() keeps them
    (under a REPLY_SCHEMA transport the reply could never name a kind, and the split was
    always dropped)."""
    seen: list[dict[str, Any] | None] = []
    cut = {
        "kind": "doc_and_web",
        "doc_question": "Who runs the Zorblat bureau, per my file?",
        "web_question": "How deep is the Quenth river now?",
    }

    async def enforcing(
        system: str, content: str, schema: dict[str, Any] | None = None
    ) -> str:
        seen.append(schema)
        return _enforce_schema(json.dumps(cut), schema)

    got = await compound.split(enforcing, "Who runs the bureau, and how deep is the river?")
    assert got is not None
    assert got.kind == "doc_and_web"
    assert got.doc_question == "Who runs the Zorblat bureau, per my file?"
    assert seen and seen[0] is compound.SPLIT_SCHEMA


# --- R1-2: ask_documents(status="web") double lookup + REPEAT_PREFIX spoken aloud ---


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

    async def fake_json(
        system: str, content: str, schema: dict[str, Any] | None = None
    ) -> str:
        verdict_calls.append(content)
        return _enforce_schema(V_MISSING_HALF, schema)

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
    assert verdict_calls == []  # nothing is paid for: the answer already carries the web part
    assert len(web_calls) == 1
    assert not any(agent_tools.REPEAT_PREFIX in s for s in spoken_events)
    assert len(spoken_events) == 1


# --- R1-3: wrong_question pays for web_answer and poisons turn_source_sentence / turn_claim ---


async def test_wrong_question_does_not_call_web_or_leak_source_and_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    monkeypatch.setattr(agent_tools, "CHAIN", None)
    monkeypatch.setattr(agent_tools, "SCRATCHPAD", None)
    agent_tools.reset_turn_lookups()

    record: list[dict[str, Any]] = []
    web_calls: list[str] = []

    async def fake_json(
        system: str, content: str, schema: dict[str, Any] | None = None
    ) -> str:
        return _enforce_schema(V_WRONG, schema)

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
    assert web_calls == []  # a reply that cannot be spoken is never looked up
    assert res["sufficiency"] == "wrong_question"
    assert res["sufficiency_web"] is False
    assert agent_tools.turn_source_sentence(res["text"]) == ""  # no stray "Source: ..."
    assert agent_tools.turn_claim() == ""  # no web_verify against an unspoken answer


# --- R1-4: cancelled speak leaks verdict task; WS disconnect holds self.lock ---


async def test_cancelled_speak_cancels_pending_verdict_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    started = asyncio.Event()
    tasks: list[asyncio.Task[Any]] = []

    async def slow_json(
        system: str, content: str, schema: dict[str, Any] | None = None
    ) -> str:
        t = asyncio.current_task()
        if t is not None:
            tasks.append(t)
        started.set()
        await asyncio.sleep(5.0)
        return _enforce_schema(V_STALE, schema)

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

    async def fake_json(
        system: str, content: str, schema: dict[str, Any] | None = None
    ) -> str:
        if system == sufficiency.SYSTEM:
            await asyncio.sleep(0.8)
            return _enforce_schema(V_STALE, schema)
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


# --- R2-1: SentenceFilter spacing and single-sentence "The ... " hold (ticket 259) ---


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

    async def fake_json(
        system: str, content: str, schema: dict[str, Any] | None = None
    ) -> str:
        return _enforce_schema(V_STALE, schema)

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


# --- R2-2: temporal.source_year strips FY before source_earlier (ticket 259) ---


@pytest.mark.xfail(
    strict=True,
    reason="R2-2: temporal.source_year strips 'FY' so source_earlier misses FY2026 in FY27",
)
def test_temporal_source_year_preserves_fiscal_year_for_source_earlier() -> None:
    file_name = "gsa/fy2026_per_diem_master_rates.xlsx"
    sy = temporal.source_year(None, file_name, file_name)
    assert sufficiency.source_earlier(sy, NOW) is True


# --- R2-3: format_citation prefix pushes answer length past SHORT_ANSWER_WORDS (ticket 259) ---


@pytest.mark.xfail(
    strict=True,
    reason="R2-3: format_citation spoken text bypasses SHORT_ANSWER_WORDS=6 in should_check",
)
async def test_short_answer_gate_uses_answer_not_citation_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    verdict_calls: list[str] = []

    async def fake_json(
        system: str, content: str, schema: dict[str, Any] | None = None
    ) -> str:
        verdict_calls.append(content)
        return _enforce_schema(V_MISSING_HALF, schema)

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


async def test_offline_server_skips_sufficiency_web_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    web_calls: list[str] = []

    async def wa(params: Any) -> None:
        web_calls.append(params.arguments.get("query", ""))
        await params.result_callback("14.2.")

    async def fake_json(
        system: str, content: str, schema: dict[str, Any] | None = None
    ) -> str:
        return _enforce_schema(V_STALE, schema)

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

    async def fake_json(
        system: str, content: str, schema: dict[str, Any] | None = None
    ) -> str:
        if system == sufficiency.SYSTEM:
            verdict_calls.append(content)
            return _enforce_schema(V_STALE, schema)
        if system == compound.SYSTEM:
            return _enforce_schema(
                json.dumps(
                    {
                        "kind": "doc_and_web",
                        "doc_question": "What is the Zorblat index in my sheet?",
                        "web_question": "What is the krona exchange rate right now?",
                    }
                ),
                schema,
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

    async def fake_json(
        system: str, content: str, schema: dict[str, Any] | None = None
    ) -> str:
        verdict_calls.append(content)
        return _enforce_schema(V_STALE, schema)

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
