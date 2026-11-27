"""Docs-first routing: ticket 210."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voice_pipeline.ask_documents import DocResult
from voice_pipeline.bench_fakes import (
    _FakeRouter,
)
from voice_pipeline.benchmark_cases import RouterScript
from voice_pipeline.doc_route import (
    document_cue,
    new_failures_prefix,
    refers_to_own_sources,
    set_cached_files,
)


class _FakeAgent:
    def __init__(self) -> None:
        self.reached = False
        self.tap = SimpleNamespace(usage={"requests": 0, "prompt": 0, "completion": 0, "cached": 0})

    async def ask(
        self, messages: Any, *, reset: bool, on_text: Any = None, **_: Any
    ) -> dict[str, Any]:
        self.reached = True
        text = "AGENT reply"
        if on_text:
            await on_text(text)
        return {"status": "ok", "text": text, "cancelled": False}

    def hide(self, names: frozenset[str]) -> None:
        pass


class _DocsFirstFixtures:
    """Build a minimal AgentServer with docs-first enabled."""

    def __init__(
        self,
        router_lines: str = "",
        doc_result: DocResult | None = None,
        router: bool = True,
    ) -> None:
        self.router_lines = router_lines
        self.doc_result = doc_result
        self.router = router
        self.doc_calls: list[str] = []
        self.server: Any = None
        self.agent: _FakeAgent | None = None
        self._url: str = ""

    async def build(self) -> str:
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.adapters.schemas.tools_schema import ToolsSchema
        from websockets.asyncio.server import serve

        from voice_pipeline import agent_tools
        from voice_pipeline.agent_server import AgentServer

        record: list[dict[str, Any]] = []
        prev_chain = agent_tools.CHAIN
        prev_scratchpad = agent_tools.SCRATCHPAD
        agent_tools.CHAIN = None
        agent_tools.SCRATCHPAD = None

        try:

            async def _fc(params: Any) -> None:
                self.doc_calls.append(params.arguments.get("query", ""))
                await params.result_callback(
                    [
                        {
                            "source": "test.md",
                            "text": "The midterm is found in section 3 of the document.",
                        }
                    ]
                )

            fc = FunctionSchema(
                name="file_context",
                description="File context",
                properties={"query": {"type": "string"}},
                required=["query"],
                handler=_fc,
            )
            agent_tools._recorded(fc, [])

            async def _ds(params: Any) -> None:
                await params.result_callback({"files": [], "failures": []})

            ds = FunctionSchema(
                name="docs_status",
                description="Docs status",
                properties={},
                required=[],
                handler=_ds,
            )
            agent_tools._recorded(ds, [])

            async def _sh(params: Any) -> None:
                await params.result_callback({"ok": True})

            sh = FunctionSchema(
                name="smart_home",
                description="Smart home",
                properties={"intent": {"type": "string"}, "slot": {"type": "string"}},
                required=["intent"],
                handler=_sh,
            )
            agent_tools._recorded(sh, [])

            async def _wa(params: Any) -> None:
                await params.result_callback("As per example.org: The capital is Lima.")

            wa = FunctionSchema(
                name="web_answer",
                description="Web answer",
                properties={"query": {"type": "string"}},
                required=["query"],
                handler=_wa,
            )
            agent_tools._recorded(wa, [])

            tools = ToolsSchema(standard_tools=[fc, ds, sh, wa])

            self.agent = _FakeAgent()
            router: Any = (
                _FakeRouter(
                    RouterScript(lines=self.router_lines)
                    if self.router_lines
                    else RouterScript(reason="normal")
                )
                if self.router
                else None
            )

            async def _fake_json(system: str, content: str) -> str:
                if self.doc_result is not None:
                    return self._doc_result_to_json(self.doc_result)
                return json.dumps({"status": "none"})

            log = Path("/tmp/test_df.log")
            log.parent.mkdir(parents=True, exist_ok=True)

            self.server = AgentServer(
                self.agent,
                record,
                log,
                tools,
                router,
                doc_request=None,
                docs_first=True,
                doc_json_request=_fake_json,
            )
            self.server.stamp = lambda: "[Now: test]"
            self.server.ledger = log

            ws_server = await serve(self.server.handle, "127.0.0.1", 0)
            port = next(iter(ws_server.sockets)).getsockname()[1]
            self._ws_server = ws_server
            self._port = port
            self._url = f"ws://127.0.0.1:{port}"
            return self._url
        finally:
            agent_tools.CHAIN = prev_chain
            agent_tools.SCRATCHPAD = prev_scratchpad

    def _doc_result_to_json(self, result: DocResult) -> str:
        if result.status == "accepted":
            return json.dumps(
                {
                    "status": "answer",
                    "answer": "The midterm is in section 3",
                    "file": "test.md",
                    "quotes": ["midterm is in section 3"],
                }
            )
        return json.dumps({"status": "none"})

    async def send(self, question: str, session: str = "") -> dict[str, Any]:
        from websockets.asyncio.client import connect

        messages = getattr(self, "_msgs", []) + [{"role": "user", "content": question}]
        async with connect(self._url) as ws:
            payload = {"messages": messages}
            if session:
                payload["session"] = session
            await ws.send(json.dumps(payload))
            texts: list[str] = []
            while True:
                frame = json.loads(await ws.recv())
                if "text" in frame:
                    texts.append(frame["text"])
                if "done" in frame or "error" in frame:
                    break
        self._msgs = messages + [{"role": "assistant", "content": " ".join(texts)}]
        log_path = self.server.log
        row = json.loads(log_path.read_text().splitlines()[-1])
        row["text"] = " ".join(texts)
        row["reached_agent"] = getattr(self.agent, "reached", getattr(self.agent, "spoke", False))
        log_path.write_text("", encoding="utf-8")
        return row

    async def close(self) -> None:
        self._ws_server.close()
        await self._ws_server.wait_closed()


@pytest.fixture
async def df_fixture() -> AsyncIterator[_DocsFirstFixtures]:
    """Docs-first test fixture."""
    fx = _DocsFirstFixtures()
    await fx.build()
    yield fx
    await fx.close()


async def _df_with_router(lines: str, result: DocResult | None) -> _DocsFirstFixtures:
    """Helper: build a fixture with given router lines and doc result."""
    fx = _DocsFirstFixtures(router_lines=lines, doc_result=result)
    await fx.build()
    return fx


# --- Test 1: router action with no doc cue -> run_routed, no docs call ---


@pytest.mark.asyncio
async def test_action_no_doc_cue() -> None:
    """Router action with no document cue runs routed; docs not called."""
    fx = await _df_with_router("TIMER|45 minutes", None)
    try:
        row = await fx.send("Set a 45 minute timer")
        assert row["route"] == "routed"
        assert len(fx.doc_calls) == 0
    finally:
        await fx.close()


# --- Test 2: action-worded doc question -> document_cue -> docs ---


@pytest.mark.asyncio
async def test_action_doc_cue() -> None:
    """Action-phrased document question triggers docs, not home command."""
    fx = await _df_with_router("TIMER|syllabus", None)
    set_cached_files(fx.server, ["course_syllabus.md"])
    assert document_cue(
        "Turn to the part of the syllabus that lists the course schedule",
        ["course_syllabus.md"],
    )
    try:
        await fx.send("Turn to the part of the syllabus that lists the course schedule")
        assert len(fx.doc_calls) > 0
    finally:
        await fx.close()


# --- Test 3: hedged fake: docs wins, model does not speak ---


@pytest.mark.asyncio
async def test_docs_wins_over_hedged_model() -> None:
    """ask_documents accepted at 1500ms beats model at 300ms."""
    from voice_pipeline.ask_documents import DocResult
    from voice_pipeline.doc_route import format_citation

    result = DocResult(
        "accepted",
        spoken=format_citation(
            "Section 3 lists the deadlines",
            "Section 3 lists the deadlines",
            {"file": "test.md"},
            ["test.md"],
        ),
        answer="Section 3 lists the deadlines",
        file="test.md",
        quotes=["Section 3 lists the deadlines"],
        sources=["test.md"],
        calls=1,
    )

    fx = await _df_with_router("", result)
    row = await fx.send("When is the midterm?")
    await fx.close()
    assert "As per" in row.get("text", "")
    assert row.get("reached_agent") is False
    assert len(fx.doc_calls) == 1


# --- Test 4: NONE -> Gemini with note / web_answer -> lookup runs ---


@pytest.mark.asyncio
async def test_none_falls_to_gemini() -> None:
    """NONE from docs falls through to Gemini with 'documents were checked' note."""
    none_result = DocResult("none", sources=["syllabus.md"], calls=1)
    fx = await _df_with_router("none|hello", none_result)
    try:
        row = await fx.send("What is the meaning of life?")
        assert fx.agent is not None
        assert fx.agent.reached is True
        assert len(fx.doc_calls) == 1
        assert row["route"] == "docs_miss"
        assert row.get("docs") == "none"
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_none_web_answer() -> None:
    """NONE with web_answer router call runs the lookup."""
    none_result = DocResult("none", sources=[], calls=1)
    fx = await _df_with_router("web_answer|capital of Peru", none_result)
    try:
        row = await fx.send("What is the capital of Peru?")
        assert row["route"] == "routed"
    finally:
        await fx.close()


# --- Test 5: per-turn cache ---


@pytest.mark.asyncio
async def test_per_turn_cache() -> None:
    """Second docs call in same turn reuses cached result."""
    call_count = 0

    async def counting_json(system: str, content: str) -> str:
        nonlocal call_count
        call_count += 1
        return json.dumps({"status": "none"})

    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema
    from websockets.asyncio.server import serve

    from voice_pipeline import agent_tools
    from voice_pipeline.agent_server import AgentServer

    prev_chain = agent_tools.CHAIN
    agent_tools.CHAIN = None
    try:

        async def _fc(params: Any) -> None:
            await params.result_callback(
                [
                    {
                        "source": "test.md",
                        "text": "Some content for testing.",
                    }
                ]
            )

        fc = FunctionSchema(
            name="file_context",
            description="File context",
            properties={"query": {"type": "string"}},
            required=["query"],
            handler=_fc,
        )

        async def _ds(params: Any) -> None:
            await params.result_callback({"files": [], "failures": []})

        ds = FunctionSchema(
            name="docs_status",
            description="Docs status",
            properties={},
            required=[],
            handler=_ds,
        )

        tools = ToolsSchema(standard_tools=[fc, ds])
        agent = _FakeAgent()
        router = _FakeRouter(RouterScript(reason="normal"))
        log = Path("/tmp/test_df_cache.log")
        log.parent.mkdir(parents=True, exist_ok=True)

        server = AgentServer(
            agent,
            [],
            log,
            tools,
            router,
            doc_request=None,
            docs_first=True,
            doc_json_request=counting_json,
        )
        server.stamp = lambda: "[Now: test]"
        server.ledger = log

        ws_server = await serve(server.handle, "127.0.0.1", 0)
        port = next(iter(ws_server.sockets)).getsockname()[1]

        from websockets.asyncio.client import connect

        async with connect(f"ws://127.0.0.1:{port}") as ws:
            # a question-shaped utterance: non-questions ("Thanks!", "Test question") skip the
            # documents in the router branch (ticket 235)
            await ws.send(
                json.dumps({"messages": [{"role": "user", "content": "What does the test say?"}]})
            )
            while True:
                frame = json.loads(await ws.recv())
                if "done" in frame or "error" in frame:
                    break

        ws_server.close()
        await ws_server.wait_closed()
        assert call_count == 1
    finally:
        agent_tools.CHAIN = prev_chain


# --- Test 6: sources per session ---


@pytest.mark.asyncio
async def test_sources_per_session() -> None:
    """Session A asks doc question; session B gets 'haven't looked yet'."""
    from voice_pipeline.ask_documents import DocResult
    from voice_pipeline.doc_route import format_citation

    result = DocResult(
        "accepted",
        spoken=format_citation(
            "Answer found",
            "Answer found",
            {"file": "a.md"},
            ["a.md"],
        ),
        answer="Answer found",
        file="a.md",
        quotes=["Answer found"],
        sources=["a.md", "b.xlsx"],
        calls=1,
    )

    fx = await _df_with_router("none|hello", result)
    try:
        await fx.send("What is the budget?", session="sess-A")
        # _sources is set from result.sources which comes from passage sources
        assert fx.server._sources.get("sess-A") == ["test.md"]

        row_b = await fx.send("Which files did you check?", session="sess-B")
        assert "haven't looked" in row_b.get("text", "").lower()

        row_a2 = await fx.send("Your sources?", session="sess-A")
        assert row_a2["route"] == "sources"
        # the sources lane speaks the title, never the filename (ticket 242)
        assert row_a2.get("text", "").strip() == "I checked your document on test."
    finally:
        await fx.close()


# --- Test 7: refers_to_own_sources ---


def test_refers_to_own_sources_negatives() -> None:
    """Negative cases for refers_to_own_sources."""
    assert not refers_to_own_sources("Which sources of funding does the grant document list?")
    assert not refers_to_own_sources("What files are required for graduation?")
    assert not refers_to_own_sources("Where did the name Madrid come from?")
    assert not refers_to_own_sources("What did the report use to calculate that?")


def test_refers_to_own_sources_positives() -> None:
    """Positive cases for refers_to_own_sources."""
    assert refers_to_own_sources("Which files did you check?")
    assert refers_to_own_sources("Where did you get that?")
    assert refers_to_own_sources("What did you use?")
    assert refers_to_own_sources("your sources?")
    assert refers_to_own_sources("Which documents did you look at?")
    assert refers_to_own_sources("Where did the answer come from?")


# --- Test 8: new_failures_prefix cap ---


def test_new_failures_prefix_cap() -> None:
    """5 failed files -> 2 sentences + 'And 3 more files I couldn't read.'."""
    status = {
        "files": [{"name": f"f{i}.md"} for i in range(5)],
        "failures": [
            {
                "name": f"f{i}.md",
                "status": "unreadable",
                "reason": f"reason {i}",
                "mtime": None,
                "size": None,
            }
            for i in range(5)
        ],
    }
    prefix, names = new_failures_prefix(status, {})
    assert len(names) == 5
    sentences = prefix.split(". ")
    assert len(sentences) == 3
    assert "And 3 more files I couldn't read" in prefix


def test_new_failures_prefix_counts_unsupported_files() -> None:
    """Unsupported-extension files are one spoken count, never named (ticket 252)."""
    status = {
        "files": [{"name": "a.json"}, {"name": "b.odt"}, {"name": "c.md"}],
        "failures": [
            {
                "name": "a.json",
                "status": "unsupported",
                "reason": "unsupported file extension: .json",
                "mtime": None,
                "size": None,
            },
            {
                "name": "b.odt",
                "status": "unsupported",
                "reason": "unsupported file extension: .odt",
                "mtime": None,
                "size": None,
            },
        ],
    }
    prefix, names = new_failures_prefix(status, {})
    assert set(names) == {"a.json", "b.odt", "c.md"}
    assert prefix == "2 files are in a format I can't read."
    assert "a.json" not in prefix and "b.odt" not in prefix

    # A third unsupported file changes the count, still one sentence, still no names.
    status["failures"].append(
        {
            "name": "d.xml",
            "status": "unsupported",
            "reason": "unsupported file extension: .xml",
            "mtime": None,
            "size": None,
        }
    )
    prefix, names = new_failures_prefix(status, {})
    assert prefix == "3 files are in a format I can't read."
    assert "d.xml" not in prefix


# --- Test 9: flag off -> existing paths unchanged ---


@pytest.mark.asyncio
async def test_docs_first_off() -> None:
    """When docs_first=False, existing paths are unchanged."""
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    from voice_pipeline import agent_tools
    from voice_pipeline.agent_server import AgentServer

    prev_chain = agent_tools.CHAIN
    agent_tools.CHAIN = None
    try:

        async def _sd(params: Any) -> None:
            await params.result_callback([])

        sd = FunctionSchema(
            name="search_docs",
            description="Search docs",
            properties={"query": {"type": "string"}},
            required=["query"],
            handler=_sd,
        )

        async def _ds(params: Any) -> None:
            await params.result_callback({"files": [], "failures": []})

        ds = FunctionSchema(
            name="docs_status",
            description="Docs status",
            properties={},
            required=[],
            handler=_ds,
        )

        tools = ToolsSchema(standard_tools=[sd, ds])
        agent = _FakeAgent()

        async def fake_doc(system: str, content: str) -> str:
            return "ANSWER: test\nQUOTE: test"

        log = Path("/tmp/test_df_off.log")
        log.parent.mkdir(parents=True, exist_ok=True)

        server = AgentServer(
            agent,
            [],
            log,
            tools,
            None,
            doc_request=fake_doc,
            docs_first=False,
        )
        server.stamp = lambda: "[Now: test]"
        server.ledger = log
        assert not server.docs_first
        assert server.doc_json_request is None
    finally:
        agent_tools.CHAIN = prev_chain


@pytest.mark.asyncio
async def test_docs_miss_fallthrough_is_logged() -> None:
    """No-router fall-through logs docs_miss, rejection brief, and unconfirmed prefix."""
    rejected = {"answer": "X", "quote": "Q" * 50, "failed": "quote"}
    res = DocResult("check_failed", rejected=rejected, sources=["syllabus.md"], calls=1)
    fx = _DocsFirstFixtures(doc_result=res, router=False)
    await fx.build()
    try:

        async def _fake_ask_docs(question: str, session: str) -> DocResult:
            return res

        fx.server._ask_docs = _fake_ask_docs
        row = await fx.send("Is the capital of France in my notes?")
        assert row["route"] == "docs_miss"
        assert row["docs"] == "check_failed"
        assert isinstance(row["doc_s"], (int, float))
        assert row["doc_calls"] == 1
        assert row["sources"] == ["syllabus.md"]
        assert row["rejected"]["failed"] == "quote"
        assert len(row["rejected"]["excerpt"]) <= 120
        assert row["text"].startswith("I couldn't confirm that in your documents. ")
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_docs_miss_open_none_has_no_prefix() -> None:
    """No-router open (non-document) question with doc_result none: no miss prefix (243)."""
    res = DocResult("none", sources=["syllabus.md"], calls=1)
    fx = _DocsFirstFixtures(doc_result=res, router=False)
    await fx.build()
    try:
        row = await fx.send("What is the capital of France?")
        assert row["route"] == "docs_miss"
        assert row["docs"] == "none"
        assert not row["text"].startswith("Your documents don't cover that.")
        assert row["text"].startswith("AGENT reply")
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_weather_question_skips_docs() -> None:
    """Weather question skips docs on no-router Pi path and goes to Gemini."""
    fx = _DocsFirstFixtures(doc_result=DocResult("none"), router=False)
    await fx.build()
    try:
        row = await fx.send("What's the weather today?")
        assert len(fx.doc_calls) == 0
        assert row["route"] == "gemini"
        assert fx.agent is not None
        assert fx.agent.reached is True
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_docs_miss_doc_scoped_refuses() -> None:
    """Doc-scoped question with doc_result none refuses without calling Gemini."""
    from voice_pipeline import doc_route

    fx = _DocsFirstFixtures(doc_result=DocResult("none"), router=False)
    await fx.build()
    try:
        doc_route.set_cached_files(fx.server, ["pet_policy.md"])
        row = await fx.send("What does the document say about the pet policy?")
        assert fx.agent is not None
        assert fx.agent.reached is False
        assert row["text"].rstrip() == "The documents don't say."
        assert row["route"] == "docs_miss"
        assert row["docs"] == "none"
    finally:
        await fx.close()


# --- Ticket 230b: file_has is wired through the real _ask_docs (fake call_tool) ---


@pytest.mark.asyncio
async def test_ask_docs_file_has_wiring() -> None:
    """The retrieved passages never spell 'Guam'; the answer is accepted only because the
    file_has call (docs_enumerate) reports the token present in the cited file."""
    fx = _DocsFirstFixtures()
    await fx.build()
    try:
        calls: list[tuple[str, dict[str, Any]]] = []

        async def fake_call_tool(name: str, args: dict[str, Any]) -> Any:
            calls.append((name, dict(args)))
            if name == "file_context":
                return [
                    {
                        "source": "test.md",
                        "text": "State: Hawaii; Average price: 0.342",
                        "part": "whole",
                    }
                ]
            if name == "docs_enumerate":
                return [{"rel_key": "test.md"}]
            return []

        async def fake_json(system: str, content: str) -> str:
            return json.dumps(
                {
                    "status": "answer",
                    "answer": "0.342",
                    "file": "test.md",
                    "quotes": ["State: Hawaii; Average price: 0.342"],
                }
            )

        fx.server.call_tool = fake_call_tool
        fx.server.doc_json_request = fake_json
        result = await fx.server._ask_docs(
            "What was the average home electricity price in Guam?", ""
        )
        assert result.status == "accepted", result.rejected
        assert ("docs_enumerate", {"query": "guam", "limit": 100}) in calls
    finally:
        await fx.close()
