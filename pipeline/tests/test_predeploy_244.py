"""Pre-deploy 244: after a docs miss, no document tools; no double prefix; empty-question guard.

BUG A: when the turn's docs verdict came from ask_documents (a miss), the Gemini/agent turn that
follows must not be offered the document tools (search_docs, file_context, docs_catalog,
docs_enumerate, docs_status) - it could re-answer from an unchecked search_docs. Web tools stay.
BUG B: the miss prefix is not said twice - if the model already leads with it, do not add it
again, and a first sentence repeated verbatim right after itself is collapsed.
BUG C: ask_documents returns an honest "none" (no model call) for an ASR echo or an utterance
too empty to be a document question; a real question (a '?' or a leading question word) and
any digit/file/term cue pass through.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voice_pipeline.agent_server import (
    DOC_MISS_HIDDEN_TOOLS,
    DOCS_NO_COVER_PREFIX,
    _join_miss_prefix,
    _strip_repeated_first_sentence,
)
from voice_pipeline.ask_documents import DocResult, ask_documents
from voice_pipeline.bench_fakes import _FakeRouter
from voice_pipeline.benchmark_cases import RouterScript

_DOC_TOOLS = {"search_docs", "file_context", "docs_catalog", "docs_enumerate", "docs_status"}
_WEB_TOOLS = {"web_answer", "wiki_summary"}


class _RecordingAgent:
    """A fake agent that records every hide() set and answers with a fixed (or prefixed) text."""

    def __init__(self, reply_text: str) -> None:
        self.reply_text = reply_text
        self.reached = False
        self.hidden: list[frozenset[str]] = []
        self.tap = SimpleNamespace(usage={"requests": 0, "prompt": 0, "completion": 0, "cached": 0})

    async def ask(
        self, messages: Any, *, reset: bool, on_text: Any = None, **_: Any
    ) -> dict[str, Any]:
        self.reached = True
        if on_text:
            await on_text(self.reply_text)
        return {"status": "ok", "text": self.reply_text, "cancelled": False}

    def hide(self, names: frozenset[str]) -> None:
        self.hidden.append(frozenset(names))


class _PredeployFixtures:
    """AgentServer with a fake router and every docs tool plus web_answer. `reply_text` is what
    the fake agent says (and returns) when it is reached; `with_race` adds search_docs."""

    def __init__(
        self,
        reply_text: str,
        router_lines: str = "",
        router_reason: str = "normal",
        with_race: bool = False,
    ) -> None:
        self.reply_text = reply_text
        self.router_lines = router_lines
        self.router_reason = router_reason
        self.with_race = with_race
        self.doc_calls: list[str] = []
        self.web_calls: list[str] = []
        self.server: Any = None
        self.agent: _RecordingAgent | None = None
        self._url = ""
        self._ws_server: Any = None

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
                    [{"source": "test.md", "text": "The answer is in section 3."}]
                )

            fc = FunctionSchema(
                name="file_context",
                description="File context",
                properties={"query": {"type": "string"}},
                required=["query"],
                handler=_fc,
            )
            fc = agent_tools._recorded(fc, record)

            async def _ds(params: Any) -> None:
                await params.result_callback({"files": [], "failures": []})

            ds = FunctionSchema(
                name="docs_status",
                description="Docs status",
                properties={},
                required=[],
                handler=_ds,
            )
            ds = agent_tools._recorded(ds, record)

            async def _dc(params: Any) -> None:
                await params.result_callback([])

            dc = FunctionSchema(
                name="docs_catalog",
                description="Docs catalog",
                properties={"max_chars": {"type": "integer"}},
                required=[],
                handler=_dc,
            )
            dc = agent_tools._recorded(dc, record)

            async def _de(params: Any) -> None:
                await params.result_callback([])

            de = FunctionSchema(
                name="docs_enumerate",
                description="Docs enumerate",
                properties={"query": {"type": "string"}, "limit": {"type": "integer"}},
                required=["query"],
                handler=_de,
            )
            de = agent_tools._recorded(de, record)

            async def _wa(params: Any) -> None:
                self.web_calls.append(params.arguments.get("query", ""))
                await params.result_callback("As per example.org: The capital is Lima.")

            wa = FunctionSchema(
                name="web_answer",
                description="Web answer",
                properties={"query": {"type": "string"}},
                required=["query"],
                handler=_wa,
            )
            wa = agent_tools._recorded(wa, record)

            schemas = [fc, ds, dc, de, wa]
            doc_request: Any = None
            if self.with_race:

                async def _sd(params: Any) -> None:
                    await params.result_callback(
                        [{"source": "decoy.md", "text": "The decoy value is 42."}]
                    )

                sd = FunctionSchema(
                    name="search_docs",
                    description="Search docs",
                    properties={"query": {"type": "string"}},
                    required=["query"],
                    handler=_sd,
                )
                sd = agent_tools._recorded(sd, record)
                schemas.append(sd)

            tools = ToolsSchema(standard_tools=schemas)
            self.agent = _RecordingAgent(self.reply_text)
            router = (
                _FakeRouter(RouterScript(lines=self.router_lines))
                if self.router_lines
                else _FakeRouter(RouterScript(reason=self.router_reason))
            )

            async def _fake_json(system: str, content: str) -> str:
                return json.dumps({"status": "none"})

            log = Path("/tmp/test_predeploy_244.log")
            log.parent.mkdir(parents=True, exist_ok=True)

            self.server = AgentServer(
                self.agent,
                record,
                log,
                tools,
                router,
                doc_request=doc_request,
                docs_first=True,
                doc_json_request=_fake_json,
            )
            self.server.stamp = lambda: "[Now: test]"
            self.server.ledger = log
            self.record = record

            ws_server = await serve(self.server.handle, "127.0.0.1", 0)
            port = next(iter(ws_server.sockets)).getsockname()[1]
            self._ws_server = ws_server
            self._url = f"ws://127.0.0.1:{port}"
            return self._url
        finally:
            agent_tools.CHAIN = prev_chain
            agent_tools.SCRATCHPAD = prev_scratchpad

    async def send(self, question: str, session: str = "") -> tuple[dict[str, Any], str]:
        """Send one turn; return (the raw log row, the joined spoken text)."""
        from websockets.asyncio.client import connect

        async with connect(self._url) as ws:
            payload: dict[str, Any] = {"messages": [{"role": "user", "content": question}]}
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
        row = json.loads(self.server.log.read_text().splitlines()[-1])
        self.server.log.write_text("", encoding="utf-8")
        return row, " ".join(texts)

    async def close(self) -> None:
        self._ws_server.close()
        await self._ws_server.wait_closed()


# --- BUG A: a docs-miss turn gets no document tools, but keeps the web tools ---


@pytest.mark.asyncio
async def test_miss_turn_hides_doc_tools_keeps_web_tools() -> None:
    """A doc question that misses hides exactly the document tools for the Gemini turn; the
    web tools stay offered, and the base set (all tools) is restored when the turn is done."""
    fx = _PredeployFixtures("AGENT reply", router_lines="none|I don't know.")
    await fx.build()
    try:
        row, _ = await fx.send("Where does my base schedule live?")
        assert row["route"] == "docs_miss"
        assert fx.agent is not None and fx.agent.reached
        # the doc tools are hidden for the Gemini turn, then the base set (all tools) is restored
        assert DOC_MISS_HIDDEN_TOOLS in fx.agent.hidden
        assert fx.agent.hidden[-1] == frozenset()  # restored: no leak past the turn
        # the hidden set is exactly the document tools; the web tools are not in it
        assert DOC_MISS_HIDDEN_TOOLS == _DOC_TOOLS
        assert not (_WEB_TOOLS & DOC_MISS_HIDDEN_TOOLS)
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_miss_turn_hide_is_per_turn() -> None:
    """The docs-miss hide does not leak: each docs-miss turn hides the doc tools and then
    restores the base set, so the next turn starts from the full tool set again."""
    fx = _PredeployFixtures("AGENT reply", router_lines="none|I don't know.")
    await fx.build()
    try:
        await fx.send("Where does my base schedule live?")  # a docs miss
        await fx.send("What is the per diem for Denver?")  # another docs miss
        assert fx.agent is not None
        # turn 1: hide the doc tools, restore; turn 2: hide the doc tools, restore
        assert fx.agent.hidden == [
            DOC_MISS_HIDDEN_TOOLS,
            frozenset(),
            DOC_MISS_HIDDEN_TOOLS,
            frozenset(),
        ]
    finally:
        await fx.close()


# --- BUG B: the miss prefix is not said twice ---


def test_join_miss_prefix_does_not_double_when_model_repeats_it() -> None:
    """The model already leads with the miss: it is not added again."""
    model_text = DOCS_NO_COVER_PREFIX + "As per the file, the rate is 165 dollars."
    out = _join_miss_prefix(DOCS_NO_COVER_PREFIX, model_text)
    assert out == "Your documents don't cover that. As per the file, the rate is 165 dollars."
    assert out.count("Your documents don't cover that.") == 1


def test_join_miss_prefix_adds_it_once_when_model_does_not_repeat_it() -> None:
    """A model reply that does not repeat the miss gets the prefix exactly once."""
    out = _join_miss_prefix(DOCS_NO_COVER_PREFIX, "As per the file, the rate is 165 dollars.")
    assert out == "Your documents don't cover that. As per the file, the rate is 165 dollars."
    assert out.count("Your documents don't cover that.") == 1


def test_strip_repeated_first_sentence_collapses_a_verbatim_repeat() -> None:
    s = "Your documents don't cover that. Your documents don't cover that. The rate is 165."
    assert _strip_repeated_first_sentence(s) == "Your documents don't cover that. The rate is 165."


def test_strip_repeated_first_sentence_leaves_distinct_sentences_alone() -> None:
    s = "Your documents don't cover that. Checking the rate for Denver now."
    assert _strip_repeated_first_sentence(s) == s


@pytest.mark.asyncio
async def test_miss_turn_reply_has_single_prefix() -> None:
    """End to end: the model repeats the miss in its reply; the recorded reply says it once."""
    fx = _PredeployFixtures(
        DOCS_NO_COVER_PREFIX + "As per the gsa file, the rate is 165 dollars.",
        router_lines="none|I don't know.",
    )
    await fx.build()
    try:
        row, _ = await fx.send("What is the per diem for Denver?")
        assert row["route"] == "docs_miss"
        assert row["text"].count("Your documents don't cover that.") == 1
        assert row["text"].startswith("Your documents don't cover that. ")
        assert "the rate is 165 dollars" in row["text"]
    finally:
        await fx.close()


# --- BUG C: the empty-question / ASR-echo guard in ask_documents ---


def _passage() -> dict[str, Any]:
    return {
        "source": "test.md",
        "id": "test.md#0",
        "section": "",
        "page": None,
        "file_date": "2026-01-01",
        "text": "The per diem rate for Denver is 165 dollars.",
        "score": 1.0,
        "coverage": 1.0,
        "part": "chunk",
    }


class _ExplodingRequest:
    """A model call that fails the test if it runs - the guard must stop before any call."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, system: str, content: str) -> str:
        self.calls += 1
        raise AssertionError("the model was called for an empty/echo question")


@pytest.mark.asyncio
async def test_echo_question_is_a_honest_miss_with_no_model_call() -> None:
    """'Community Community Community Community': no retrieval, no model call, status none."""
    requests = _ExplodingRequest()
    context_calls: list[str] = []

    async def context(q: str) -> list[dict[str, Any]]:
        context_calls.append(q)
        return [_passage()]

    result = await ask_documents(
        "Community Community Community Community",
        context=context,
        request=requests,
    )
    assert isinstance(result, DocResult)
    assert result.status == "none"
    assert result.sources == []
    assert requests.calls == 0
    assert context_calls == []  # the guard returns before any retrieval


@pytest.mark.asyncio
async def test_single_word_echo_is_a_honest_miss() -> None:
    """A repeated word with fewer than 2 content words and no question mark is a miss."""
    requests = _ExplodingRequest()

    async def context(q: str) -> list[dict[str, Any]]:
        return [_passage()]

    result = await ask_documents("Community Community", context=context, request=requests)
    assert result.status == "none"
    assert requests.calls == 0


@pytest.mark.asyncio
async def test_one_word_content_question_still_goes_through() -> None:
    """'Texas?' is a real (if short) question: the guard must not drop it, so the model is
    actually called."""
    calls = {"n": 0}

    async def request(system: str, content: str) -> str:
        calls["n"] += 1
        return json.dumps({"status": "none"})

    async def context(q: str) -> list[dict[str, Any]]:
        return [_passage()]

    result = await ask_documents("Texas?", context=context, request=request)
    assert calls["n"] == 1  # the model was reached: the guard let the question through
    assert result.status == "none"  # an honest miss from the model, not from the guard


@pytest.mark.asyncio
async def test_digit_question_still_goes_through() -> None:
    """A digit keeps the question even when it is short (a real reference, not an echo)."""
    calls = {"n": 0}

    async def request(system: str, content: str) -> str:
        calls["n"] += 1
        return json.dumps({"status": "none"})

    async def context(q: str) -> list[dict[str, Any]]:
        return [_passage()]

    result = await ask_documents("GS 15 step 11", context=context, request=request)
    assert calls["n"] == 1
    assert result.status == "none"
