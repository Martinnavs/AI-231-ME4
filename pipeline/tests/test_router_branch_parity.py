"""Router-branch docs parity: the document half of the ME3 router branch gets the no-router
branch's GENERAL lane and no-answer policy (ticket 233). The router keeps commands and web."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voice_pipeline.bench_fakes import _FakeRouter
from voice_pipeline.benchmark_cases import RouterScript


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


_CATALOG = [
    {"folder": "a", "rel_key": "a/guide.md", "line": "a/guide.md | Guide | markdown"},
    {"folder": "b", "rel_key": "b/notes.txt", "line": "b/notes.txt | Notes | text"},
]


class _RouterParityFixtures:
    """AgentServer with a fake router and every docs tool (file_context, docs_catalog, ...)."""

    def __init__(self, reply: str, router_lines: str = "", router_reason: str = "normal") -> None:
        self.reply = reply
        self.router_lines = router_lines
        self.router_reason = router_reason
        self.doc_calls: list[str] = []
        self.web_calls: list[str] = []
        self.server: Any = None
        self.agent: _FakeAgent | None = None
        self._url = ""

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
                await params.result_callback(list(_CATALOG))

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

            async def _sh(params: Any) -> None:
                await params.result_callback({"ok": True})

            sh = FunctionSchema(
                name="smart_home",
                description="Smart home",
                properties={"intent": {"type": "string"}, "slot": {"type": "string"}},
                required=["intent"],
                handler=_sh,
            )
            sh = agent_tools._recorded(sh, record)

            tools = ToolsSchema(standard_tools=[fc, ds, dc, de, wa, sh])
            self.agent = _FakeAgent()
            router = (
                _FakeRouter(RouterScript(lines=self.router_lines))
                if self.router_lines
                else _FakeRouter(RouterScript(reason=self.router_reason))
            )

            async def _fake_json(system: str, content: str) -> str:
                return self.reply

            log = Path("/tmp/test_rbp.log")
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
            self.record = record

            ws_server = await serve(self.server.handle, "127.0.0.1", 0)
            port = next(iter(ws_server.sockets)).getsockname()[1]
            self._ws_server = ws_server
            self._port = port
            self._url = f"ws://127.0.0.1:{port}"
            return self._url
        finally:
            agent_tools.CHAIN = prev_chain
            agent_tools.SCRATCHPAD = prev_scratchpad

    async def send(self, question: str, session: str = "") -> dict[str, Any]:
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
        row["text"] = " ".join(texts)
        row["reached_agent"] = getattr(self.agent, "reached", False)
        self.server.log.write_text("", encoding="utf-8")
        return row

    async def close(self) -> None:
        self._ws_server.close()
        await self._ws_server.wait_closed()


# --- General lane: the router branch runs the same _ask_general turn as the no-router branch ---


@pytest.mark.asyncio
async def test_router_general_question_goes_docs_general() -> None:
    """Router branch + general question: route docs_general, the catalogue answer, no docs call."""
    from voice_pipeline.doc_route import is_general_question

    assert is_general_question("What are all the files about?")
    reply = json.dumps({"status": "answer", "answer": "You have 2 files in 2 folders: a and b."})
    fx = _RouterParityFixtures(reply)
    await fx.build()
    try:
        row = await fx.send("What are all the files about?")
        assert row["route"] == "docs_general"
        assert "You have 2 files in 2 folders: a and b." in row["text"]
        assert any(e.get("tool") == "docs_catalog" for e in fx.record)
        assert fx.doc_calls == []
        assert row["reached_agent"] is False
    finally:
        await fx.close()


# --- No-answer policy: a doc-scoped 'none' refuses, even when the router wanted the web ---


@pytest.mark.asyncio
async def test_router_doc_scoped_none_refuses() -> None:
    """Router branch + doc-scoped none: exactly 'The documents don't say.', Gemini never reached."""
    fx = _RouterParityFixtures(json.dumps({"status": "none"}))
    await fx.build()
    try:
        from voice_pipeline import doc_route

        doc_route.set_cached_files(fx.server, ["pet_policy.md"])
        row = await fx.send("What does the document say about the pet policy?")
        assert row["route"] == "docs_miss"
        assert row["text"].rstrip() == "The documents don't say."
        assert row["docs"] == "none"
        assert row["reached_agent"] is False
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_router_doc_scoped_none_beats_web_call() -> None:
    """The router asked for web_answer, but a doc-scoped none refuses and the web never runs."""
    fx = _RouterParityFixtures(json.dumps({"status": "none"}), router_lines="web_answer|pet policy")
    await fx.build()
    try:
        from voice_pipeline import doc_route

        doc_route.set_cached_files(fx.server, ["pet_policy.md"])
        row = await fx.send("What does the document say about the pet policy?")
        assert row["route"] == "docs_miss"
        assert row["text"].rstrip() == "The documents don't say."
        assert fx.web_calls == []
        assert row["reached_agent"] is False
    finally:
        await fx.close()


# --- No-answer policy: an open 'none' falls to Gemini with the prefix, as the no-router branch ---


@pytest.mark.asyncio
async def test_router_open_none_gets_no_prefix() -> None:
    """Router branch + open (non-document) none: no miss prefix is spoken (243); the docs
    verdict is still logged."""
    fx = _RouterParityFixtures(json.dumps({"status": "none"}))
    await fx.build()
    try:
        row = await fx.send("What is the capital of France?")
        assert row["route"] == "docs_miss"
        assert row["docs"] == "none"
        assert not row["text"].startswith("Your documents don't cover that.")
        assert row["reached_agent"] is True
    finally:
        await fx.close()


# --- Router authority is untouched: actions and (non-refused) web keep their current behaviour ---


@pytest.mark.asyncio
async def test_router_action_command_never_touches_docs() -> None:
    """A router ACTION command stays 'routed' and never asks the documents."""
    fx = _RouterParityFixtures(json.dumps({"status": "none"}), router_lines="TIMER|45 minutes")
    await fx.build()
    try:
        row = await fx.send("Set a 45 minute timer")
        assert row["route"] == "routed"
        assert fx.doc_calls == []
        assert row["reached_agent"] is False
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_router_web_after_open_miss_still_routed() -> None:
    """An open (not doc-scoped) none with web_answer calls still runs the web lookup."""
    fx = _RouterParityFixtures(
        json.dumps({"status": "none"}), router_lines="web_answer|capital of Peru"
    )
    await fx.build()
    try:
        row = await fx.send("What is the capital of Peru?")
        assert row["route"] == "routed"
        assert fx.web_calls
    finally:
        await fx.close()
