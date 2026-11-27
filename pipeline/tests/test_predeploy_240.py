"""Pre-deploy 240: an open research chain must not capture document questions.

FIX A: the chain branch yields a turn to the checked document path when docs-first is on
and the utterance names a file, or is a question while the chain is not mid-protocol (in
the awaiting_* phases a turn answers the chain's own question, so the chain keeps it). A
yielding turn does not cancel the chain: it counts as one unrelated turn, and three of
them expire the chain.
FIX B: a plain web_answer is not chain work: the turn does not refresh the chain's
updated_at and counts as unrelated; chain_update and web_research still do.
FIX C: a stale chain.json from a previous run (older than EXPIRE_S) is reset to idle when
the service starts (ChainState.__init__ -> load -> active), so it cannot capture the
first turn.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voice_pipeline.bench_fakes import _FakeRouter, _ScriptedRequest
from voice_pipeline.benchmark_cases import ModelCall, RouterScript
from voice_pipeline.chain_state import EXPIRE_S, ChainState
from voice_pipeline.doc_route import set_cached_files

_BRIEFING_QUESTION = "The UK prime minister in my briefing notes, is that still right?"


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


class _PredeployFixtures:
    """AgentServer with a fake router, every docs tool and an optional open chain, so the
    chain branch and the docs-first block are both reachable."""

    def __init__(self, reply: str, chain: ChainState | None = None) -> None:
        self.reply = reply
        self.chain = chain
        self.doc_calls: list[str] = []
        self.web_calls: list[str] = []
        self.server: Any = None
        self.agent: Any = None
        self.tools_schema: Any = None
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

            fc = agent_tools._recorded(
                FunctionSchema(
                    name="file_context",
                    description="File context",
                    properties={"query": {"type": "string"}},
                    required=["query"],
                    handler=_fc,
                ),
                record,
            )

            async def _ds(params: Any) -> None:
                await params.result_callback({"files": [], "failures": []})

            ds = agent_tools._recorded(
                FunctionSchema(
                    name="docs_status",
                    description="Docs status",
                    properties={},
                    required=[],
                    handler=_ds,
                ),
                record,
            )

            async def _dc(params: Any) -> None:
                await params.result_callback([])

            dc = agent_tools._recorded(
                FunctionSchema(
                    name="docs_catalog",
                    description="Docs catalog",
                    properties={"max_chars": {"type": "integer"}},
                    required=[],
                    handler=_dc,
                ),
                record,
            )

            async def _de(params: Any) -> None:
                await params.result_callback([])

            de = agent_tools._recorded(
                FunctionSchema(
                    name="docs_enumerate",
                    description="Docs enumerate",
                    properties={"query": {"type": "string"}, "limit": {"type": "integer"}},
                    required=["query"],
                    handler=_de,
                ),
                record,
            )

            async def _wa(params: Any) -> None:
                self.web_calls.append(params.arguments.get("query", ""))
                await params.result_callback("As per example.org: The capital is Lima.")

            wa = agent_tools._recorded(
                FunctionSchema(
                    name="web_answer",
                    description="Web answer",
                    properties={"query": {"type": "string"}},
                    required=["query"],
                    handler=_wa,
                ),
                record,
            )

            tools = ToolsSchema(standard_tools=[fc, ds, dc, de, wa])
            self.tools_schema = tools
            self.agent = _FakeAgent()
            router = _FakeRouter(RouterScript(reason="normal"))

            async def _fake_json(
                system: str, content: str, schema: dict[str, Any] | None = None
            ) -> str:
                return self.reply

            log = Path("/tmp/test_predeploy_240.log")
            log.parent.mkdir(parents=True, exist_ok=True)

            self.server = AgentServer(
                self.agent,
                record,
                log,
                tools,
                router,
                chain=self.chain,
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


# --- FIX A: an open chain yields document questions to the checked document path ---


@pytest.mark.asyncio
async def test_a_briefing_note_question_yields_from_a_researching_chain(tmp_path: Path) -> None:
    """(a) A chain open in 'researching' must not capture a question about a briefing note:
    it reaches the document path, the agent never sees it, and the chain is not cancelled -
    the turn counts as one unrelated turn."""
    chain = ChainState(tmp_path)
    chain.update("researching", subject="UK politics", facts=["the note says Starmer"])
    fx = _PredeployFixtures(json.dumps({"status": "none"}), chain=chain)
    await fx.build()
    try:
        set_cached_files(fx.server, ["briefing_notes.md"])
        row = await fx.send(_BRIEFING_QUESTION)
        assert row["route"] != "chain", row["route"]
        assert row["route"] == "docs_miss"
        assert row["reached_agent"] is False
        assert fx.doc_calls, "the checked document path ran"
        assert chain.data["phase"] == "researching"  # not cancelled
        assert chain.data["unrelated_turns"] == 1
        assert chain.active()
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_a_file_cue_question_yields_from_a_researching_chain(tmp_path: Path) -> None:
    """(b) 'What does the census file say about Texas?' reaches the document path while a
    research chain is open."""
    chain = ChainState(tmp_path)
    chain.update("researching", subject="census", facts=["f"])
    fx = _PredeployFixtures(json.dumps({"status": "none"}), chain=chain)
    await fx.build()
    try:
        set_cached_files(fx.server, ["census_2026.xlsx"])
        row = await fx.send("What does the census file say about Texas?")
        assert row["route"] != "chain", row["route"]
        assert row["route"] == "docs_miss"
        assert row["reached_agent"] is False
        assert fx.doc_calls
        assert chain.active() and chain.data["unrelated_turns"] == 1
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_a_mid_protocol_chain_keeps_the_answer_to_its_own_question(tmp_path: Path) -> None:
    """(c) In 'awaiting_fact_approval' the next utterance answers the chain's question:
    'yes' and 'approve' stay route 'chain', the documents are never consulted."""
    chain = ChainState(tmp_path)
    chain.update("awaiting_fact_approval", subject="Ana Cruz", facts=["f1"])
    fx = _PredeployFixtures(json.dumps({"status": "none"}), chain=chain)
    await fx.build()
    try:
        set_cached_files(fx.server, ["briefing_notes.md"])
        for said in ("yes", "approve"):
            row = await fx.send(said)
            assert row["route"] == "chain", (said, row["route"])
            assert row["reached_agent"] is True, said
        assert fx.doc_calls == []
        assert chain.data["phase"] == "awaiting_fact_approval"
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_three_unrelated_yields_expire_the_chain(tmp_path: Path) -> None:
    """(d) Each yielded turn counts as one unrelated turn; the chain survives two of them
    (still yielding to the documents) and the third expires it."""
    chain = ChainState(tmp_path)
    chain.update("researching", subject="census", facts=["f"])
    fx = _PredeployFixtures(json.dumps({"status": "none"}), chain=chain)
    await fx.build()
    try:
        set_cached_files(fx.server, ["census_2026.xlsx"])
        questions = (
            "What does the census file say about Texas?",
            "What does the census file say about Florida?",
            "What does the census file say about Georgia?",
        )
        for said in questions:
            row = await fx.send(said)
            assert row["route"] != "chain", said
            assert row["route"] == "docs_miss", said
            if said in questions[:2]:  # the third one expires the chain: counter resets to idle
                assert chain.active(), said
                assert chain.data["unrelated_turns"] == questions.index(said) + 1, said
        assert not chain.active()
        assert chain.data["phase"] == "idle"
    finally:
        await fx.close()


# --- FIX B: a plain web_answer is not chain work ---


@pytest.mark.asyncio
async def test_a_plain_web_answer_does_not_refresh_the_chain(tmp_path: Path) -> None:
    """(e) A turn answered with web_answer alone does not refresh the chain's updated_at;
    it counts as an unrelated turn so the chain can expire. The turn is still handled by
    the chain (route 'chain'), web_answer stays usable inside the chain."""
    chain = ChainState(tmp_path)
    chain.update("researching", subject="miss universe", facts=["f"])
    fx = _PredeployFixtures(json.dumps({"status": "none"}), chain=chain)
    await fx.build()
    try:
        fx.server.agent = _ScriptedRequest(
            fx.tools_schema,
            calls=(ModelCall("web_answer", {"query": "who won miss universe 2026"}),),
            text="As per example.org: Jennifer Barreto won Miss Universe 2026.",
        )
        updated_before = chain.data["updated_at"]
        row = await fx.send("Keep researching.")
        assert row["route"] == "chain", row["route"]
        assert fx.web_calls == ["who won miss universe 2026"]
        assert chain.data["updated_at"] == updated_before  # web_answer is not chain work
        assert chain.data["unrelated_turns"] == 1
        assert chain.active()
    finally:
        await fx.close()


# --- FIX C: a stale chain.json from a previous run does not capture the first turn ---


def _stale_chain_file(path: Path) -> None:
    data = {
        "phase": "researching",
        "subject": "miss universe",
        "facts": ["Jennifer Barreto won Miss Universe 2026"],
        "items": [],
        "gaps": [],
        "job_id": "",
        "artifact_path": "",
        "unrelated_turns": 0,
        "done": False,
        "focus": {"entity": "", "tool": ""},
        "updated_at": time.time() - EXPIRE_S - 60,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_a_stale_chain_file_is_idle_when_the_state_loads(tmp_path: Path) -> None:
    """The ChainState constructor (the service's constructor site, `run`) loads the file and
    resets anything older than EXPIRE_S to idle, rewriting the file."""
    _stale_chain_file(tmp_path / "chain.json")
    chain = ChainState(tmp_path)
    assert not chain.active()
    assert chain.data["phase"] == "idle"
    assert json.loads((tmp_path / "chain.json").read_text())["phase"] == "idle"


@pytest.mark.asyncio
async def test_a_stale_chain_json_does_not_capture_the_first_turn(tmp_path: Path) -> None:
    """A chain left open by a previous run (600 s stale) must not route the first turn of
    the new service to 'chain': the document question reaches the document path."""
    _stale_chain_file(tmp_path / "chain.json")
    chain = ChainState(tmp_path)  # as built at service start
    fx = _PredeployFixtures(json.dumps({"status": "none"}), chain=chain)
    await fx.build()
    try:
        set_cached_files(fx.server, ["briefing_notes.md"])
        row = await fx.send(_BRIEFING_QUESTION)
        assert row["route"] != "chain", row["route"]
        assert row["route"] == "docs_miss"
        assert row["reached_agent"] is False
        assert chain.data["phase"] == "idle"
    finally:
        await fx.close()
