"""GENERAL-lane routing: is_general_question + the docs_general turn (ticket 221b)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voice_pipeline.doc_route import EMPTY_FOLDER_REPLY, is_general_question

GENERAL_V2 = Path.home() / "rag-testsets" / "golden-v2" / "general-v2.jsonl"
GOLDEN_V2 = Path.home() / "rag-testsets" / "golden-v2" / "golden-v2.jsonl"


def _general_q(id_: str) -> str:
    for line in GENERAL_V2.read_text().splitlines():
        row = json.loads(line)
        if row["id"] == id_:
            return row["q"]
    raise AssertionError(f"missing {id_}")


@pytest.mark.parametrize("id_", ["g2-g01", "g2-g02", "g2-g04", "g2-g05", "g2-g06"])
def test_general_rows_route_general(id_: str) -> None:
    assert is_general_question(_general_q(id_))


def test_single_file_summarise_stays_specific() -> None:
    assert not is_general_question(_general_q("g2-g03"))


def test_specific_golden_rows_stay_specific() -> None:
    for line in GOLDEN_V2.read_text().splitlines():
        assert not is_general_question(json.loads(line)["q"])


@pytest.mark.parametrize(
    "q",
    [
        "How many hours does a full-time TTS employee work in a two-week pay period?",
        "What does the parking policy document say about visitor parking?",
        "Which city is the university whose fall commencement is December 19 in my files?",
        "The flake8 tox file lists its environments. How many are there, and what's the "
        "latest tox version?",
    ],
)
def test_specific_controls_stay_specific(q: str) -> None:
    assert not is_general_question(q)


def test_over_carry_question_stays_specific() -> None:
    assert not is_general_question("What's the capital of Australia?")


_CATALOG = [
    {"folder": "a", "rel_key": "a/guide.md", "line": "a/guide.md | Guide | markdown"},
    {"folder": "b", "rel_key": "b/notes.txt", "line": "b/notes.txt | Notes | text"},
]


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


class _GenFixtures:
    """A no-router AgentServer with docs_catalog/docs_enumerate (ticket 221b)."""

    def __init__(self, reply: str, catalog: list[dict[str, Any]] | None = None) -> None:
        self.reply = reply
        self.catalog = _CATALOG if catalog is None else catalog
        self.doc_calls: list[str] = []
        self.json_calls: list[tuple[str, str]] = []
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
                await params.result_callback(list(self.catalog))

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

            tools = ToolsSchema(standard_tools=[fc, ds, dc, de])

            self.agent = _FakeAgent()

            async def _fake_json(system: str, content: str) -> str:
                self.json_calls.append((system, content))
                return self.reply

            log = Path("/tmp/test_gen.log")
            log.parent.mkdir(parents=True, exist_ok=True)

            self.server = AgentServer(
                self.agent,
                record,
                log,
                tools,
                None,
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


@pytest.mark.asyncio
async def test_general_question_routes_docs_general() -> None:
    reply = json.dumps({"status": "answer", "answer": "You have 2 files in 2 folders: a and b."})
    fx = _GenFixtures(reply)
    await fx.build()
    try:
        row = await fx.send("What are all the files about?")
        assert row["route"] == "docs_general"
        assert any(e.get("tool") == "docs_catalog" for e in fx.record)
        assert fx.doc_calls == []
        assert row["reached_agent"] is False
        assert "You have 2 files in 2 folders: a and b." in row["text"]
        assert fx.json_calls[0][1].splitlines()[-1] == "Question: What are all the files about?"
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_general_bad_json_speaks_plain_list() -> None:
    fx = _GenFixtures("not json")
    await fx.build()
    try:
        row = await fx.send("What are all the files about?")
        assert row["route"] == "docs_general"
        assert row["fallback"] == "plain_list"
        assert "I have 2 files in 2 folders: a (1), b (1)." in row["text"]
        assert row["reached_agent"] is False
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_general_empty_catalog_speaks_empty_folder() -> None:
    fx = _GenFixtures("", catalog=[])
    await fx.build()
    try:
        row = await fx.send("What are all the files about?")
        assert row["route"] == "docs_general"
        assert row["text"] == EMPTY_FOLDER_REPLY
        assert row["reached_agent"] is False
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_specific_question_unchanged() -> None:
    fx = _GenFixtures(json.dumps({"status": "none"}))
    await fx.build()
    try:
        row = await fx.send(
            "How many hours does a full-time TTS employee work in a two-week pay period?"
        )
        assert row["route"] == "docs_miss"
        assert len(fx.doc_calls) >= 1
        assert row["reached_agent"] is True
    finally:
        await fx.close()
