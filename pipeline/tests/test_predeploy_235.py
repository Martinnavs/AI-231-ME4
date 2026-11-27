"""Pre-deploy 235: the router-branch docs fixes.

FIX A: the router branch skips the documents for non-questions ("Thanks!", "okay",
"goodbye"), as the no-router branch already does.
FIX B: after a docs miss the web fallback does not re-run the legacy doc_race (it could
speak the decoy ask_documents just rejected) and keeps the miss prefix.
FIX C: _ask_docs' _file_has fails open: an error, a list cut at the limit, a truncated or
undecodable result cannot prove a file is absent.
FIX D: a GENERAL-lane 'none' does not claim the folder is empty when docs_status lists files.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voice_pipeline.bench_fakes import _FakeRouter
from voice_pipeline.benchmark_cases import RouterScript
from voice_pipeline.doc_route import EMPTY_FOLDER_REPLY

_CAT_DEFAULT = [
    {"folder": "a", "rel_key": "a/guide.md", "line": "a/guide.md | Guide | markdown"},
    {"folder": "b", "rel_key": "b/notes.txt", "line": "b/notes.txt | Notes | text"},
]

_ANSWER_JSON = json.dumps(
    {
        "status": "answer",
        "answer": "0.342",
        "file": "test.md",
        "quotes": ["State: Hawaii; Average price: 0.342"],
    }
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


class _PredeployFixtures:
    """AgentServer with a fake router and every docs tool. `with_race` also adds search_docs
    and a doc_request, so run_routed's legacy doc_race branch is actually reachable."""

    def __init__(
        self,
        reply: str,
        router_lines: str = "",
        router_reason: str = "normal",
        status_files: list[str] | None = None,
        catalog: list[dict[str, Any]] | None = None,
        with_race: bool = False,
    ) -> None:
        self.reply = reply
        self.router_lines = router_lines
        self.router_reason = router_reason
        self.status_files = status_files
        self.catalog = _CAT_DEFAULT if catalog is None else catalog
        self.with_race = with_race
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
                files = [{"name": n, "status": "ok"} for n in (self.status_files or [])]
                await params.result_callback({"files": files, "failures": []})

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

                async def _doc_request(system: str, content: str) -> str:
                    return json.dumps(
                        {
                            "answer": "The decoy value is 42.",
                            "file": "decoy.md",
                            "quotes": ["The decoy value is 42."],
                        }
                    )

                doc_request = _doc_request

            tools = ToolsSchema(standard_tools=schemas)
            self.agent = _FakeAgent()
            router = (
                _FakeRouter(RouterScript(lines=self.router_lines))
                if self.router_lines
                else _FakeRouter(RouterScript(reason=self.router_reason))
            )

            async def _fake_json(
                system: str, content: str, schema: dict[str, Any] | None = None
            ) -> str:
                return self.reply

            log = Path("/tmp/test_predeploy_235.log")
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


# --- FIX A: the router branch does not ask the documents for non-questions ---


@pytest.mark.asyncio
async def test_router_non_questions_skip_docs() -> None:
    """Router branch + "Thanks!"/"okay"/"goodbye": zero file_context calls, no miss prefix,
    the router's own reply is spoken."""
    fx = _PredeployFixtures(json.dumps({"status": "none"}), router_lines="none|You're welcome.")
    await fx.build()
    try:
        for said in ("Thanks!", "okay", "goodbye"):
            row = await fx.send(said)
            assert row["route"] == "routed", said
            assert row["text"].rstrip() == "You're welcome.", said
            assert not row["text"].startswith("Your documents don't cover that"), said
            assert row["reached_agent"] is False, said
        assert fx.doc_calls == []
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_router_doc_question_still_calls_docs_once() -> None:
    """Router branch + a real doc question: file_context is called exactly once and the miss
    keeps its prefix."""
    fx = _PredeployFixtures(json.dumps({"status": "none"}), router_lines="none|I don't know.")
    await fx.build()
    try:
        row = await fx.send("Where does my base schedule live, in my notes?")
        assert len(fx.doc_calls) == 1
        assert row["route"] == "docs_miss"
        assert row["text"].startswith("Your documents don't cover that. ")
    finally:
        await fx.close()


# --- FIX B: a docs miss falls to the web without the legacy doc_race, keeping the prefix ---


@pytest.mark.asyncio
async def test_router_docs_miss_web_fallback_skips_doc_race() -> None:
    """Router branch + web_answer + docs miss: doc_race is never called, the decoy is not
    spoken, the miss prefix is kept, and file_context ran exactly once."""
    fx = _PredeployFixtures(
        json.dumps({"status": "none"}),
        router_lines="web_answer|capital of Peru",
        with_race=True,
    )
    await fx.build()
    try:
        race_calls: list[str] = []

        async def decoy_race(user: str) -> tuple[str | None, dict[str, Any]]:
            race_calls.append(user)
            return "As per decoy.md: The decoy value is 42.", {"docs": "accepted", "doc_s": 0.1}

        fx.server.doc_race = decoy_race
        row = await fx.send("Is the capital of Peru in my notes?")
        assert race_calls == []
        assert "decoy" not in row["text"].lower()
        assert row["text"].startswith("Your documents don't cover that. ")
        assert "The capital is Lima." in row["text"]
        assert fx.web_calls == ["capital of Peru"]
        assert len(fx.doc_calls) == 1
        assert row["route"] == "routed"
        assert row["docs"] == "none"
        assert row["reached_agent"] is False
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_router_docs_miss_check_failed_web_fallback_unconfirmed_prefix() -> None:
    """A check_failed verdict (a rejected answer) falls to the web with the UNCONFIRMED prefix."""
    fx = _PredeployFixtures(
        _ANSWER_JSON_WITH_BAD_QUOTE(),
        router_lines="web_answer|capital of Peru",
        with_race=True,
    )
    await fx.build()
    try:
        race_calls: list[str] = []

        async def decoy_race(user: str) -> tuple[str | None, dict[str, Any]]:
            race_calls.append(user)
            return "As per decoy.md: The decoy value is 42.", {"docs": "accepted", "doc_s": 0.1}

        fx.server.doc_race = decoy_race
        row = await fx.send("Is the capital of Peru in my notes?")
        assert race_calls == []
        assert row["docs"] == "check_failed"
        assert row["text"].startswith("I couldn't confirm that in your documents. ")
        assert "The capital is Lima." in row["text"]
        assert row["route"] == "routed"
    finally:
        await fx.close()


def _ANSWER_JSON_WITH_BAD_QUOTE() -> str:
    """An 'answer' reply whose quote is not in the passage: the checks fail (check_failed)."""
    return json.dumps(
        {
            "status": "answer",
            "answer": "It is in section 3",
            "file": "test.md",
            "quotes": ["a phrase that is not in the passage at all"],
        }
    )


# --- FIX C: _file_has fails open ---


@pytest.mark.parametrize(
    ("enumerate_result", "expected_status"),
    [
        pytest.param("RAISE", "accepted", id="exception-fails-open"),
        pytest.param(
            [{"rel_key": f"f{i}.md"} for i in range(100)],
            "accepted",
            id="100-rows-without-file-fails-open",
        ),
        pytest.param(
            [{"rel_key": f"f{i}.md"} for i in range(5)],
            "check_failed",
            id="5-rows-without-file-rejects",
        ),
        pytest.param(
            [{"rel_key": f"f{i}.md"} for i in range(4)] + [{"rel_key": "test.md"}],
            "accepted",
            id="5-rows-with-file-accepted",
        ),
        pytest.param(
            '[{"rel_key": "a.md"}] [truncated: 40 more]',
            "accepted",
            id="truncated-text-fails-open",
        ),
    ],
)
@pytest.mark.asyncio
async def test_ask_docs_file_has_fails_open(enumerate_result: Any, expected_status: str) -> None:
    """The passages never spell 'Guam': the answer is accepted only when _file_has reports the
    token present in the cited file (or cannot prove it absent); a complete 5-row list without
    the file still rejects."""
    fx = _PredeployFixtures(json.dumps({"status": "none"}))
    await fx.build()
    try:

        async def fake_call_tool(name: str, args: dict[str, Any]) -> Any:
            if name == "file_context":
                return [
                    {
                        "source": "test.md",
                        "text": "State: Hawaii; Average price: 0.342",
                        "part": "whole",
                    }
                ]
            if name == "docs_enumerate":
                if enumerate_result == "RAISE":
                    raise RuntimeError("docs_enumerate blew up")
                return enumerate_result
            return []

        async def fake_json(system: str, content: str) -> str:
            return _ANSWER_JSON

        fx.server.call_tool = fake_call_tool
        fx.server.doc_json_request = fake_json
        result = await fx.server._ask_docs(
            "What was the average home electricity price in Guam?", "", fx.server.turn_now
        )
        assert result.status == expected_status, (result.status, result.rejected)
        if expected_status == "check_failed":
            assert result.rejected is not None
            assert result.rejected["failed"] == "key_absent"
    finally:
        await fx.close()


# --- FIX D: no false "folder is empty" in the GENERAL lane ---


@pytest.mark.asyncio
async def test_general_miss_with_files_says_file_count() -> None:
    """docs_catalog returns [] but docs_status lists 3 files: the reply says '3 files' and not
    that the folder is empty."""
    fx = _PredeployFixtures(
        "",
        status_files=["a/guide.md", "b/notes.txt", "c/plan.md"],
        catalog=[],
    )
    await fx.build()
    try:
        row = await fx.send("What are all the files about?")
        assert row["route"] == "docs_general"
        assert "3 files" in row["text"]
        assert "empty" not in row["text"]
        assert row["text"].rstrip() == "I have 3 files; ask me about a topic or a folder."
        assert row["reached_agent"] is False
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_general_miss_with_no_files_says_empty() -> None:
    """docs_status lists 0 files: the GENERAL-lane 'none' still gets the empty-folder reply."""
    fx = _PredeployFixtures("", catalog=[])
    await fx.build()
    try:
        row = await fx.send("What are all the files about?")
        assert row["route"] == "docs_general"
        assert row["text"] == EMPTY_FOLDER_REPLY
        assert row["reached_agent"] is False
    finally:
        await fx.close()
