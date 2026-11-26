"""Shared test setup: no test may depend on the apps that happen to exist on this machine."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from voice_pipeline import agent_tools, call_policy
from voice_pipeline.agent_server import AgentServer
from voice_pipeline.hedge import HedgedAgent, HedgeGate, gated_tools
from voice_pipeline.router import Routed, spec

USER = "Who is the CEO of Nvidia?"
LOOKED_UP = "As per CNN and FT, Jensen Huang is the CEO of Nvidia."
LOOKUP = "web_answer|who is the ceo of nvidia?"


@pytest.fixture(autouse=True)
def _no_real_apps(monkeypatch: pytest.MonkeyPatch) -> None:
    """known_apps reads ~/agent_sandbox/workspace/apps; a live session adds apps such as about_me
    there, which then changed unrelated verdicts. A test that needs apps sets its own."""
    monkeypatch.setattr(call_policy, "known_apps", lambda: [])


async def _serve_turn(
    server: AgentServer, text: str = USER, **request: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The frames of one turn and the row it logged, through the real `AgentServer.handle`."""
    ws_server = await serve(server.handle, "127.0.0.1", 0)
    url = f"ws://127.0.0.1:{next(iter(ws_server.sockets)).getsockname()[1]}"
    frames: list[dict[str, Any]] = []
    async with connect(url) as ws:
        await ws.send(json.dumps({"messages": [{"role": "user", "content": text}], **request}))
        while True:
            frames.append(json.loads(await ws.recv()))
            if "done" in frames[-1] or "error" in frames[-1]:
                break
    ws_server.close()
    await ws_server.wait_closed()
    return frames, json.loads(server.log.read_text().splitlines()[-1])


class Router:
    def __init__(self, lines: str = LOOKUP) -> None:
        self.lines = lines

    async def route(self, history: Any, user: str, names: Any) -> Routed:
        return Routed(spec.parse_output(self.lines), self.lines, 0.1, None, user)


class Scripted:
    """A model request that waits `delay` s, then calls `tool` (through its hedge-gated handler)."""

    def __init__(self, gated: ToolsSchema, delay: float, tool: str | None, prompt: int) -> None:
        self.handlers = {t.name: t.handler for t in gated.standard_tools}
        self.delay, self.tool, self.prompt = delay, tool, prompt
        self.cancelled = asyncio.Event()
        self.tap = SimpleNamespace(usage={"requests": 0, "prompt": 0, "completion": 0, "cached": 0})

    async def ask(self, messages: Any, *, reset: bool, on_text: Any, **_: Any) -> dict[str, Any]:
        self.cancelled.clear()
        self.tap.usage["requests"] += 1
        self.tap.usage["prompt"] += self.prompt
        try:
            await asyncio.wait_for(self.cancelled.wait(), self.delay)
            return {"status": "ok", "text": "", "cancelled": True}
        except TimeoutError:
            pass
        got: list[Any] = []

        async def keep(result: Any, **_kw: Any) -> None:
            got.append(result)

        if self.tool:
            # same query as the router's (LOOKUP), aside from case/punctuation: the model asks
            # the same thing the router is looking up (243 keeps per-query keys, so the F3
            # hedge-race duplicate stays a duplicate of one query)
            params = SimpleNamespace(arguments={"query": "who is the ceo of nvidia?"},
                                      result_callback=keep,  # fmt: skip
                                      tool_call_id="call-1", function_name=self.tool)  # fmt: skip
            await self.handlers[self.tool](params)
        text = str(got[0]) if got else "Jensen Huang."
        await on_text(text)
        return {"status": "ok", "text": text}

    async def cancel(self) -> None:
        self.cancelled.set()


class World:
    """A hedged agent server over fake agents, a fake router and `_recorded` fake lookups."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, model_wins: bool):
        monkeypatch.setattr(agent_tools, "CHAIN", None)
        monkeypatch.setattr(agent_tools, "SCRATCHPAD", None)
        self.ledger = tmp_path / "ledger.jsonl"
        self.router_delay = 30.0 if model_wins else 0.0
        self.record: list[dict[str, Any]] = []
        schema = FunctionSchema(name="web_answer", description="d", properties={}, required=[],
                                handler=self.lookup)  # fmt: skip
        tools = ToolsSchema(standard_tools=[agent_tools._recorded(schema, self.record)])
        gate = HedgeGate()
        self.agents = [
            Scripted(gated_tools(tools, gate, 0), 30.0, None, 80),  # never answers: the loser
            Scripted(gated_tools(tools, gate, 1), 0.0 if model_wins else 0.3, "web_answer", 100),
        ]
        hedged = HedgedAgent(self.agents, gate, 0.0, hold_s=0.0)  # type: ignore[arg-type]
        self.server = AgentServer(hedged, self.record, tmp_path / "20261118-120000.jsonl", tools,
                                  Router())  # fmt: skip
        self.server.ledger = self.ledger
        self.server.stamp = lambda: "[Now: test]"

    async def lookup(self, params: Any) -> None:
        if params.tool_call_id == "router":
            await asyncio.sleep(self.router_delay)
        self.append({"month": "2026-11", "searches": 1, "by": params.tool_call_id})
        await params.result_callback(LOOKED_UP)

    def append(self, row: dict[str, Any]) -> None:
        with self.ledger.open("a") as sink:
            sink.write(json.dumps(row) + "\n")


@pytest.fixture
def serve_turn() -> Callable[..., Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]]]:
    """Return the coroutine function that serves one turn over a real local WebSocket."""
    return _serve_turn


@pytest.fixture
def make_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], World]:
    """Factory fixture building isolated hedged `World` servers."""
    count = 0

    def _make(model_wins: bool) -> World:
        nonlocal count
        count += 1
        root = tmp_path / f"world-{count}"
        root.mkdir(exist_ok=True)
        return World(root, monkeypatch, model_wins=model_wins)

    return _make
