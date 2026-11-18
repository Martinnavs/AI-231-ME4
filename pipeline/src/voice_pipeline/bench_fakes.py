"""Offline benchmark driver using scripted fakes through the real ``AgentServer.handle``."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from voice_pipeline import agent_tools, doc_answer, question
from voice_pipeline.agent_server import AgentServer
from voice_pipeline.benchmark_cases import (
    Case,
    ModelCall,
    OfflineFixture,
    RouterScript,
    ToolScript,
)
from voice_pipeline.hedge import HedgedAgent, HedgeGate, gated_tools
from voice_pipeline.router import Routed, spec

RUN_STEM = "20261118-120000"
_NON_TOOL_ROUTER_CALLS = frozenset({"none", "escalate", "get_current_time"})


def _collect_tool_names(offline: OfflineFixture) -> list[str]:
    """Return deduplicated tool names needed by ``offline`` in declaration order."""
    names: list[str] = []

    def add(name: str) -> None:
        if name and name not in names:
            names.append(name)

    for name in offline.tools:
        add(name)

    if offline.router is not None and offline.router.lines:
        try:
            calls = spec.parse_output(offline.router.lines)
        except spec.LineError:
            calls = []
        for call in calls:
            if call.name in _NON_TOOL_ROUTER_CALLS:
                continue
            add("smart_home" if call.name in spec.HOME_INTENTS else call.name)

    for model_call in offline.model.calls:
        add(model_call.tool)

    return names


def _tool_script_for(name: str, tools: Mapping[str, ToolScript]) -> ToolScript:
    if name in tools:
        return tools[name]
    return ToolScript(result={"ok": True} if name == "smart_home" else "ok")


def _make_tool_schema(
    name: str,
    script: ToolScript,
    ledger_path: Path,
    record: list[dict[str, Any]],
) -> FunctionSchema:
    """Create a fake MCP-style ``FunctionSchema`` wrapped with ``agent_tools._recorded``."""

    async def handler(params: Any) -> None:
        is_router = getattr(params, "tool_call_id", None) == "router"
        override = script.by_router if is_router else None
        delay_s = (
            override.delay_s
            if override is not None and override.delay_s is not None
            else script.delay_s
        )
        ledger_rows = (
            override.ledger
            if override is not None and override.ledger is not None
            else script.ledger
        )
        result = (
            override.result
            if override is not None and override.result is not None
            else script.result
        )
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        if ledger_rows:
            with ledger_path.open("a", encoding="utf-8") as sink:
                for row in ledger_rows:
                    sink.write(json.dumps(row) + "\n")
        await params.result_callback(result)

    schema = FunctionSchema(
        name=name,
        description=f"Offline fake tool {name}",
        properties={},
        required=[],
        handler=handler,
    )
    return agent_tools._recorded(schema, record)


class _FakeRouter:
    """Scripted ``LocalRouter`` returning ``Routed`` from ``lines`` or ``reason``."""

    def __init__(self, script: RouterScript) -> None:
        self.script = script

    async def route(self, history: Any, user: str, names: Any) -> Routed:
        del history, names
        if self.script.delay_s > 0:
            await asyncio.sleep(self.script.delay_s)
        if self.script.reason is not None:
            return Routed(seconds=self.script.delay_s, reason=self.script.reason, user=user)
        lines = self.script.lines or ""
        try:
            calls = spec.parse_output(lines)
        except spec.LineError as exc:
            return Routed(
                lines=lines,
                seconds=self.script.delay_s,
                reason=f"parse: {exc}",
                user=user,
            )
        for call in calls:
            if call.name in spec.HOME_INTENTS and call.args:
                call.args = [" ".join(p.strip() for p in call.args[0].split("|") if p.strip())]
        escalate = next((c for c in calls if c.name == "escalate"), None)
        if escalate is not None:
            reason = escalate.args[0] if escalate.args else "escalate"
            return Routed(
                calls=calls, lines=lines, seconds=self.script.delay_s, reason=reason, user=user
            )
        return Routed(calls=calls, lines=lines, seconds=self.script.delay_s, reason=None, user=user)


class _ScriptedRequest:
    """Fake model request executing scripted tool calls then speaking ``text``."""

    def __init__(
        self,
        tools: ToolsSchema,
        *,
        delay_s: float = 0.0,
        calls: tuple[ModelCall, ...] = (),
        text: str = "",
        parallel: bool = False,
        never_answer: bool = False,
    ) -> None:
        self.handlers = {t.name: t.handler for t in tools.standard_tools}
        self.delay_s = delay_s
        self.calls = calls
        self.text = text
        self.parallel = parallel
        self.never_answer = never_answer
        self.cancelled = asyncio.Event()
        self.tap = SimpleNamespace(usage={"requests": 0, "prompt": 0, "completion": 0, "cached": 0})

    def hide(self, names: frozenset[str]) -> None:
        del names

    async def cancel(self) -> None:
        self.cancelled.set()

    async def ask(
        self,
        messages: Any,
        *,
        reset: bool,
        on_text: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        del messages, reset
        self.cancelled.clear()
        self.tap.usage["requests"] += 1
        if self.never_answer:
            await self.cancelled.wait()
            return {"status": "ok", "text": "", "cancelled": True}
        if self.delay_s > 0:
            try:
                await asyncio.wait_for(self.cancelled.wait(), self.delay_s)
                return {"status": "ok", "text": "", "cancelled": True}
            except TimeoutError:
                pass
        if self.cancelled.is_set():
            return {"status": "ok", "text": "", "cancelled": True}

        got: list[Any] = []

        async def keep(result: Any, *, properties: Any = None) -> None:
            del properties
            got.append(result)

        def build_call(idx: int, call: ModelCall) -> tuple[Any, Any]:
            handler = self.handlers.get(call.tool)
            if handler is None:
                raise KeyError(f"unknown offline tool {call.tool!r}")
            params: Any = SimpleNamespace(
                arguments=dict(call.args),
                result_callback=keep,
                tool_call_id=f"model-{idx}",
                function_name=call.tool,
            )
            return handler, params

        if self.parallel:
            if self.cancelled.is_set():
                return {"status": "ok", "text": "", "cancelled": True}
            pending = [
                handler(params)
                for idx, call in enumerate(self.calls, start=1)
                for handler, params in [build_call(idx, call)]
            ]
            if pending:
                await asyncio.gather(*pending)
        else:
            for idx, call in enumerate(self.calls, start=1):
                if self.cancelled.is_set():
                    return {"status": "ok", "text": "", "cancelled": True}
                handler, params = build_call(idx, call)
                await handler(params)

        if self.cancelled.is_set():
            return {"status": "ok", "text": "", "cancelled": True}
        if self.text and on_text is not None:
            await on_text(self.text)
        return {"status": "ok", "text": self.text}


def search_docs_passage(
    source: str,
    text: str,
    *,
    chunk_id: str | None = None,
    section: str = "",
    page: int | None = None,
    file_date: str = "2026-11-18",
    score: float = 2.0,
    coverage: float = 1.0,
) -> dict[str, Any]:
    """Return one T1-shaped ``search_docs`` passage dict."""
    return {
        "source": source,
        "id": chunk_id if chunk_id is not None else f"{source}#0",
        "section": section,
        "page": page,
        "file_date": file_date,
        "text": text,
        "score": float(score),
        "coverage": float(coverage),
    }


def fake_doc_request(reply: str) -> doc_answer.Request:
    """Return an offline ``doc_request`` callable returning the scripted ``reply``."""

    async def request(system: str, content: str) -> str:
        del system, content
        return reply

    return request


async def run_offline_case(case: Case, workdir: Path | str) -> dict[str, Any]:
    """Run one offline ``Case`` through the real ``AgentServer.handle`` and return its T1 row."""
    if case.offline is None:
        raise ValueError(f"case {case.id!r} has no offline fixture")

    offline = case.offline
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / f"{RUN_STEM}.jsonl"
    ledger_path = work / "ledger.jsonl"
    ledger_path.write_text("", encoding="utf-8")

    prev_chain = agent_tools.CHAIN
    prev_scratchpad = agent_tools.SCRATCHPAD
    prev_web_answer = list(agent_tools._web_answer)
    prev_question = question.current
    agent_tools.CHAIN = None
    agent_tools.SCRATCHPAD = None
    agent_tools._web_answer.clear()
    agent_tools.reset_turn_lookups()
    try:
        record: list[dict[str, Any]] = []
        tool_names = _collect_tool_names(offline)
        schemas = [
            _make_tool_schema(name, _tool_script_for(name, offline.tools), ledger_path, record)
            for name in tool_names
        ]
        tools = ToolsSchema(standard_tools=schemas)

        agent: Any
        if offline.hedged:
            gate = HedgeGate()
            agents: list[Any] = [
                _ScriptedRequest(gated_tools(tools, gate, 0), never_answer=True),
                _ScriptedRequest(
                    gated_tools(tools, gate, 1),
                    delay_s=offline.model.delay_s,
                    calls=offline.model.calls,
                    text=offline.model.text,
                    parallel=offline.model.parallel,
                ),
            ]
            agent = HedgedAgent(agents, gate, 0.0, hold_s=0.0)
        else:
            agent = _ScriptedRequest(
                tools,
                delay_s=offline.model.delay_s,
                calls=offline.model.calls,
                text=offline.model.text,
                parallel=offline.model.parallel,
            )

        router: Any = _FakeRouter(offline.router) if offline.router is not None else None
        doc_request = fake_doc_request(offline.model.text) if "search_docs" in tool_names else None
        server = AgentServer(agent, record, log_path, tools, router, doc_request=doc_request)
        server.ledger = ledger_path
        if offline.stamp:
            server.stamp = lambda: "[Now: test]"

        ws_server = await serve(server.handle, "127.0.0.1", 0)
        try:
            port = next(iter(ws_server.sockets)).getsockname()[1]
            url = f"ws://127.0.0.1:{port}"
            messages = [{"role": t.role, "content": t.content} for t in case.turns]
            async with connect(url) as ws:
                await ws.send(json.dumps({"messages": messages}))
                while True:
                    frame = json.loads(await ws.recv())
                    if "done" in frame or "error" in frame:
                        break
        finally:
            ws_server.close()
            await ws_server.wait_closed()
        if server.checks:
            await asyncio.gather(*server.checks, return_exceptions=True)

        row: dict[str, Any] = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
        return row
    finally:
        agent_tools.reset_turn_lookups()
        agent_tools._web_answer[:] = prev_web_answer
        agent_tools.CHAIN = prev_chain
        agent_tools.SCRATCHPAD = prev_scratchpad
        question.current = prev_question
