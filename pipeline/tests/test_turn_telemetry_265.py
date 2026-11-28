"""Turn telemetry and docs-miss prefix meta tests for agent_server (ticket 265)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from voice_pipeline import agent_tools
from voice_pipeline.agent_server import DOCS_NO_COVER_PREFIX, AgentServer
from voice_pipeline.bench_fakes import _FakeRouter
from voice_pipeline.benchmark_cases import RouterScript


class _ScriptedAgent:
    def __init__(self, reply_text: str, delay_s: float = 0.0) -> None:
        self.reply_text = reply_text
        self.delay_s = delay_s
        self.tap = SimpleNamespace(
            usage={"requests": 0, "prompt": 0, "completion": 0, "cached": 0}
        )

    async def ask(
        self, messages: Any, *, reset: bool, on_text: Any = None, **_: Any
    ) -> dict[str, Any]:
        del messages, reset
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        if on_text is not None:
            await on_text(self.reply_text)
        return {"status": "ok", "text": self.reply_text, "cancelled": False}

    def hide(self, names: frozenset[str]) -> None:
        del names


async def _send_turn(url: str, question: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    async with connect(url) as ws:
        await ws.send(json.dumps({"messages": [{"role": "user", "content": question}]}))
        while True:
            frame = json.loads(await ws.recv())
            frames.append(frame)
            if frame.get("done") or frame.get("error"):
                break
    return frames, frames[-1]


async def test_turn_row_logs_identical_turn_id_and_monotonic_bounds(tmp_path: Path) -> None:
    log_path = tmp_path / "bench_run.jsonl"
    agent = _ScriptedAgent("Lights on.", delay_s=0.02)
    server = AgentServer(agent, [], log_path)
    ws_server = await serve(server.handle, "127.0.0.1", 0)
    port = next(iter(ws_server.sockets)).getsockname()[1]
    try:
        frames, done_frame = await _send_turn(f"ws://127.0.0.1:{port}", "Turn on the lights")
    finally:
        ws_server.close()
        await ws_server.wait_closed()

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert done_frame["done"] is True
    assert isinstance(done_frame["turn"], str) and done_frame["turn"]
    assert row["turn"] == done_frame["turn"] == "bench_run-0001"
    assert isinstance(row["t_start_mono_ns"], int)
    assert isinstance(row["t_end_mono_ns"], int)
    assert 0 < row["t_start_mono_ns"] <= row["t_end_mono_ns"]
    assert (row["t_end_mono_ns"] - row["t_start_mono_ns"]) >= 15_000_000
    expected_keys = (
        "time", "question", "calls", "cancelled", "run", "turn", "route", "tokens", "ledger"
    )
    for key in expected_keys:
        assert key in row
    assert [f["text"] for f in frames if "text" in f] == ["Lights on."]


async def test_docs_miss_prefix_carries_meta_prefix_in_docs_first_mode(tmp_path: Path) -> None:
    log_path = tmp_path / "miss_docs_first.jsonl"
    record: list[dict[str, Any]] = []

    async def _fc(params: Any) -> None:
        await params.result_callback([])

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
    tools = ToolsSchema(standard_tools=[fc])

    async def _fake_json(system: str, content: str) -> str:
        del system, content
        return json.dumps({"status": "none"})

    agent = _ScriptedAgent("The capital of Peru is Lima.")
    server = AgentServer(
        agent,
        record,
        log_path,
        tools=tools,
        docs_first=True,
        doc_json_request=_fake_json,
    )
    ws_server = await serve(server.handle, "127.0.0.1", 0)
    port = next(iter(ws_server.sockets)).getsockname()[1]
    try:
        frames, done_frame = await _send_turn(
            f"ws://127.0.0.1:{port}",
            "According to the quarterly memo, what's the capital of Peru?",
        )
    finally:
        ws_server.close()
        await ws_server.wait_closed()

    text_frames = [f for f in frames if "text" in f]
    assert text_frames[0] == {"text": DOCS_NO_COVER_PREFIX, "meta": "prefix"}
    assert text_frames[1]["text"].strip() == "The capital of Peru is Lima."
    assert "meta" not in text_frames[1]
    assert "".join(f["text"] for f in text_frames).strip() == (
        DOCS_NO_COVER_PREFIX + "The capital of Peru is Lima."
    )
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["turn"] == done_frame["turn"]
    assert row["text"].startswith(DOCS_NO_COVER_PREFIX)


async def test_docs_miss_prefix_carries_meta_prefix_in_routed_web_fallback(tmp_path: Path) -> None:
    log_path = tmp_path / "miss_routed_web.jsonl"
    record: list[dict[str, Any]] = []

    async def _fc(params: Any) -> None:
        await params.result_callback([])

    async def _wa(params: Any) -> None:
        await params.result_callback("As per example.org: Lima is the capital of Peru.")

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
    tools = ToolsSchema(standard_tools=[fc, wa])
    router = _FakeRouter(RouterScript(lines="web_answer|capital of Peru"))

    async def _fake_json(system: str, content: str) -> str:
        del system, content
        return json.dumps({"status": "none"})

    agent = _ScriptedAgent("Unused.")
    server = AgentServer(
        agent,
        record,
        log_path,
        tools=tools,
        router=router,  # type: ignore[arg-type]
        docs_first=True,
        doc_json_request=_fake_json,
    )
    ws_server = await serve(server.handle, "127.0.0.1", 0)
    port = next(iter(ws_server.sockets)).getsockname()[1]
    try:
        frames, done_frame = await _send_turn(
            f"ws://127.0.0.1:{port}",
            "According to the quarterly memo, what's the capital of Peru?",
        )
    finally:
        ws_server.close()
        await ws_server.wait_closed()

    text_frames = [f for f in frames if "text" in f]
    assert text_frames[0] == {"text": DOCS_NO_COVER_PREFIX, "meta": "prefix"}
    assert all(f.get("meta") != "prefix" for f in text_frames[1:])
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["turn"] == done_frame["turn"]
    assert row["text"].startswith(DOCS_NO_COVER_PREFIX)
