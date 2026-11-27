"""Citation, page, and file_context trace logging tests (ticket 266)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from voice_pipeline import agent_tools, trace
from voice_pipeline.agent_server import AgentServer
from voice_pipeline.ask_documents import DocResult, ask_documents
from voice_pipeline.bench_fakes import _FakeRouter
from voice_pipeline.benchmark_cases import RouterScript
from voice_pipeline.doc_answer import passages_with_meta


def _passage(
    source: str,
    text: str,
    *,
    page: int | None = None,
    part: str = "chunk",
    passage_id: str | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "id": passage_id or f"{source}#0",
        "section": "",
        "page": page,
        "file_date": "2026-01-01",
        "text": text,
        "score": 1.0,
        "coverage": 1.0,
        "part": part,
    }


class _ScriptedAgent:
    def __init__(self, reply_text: str = "Unused.") -> None:
        self.reply_text = reply_text
        self.tap = SimpleNamespace(usage={"requests": 0, "prompt": 0, "completion": 0, "cached": 0})

    async def ask(
        self, messages: Any, *, reset: bool, on_text: Any = None, **_: Any
    ) -> dict[str, Any]:
        del messages, reset
        if on_text is not None:
            await on_text(self.reply_text)
        return {"status": "ok", "text": self.reply_text, "cancelled": False}

    def hide(self, names: frozenset[str]) -> None:
        del names


async def _send_turn(url: str, question: str) -> dict[str, Any]:
    async with connect(url) as ws:
        await ws.send(json.dumps({"messages": [{"role": "user", "content": question}]}))
        while True:
            frame = json.loads(await ws.recv())
            if frame.get("done") or frame.get("error"):
                return dict(frame)


async def test_pdf_fixture_resolves_cited_pages_and_file() -> None:
    # Include a #grep passage (page=None) containing the same line before the #chunk (page=3)
    # to mirror DbIndex.file_context output on large PDFs.
    raw = [
        _passage(
            "docs/annual_report.pdf",
            "L12: The operating margin reached 18 percent in Q3.",
            page=None,
            part="grep",
            passage_id="docs/annual_report.pdf#grep",
        ),
        _passage(
            "docs/annual_report.pdf",
            "Summary of annual revenue across divisions.",
            page=1,
            passage_id="docs/annual_report.pdf#0",
        ),
        _passage(
            "docs/annual_report.pdf",
            "The operating margin reached 18 percent in Q3.",
            page=3,
            passage_id="docs/annual_report.pdf#2",
        ),
    ]
    passages = passages_with_meta(raw)

    async def _ctx(q: str) -> list[dict[str, Any]]:
        del q
        return passages

    async def _req(system: str, content: str) -> str:
        del system, content
        return json.dumps(
            {
                "status": "answer",
                "answer": "18 percent.",
                "file": "docs/annual_report.pdf",
                "quotes": ["operating margin reached 18 percent in Q3"],
            }
        )

    res = await ask_documents("What was the Q3 operating margin?", context=_ctx, request=_req)
    assert res.status == "accepted"
    assert res.file == "docs/annual_report.pdf"
    assert res.cited_pages == [3]
    assert res.quote_files == ["docs/annual_report.pdf"]
    assert res.quotes == ["operating margin reached 18 percent in Q3"]


async def test_quotes_spanning_two_files_preserve_first_seen_order_and_deduplicate() -> None:
    raw = [
        _passage("alpha.pdf", "Alpha revenue was 10 million in 2025.", page=2),
        _passage("beta.pdf", "Beta revenue was 25 million in 2025.", page=4),
        _passage("beta.pdf", "Beta headcount grew to 120 engineers.", page=4),
    ]

    async def _ctx(q: str) -> list[dict[str, Any]]:
        del q
        return raw

    async def _req(system: str, content: str) -> str:
        del system, content
        return json.dumps(
            {
                "status": "answer",
                "answer": "35 million total.",
                "file": "alpha.pdf",
                "quotes": [
                    "Beta revenue was 25 million in 2025",
                    "Beta headcount grew to 120 engineers",
                    "Alpha revenue was 10 million in 2025",
                ],
                "calc": "25 + 10",
            }
        )

    res = await ask_documents("What was combined revenue in 2025?", context=_ctx, request=_req)
    assert res.status == "accepted"
    assert res.file == "alpha.pdf"
    assert res.quote_files == ["beta.pdf", "alpha.pdf"]
    assert res.cited_pages == [4, 2]


async def test_context_without_page_metadata_leaves_cited_pages_empty() -> None:
    raw = [
        _passage("notes.md", "The office Wi-Fi password is cedar-falcon-42.", page=None),
        {"source": "backup.txt", "text": "Backup note with no page field."},
    ]

    async def _ctx(q: str) -> list[dict[str, Any]]:
        del q
        return passages_with_meta(raw)

    async def _req(system: str, content: str) -> str:
        del system, content
        return json.dumps(
            {
                "status": "answer",
                "answer": "cedar-falcon-42.",
                "file": "notes.md",
                "quotes": ["office Wi-Fi password is cedar-falcon-42"],
            }
        )

    res = await ask_documents("What is the office Wi-Fi password?", context=_ctx, request=_req)
    assert res.status == "accepted"
    assert res.file == "notes.md"
    assert res.cited_pages == []
    assert res.quote_files == ["notes.md"]


def test_doc_result_defaults_new_citation_fields_empty() -> None:
    res = DocResult("none")
    assert res.cited_pages == []
    assert res.quote_files == []


@pytest.mark.parametrize("use_router", [False, True])
async def test_agent_server_docs_first_turn_log_contains_citation_fields(
    tmp_path: Path, use_router: bool
) -> None:
    log_path = tmp_path / f"docs_first_{use_router}.jsonl"
    record: list[dict[str, Any]] = []
    passages = [
        _passage(
            "handbook.pdf",
            "Employees receive 20 paid vacation days each year.",
            page=3,
            passage_id="handbook.pdf#2",
        ),
    ]

    async def _fc(params: Any) -> None:
        await params.result_callback(passages)

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
        return json.dumps(
            {
                "status": "answer",
                "answer": "20 paid vacation days.",
                "file": "handbook.pdf",
                "quotes": ["Employees receive 20 paid vacation days each year"],
            }
        )

    router = _FakeRouter(RouterScript(lines="none|")) if use_router else None
    server = AgentServer(
        _ScriptedAgent(),
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
        done_frame = await _send_turn(
            f"ws://127.0.0.1:{port}",
            "How many paid vacation days do employees receive?",
        )
    finally:
        ws_server.close()
        await ws_server.wait_closed()

    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert done_frame["done"] is True
    assert row["route"] == "docs"
    assert row["docs"] == "accepted"
    assert row["status"] == "ok"
    assert row["sources"] == ["handbook.pdf"]
    assert row["doc_calls"] == 1
    assert "doc_s" in row and "fallback" in row
    assert row["cited_file"] == "handbook.pdf"
    assert row["cited_pages"] == [3]
    assert row["quote_files"] == ["handbook.pdf"]
    assert row["quotes"] == ["Employees receive 20 paid vacation days each year"]
    assert row["calls"][0]["tool"] == "file_context"
    assert "result_text" in row["calls"][0]


@pytest.mark.parametrize("tool_name", ["file_context", "docs_catalog", "docs_enumerate"])
async def test_doc_trace_tools_keep_result_text_truncated_to_2000_chars(
    monkeypatch: pytest.MonkeyPatch, tool_name: str
) -> None:
    monkeypatch.setattr(agent_tools, "CHAIN", None)
    agent_tools.reset_turn_lookups()
    record: list[dict[str, Any]] = []
    payload = [{"source": "big.pdf", "text": "x" * 5000}]

    async def inner(params: Any) -> None:
        await params.result_callback(payload)

    schema = FunctionSchema(
        name=tool_name, description="d", properties={}, required=[], handler=inner
    )
    handler = agent_tools._recorded(schema, record).handler
    assert handler is not None

    async def sink(value: Any, *, properties: Any = None) -> None:
        del value, properties

    params = SimpleNamespace(
        arguments={"query": "q"},
        result_callback=sink,
        tool_call_id="router",
        function_name=tool_name,
    )
    await handler(params)  # type: ignore[arg-type]
    agent_tools.reset_turn_lookups()

    full_text = trace.text_of(payload)
    assert len(full_text) > 2000
    assert record[0]["result_chars"] == len(full_text)
    assert len(record[0]["result_text"]) == trace.RESULT_CAP == 2000
    assert record[0]["result_text"] == full_text[:2000]
