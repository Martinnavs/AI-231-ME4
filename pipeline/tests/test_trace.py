"""Per-turn trace row tests (docs/TURN-TRACE.md): ids, spans, tokens, ledger, second look."""

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from conftest import LOOKED_UP, USER
from voice_pipeline import agent_tools, question, trace
from voice_pipeline.agent import Agent
from voice_pipeline.agent_server import AgentServer

DOCS = Path(__file__).resolve().parents[2] / "docs/TURN-TRACE.md"


async def test_a_hedged_turn_spans_match_the_calls_made(make_world: Any, serve_turn: Any) -> None:
    """The model wins the race: the router's lookup is cancelled, the model's runs."""
    world = make_world(model_wins=True)
    _, row = await serve_turn(world.server)
    spans = [(c["by"], c["tool"], c["status"]) for c in row["calls"]]
    assert spans == [("router", "web_answer", "cancelled"), ("model", "web_answer", "ok")]
    router_span, model_span = row["calls"]
    assert "result_sha" not in router_span and "result_text" not in router_span
    assert model_span["result_text"] == LOOKED_UP and model_span["result_chars"] == len(LOOKED_UP)
    assert re.fullmatch(r"[0-9a-f]{12}", model_span["result_sha"])
    assert model_span["sources"] == ["CNN", "FT"] and model_span["only_one_source"] is False
    assert 0 <= router_span["t0"] <= model_span["t0"] + 1 and model_span["s"] >= 0
    assert row["route"] == "routed" and row["tail"] == ["note", "now"]
    # the model's request won; both requests (loser too) are counted
    assert row["tokens"] == {"requests": 2, "prompt": 180, "completion": 0, "cached": 0}
    assert row["hedge_winner"] == 1


async def test_the_router_winning_leaves_one_router_span_and_the_ledger_slice(
    make_world: Any, serve_turn: Any
) -> None:
    world = make_world(model_wins=False)
    world.append({"month": "2026-11", "searches": 1, "by": "before the turn"})
    before = world.ledger.stat().st_size
    frames, row = await serve_turn(world.server)
    world.append({"month": "2026-11", "searches": 1, "by": "after the turn"})
    assert [(c["by"], c["tool"], c["status"]) for c in row["calls"]] == [
        ("router", "web_answer", "ok")
    ]
    assert row["hedge_winner"] == "router"
    ledger = row["ledger"]
    assert [r["by"] for r in ledger["rows"]] == ["router"]  # not the rows before or after
    assert ledger["from"] == before and ledger["to"] > before and ledger["n"] == 1
    assert ledger["path"] == str(world.ledger)
    assert frames[-1] == {"done": True, "turn": "20261118-120000-0001"}
    assert row["turn"] == "20261118-120000-0001" and row["run"] == "20261118-120000"
    _, second = await serve_turn(world.server)
    assert second["turn"] == "20261118-120000-0002"
    assert second["ledger"]["from"] > ledger["to"]  # the "after" row is outside both slices
    assert [r["by"] for r in second["ledger"]["rows"]] == ["router"]


async def test_the_existing_row_and_call_keys_are_kept(make_world: Any, serve_turn: Any) -> None:
    world = make_world(model_wins=False)
    _, row = await serve_turn(world.server)
    for key in ("time", "question", "calls", "cancelled", "status", "text", "first_call_s",
                "first_text_s", "end_s", "routed", "router_s", "router_lines"):  # fmt: skip
        assert key in row
    assert {"tool", "args", "by", "t0", "s", "status"} <= set(row["calls"][0])
    assert row["calls"][0]["args"] == {"query": "who is the ceo of nvidia?"}


def _recorded_lookup(
    record: list[dict[str, Any]], result: Any, name: str = "web_answer", boom: bool = False
) -> Any:
    async def inner(params: Any) -> None:
        if boom:
            raise RuntimeError("down")
        await params.result_callback(result)

    schema = FunctionSchema(name=name, description="d", properties={}, required=[], handler=inner)
    handler = agent_tools._recorded(schema, record).handler
    assert handler is not None
    return handler


def _params(call_id: str = "model-1", **arguments: Any) -> Any:
    async def sink(value: Any, *, properties: Any = None) -> None:
        return None

    return SimpleNamespace(arguments=arguments, result_callback=sink, tool_call_id=call_id,
                           function_name="x")  # fmt: skip


async def test_a_lookup_result_is_kept_to_2000_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_tools, "CHAIN", None)
    agent_tools.reset_turn_lookups()
    record: list[dict[str, Any]] = []
    await _recorded_lookup(record, "x" * 5000)(_params(query="q"))
    agent_tools.reset_turn_lookups()
    assert len(record[0]["result_text"]) == trace.RESULT_CAP == 2000
    assert record[0]["result_chars"] == 5000


async def test_other_tools_keep_no_result_text_and_a_second_lookup_is_joined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_tools, "CHAIN", None)
    agent_tools.reset_turn_lookups()
    question.current = "tell me about it"
    record: list[dict[str, Any]] = []
    await _recorded_lookup(record, "Paris.", name="clock")(_params())
    lookup = _recorded_lookup(record, LOOKED_UP)
    await lookup(_params(query="a"))
    await lookup(_params(query="b"))
    agent_tools.reset_turn_lookups()
    assert "result_text" not in record[0] and record[0]["result_chars"] == 6
    assert [c["status"] for c in record[1:]] == ["ok", "joined"]
    assert record[2]["cached"] is True


async def test_a_failed_and_a_raising_lookup_are_marked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_tools, "CHAIN", None)
    agent_tools.reset_turn_lookups()
    record: list[dict[str, Any]] = []
    await _recorded_lookup(record, "I couldn't find that.")(_params(query="q"))
    agent_tools.reset_turn_lookups()
    with pytest.raises(RuntimeError):
        await _recorded_lookup(record, "", boom=True)(_params(query="q"))
    agent_tools.reset_turn_lookups()
    assert [c["status"] for c in record] == ["failed", "failed"]
    assert "result_sha" not in record[1]


async def test_a_home_command_the_tool_records_itself_gets_a_span(tmp_path: Path) -> None:
    record: list[dict[str, Any]] = []

    async def handler(params: Any) -> None:
        record.append({"intent": "VOLUME", "slot": "up"})
        await params.result_callback({"ok": True})

    tools = SimpleNamespace(standard_tools=[SimpleNamespace(name="smart_home", handler=handler)])
    server = AgentServer(None, record, tmp_path / "l.jsonl", tools)
    trace.begin_turn()
    await server.call_tool("smart_home", {})
    assert record[0]["by"] == "router" and record[0]["status"] == "ok" and record[0]["s"] >= 0


@pytest.mark.parametrize(
    ("given", "kept"),
    [("abc", "abc"), ("s" * 64, "s" * 64), ("s" * 100, "s" * 64), (123, None), (None, None),
     (["a"], None), ({"a": 1}, None), ("", None), (True, None)],
)  # fmt: skip
def test_the_client_session_is_an_untrusted_string_capped_at_64(given: Any, kept: Any) -> None:
    assert trace.clean_session(given) == kept


async def test_the_session_reaches_the_row_only_when_it_is_a_string(
    make_world: Any, serve_turn: Any
) -> None:
    world = make_world(model_wins=False)
    _, row = await serve_turn(world.server, session="s" * 100)
    assert row["session"] == "s" * 64
    _, row = await serve_turn(world.server, session={"x": 1})
    assert "session" not in row
    _, row = await serve_turn(world.server)
    assert "session" not in row


class Verifying(AgentServer):
    def __init__(self, tmp: Path) -> None:
        super().__init__(None, [], tmp / "20261118-130000.jsonl", router=None)
        self.tools = {"web_verify": object()}
        self.ledger = tmp / "ledger.jsonl"
        self.checked = asyncio.Event()

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        with self.ledger.open("a") as sink:
            sink.write(json.dumps({"month": "2026-11", "searches": 1, "check": "DIFFERS"}) + "\n")
        return "VERDICT: SUPPORTED"


async def test_the_second_look_is_logged_apart_with_its_turn_id_and_its_own_ledger_rows(
    tmp_path: Path, serve_turn: Any
) -> None:
    class Agent_:
        async def ask(self, messages: Any, *, on_text: Any, **_: Any) -> dict[str, Any]:
            agent_tools._turn_lookups.append(LOOKED_UP)
            await on_text("Jensen Huang.")
            return {"status": "ok", "text": "Jensen Huang."}

        async def cancel(self) -> None:
            return None

    server = Verifying(tmp_path)
    server.agent = Agent_()
    agent_tools.reset_turn_lookups()
    frames, row = await serve_turn(server)
    await asyncio.gather(*server.checks)
    agent_tools.reset_turn_lookups()
    assert "second_opinion" not in row
    checks = tmp_path / "checks" / "20261118-130000.jsonl"
    check = json.loads(checks.read_text().splitlines()[-1])
    assert check["turn"] == frames[-1]["turn"] == "20261118-130000-0001"
    assert check["verdict"] == "VERDICT: SUPPORTED" and check["question"] == USER
    assert [r["check"] for r in check["ledger"]["rows"]] == ["DIFFERS"]
    assert len(server.log.read_text().splitlines()) == 1  # the turn file holds turns only


class FakeLLM(FrameProcessor):
    """Stands in for the Gemini service: answers every context with one sentence and its usage."""

    async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        usage = LLMTokenUsage(prompt_tokens=120, completion_tokens=7, total_tokens=127,
                              cache_read_input_tokens=100)  # fmt: skip
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(text="Hello there."))
        await self.push_frame(MetricsFrame(data=[LLMUsageMetricsData(processor="p", value=usage)]))
        await self.push_frame(LLMFullResponseEndFrame())


async def _first_text_and_usage(usage: bool) -> tuple[float, dict[str, int]]:
    agent = Agent(FakeLLM(), ToolsSchema(standard_tools=[]), usage=usage)  # type: ignore[arg-type]
    await agent.start()
    try:
        before = trace.usage_total(agent)
        result = await agent.ask([{"role": "user", "content": "hi"}], reset=True)
        return float(result["first_text_s"]), trace.usage_delta(before, trace.usage_total(agent))
    finally:
        await agent.stop()


async def test_a_metrics_frame_through_a_real_agent_gives_tokens_and_no_extra_wait() -> None:
    plain_s, _ = await _first_text_and_usage(False)
    traced_s, tokens = await _first_text_and_usage(True)
    assert tokens == {"requests": 1, "prompt": 120, "completion": 7, "cached": 100}
    assert tokens["prompt"] > 0
    assert traced_s < plain_s + 0.25  # usage bookkeeping adds no measurable wait


def test_the_usage_of_a_hedged_agent_sums_every_request_and_a_plain_one_counts_zero() -> None:
    def one(n: int) -> Any:
        usage = {"requests": n, "prompt": 10 * n, "completion": n, "cached": 0}
        return SimpleNamespace(tap=SimpleNamespace(usage=usage))

    assert trace.usage_total(SimpleNamespace(agents=[one(1), one(2)])) == {
        "requests": 3, "prompt": 30, "completion": 3, "cached": 0}  # fmt: skip
    assert trace.usage_total(object()) == {"requests": 0, "prompt": 0, "completion": 0,
                                           "cached": 0}  # fmt: skip


def test_the_ledger_slice_is_capped_and_survives_a_missing_or_torn_file(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    assert trace.ledger_size(path) == 0
    assert trace.ledger_slice(path, 0, 0)["rows"] == []
    path.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(12)) + "not json\n")
    sliced = trace.ledger_slice(path, 0, trace.ledger_size(path))
    assert sliced["n"] == 12 and [r["i"] for r in sliced["rows"]] == list(range(10))
    assert trace.ledger_slice(path, 5, 3)["rows"] == []  # a ledger that shrank


async def test_the_documented_example_has_the_keys_of_a_real_row(
    make_world: Any, serve_turn: Any
) -> None:
    text = DOCS.read_text()
    block = re.search(r"<!-- example-row -->\s*```json\n(.*?)\n```", text, re.S)
    assert block is not None
    example = json.loads(block.group(1))
    world = make_world(model_wins=True)
    _, row = await serve_turn(world.server, session="abc")
    assert set(example) == set(row)
    assert set(example["ledger"]) == set(row["ledger"])
    assert set(example["tokens"]) == set(row["tokens"])
    assert [set(c) for c in example["calls"]] == [set(c) for c in row["calls"]]
