"""What every Gemini request carries (2026-11-14): the [Now] line, the model's own lookup query,
one unrelated-turn count per turn, and "turn it up" after a temperature command."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from voice_pipeline import agent_tools, fast_path, question, venue
from voice_pipeline.agent_server import AgentServer, now_line
from voice_pipeline.agent_tools import _recorded
from voice_pipeline.chain_state import ChainState
from voice_pipeline.hedge import HedgedAgent, HedgeGate
from voice_pipeline.router import Routed

PROMPT = Path(__file__).resolve().parents[1] / "config/eval/system_agent.txt"
CASE_F = PROMPT.with_name("system_agent_case_f.txt")
NOW = "[Now: Saturday, November 14, 2026, 7:05 PM, Asia/Manila. Place: Cebu City.]"


class Agent:
    """Records what Gemini was sent; can add a tool call to the shared record, as a tool does."""

    def __init__(self, record: list[dict[str, Any]] | None = None, call: str = "") -> None:
        self.asked: list[list[dict[str, Any]]] = []
        self.record, self.call = record, call

    async def ask(self, messages: list[dict[str, Any]], *, on_text: Any, **_: Any) -> dict:
        self.asked.append(messages)
        if self.record is not None and self.call:
            self.record.append({"tool": self.call, "args": {}})
        await on_text("ok.")
        return {"status": "ok", "text": "ok.", "first_call_s": 0, "first_text_s": 0, "end_s": 0}

    async def cancel(self) -> None:
        return None


class Hedged(HedgedAgent):
    def __init__(self) -> None:
        super().__init__([], HedgeGate())
        self.asked: list[list[dict[str, Any]]] = []

    async def ask(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict:
        self.asked.append(messages)
        await kwargs["on_text"]("ok.")
        return {"status": "ok", "text": "ok.", "first_call_s": 0, "first_text_s": 0, "end_s": 0}

    async def cancel(self) -> None:
        return None


class Router:
    """Sends every turn to Gemini and remembers what it was asked."""

    def __init__(self) -> None:
        self.users: list[str] = []

    async def route(self, history: Any, user: str, names: Any) -> Routed:
        self.users.append(user)
        return Routed(reason="test: send to Gemini")


async def turn(server: AgentServer, *messages: tuple[str, str]) -> None:
    ws_server = await serve(server.handle, "127.0.0.1", 0)
    url = f"ws://127.0.0.1:{next(iter(ws_server.sockets)).getsockname()[1]}"
    sent = [{"role": role, "content": text} for role, text in messages]
    async with connect(url) as ws:
        await ws.send(json.dumps({"messages": sent}))
        while "done" not in json.loads(await ws.recv()):
            pass
    ws_server.close()


def server_for(tmp_path: Path, agent: Any, **kw: Any) -> AgentServer:
    server = AgentServer(agent, kw.pop("record", []), tmp_path / "log.jsonl", **kw)
    server.stamp = lambda: NOW
    return server


def test_the_now_line_names_the_date_the_time_and_the_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    monkeypatch.setattr(venue, "current", venue.Venue("Cebu City"))
    at = datetime(2026, 11, 14, 19, 5, tzinfo=ZoneInfo(fast_path.TIMEZONE))
    assert now_line(at) == NOW
    monkeypatch.setattr(venue, "current", venue.Venue("Cebu City", "Lahug", True))
    assert now_line(at).endswith("Place: Barangay Lahug, Cebu City.]")  # else the city (above)


async def test_every_gemini_request_ends_with_the_now_line_and_the_router_does_not_see_it(
    tmp_path: Path,
) -> None:
    agent, router = Agent(), Router()
    await turn(server_for(tmp_path, agent, router=router), ("user", "tell me a joke"))
    assert agent.asked == [[{"role": "user", "content": "tell me a joke\n" + NOW}]]
    assert router.users == ["tell me a joke"]


async def test_the_now_line_comes_after_the_chain_block(tmp_path: Path) -> None:
    chain = ChainState(tmp_path)
    chain.update("awaiting_name", subject="me")
    agent = Agent()
    await turn(server_for(tmp_path, agent, router=Router(), chain=chain), ("user", "Ana Cruz"))
    assert agent.asked[0][-1]["content"] == "Ana Cruz\n" + chain.block() + "\n" + NOW


async def test_a_hedged_request_carries_only_the_now_line(tmp_path: Path) -> None:
    agent = Hedged()
    await turn(server_for(tmp_path, agent, router=Router()), ("user", "tell me a joke"))
    assert agent.asked == [[{"role": "user", "content": "tell me a joke\n" + NOW}]]


async def test_without_a_stamp_nothing_changes(tmp_path: Path) -> None:
    agent = Agent()
    server = AgentServer(agent, [], tmp_path / "log.jsonl", router=Router())
    await turn(server, ("user", "tell me a joke"))
    assert agent.asked == [[{"role": "user", "content": "tell me a joke"}]]


async def test_one_routed_side_turn_counts_once_and_the_chain_survives_two(tmp_path: Path) -> None:
    chain = ChainState(tmp_path)
    chain.update("researching", subject="x")
    record: list[dict[str, Any]] = []
    agent = Agent(record, "weather")
    server = server_for(tmp_path, agent, record=record, chain=chain)
    # a weather question while researching: the model calls the tool once per turn
    agent_tools.CHAIN = chain
    schema = FunctionSchema(name="weather", description="d", properties={}, required=[],
                            handler=lambda p: p.result_callback("sunny"))  # fmt: skip

    async def sink(value: Any, *, properties: Any = None) -> None:
        return None

    wrapped = _recorded(schema, record)
    for expected in (1, 2):
        await wrapped.handler(SimpleNamespace(arguments={}, result_callback=sink))
        record.pop()  # the agent below adds the entry the server counts at the end of the turn
        await turn(server, ("user", "what is the weather"))
        assert chain.data["unrelated_turns"] == expected
    assert chain.active()


async def test_the_models_query_is_kept_when_the_question_points_back_and_only_then(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, Any]] = []

    async def web_answer(params: Any) -> None:
        sent.append(dict(params.arguments))
        await params.result_callback("x", properties=None)

    async def research(params: Any) -> None:
        raise AssertionError("web_research ran")

    monkeypatch.setattr(agent_tools, "_web_answer", [web_answer])
    monkeypatch.setattr(agent_tools, "CHAIN", None)
    schema = FunctionSchema(name="web_research", description="d", properties={}, required=[],
                            handler=research)  # fmt: skip
    wrapped = _recorded(schema, [])

    async def callback(result: Any, *, properties: Any = None) -> None:
        return None

    cases = (  # (what was said, the model's arguments, the query that is searched)
        (
            "Who's his wife?",
            {"query": "Who is Jensen Huang's wife?"},
            "Who is Jensen Huang's wife?",
        ),
        ("Who's his wife?", {"query": "  "}, "Who's his wife?"),
        ("Who's his wife?", {}, "Who's his wife?"),
        ("Who is the CEO of Nvidia?", {"query": "nvidia ceo"}, "Who is the CEO of Nvidia?"),
    )
    for said, arguments, expected in cases:
        question.current = said
        await wrapped.handler(SimpleNamespace(arguments=arguments, result_callback=callback))
        assert sent[-1] == {"query": expected}


@pytest.mark.parametrize(
    ("before", "turns_up"),
    [
        ({"intent": "TEMPERATURE", "slot": "24 degrees"}, True),
        ({"intent": "BRIGHTNESS", "slot": "40 percent"}, True),
        ({"intent": "VOLUME_UP"}, False),
        ({"intent": "LIGHT_ON"}, False),
        ({"tool": "weather", "args": {}}, False),
    ],
)
async def test_turn_it_up_after_a_temperature_or_brightness_command_goes_to_the_model(
    tmp_path: Path, before: dict[str, Any], turns_up: bool
) -> None:
    tools = SimpleNamespace(standard_tools=[SimpleNamespace(name="smart_home")])
    handled: list[Any] = []

    async def smart_home(params: Any) -> None:
        handled.append(params.arguments)
        await params.result_callback("ok")

    tools.standard_tools[0].handler = smart_home
    agent, router = Agent(), Router()
    record = [before]
    server = server_for(tmp_path, agent, record=record, tools=tools, router=router)
    server.tools = {"smart_home": tools.standard_tools[0]}
    await turn(
        server,
        ("user", "set it to 24 degrees"),
        ("assistant", "Temperature set to 24 degrees."),
        ("user", "turn it up"),
    )
    if turns_up:
        assert len(agent.asked) == 1 and not router.users and not handled
    else:
        assert not agent.asked and handled  # the fast path: the volume


async def test_turn_it_up_with_no_earlier_reply_stays_the_volume(tmp_path: Path) -> None:
    server = server_for(tmp_path, Agent(), record=[{"intent": "TEMPERATURE", "slot": "24"}])
    assert not server.refers_to_setting("turn it up", [{"role": "user", "content": "x"}], 1)
    assert not server.refers_to_setting("turn it on", [{"role": "assistant", "content": "x"}], 1)


def test_the_prompt_has_the_moved_rules_the_cannot_block_and_the_now_note() -> None:
    base, case_f = PROMPT.read_text(), CASE_F.read_text()
    for rule in ('"No, I mean ..." corrects your last action', '"Got it", "thanks", "mm-hmm"'):
        assert rule in base and rule not in case_f
    assert "You cannot:" in base and "[Now: ...]" in base and "never read it out" in base
    assert '"Turn it up" or "turn it down" right after a temperature' in base
