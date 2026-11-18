"""A fact that can change is looked up and sourced, never answered from memory (2026-11-16)."""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from voice_pipeline import agent_tools, call_policy, question
from voice_pipeline.agent_server import AgentServer
from voice_pipeline.router import Routed

ROOT_FILE = Path(__file__).resolve().parents[2] / "src/voice_agent/mcp_servers/gemini_grounding.py"


@pytest.mark.parametrize(
    "text",
    ["Who's the CEO of Nvidia?", "Who is the mayor here", "What is the latest version of Python",
     "Who won the last NBA championship", "How much is the price of rice now"],
)  # fmt: skip
def test_a_question_about_something_that_changes(text: str) -> None:
    assert call_policy.changeable_question(text)


@pytest.mark.parametrize(
    "text",
    ["What is the capital of France?", "Who wrote Noli Me Tangere", "Play the new album",
     "Turn on the lights now", "Tell me a joke", "Who painted the Mona Lisa"],
)  # fmt: skip
def test_settled_facts_and_commands_are_not(text: str) -> None:
    assert not call_policy.changeable_question(text)


def test_the_word_list_is_the_one_the_grounding_server_uses() -> None:
    """The two packages cannot import each other: read the other's pattern out of its source."""
    tree = ast.parse(ROOT_FILE.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "CHANGEABLE"
            and isinstance(node.value, ast.Call)
        ):
            assert ast.literal_eval(node.value.args[0]) == call_policy.CHANGEABLE.pattern
            return
    raise AssertionError("CHANGEABLE not found in gemini_grounding.py")


class Agent:
    def __init__(self, record: list[dict[str, Any]], call: str = "") -> None:
        self.asked: list[list[dict[str, Any]]] = []
        self.record, self.call = record, call

    async def ask(self, messages: list[dict[str, Any]], *, on_text: Any, **_: Any) -> dict:
        self.asked.append(messages)
        if self.call:
            self.record.append({"tool": self.call, "args": {}})
        await on_text("Jensen Huang.")
        return {"status": "ok", "text": "Jensen Huang.", "first_call_s": 0, "first_text_s": 0,
                "end_s": 0}  # fmt: skip

    async def cancel(self) -> None:
        return None


class Router:
    async def route(self, history: Any, user: str, names: Any) -> Routed:
        return Routed(reason="test: send to Gemini")


async def turn(server: AgentServer, text: str) -> None:
    ws_server = await serve(server.handle, "127.0.0.1", 0)
    url = f"ws://127.0.0.1:{next(iter(ws_server.sockets)).getsockname()[1]}"
    async with connect(url) as ws:
        await ws.send(json.dumps({"messages": [{"role": "user", "content": text}]}))
        while "done" not in json.loads(await ws.recv()):
            pass
    ws_server.close()


async def test_a_changeable_question_sent_to_the_model_carries_the_note(tmp_path: Path) -> None:
    record: list[dict[str, Any]] = []
    agent = Agent(record)
    await turn(AgentServer(agent, record, tmp_path / "l.jsonl", router=Router()),
               "Who's the CEO of Nvidia?")  # fmt: skip
    assert (
        agent.asked[0][-1]["content"] == "Who's the CEO of Nvidia?\n" + call_policy.CHANGEABLE_NOTE
    )
    asked = AgentServer(Agent(record), record, tmp_path / "m.jsonl", router=Router())
    plain = asked.agent
    await turn(asked, "tell me a joke")
    assert plain.asked[0] == [{"role": "user", "content": "tell me a joke"}]


async def test_an_answer_with_no_lookup_is_flagged_in_the_log(tmp_path: Path) -> None:
    for call, flagged in (("", True), ("web_answer", False)):
        record: list[dict[str, Any]] = []
        log = tmp_path / f"{call or 'none'}.jsonl"
        await turn(AgentServer(Agent(record, call), record, log, router=Router()),
                   "Who's the CEO of Nvidia?")  # fmt: skip
        assert json.loads(log.read_text().splitlines()[-1]).get("unchecked", False) is flagged


LOOKED_UP = "Jensen Huang is the CEO of Nvidia. Sources: CNN and FT."


@pytest.mark.parametrize(
    ("spoken", "cited"),
    [
        ("The CEO of Nvidia is Jensen Huang.", "Sources: CNN and FT."),
        ("According to CNN, it is Jensen Huang.", ""),
        ("Jensen Huang. Source: Nvidia.", ""),
        ("It is Jensen Huang, FT says.", ""),
        ("Per CNN: Jensen Huang.", ""),
    ],
)
def test_the_source_sentence_is_added_only_when_the_reply_names_none(
    spoken: str, cited: str
) -> None:
    agent_tools.reset_turn_lookups()
    agent_tools._turn_lookups.append(LOOKED_UP)
    assert agent_tools.turn_source_sentence(spoken) == cited
    agent_tools.reset_turn_lookups()


def test_a_publisher_that_is_also_in_the_question_is_not_a_citation() -> None:
    agent_tools.reset_turn_lookups()
    agent_tools._turn_lookups.append("Jensen Huang is the CEO. Sources: Nvidia and SEC.")
    spoken = "Checking the CEO of Nvidia. The CEO of Nvidia is Jensen Huang."
    assert agent_tools.turn_source_sentence(spoken) == "Sources: Nvidia and SEC."
    assert agent_tools.turn_source_sentence("Nvidia says it is Jensen Huang.") == ""
    agent_tools.reset_turn_lookups()


def test_no_lookup_or_a_lookup_without_a_source_adds_nothing() -> None:
    agent_tools.reset_turn_lookups()
    assert agent_tools.turn_source_sentence("Hello.") == ""
    agent_tools._turn_lookups.append("Paris is the capital of France.")
    assert agent_tools.turn_source_sentence("Paris.") == ""
    agent_tools.reset_turn_lookups()
    agent_tools._turn_lookups.append("Mayor Garcia. Source: Wikipedia. Only one source found.")
    assert (
        agent_tools.turn_source_sentence("Garcia is mayor.")
        == "Source: Wikipedia. Only one source found."
    )
    agent_tools.reset_turn_lookups()


class LookingUp(Agent):
    """Does what the front model did live: gets a lookup with its source and rewords it."""

    async def ask(self, messages: list[dict[str, Any]], *, on_text: Any, **_: Any) -> dict:
        agent_tools._turn_lookups.append(LOOKED_UP)
        await on_text("The CEO of Nvidia is Jensen Huang.")
        return {"status": "ok", "text": "The CEO of Nvidia is Jensen Huang.", "first_call_s": 0,
                "first_text_s": 0, "end_s": 0}  # fmt: skip


async def test_the_server_speaks_the_source_the_model_dropped(tmp_path: Path) -> None:
    server = AgentServer(LookingUp([]), [], tmp_path / "l.jsonl", router=Router())
    ws_server = await serve(server.handle, "127.0.0.1", 0)
    url = f"ws://127.0.0.1:{next(iter(ws_server.sockets)).getsockname()[1]}"
    frames: list[dict[str, Any]] = []
    async with connect(url) as ws:
        await ws.send(json.dumps({"messages": [{"role": "user", "content": "Who's the CEO?"}]}))
        while "done" not in (frame := json.loads(await ws.recv())):
            frames.append(frame)
    ws_server.close()
    assert [str(f.get("text")).strip() for f in frames] == [
        "The CEO of Nvidia is Jensen Huang.",
        "Sources: CNN and FT.",
    ]
    row = json.loads((tmp_path / "l.jsonl").read_text().splitlines()[-1])
    assert row["text"].endswith("Sources: CNN and FT.") and row["cited_by_server"] is True


async def test_a_lookup_cancelled_mid_flight_does_not_leave_the_waiting_call_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The router's lookup and the model's run together; when the model wins the hedge race the
    router's is cancelled. The model's call was handed "(already looked up this turn) " and
    nothing, and answered from memory. It must run the lookup itself."""
    import asyncio  # noqa: PLC0415

    monkeypatch.setattr(agent_tools, "CHAIN", None)
    calls: list[str] = []
    got: list[Any] = []

    async def inner(params: Any) -> None:
        calls.append(params.tool_call_id)
        if params.tool_call_id == "router":
            await asyncio.sleep(5)  # cancelled before it answers
        await params.result_callback("Lip-Bu Tan is the CEO of Intel. Sources: Intel and CNBC.")

    async def sink(value: Any, *, properties: Any = None) -> None:
        got.append(value)

    schema = FunctionSchema(name="web_answer", description="d", properties={}, required=[],
                            handler=inner)  # fmt: skip
    handler = agent_tools._recorded(schema, []).handler
    assert handler is not None
    agent_tools.reset_turn_lookups()
    question.current = "Who is the CEO of Intel?"

    def params(call_id: str) -> Any:
        return SimpleNamespace(arguments={"query": "CEO of Intel"}, result_callback=sink,
                               tool_call_id=call_id, function_name="web_answer")  # fmt: skip

    router = asyncio.create_task(handler(params("router")))
    await asyncio.sleep(0.05)
    model = asyncio.create_task(handler(params("model")))
    await asyncio.sleep(0.05)
    router.cancel()
    await model
    assert calls == ["router", "model"]  # the model's call ran its own lookup
    assert (
        got and "Lip-Bu Tan" in str(got[-1]) and not str(got[-1]).startswith("(already looked up")
    )
    assert agent_tools.turn_source_sentence("Lip-Bu Tan.") == "Sources: Intel and CNBC."
    agent_tools.reset_turn_lookups()


class Verifying(AgentServer):
    """An agent server whose web_verify answers as scripted."""

    def __init__(self, tmp: Path, answer: str) -> None:
        super().__init__(LookingUp([]), [], tmp / "l.jsonl", router=Router())
        self.answer, self.verified = answer, []
        self.tools = {"web_verify": object()}

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        self.verified.append((name, args))
        return self.answer


async def _settled(server: Verifying) -> None:
    await turn(server, "Who's the CEO of Nvidia?")
    await asyncio.gather(*server.checks)


async def test_a_reply_the_second_look_differs_from_becomes_a_notice(tmp_path: Path) -> None:
    server = Verifying(
        tmp_path,
        "VERDICT: DIFFERS\nCORRECTION: Colette Kress is the interim CEO. Sources: SEC.",
    )
    await _settled(server)
    assert server.verified == [
        ("web_verify", {"question": "Who's the CEO of Nvidia?",
                        "claim": "Jensen Huang is the CEO of Nvidia."})
    ]  # fmt: skip
    assert [n["text"] for n in server.unacked.values()] == [
        "A second check found something different. Colette Kress is the interim CEO. Sources: SEC."
    ]
    check = json.loads((tmp_path / "checks" / "l.jsonl").read_text().splitlines()[-1])
    assert check["verdict"] == "VERDICT: DIFFERS" and check["turn"] == "l-0001"
    assert "second_opinion" not in (tmp_path / "l.jsonl").read_text()


@pytest.mark.parametrize("answer", ["VERDICT: SUPPORTED", "VERDICT: UNVERIFIED",
                                    "SKIP: the allowance of 100 checks is used", ""])  # fmt: skip
async def test_anything_but_a_contradiction_is_silent(tmp_path: Path, answer: str) -> None:
    server = Verifying(tmp_path, answer)
    await _settled(server)
    assert not server.unacked and server.verified  # asked, nothing said


async def test_no_check_without_the_tool_or_without_a_sourced_lookup(tmp_path: Path) -> None:
    plain = AgentServer(LookingUp([]), [], tmp_path / "a.jsonl", router=Router())
    await turn(plain, "Who's the CEO of Nvidia?")
    assert not plain.checks and not plain.unacked  # no web_verify tool: nothing to ask

    server = Verifying(tmp_path, "VERDICT: DIFFERS\nCORRECTION: x.")
    server.agent = Agent([])  # a turn with no lookup: no claim, no check
    await turn(server, "tell me a joke")
    assert not server.verified


async def test_the_verify_call_is_not_recorded_as_a_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    async def inner(params: Any) -> None:
        seen.append(dict(params.arguments))
        await params.result_callback("VERDICT: SUPPORTED")

    async def sink(value: Any, *, properties: Any = None) -> None:
        return None

    record: list[dict[str, Any]] = []
    schema = FunctionSchema(name="web_verify", description="d", properties={}, required=[],
                            handler=inner)  # fmt: skip
    handler = agent_tools._recorded(schema, record).handler
    assert handler is not None
    await handler(SimpleNamespace(arguments={"question": "q", "claim": "c", "say": "x"},
                                  result_callback=sink, tool_call_id="router"))  # fmt: skip
    assert record == [] and seen == [{"question": "q", "claim": "c"}]
    assert "web_verify" in agent_tools.MODEL_HIDDEN  # the model never sees it


def test_the_verify_allowance_reaches_the_child_only_when_set() -> None:
    assert "--verify-max" not in agent_tools.tavily_command({})
    assert agent_tools.tavily_command({"VOICE_AGENT_VERIFY_MAX": "100"})[-2:] == [
        "--verify-max",
        "100",
    ]


@pytest.mark.parametrize(
    ("lookup", "spoken", "cited"),
    [
        ("As per CNN and FT, Jensen Huang is the CEO.", "The CEO is Jensen Huang.",
         "Sources: CNN and FT."),
        ("As per CNN and FT, Jensen Huang is the CEO.", "As per CNN, Jensen Huang is the CEO.", ""),
        ("As per Wikipedia (the only source I found), Garcia is mayor.", "Garcia is mayor.",
         "Source: Wikipedia. Only one source found."),
        ("Researched on 2026-11-16 (no links).\nHuang resigned.\nPages used: sec.gov; cnn.com\n"
         "Sources: SEC and CNN.", "Huang resigned on November 12.", "Sources: SEC and CNN."),
        ("Researched on 2026-11-16 (no links).\nHuang resigned, SEC says.\nSources: SEC and CNN.",
         "Huang resigned, SEC says.", ""),
    ],
)  # fmt: skip
def test_the_lead_in_and_the_research_sources_line_are_both_read(
    lookup: str, spoken: str, cited: str
) -> None:
    agent_tools.reset_turn_lookups()
    agent_tools._turn_lookups.append(lookup)
    assert agent_tools.turn_source_sentence(spoken) == cited
    agent_tools.reset_turn_lookups()


def test_the_claim_to_check_has_no_source_marks() -> None:
    agent_tools.reset_turn_lookups()
    agent_tools._turn_lookups.append("As per CNN and FT, Jensen Huang is the CEO of Nvidia.")
    assert agent_tools.turn_claim() == "Jensen Huang is the CEO of Nvidia."
    agent_tools.reset_turn_lookups()
    agent_tools._turn_lookups.append("Jensen Huang is the CEO of Nvidia. Sources: CNN and FT.")
    assert agent_tools.turn_claim() == "Jensen Huang is the CEO of Nvidia."
    agent_tools.reset_turn_lookups()


async def test_a_hedged_changeable_turn_carries_the_note_before_the_now_line(
    make_world: Any, serve_turn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2a: both hedged requests start before the router decides, and `stamped()` is what they
    get, so the note must be in what the (fake) agents actually received, ahead of the date."""
    from conftest import Scripted

    seen: list[str] = []
    ask = Scripted.ask

    async def spy(self: Any, messages: Any, **kw: Any) -> dict[str, Any]:
        seen.append(str(messages[-1]["content"]))
        return await ask(self, messages, **kw)

    monkeypatch.setattr(Scripted, "ask", spy)
    world = make_world(model_wins=True)
    _, row = await serve_turn(world.server)  # "Who is the CEO of Nvidia?"
    assert len(seen) == 2
    for content in seen:
        assert content.endswith("\n" + call_policy.CHANGEABLE_NOTE + "\n[Now: test]")
    assert row["tail"] == ["note", "now"]


async def test_a_hedged_unchanging_turn_is_stamped_without_the_note(
    make_world: Any, serve_turn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conftest import Scripted

    seen: list[str] = []
    ask = Scripted.ask

    async def spy(self: Any, messages: Any, **kw: Any) -> dict[str, Any]:
        seen.append(str(messages[-1]["content"]))
        return await ask(self, messages, **kw)

    monkeypatch.setattr(Scripted, "ask", spy)
    world = make_world(model_wins=True)
    _, row = await serve_turn(world.server, "Thanks, that is all.")
    assert seen and all(c == "Thanks, that is all.\n[Now: test]" for c in seen)
    assert row["tail"] == ["now"]


def test_stamped_adds_the_note_even_without_a_clock(make_world: Any) -> None:
    server = make_world(model_wins=True).server
    server.stamp = None
    changeable = [{"role": "user", "content": "Who is the CEO of Nvidia?"}]
    out = server.stamped(changeable)
    assert out[-1]["content"] == "Who is the CEO of Nvidia?\n" + call_policy.CHANGEABLE_NOTE
    assert server.tail == ["note"]
    settled = [{"role": "user", "content": "Thanks, that is all."}]
    assert server.stamped(settled) == settled


async def test_a_hedged_router_win_leaves_the_router_answer_and_records_what_the_models_got(
    make_world: Any, serve_turn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conftest import Scripted

    seen: list[str] = []
    ask = Scripted.ask

    async def spy(self: Any, messages: Any, **kw: Any) -> dict[str, Any]:
        seen.append(str(messages[-1]["content"]))
        return await ask(self, messages, **kw)

    monkeypatch.setattr(Scripted, "ask", spy)
    world = make_world(model_wins=False)
    _, row = await serve_turn(world.server)
    assert row["hedge_winner"] == "router"
    assert [(c["by"], c["status"]) for c in row["calls"]] == [("router", "ok")]
    assert seen and all(
        c.endswith("\n" + call_policy.CHANGEABLE_NOTE + "\n[Now: test]") for c in seen
    )
    assert row["tail"] == ["note", "now"]


async def test_the_tail_is_per_turn_not_carried_over_from_a_changeable_turn(
    make_world: Any, serve_turn: Any
) -> None:
    world = make_world(model_wins=True)
    _, first = await serve_turn(world.server)
    _, second = await serve_turn(world.server, "Thanks, that is all.")
    assert first["tail"] == ["note", "now"]
    assert second["tail"] == ["now"]


@pytest.mark.parametrize("text", ["Who is the CEO of Nvidia?", "Thanks, that is all."])
@pytest.mark.parametrize("stamp", [True, False])
def test_the_hedged_suffix_is_the_same_as_steers_without_a_chain(
    make_world: Any, text: str, stamp: bool
) -> None:
    server = make_world(model_wins=True).server
    if not stamp:
        server.stamp = None
    messages = [{"role": "user", "content": text}]
    steered = server.steer(messages, case_f=False)
    steer_tail = list(server.tail)
    assert server.stamped(messages) == steered
    assert server.tail == steer_tail
    if stamp and text.startswith("Who"):
        assert steered[-1]["content"].index(call_policy.CHANGEABLE_NOTE) < steered[-1][
            "content"
        ].index("[Now:")
        assert steered[-1]["content"].endswith("[Now: test]")


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "Who is the CEO of Nvidia?"}],
        [{"role": "user", "content": [{"type": "text", "text": "Who is the CEO of Nvidia?"}]}],
    ],
)
def test_stamped_leaves_what_it_cannot_suffix_alone(make_world: Any, messages: list[Any]) -> None:
    server = make_world(model_wins=True).server
    server.tail = ["sentinel"]
    before = json.dumps(messages)
    assert server.stamped(messages) == messages
    assert server.tail == ["sentinel"]
    assert json.dumps(messages) == before


def test_stamped_does_not_mutate_the_callers_messages(make_world: Any) -> None:
    server = make_world(model_wins=True).server
    messages = [{"role": "user", "content": "Who is the CEO of Nvidia?"}]
    out = server.stamped(messages)
    assert messages == [{"role": "user", "content": "Who is the CEO of Nvidia?"}]
    assert out is not messages and out[-1] is not messages[-1]
