"""A fact that can change is looked up and sourced, never answered from memory (2026-11-16)."""

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from voice_pipeline import call_policy
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
