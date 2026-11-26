"""Pre-deploy 241: the docs gate is inverted from "skip non-questions" to "skip pleasantries".

Fix 235's router-branch gate (`not question_like and not document_cue`) skipped the
documents for every utterance that did not start with a question word, which dropped real
questions with spoken lead-ins ("Got it, what's the office Wi-Fi password?", "On which
day ...", "The ATSDR module mentions a law from 2010. Who signed it?"). 241 skips only a
closed-list pleasantry and sends everything else to the documents, exactly as before 235:

- `is_pleasantry`: closed list of greetings/acknowledgements (<= 6 words, every clause on
  the list); a pleasantry carrying a question word, a digit, or a document cue is False.
- `question_like`: leading fillers stripped first ("got it", "sorry I mean", ...), the
  start words gained "on which"/"in which"/.../"do you know", and a wh-word or auxiliary
  starting a clause after a comma/period (or sitting anywhere, for a garbled start) counts.
  Home commands and pure pleasantries stay False.
- `config/entities.json`: "Wi-Fi", "per diem", "ATSDR", "QCAS", "mayor race" plus their
  misheard spoken variants.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voice_pipeline.bench_fakes import _FakeRouter
from voice_pipeline.benchmark_cases import RouterScript
from voice_pipeline.doc_route import is_pleasantry, question_like

# --- table test: the session's utterances, what the gate must do with each ---

PLEASANTRIES = [
    # the ones the ticket names
    "Hello, good evening.",
    "Good evening.",
    "Say OK.",
    "Sorry.",
    "Thanks.",
    "Okay.",
    "Got it. Goodbye.",
    # the rest of the closed list
    "Good morning.",
    "Good afternoon.",
    "Good night.",
    "Thank you.",
    "Thank you very much.",
    "Thanks a lot.",
    "Ok.",
    "All right.",
    "Got it.",
    "Understood.",
    "Sure.",
    "Yes.",
    "Yeah.",
    "No.",
    "Nope.",
    "Right.",
    "Cool.",
    "Great.",
    "Nice.",
    "Goodbye.",
    "Bye.",
    "See you.",
    "Never mind.",
    "My bad.",
    "That's all.",
    "That is all.",
    "Nothing else.",
    "You're welcome.",
    "Stop.",
    "Hmm.",
    "Okay, thanks.",
    "Ok thanks.",
    "Hello.",
    "Hi.",
    "Hey.",
    "Have a nice day.",
]

REAL_QUESTIONS = [
    # the six the old gate dropped (Cause 2, docs/ME4-VOICE-SESSION-FINDINGS-2026-11-26.md)
    "Got it, what's the office Wi-Fi password?",
    "Sorry, I mean who won the mayor race?",
    "On which day do NYU Abu Dhabi classes follow a Monday schedule?",
    "The At SDR module mentions a law from 2010. Who signed it?",
    "Broadly speaking, what are all the documents about?",
    "This was California's commercial electricity price in August 2026.",
    # spoken lead-ins the new question_like must survive
    "Understood, who is Japan's prime minister?",
    "Can you try again?",
]


@pytest.mark.parametrize("said", PLEASANTRIES)
def test_pleasantries_are_pleasantries(said: str) -> None:
    assert is_pleasantry(said), said


@pytest.mark.parametrize("said", PLEASANTRIES)
def test_pleasantries_are_not_questions(said: str) -> None:
    """Pure pleasantries stay question_like False: the no-router branch still lets them
    fall through to Gemini, and the router branch skips the documents for them."""
    assert not question_like(said), said


@pytest.mark.parametrize("said", REAL_QUESTIONS)
def test_real_questions_are_not_pleasantries(said: str) -> None:
    assert not is_pleasantry(said), said


@pytest.mark.parametrize("said", REAL_QUESTIONS)
def test_real_questions_are_question_like(said: str) -> None:
    assert question_like(said), said


@pytest.mark.parametrize(
    "said",
    [
        # a repair phrase: is_pleasantry False is required, question_like may be either
        "Sorry, I mean T T S.",
        # a greeting that carries a question word
        "Okay, who won?",
        # a greeting that carries a digit
        "Sorry, I mean the law from 2010.",
        # a greeting that carries a document cue
        "Thanks, the document says so.",
        # statements
        "The meeting is at three.",
        "I have no idea.",
        "That was great.",
    ],
)
def test_non_pleasantry_utterances_are_not_pleasantries(said: str) -> None:
    assert not is_pleasantry(said), said


@pytest.mark.parametrize(
    "said",
    [
        "The meeting is at three.",  # an auxiliary anywhere: worth a docs check
        "I have no idea.",
        "That was great.",
        "The UK prime minister in my briefing notes, is that still right?",
        "Who won Miss Universe 2026?",
        "Sorry, I mean, the ATSDR module mentions a law from 2010. Who signed it?",
        "Tell me who won the mayor race.",
        "In which section is the deadline?",
        "For which course is this syllabus?",
        "How come the file is missing?",
        "Do you know where the schedule lives?",
    ],
)
def test_question_like_gains(said: str) -> None:
    assert question_like(said), said


@pytest.mark.parametrize(
    "said",
    [
        "Turn on the living room lights",  # home commands are not questions
        "Play some jazz",
        "Set a timer for ten minutes",
        "Please turn it down",
        "Tell me about California electricity prices.",  # a request without a question word
    ],
)
def test_commands_and_requests_stay_not_question_like(said: str) -> None:
    assert not question_like(said), said


def test_pleasantry_cannot_hide_a_question_digit_or_cue() -> None:
    assert not is_pleasantry("Okay, what's the password?")
    assert not is_pleasantry("Got it, the file from 2010.")
    assert not is_pleasantry("Good evening, check the document.")
    assert not is_pleasantry("")
    assert not is_pleasantry("   ")


# --- routed server: the lead-in question reaches the documents, the greeting does not ---


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
    """AgentServer with a fake router and the docs tools, style of test_predeploy_235."""

    def __init__(self, reply: str, router_lines: str = "") -> None:
        self.reply = reply
        self.router_lines = router_lines
        self.doc_calls: list[str] = []
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
                await params.result_callback("As per example.org: The capital is Lima.")

            wa = FunctionSchema(
                name="web_answer",
                description="Web answer",
                properties={"query": {"type": "string"}},
                required=["query"],
                handler=_wa,
            )
            wa = agent_tools._recorded(wa, record)

            tools = ToolsSchema(standard_tools=[fc, ds, de, wa])
            self.agent = _FakeAgent()
            router = _FakeRouter(RouterScript(lines=self.router_lines))

            async def _fake_json(system: str, content: str) -> str:
                return self.reply

            log = Path("/tmp/test_predeploy_241.log")
            log.parent.mkdir(parents=True, exist_ok=True)

            self.server = AgentServer(
                self.agent,
                record,
                log,
                tools,
                router,
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
            self._port = port
            self._url = f"ws://127.0.0.1:{port}"
            return self._url
        finally:
            agent_tools.CHAIN = prev_chain
            agent_tools.SCRATCHPAD = prev_scratchpad

    async def send(self, question: str) -> dict[str, Any]:
        from websockets.asyncio.client import connect

        async with connect(self._url) as ws:
            payload: dict[str, Any] = {"messages": [{"role": "user", "content": question}]}
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
async def test_lead_in_question_reaches_the_documents() -> None:
    """The 235 bug: "Got it, what's the office Wi-Fi password?" must reach the docs call
    (the old question_like gate skipped it); the miss keeps its prefix."""
    fx = _PredeployFixtures(json.dumps({"status": "none"}), router_lines="none|You're welcome.")
    await fx.build()
    try:
        row = await fx.send("Got it, what's the office Wi-Fi password?")
        assert fx.doc_calls == ["Got it, what's the office Wi-Fi password?"]
        assert row["route"] == "docs_miss"
        assert row["text"].startswith("Your documents don't cover that. ")
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_pleasanties_skip_the_documents() -> None:
    """Closed-list pleasantries: route "routed", zero docs calls, the router's own reply
    is spoken with no miss prefix."""
    fx = _PredeployFixtures(json.dumps({"status": "none"}), router_lines="none|You're welcome.")
    await fx.build()
    try:
        for said in ("Good evening.", "Got it. Goodbye.", "Thanks."):
            row = await fx.send(said)
            assert row["route"] == "routed", said
            assert row["text"].rstrip() == "You're welcome.", said
            assert not row["text"].startswith("Your documents don't cover that"), said
            assert row["reached_agent"] is False, said
        assert fx.doc_calls == []
    finally:
        await fx.close()
