"""Pre-deploy 247: a compound document+web question is split, each half is answered in
its own lane, and the two are combined.

Before 247 such a question was answered by one lane only: the document check saw the
whole compound question, missed, and the web answered only its half (the document half
was lost). Now a docs-miss turn that references the user's documents pays for ONE call
to the document lane's own cheap JSON model (the split); the document half is answered
in the checked document lane (its checks stand) and the web half through the web lookup
(ledger, hedge and the directness check apply); the halves are combined and spoken.

Fail-open: no split for turns without a document reference, at most one split per turn,
and a split error, timeout, 'single' verdict, unusable half or an exceeded budget leaves
the existing docs-miss answer byte-for-byte.

The tests use synthetic names so they prove the behaviour on an unseen corpus.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voice_pipeline import compound, doc_route
from voice_pipeline.agent_server import AgentServer
from voice_pipeline.ask_documents import DocResult


def _enforce_schema(reply: str, schema: dict[str, Any] | None) -> str:
    """Like the production transport's structured output: a reply that carries a key the
    schema it was given lacks cannot happen, so the scripted reply is checked the same way
    (ticket 258, R1-1: a schema-ignoring fake is what masked the hardcoded REPLY_SCHEMA)."""
    if schema is None:
        return reply
    try:
        data = json.loads(reply)
    except ValueError:
        return reply  # unparsable: not a schema violation; the caller's fail-open case
    if isinstance(data, dict) and set(data) - set(schema["properties"]):
        raise AssertionError("the scripted reply carries keys outside the schema it was given")
    return reply

# --- the synthetic corpus: two files with invented names ---
FILE_A = "country_sheet_2031.csv"
FILE_B = "briefing_notes_2031.md"
PASSAGE_A = "Zorblat index: 12.5\nSmallest: Norbria, 12,000 people"
PASSAGE_B = "Bureau chief: Keira Moss\nPrepared for the quarterly review"
CATALOG = [
    {"folder": "a", "rel_key": "a/" + FILE_A, "line": f"a/{FILE_A} | Country sheet | csv"},
    {"folder": "b", "rel_key": "b/" + FILE_B, "line": f"b/{FILE_B} | Briefing notes | markdown"},
]
# the web lookup's own honest line (evaluators.HEDGE); a reply ending this way did not
# actually check the web
HEDGE = "I couldn't confirm that with a search, so I won't guess."

# --- the scripted splits (the patterns are what is under test, not the words) ---
SPLIT_THEN_WEB = json.dumps(
    {
        "kind": "doc_then_web",
        "doc_question": "Which country does my country sheet list as the smallest?",
        "web_question": "What is the capital of {doc}?",
    }
)
SPLIT_AND_WEB = json.dumps(
    {
        "kind": "doc_and_web",
        "doc_question": "What is the Zorblat index in my table?",
        "web_question": "What is today's exchange rate for the krona?",
    }
)
SPLIT_VS_WEB = json.dumps(
    {
        "kind": "doc_vs_web",
        "doc_question": "Who does my briefing notes name as the bureau chief?",
        "web_question": "Who is the current chief of the Zorblat bureau?",
    }
)
SPLIT_SINGLE = json.dumps({"kind": "single", "doc_question": None, "web_question": None})
SPLIT_NO_KIND = json.dumps({"status": "none"})  # a reply without a kind: unparsable for us

# --- the scripted document-lane replies (a quote in the passage passes the checks) ---
DOC_A = json.dumps(
    {"status": "answer", "answer": "Norbria", "file": FILE_A, "quotes": ["Smallest: Norbria"]}
)
DOC_INDEX = json.dumps(
    {"status": "answer", "answer": "12.5", "file": FILE_A, "quotes": ["Zorblat index: 12.5"]}
)
DOC_B = json.dumps(
    {
        "status": "answer",
        "answer": "Keira Moss",
        "file": FILE_B,
        "quotes": ["Bureau chief: Keira Moss"],
    }
)
DOC_BAD_QUOTE = json.dumps(
    {
        "status": "answer",
        "answer": "12.5",
        "file": FILE_A,
        "quotes": ["a phrase that is not in the passage at all"],
    }
)

Q_THEN_WEB = "What's the capital of the smallest country in my country sheet?"
Q_AND_WEB = "The Zorblat index is in my table. What's today's exchange rate for the krona?"
Q_VS_WEB = "The bureau chief in my briefing notes, is that still right?"
Q_NO_DOC_REF = "What's today's exchange rate for the krona?"


class _Rec:
    def __init__(self) -> None:
        self.split_calls: list[str] = []
        self.doc_lane_calls: list[str] = []
        self.web_calls: list[str] = []
        self.ctx_calls: list[str] = []


class _MissAgent:
    def __init__(self, text: str) -> None:
        self.text = text
        self.reached = False
        self.tap = SimpleNamespace(usage={"requests": 0, "prompt": 0, "completion": 0, "cached": 0})

    async def ask(
        self, messages: Any, *, reset: bool, on_text: Any = None, **_: Any
    ) -> dict[str, Any]:
        self.reached = True
        if on_text:
            await on_text(self.text)
        return {"status": "ok", "text": self.text, "cancelled": False}

    def hide(self, names: frozenset[str]) -> None:
        pass


def make_server(
    tmp_path: Path,
    *,
    split_reply: str,
    doc_reply: str,
    web_reply: str,
    agent_text: str = "The agent answers.",
    split_sleep_s: float = 0.0,
    split_error: bool = False,
    ctx_sleep_s: float = 0.0,
) -> tuple[AgentServer, _Rec]:
    """An AgentServer with scripted document and web tools. The same fake
    ``doc_json_request`` serves the split call and the document lane, told apart by the
    system prompt (the split gets ``compound.SYSTEM``)."""
    rec = _Rec()

    async def fc(params: Any) -> None:
        rec.ctx_calls.append(params.arguments.get("query", ""))
        if ctx_sleep_s:
            await asyncio.sleep(ctx_sleep_s)
        await params.result_callback(
            [
                {"source": "a/" + FILE_A, "text": PASSAGE_A, "part": "whole"},
                {"source": "b/" + FILE_B, "text": PASSAGE_B, "part": "whole"},
            ]
        )

    async def ds(params: Any) -> None:
        await params.result_callback(
            {
                "files": [
                    {"name": "a/" + FILE_A, "status": "ok"},
                    {"name": "b/" + FILE_B, "status": "ok"},
                ],
                "failures": [],
            }
        )

    async def dc(params: Any) -> None:
        await params.result_callback(list(CATALOG))

    async def de(params: Any) -> None:
        await params.result_callback([{"rel_key": "a/" + FILE_A}, {"rel_key": "b/" + FILE_B}])

    async def wa(params: Any) -> None:
        rec.web_calls.append(params.arguments.get("query", ""))
        await params.result_callback(web_reply)

    async def fake_json(
        system: str, content: str, schema: dict[str, Any] | None = None
    ) -> str:
        if system == compound.SYSTEM:
            assert schema is compound.SPLIT_SCHEMA  # R1-1: the split passes its own schema
            rec.split_calls.append(content)
            if split_error:
                raise RuntimeError("the split model is down")
            if split_sleep_s:
                await asyncio.sleep(split_sleep_s)
            return _enforce_schema(split_reply, schema)
        rec.doc_lane_calls.append(content)
        return doc_reply

    tools = SimpleNamespace(
        standard_tools=[
            SimpleNamespace(name=n, handler=h)
            for n, h in (
                ("file_context", fc),
                ("docs_status", ds),
                ("docs_catalog", dc),
                ("docs_enumerate", de),
                ("web_answer", wa),
            )
        ]
    )
    server = AgentServer(
        _MissAgent(agent_text),
        [],
        tmp_path / "247.jsonl",
        tools=tools,
        doc_json_request=fake_json,
        docs_first=True,
    )
    doc_route.set_cached_files(server, [c["rel_key"] for c in CATALOG])
    return server, rec


async def miss_turn(
    server: AgentServer,
    said: str,
    status: str,
    web_fallback: Any = None,
) -> dict[str, Any]:
    """One docs-miss turn through the real `_docs_miss_turn`, with a scripted verdict."""

    async def on_text(text: str) -> None:
        pass

    return await server._docs_miss_turn(
        DocResult(status),
        0.1,
        said,
        [{"role": "user", "content": said}],
        on_text,
        None,
        None,
        web_fallback=web_fallback,
    )


# --- doc_then_web: the document's answer fills the {doc} placeholder ---


async def test_doc_then_web_fills_the_placeholder(tmp_path: Path) -> None:
    server, rec = make_server(
        tmp_path,
        split_reply=SPLIT_THEN_WEB,
        doc_reply=DOC_A,
        web_reply="The capital is Norbeth.",
    )
    res = await miss_turn(server, Q_THEN_WEB, "none")
    assert rec.split_calls == [f"Question: {Q_THEN_WEB}"]  # one split call
    assert rec.web_calls == ["What is the capital of Norbria?"]  # {doc} filled with the core
    assert res["compound"] is True
    assert res["compound_kind"] == "doc_then_web"
    assert res["docs"] == "accepted"  # the doc half's verdict is logged
    assert res["text"] == "As per country_sheet_2031.csv: Norbria. The capital is Norbeth."


# --- doc_and_web: both independent halves are spoken, doc first ---


async def test_doc_and_web_speaks_both_halves(tmp_path: Path) -> None:
    server, rec = make_server(
        tmp_path,
        split_reply=SPLIT_AND_WEB,
        doc_reply=DOC_INDEX,
        web_reply="The krona trades at 0.11 dollars.",
    )
    res = await miss_turn(server, Q_AND_WEB, "none")
    assert res["compound_kind"] == "doc_and_web"
    assert res["docs"] == "accepted"
    assert res["text"] == "As per country_sheet_2031.csv: 12.5. The krona trades at 0.11 dollars."
    assert res["text"].index("12.5") < res["text"].index("krona")  # doc sentence, then web
    assert rec.web_calls == ["What is today's exchange rate for the krona?"]


# --- doc_vs_web: the listener hears both facts (the document's and the current one) ---


async def test_doc_vs_web_speaks_both_facts(tmp_path: Path) -> None:
    server, rec = make_server(
        tmp_path,
        split_reply=SPLIT_VS_WEB,
        doc_reply=DOC_B,
        web_reply="The current chief of the Zorblat bureau is Keira Moss.",
    )
    res = await miss_turn(server, Q_VS_WEB, "check_failed")
    assert res["compound_kind"] == "doc_vs_web"
    assert res["docs"] == "accepted"
    assert res["text"] == (
        "As per briefing_notes_2031.md: Keira Moss. "
        "The current chief of the Zorblat bureau is Keira Moss."
    )
    assert rec.web_calls == ["Who is the current chief of the Zorblat bureau?"]


# --- the document half is not accepted: an honest line, then the web half ---


async def test_a_rejected_doc_half_gets_the_honest_line_plus_the_web(tmp_path: Path) -> None:
    server, rec = make_server(
        tmp_path,
        split_reply=SPLIT_AND_WEB,
        doc_reply=DOC_BAD_QUOTE,  # a quote not in the passage: the checks reject it
        web_reply="The krona trades at 0.11 dollars.",
    )
    res = await miss_turn(server, Q_AND_WEB, "none")
    assert res["compound"] is True
    assert res["compound_kind"] == "doc_and_web"
    assert res["docs"] == "check_failed"  # the doc half's verdict is logged
    assert res["text"] == "Your documents don't say. The krona trades at 0.11 dollars."
    assert rec.web_calls == ["What is today's exchange rate for the krona?"]


# --- the web half could not be checked: the doc half plus a plain admission ---


async def test_a_hedged_web_keeps_the_doc_half_and_says_so(tmp_path: Path) -> None:
    server, rec = make_server(
        tmp_path, split_reply=SPLIT_AND_WEB, doc_reply=DOC_INDEX, web_reply=HEDGE
    )
    res = await miss_turn(server, Q_AND_WEB, "none")
    assert res["compound"] is True
    assert res["text"] == "As per country_sheet_2031.csv: 12.5. I couldn't check the web part."
    assert len(rec.web_calls) == 1  # the hedged lookup still ran (and was not retried here)


# --- the split fails: the existing miss answer stands, byte-for-byte ---


@pytest.mark.parametrize(
    ("split_reply", "split_sleep_s", "split_error"),
    [
        pytest.param(SPLIT_NO_KIND, 0.0, False, id="no-kind"),
        pytest.param(SPLIT_SINGLE, 0.0, False, id="single"),
        pytest.param(SPLIT_SINGLE, 2.5, False, id="timeout"),
        pytest.param(SPLIT_SINGLE, 0.0, True, id="error"),
    ],
)
async def test_a_failed_split_keeps_the_existing_miss_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    split_reply: str,
    split_sleep_s: float,
    split_error: bool,
) -> None:
    server, rec = make_server(
        tmp_path,
        split_reply=split_reply,
        doc_reply=DOC_INDEX,
        web_reply="X",
        split_sleep_s=split_sleep_s,
        split_error=split_error,
    )
    res = await miss_turn(server, "What is the Zorblat index in my table?", "none")
    assert len(rec.split_calls) == 1  # the split was paid for once, then it failed

    monkeypatch.setattr(compound, "COMPOUND", False)  # the control: no compound code at all
    server2, rec2 = make_server(
        tmp_path, split_reply=split_reply, doc_reply=DOC_INDEX, web_reply="X"
    )
    ctrl = await miss_turn(server2, "What is the Zorblat index in my table?", "none")
    assert rec2.split_calls == []

    assert res["text"] == "The documents don't say."
    # everything the old path logs is the same (doc_s is measured, so it is out of scope)
    assert {k: v for k, v in res.items() if k != "doc_s"} == {
        k: v for k, v in ctrl.items() if k != "doc_s"
    }


# --- the cheap gates: no document reference, no split ---


async def test_no_split_without_a_document_reference(tmp_path: Path) -> None:
    server, rec = make_server(
        tmp_path,
        split_reply=SPLIT_AND_WEB,
        doc_reply=DOC_INDEX,
        web_reply="The krona trades at 0.11 dollars.",
    )
    res = await miss_turn(server, Q_NO_DOC_REF, "none")
    assert rec.split_calls == []  # the split is not paid for
    assert res["text"] == "The agent answers."  # the plain non-document miss, as before
    assert res["docs"] == "none"


# --- at most one split per turn: the per-turn flag ---


async def test_the_split_runs_at_most_once_per_turn(tmp_path: Path) -> None:
    server, rec = make_server(
        tmp_path,
        split_reply=SPLIT_AND_WEB,
        doc_reply=DOC_INDEX,
        web_reply="The krona trades at 0.11 dollars.",
    )
    res = await miss_turn(server, Q_AND_WEB, "none")
    assert res["compound"] is True
    assert len(rec.split_calls) == 1
    # the flag is still set (the turn is not over): a second miss pays nothing
    res2 = await miss_turn(server, Q_AND_WEB, "none")
    assert len(rec.split_calls) == 1
    assert "compound" not in res2
    assert res2["text"] == "The documents don't say."  # the existing miss answer
    # the next turn resets the flag (handle() does this per turn)
    server._compound_attempted = False
    res3 = await miss_turn(server, Q_AND_WEB, "none")
    assert len(rec.split_calls) == 2
    assert res3["compound"] is True


# --- the total budget: beyond it the compound answer gives up (fail-open) ---


async def test_budget_exceeded_fails_open_to_the_existing_miss_answer(tmp_path: Path) -> None:
    server, rec = make_server(
        tmp_path,
        split_reply=SPLIT_THEN_WEB,
        doc_reply=DOC_A,
        web_reply="The capital is Norbeth.",
        ctx_sleep_s=7.0,  # the document half takes the whole 7 s budget
    )
    t0 = time.perf_counter()
    res = await miss_turn(server, Q_THEN_WEB, "none")
    elapsed = time.perf_counter() - t0
    assert elapsed < 7.5  # the budget is a cap, and the answer gives up before it is blown
    assert len(rec.split_calls) == 1
    assert rec.web_calls == []  # the web half never ran
    assert res["text"] == "The documents don't say."  # the existing miss answer stands
    assert res["docs"] == "none"


# --- the router path: _docs_miss_turn with the web fallback reaches the same hook ---


async def test_the_router_web_fallback_path_reaches_the_hook(tmp_path: Path) -> None:
    # The router branch handles its docs miss through
    # _docs_miss_turn(..., web_fallback=_web_or_none); the compound hook must run before
    # the web fallback and win it.
    server, rec = make_server(
        tmp_path,
        split_reply=SPLIT_THEN_WEB,
        doc_reply=DOC_A,
        web_reply="The capital is Norbeth.",
    )
    fb_calls: list[str] = []

    async def web_fallback(prefix: str) -> dict[str, Any]:
        fb_calls.append(prefix)
        return {"status": "ok", "text": "The capital is Lima.", "routed": True}

    res = await miss_turn(server, Q_THEN_WEB, "check_failed", web_fallback=web_fallback)
    assert fb_calls == []  # the compound answer won before the web fallback
    assert server.route == "docs_miss"  # the route stays docs_miss
    assert res["compound"] is True
    assert res["compound_kind"] == "doc_then_web"
    assert res["text"] == "As per country_sheet_2031.csv: Norbria. The capital is Norbeth."

    # a 'single' split falls through to the web fallback, as before 247 (the fallback is
    # reached with check_failed; a 'none' with a document cue takes the DOCS_DONT_SAY line)
    server2, rec2 = make_server(tmp_path, split_reply=SPLIT_SINGLE, doc_reply=DOC_A, web_reply="X")
    fb2: list[str] = []

    async def web_fallback2(prefix: str) -> dict[str, Any]:
        fb2.append(prefix)
        return {"status": "ok", "text": "The capital is Lima.", "routed": True}

    res2 = await miss_turn(server2, Q_THEN_WEB, "check_failed", web_fallback=web_fallback2)
    assert len(rec2.split_calls) == 1
    assert fb2 == ["I couldn't confirm that in your documents. "]  # as before 247
    assert "compound" not in res2
    assert res2["text"] == "The capital is Lima."
    assert res2["docs"] == "check_failed"
