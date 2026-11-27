"""Pre-deploy 242: the spoken source is the document title, not the filename.

The text-mode docs lead ('As per <file>: <answer>') is kept for the offline evals; only the
SPOKEN text (the voice-side spoken_form filter) and the sources lane speak titles. The
synthetic tests below use made-up names (alpha_report_2031.pdf, the Zorblat memo) so the
behaviour is proven on an unseen corpus; the final section replays session-log utterances
as regression data.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voice_pipeline.bench_fakes import _FakeRouter
from voice_pipeline.benchmark_cases import RouterScript
from voice_pipeline.doc_route import (
    checked_sources_sentence,
    format_citation,
    spoken_title,
)
from voice_pipeline.spoken_form import spoken_form

_SYNTHETIC_FILES = [
    "alpha_report_2031.pdf",
    "folder/beta-memo.txt",
    "2024_01_travel_policy.docx",
    "2024_budget.xlsx",
    "plain notes.md",
    "the Zorblat memo.html",
]


# --- 1. spoken_title: titles, never filenames ---


@pytest.mark.parametrize(
    ("file_path", "title", "want"),
    [
        ("alpha_report_2031.pdf", None, "alpha report 2031"),
        ("folder/beta-memo.txt", None, "beta memo"),
        ("2024_01_travel_policy.docx", None, "travel policy"),
        ("2024_budget.xlsx", None, "budget"),
        ("plain notes.md", None, "plain notes"),
        ("x/y_2031-notes.txt", None, "y 2031 notes"),
        ("the Zorblat memo.html", None, "the Zorblat memo"),
        # the card's catalog title wins when given (case kept, whitespace collapsed)
        ("any.pdf", "  The   Zorblat  memo ", "The Zorblat memo"),
        # a trailing file extension or punctuation on the title is dropped
        ("any.pdf", "the zorblat memo.pdf", "the zorblat memo"),
        ("any.pdf", "the zorblat memo, revised.", "the zorblat memo, revised"),
        # an empty/blank title falls back to the stem
        ("any.pdf", "   ", "any"),
        ("any.pdf", None, "any"),
    ],
)
def test_spoken_title(file_path: str, title: str | None, want: str) -> None:
    assert spoken_title(file_path, title) == want


@pytest.mark.parametrize("file_path", _SYNTHETIC_FILES)
def test_spoken_title_never_outputs_filename_parts(file_path: str) -> None:
    words = spoken_title(file_path)
    assert words
    assert "." not in words and "_" not in words
    ext = Path(file_path).suffix.lower().lstrip(".")
    if ext:
        assert ext not in words.split()  # the extension word is never spoken


def test_spoken_title_cuts_long_titles_on_a_word_boundary() -> None:
    long_title = "the quick brown fox jumps over the lazy dog near the wide river bank"
    assert len(long_title) > 60
    out = spoken_title("any.pdf", long_title)
    assert len(out) <= 60
    assert out in long_title and long_title[len(out)] == " "  # a word-boundary prefix
    assert not out.endswith(" ")


# --- 2. format_citation: the text-mode lead is unchanged (the evals parse it) ---


def test_format_citation_text_lead_unchanged() -> None:
    meta = {"file": "corpus/alpha_report_2031.pdf", "section": "Plan", "date": "2031-01-02"}
    assert (
        format_citation("The value is 42.", "q", meta, ["corpus/alpha_report_2031.pdf"])
        == "As per alpha_report_2031.pdf: The value is 42."
    )
    # the unknown-file fallback lead
    assert format_citation("The value is 42.", "q", None, ()) == (
        "As per your documents: The value is 42."
    )


# --- 3. spoken_form: the SPOKEN lead is the title, and numbers stay sayable ---


@pytest.mark.parametrize("file_path", _SYNTHETIC_FILES)
def test_spoken_lead_matches_spoken_title(file_path: str) -> None:
    assert spoken_form(f"As per {file_path}: The value is 42.") == (
        f"From your {spoken_title(file_path)} document: The value is 42."
    )


def test_spoken_leads_fallback_and_year_tag() -> None:
    assert spoken_form("As per your documents: 3 incidents") == ("From your documents: 3 incidents")
    assert spoken_form("As per gs_plan.md (2026): 45,980") == (
        "From your gs plan document from 2026: 45,980"
    )


def test_spoken_lead_with_card_title() -> None:
    title = "The Zorblat Field Guide: Handling, Storage and Care"
    assert (
        spoken_form("As per zfieldguide.pdf: The rate is 72.5 cents a mile.", doc_title=title)
        == f"From your document on {title}: The rate is 72.5 cents a mile."
    )
    # a table description is not a title: the stem stands
    assert (
        spoken_form("As per inventory_2031.xlsx: 40 units", doc_title="A table with item and count")
        == "From your inventory 2031 document: 40 units"
    )
    # the web lead has no file extension and is untouched
    assert (
        spoken_form("As per Wikipedia and CNN, the capital is Lima.")
        == "As per Wikipedia and CNN, the capital is Lima."
    )


def test_spoken_form_negative_numbers_and_units() -> None:
    assert spoken_form("The reading is -215.") == "The reading is minus 215."
    assert spoken_form("The balance is -72.5 cents.") == "The balance is minus 72.5 cents."
    # cents stay readable; ranges and hyphenated names are not negative numbers
    assert spoken_form("The rate is 72.5 cents per unit.") == "The rate is 72.5 cents per unit."
    assert spoken_form("Steps 5-10 apply.") == "Steps 5-10 apply."
    assert spoken_form("AC 60-22 applies.") == "AC 60-22 applies."


def test_spoken_form_new_leads_are_idempotent() -> None:
    for text in (
        "From your alpha report 2031 document: The value is 42.",
        "From your document on The Zorblat Field Guide: Handling, Storage and Care: ok.",
        "From your documents: 3 incidents.",
        "The reading is minus 215.",
    ):
        assert spoken_form(text) == text


# --- 4. the sources lane speaks titles, not filenames ---


def test_checked_sources_sentence_forms() -> None:
    assert checked_sources_sentence(["launch_plan_2031.pdf"], 5) == (
        "I checked your document on launch plan 2031."
    )
    assert checked_sources_sentence(["launch_plan_2031.pdf", "beta_memo.txt"], 5) == (
        "I checked your documents on launch plan 2031, beta memo."
    )
    # the same name in two folders is said once
    assert checked_sources_sentence(["a/notes.md", "b/notes.md"], 5) == (
        "I checked your document on notes."
    )
    assert (
        checked_sources_sentence(["p1/a.md", "p2/b.md", "p3/c.md", "p4/d.md"], 3)
        == "I checked your documents on a, b, c and 1 more."
    )
    assert checked_sources_sentence([], 3) == "I haven't looked anything up yet."


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


class _Predeploy242Fixtures:
    """A routed AgentServer with docs-first on and a scripted file_context, in the style of
    test_predeploy_235. `send` keeps the message history so the sources follow-up sees the
    assistant's previous line."""

    def __init__(self, passages: list[dict[str, Any]]) -> None:
        self.passages = passages
        self.doc_calls: list[str] = []
        self.server: Any = None
        self.agent: _FakeAgent | None = None
        self._url = ""
        self._msgs: list[dict[str, str]] = []

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
                await params.result_callback(list(self.passages))

            fc = FunctionSchema(
                name="file_context",
                description="File context",
                properties={"query": {"type": "string"}},
                required=["query"],
                handler=_fc,
            )
            fc = agent_tools._recorded(fc, record)

            names = [p["source"] for p in self.passages]

            async def _ds(params: Any) -> None:
                await params.result_callback(
                    {"files": [{"name": n, "status": "ok"} for n in names], "failures": []}
                )

            ds = FunctionSchema(
                name="docs_status",
                description="Docs status",
                properties={},
                required=[],
                handler=_ds,
            )
            ds = agent_tools._recorded(ds, record)

            async def _dc(params: Any) -> None:
                await params.result_callback(
                    [
                        {
                            "rel_key": n,
                            "line": f"{n} | {n.rsplit('/', 1)[-1]} | doc",
                        }
                        for n in names
                    ]
                )

            dc = FunctionSchema(
                name="docs_catalog",
                description="Docs catalog",
                properties={"max_chars": {"type": "integer"}},
                required=[],
                handler=_dc,
            )
            dc = agent_tools._recorded(dc, record)

            async def _de(params: Any) -> None:
                await params.result_callback([{"rel_key": n} for n in names])

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

            tools = ToolsSchema(standard_tools=[fc, ds, dc, de, wa])
            self.agent = _FakeAgent()
            router: Any = _FakeRouter(RouterScript(lines="none|hello"))

            async def _fake_json(
                system: str, content: str, schema: dict[str, Any] | None = None
            ) -> str:
                return json.dumps(
                    {
                        "status": "answer",
                        "answer": "The launch date is April 9.",
                        "file": self.passages[0]["source"],
                        "quotes": ["The launch date is April 9."],
                    }
                )

            log = Path("/tmp/test_predeploy_242.log")
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

            ws_server = await serve(self.server.handle, "127.0.0.1", 0)
            port = next(iter(ws_server.sockets)).getsockname()[1]
            self._ws_server = ws_server
            self._url = f"ws://127.0.0.1:{port}"
            return self._url
        finally:
            agent_tools.CHAIN = prev_chain
            agent_tools.SCRATCHPAD = prev_scratchpad

    async def send(self, question: str, session: str = "") -> dict[str, Any]:
        from websockets.asyncio.client import connect

        messages = self._msgs + [{"role": "user", "content": question}]
        async with connect(self._url) as ws:
            payload: dict[str, Any] = {"messages": messages}
            if session:
                payload["session"] = session
            await ws.send(json.dumps(payload))
            texts: list[str] = []
            while True:
                frame = json.loads(await ws.recv())
                if "text" in frame:
                    texts.append(frame["text"])
                if "done" in frame or "error" in frame:
                    break
        self._msgs = messages + [{"role": "assistant", "content": " ".join(texts)}]
        row = json.loads(self.server.log.read_text().splitlines()[-1])
        row["text"] = " ".join(texts)
        self.server.log.write_text("", encoding="utf-8")
        return row

    async def close(self) -> None:
        self._ws_server.close()
        await self._ws_server.wait_closed()


def _passages(sources: list[str]) -> list[dict[str, Any]]:
    """The cited source carries the answer; the others are just checked sources."""
    texts = [
        "The launch date is April 9.",
        "The budget is 12 million dollars.",
        "The office is on the third floor.",
        "The total is 900 units.",
    ]
    return [{"source": s, "text": texts[i % len(texts)]} for i, s in enumerate(sources)]


@pytest.mark.asyncio
async def test_sources_lane_speaks_titles_not_filenames() -> None:
    """A docs answer followed by a sources follow-up: the follow-up says the document
    titles, never the filenames."""
    fx = _Predeploy242Fixtures(
        _passages(["alpha_report_2031.pdf", "beta_memo.txt", "gamma_notes.md", "delta_ledger.xlsx"])
    )
    await fx.build()
    try:
        row_1 = await fx.send("What is the launch date?", session="s1")
        assert row_1["route"] == "docs"
        assert row_1["docs"] == "accepted"
        # the text-mode lead is what the evals parse: unchanged
        assert row_1["text"].strip() == "As per alpha_report_2031.pdf: The launch date is April 9."

        row_2 = await fx.send("Your sources?", session="s1")
        assert row_2["route"] == "sources"
        assert row_2["text"].strip() == (
            "I checked your documents on alpha report 2031, beta memo, gamma notes and 1 more."
        )
        for name in ("alpha_report_2031", "beta_memo", ".pdf", ".txt", ".md", ".xlsx"):
            assert name not in row_2["text"]
    finally:
        await fx.close()


@pytest.mark.asyncio
async def test_sources_lane_single_file_singular() -> None:
    """One checked file: the singular 'I checked your document on X.'"""
    fx = _Predeploy242Fixtures(_passages(["launch_plan_2031.pdf"]))
    await fx.build()
    try:
        row_1 = await fx.send("What is the launch date?", session="s1")
        assert row_1["route"] == "docs"
        row_2 = await fx.send("Your sources?", session="s1")
        assert row_2["route"] == "sources"
        assert row_2["text"].strip() == "I checked your document on launch plan 2031."
        assert ".pdf" not in row_2["text"] and "launch_plan" not in row_2["text"]
    finally:
        await fx.close()


# --- 5. regression data: session-log utterances (the 48-file corpus) ---


class TestRegressionSessionLog:
    """Regression data, kept separate from the synthetic tests above: the exact utterances
    from the owner's session log that started this ticket. These names belong to the test
    corpus and appear nowhere in the source code."""

    def test_liberal_studies_calendar_spoken_title(self) -> None:
        assert spoken_title("liberal_studies_academic_calendar.html") == (
            "liberal studies academic calendar"
        )
        assert (
            spoken_form("As per liberal_studies_academic_calendar.html: Midterms are in May.")
            == "From your liberal studies academic calendar document: Midterms are in May."
        )

    def test_irs_publication_title_is_kept(self) -> None:
        title = "IRS Publication 463: Travel, Gift, and Car Expenses"
        assert spoken_title("travel_expenses_guide.pdf", title) == title
        assert (
            spoken_form(
                "As per travel_expenses_guide.pdf: The standard rate is 72.5 cents a mile.",
                doc_title=title,
            )
            == f"From your document on {title}: The standard rate is 72.5 cents a mile."
        )
