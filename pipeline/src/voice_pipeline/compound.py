"""Compound document+web questions, split and answered per lane (ticket 247).

A question that needs both a fact from the user's documents and the web (or a document
fact used inside a web question) used to be answered by one lane only: the document check
saw the whole compound question, missed, and the web answered only its half. When a
docs-miss turn references the documents, the question is split by ONE call to the same
cheap JSON model the document lane uses (``doc_json_request``); the document half goes
through the checked document lane (``AgentServer._ask_docs``) and the web half through
the web lookup (``AgentServer.lookup``, so the ledger, the hedge and the directness check
all apply); the halves are combined and spoken.

Fail-open everywhere: the split is not run for turns without a document reference, at
most once per turn (the server's per-turn flag), and any split error, timeout, a
``single`` verdict, an unusable half, or the total budget is exceeded returns None and
the existing docs-miss answer stands unchanged.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from voice_pipeline.ask_documents import DocResult, JsonRequest

COMPOUND = True

# the split call: the document lane's own cheap JSON model, no retrieval
SPLIT_TIMEOUT_S = 2.0
# the whole compound answer (split + document half + web lookup), from the miss on
COMPOUND_BUDGET_S = 7.0

# the web lookup's honest hedge (copied from evaluators.HEDGE, which copies the web
# server's own line; the AST test pins them together). A lookup that ends this way has
# not actually checked the web.
HEDGE_TAIL = "I couldn't confirm that with a search, so I won't guess."
# spoken instead of the web half when it could not be checked (hedged or failed)
WEB_MISS_TAIL = "I couldn't check the web part."
# the document half was not accepted but the web half was (kind doc_and_web only)
DOCS_DONT_SAY = "Your documents don't say. "

# Generic English only; the examples use invented entities. No test-corpus words.
SYSTEM = (
    "You do not answer questions; you split them. Decide whether a spoken question needs "
    "both the user's own documents and the web. It is compound when it (1) needs a fact "
    "from the user's documents as an ingredient of a web question, (2) asks two "
    "independent questions, one answerable from the user's documents and one from the "
    "web, or (3) quotes a fact the user's documents give and asks whether it is still "
    "right or has changed. Reply with JSON only, an object with exactly the keys kind, "
    "doc_question and web_question. kind is 'single' (both questions null) when one "
    "source alone can answer; 'doc_then_web' when the web question needs the document's "
    "answer, in which case web_question holds the literal placeholder {doc} exactly "
    "where that answer goes; 'doc_and_web' when the two halves are independent; "
    "'doc_vs_web' when the user quotes a document fact and asks whether it is still "
    "true, in which case web_question is the current-facts question. doc_question is a "
    "standalone question answerable from the user's documents alone, keeping the "
    "user's reference to their file, notes or table. Examples: "
    "'Who runs the Zorblat bureau, and how deep is the Quenth river now?' -> "
    '{"kind": "doc_and_web", "doc_question": "Who runs the Zorblat bureau, per my file?", '
    '"web_question": "How deep is the Quenth river now?"} '
    "'What is the capital of the smallest country in my country sheet?' -> "
    '{"kind": "doc_then_web", "doc_question": "Which country does my country sheet list '
    'as the smallest?", "web_question": "What is the capital of {doc}?"} '
    "'The water level my field log gives for the Quenth river, is that still right?' -> "
    '{"kind": "doc_vs_web", "doc_question": "What water level does my field log give for '
    'the Quenth river?", "web_question": "What is the current water level of the Quenth '
    'river?"}'
)

_KINDS = frozenset({"single", "doc_then_web", "doc_and_web", "doc_vs_web"})

# the split's reply shape, enforced by the transport's structured output (R1-1): the
# document lane's own REPLY_SCHEMA has no `kind` key, so the split call must ask for this
# one, or the reply can never name a kind and the split is always dropped
SPLIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": sorted(_KINDS)},
        "doc_question": {"type": ["string", "null"]},
        "web_question": {"type": ["string", "null"]},
    },
    "required": ["kind", "doc_question", "web_question"],
}

_FIRST_SENTENCE = re.compile(r"^(.*?[.!?])\s")


@dataclass
class Split:
    """A compound question in its two lanes. ``web_question`` may hold the literal
    placeholder ``{doc}`` where the document's answer goes (kind doc_then_web)."""

    kind: str
    doc_question: str
    web_question: str


async def split(request: JsonRequest | None, question: str) -> Split | None:
    """ONE call to the document lane's JSON model: the split, or None. Any error,
    unparsable reply, or timeout is None (the existing miss answer then stands)."""
    if request is None:
        return None
    try:
        raw = await asyncio.wait_for(
            request(SYSTEM, f"Question: {question}", SPLIT_SCHEMA), timeout=SPLIT_TIMEOUT_S
        )
    except Exception:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    kind = str(data.get("kind") or "").strip()
    if kind == "single":  # not compound: nothing changes
        return None
    if kind not in _KINDS:
        return None
    doc_q = str(data.get("doc_question") or "").strip()
    web_q = str(data.get("web_question") or "").strip()
    if not doc_q or not web_q:
        return None
    return Split(kind=kind, doc_question=doc_q, web_question=web_q)


def _doc_core(half: DocResult) -> str:
    """The document answer's core for the ``{doc}`` placeholder: the first sentence,
    the citation lead stripped (the same ``partition(": ")`` the document lane uses on
    its own citation)."""
    text = (half.answer or half.spoken or "").strip()
    if text.lower().startswith("as per"):
        text = text.partition(": ")[2].strip()
    m = _FIRST_SENTENCE.match(text)
    if m:
        text = m.group(1)
    return text.strip()


def _hedged(text: str) -> bool:
    """The lookup did not actually check the web: it came back with the hedge line."""
    return HEDGE_TAIL in str(text)


def _join(parts: list[str]) -> str:
    """Document sentence(s) then web sentence(s), one space apart, periods supplied."""
    out = ""
    for part in parts:
        part = (part or "").strip()
        if not part:
            continue
        if out and out[-1] not in ".!?":
            out += "."
        out = f"{out} {part}" if out else part
    return out


def _fields(half: DocResult, doc_s: float) -> dict[str, Any]:
    """The turn-log fields of the document half (None values dropped, as
    agent_server._docs_miss_fields does)."""
    fields: dict[str, Any] = {
        "docs": half.status,
        "doc_s": doc_s,
        "doc_calls": half.calls,
        "fallback": half.fallback,
        "sources": half.sources,
    }
    if half.rejected:
        brief = half.rejected.get("quote") or half.rejected.get("answer") or ""
        fields["rejected"] = {"failed": half.rejected.get("failed") or "", "excerpt": brief[:120]}
    return {k: v for k, v in fields.items() if v is not None}


async def answer(
    server: Any,
    said: str,
    doc_s: float,
    session: str,
    on_text: Any,
) -> dict[str, Any] | None:
    """The compound answer for a docs-miss turn that references the documents: split the
    question, answer the document half in the checked document lane and the web half in
    the web lookup, combine, and speak. None when nothing can be said better than the
    existing miss answer (fail-open). Total budget COMPOUND_BUDGET_S."""
    started = time.perf_counter()

    def remaining() -> float:
        return COMPOUND_BUDGET_S - (time.perf_counter() - started)

    cut = await split(server.doc_json_request, said)
    if cut is None or remaining() <= 0:
        return None

    # (a) the document half, in the checked document lane (its checks stand)
    half: DocResult | None
    timed_out = False
    try:
        half = await asyncio.wait_for(
            server._ask_docs(cut.doc_question, session, server.turn_now), remaining()
        )
    except TimeoutError:
        half, timed_out = None, True
    except Exception:
        half = None
    half = half if half is not None else DocResult("timeout" if timed_out else "error")
    doc_ok = half.status == "accepted"
    doc_text = half.spoken or ""

    # (b) the web half: the document's answer fills the {doc} placeholder; without an
    # accepted document answer a doc_then_web question cannot be asked at all
    web_q = cut.web_question
    if "{doc}" in web_q:
        core = _doc_core(half) if doc_ok else ""
        if not core:
            return None
        web_q = web_q.replace("{doc}", core)
    web_text: str | None
    if remaining() <= 0:
        web_text = None
    else:
        try:
            got = await asyncio.wait_for(server.lookup("web_answer", {"query": web_q}), remaining())
            web_text = None if got is None else str(got)
        except (TimeoutError, Exception):
            web_text = None
    doc_s_total = round(doc_s + (time.perf_counter() - started), 3)

    # (c) combine: the document sentence(s) then the web sentence(s)
    if doc_ok and web_text is not None and not _hedged(web_text):
        text = _join([doc_text, web_text])
    elif doc_ok:  # the web half could not be checked
        text = _join([doc_text, WEB_MISS_TAIL])
    elif web_text is not None and not _hedged(web_text) and cut.kind == "doc_and_web":
        text = DOCS_DONT_SAY + web_text
    else:
        return None
    # (d) speak, with the document half's result fields (the route stays docs_miss:
    # the caller already set it)
    fields = _fields(half, doc_s_total)
    fields["compound"] = True
    fields["compound_kind"] = cut.kind
    result: dict[str, Any] = await server.spoken(text, on_text, **fields)
    return result
