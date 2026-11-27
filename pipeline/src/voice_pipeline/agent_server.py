"""Agent server: the agent behind a local WebSocket, for `doc_chat --llm agent`.

    LAN_SPEECH_TOKEN=... uv run python -m voice_pipeline.agent_server [--port 8781]
        [--live URL] [--pi-url URL]

Protocol (the client is src/doc_chat/agent_client.py in the root project), one request per
connection: {"messages": [...]} in; {"text": ...} pieces, then {"done": true} or {"error": ...}
out. The server owns the system prompt, so client system messages are dropped. Requests are
served one at a time; a client that disconnects before "done" cancels its turn. Every turn is
logged to out/agent_server/<start time>.jsonl with its question, tool calls and timings (row
schema: docs/TURN-TRACE.md).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request as HttpRequest
from websockets.http11 import Response as HttpResponse

from voice_pipeline import (
    agent_tools,
    call_policy,
    compound,
    doc_route,
    fast_path,
    net,
    question,
    search_count,
    selfinfo,
    sufficiency,
    temporal,
    trace,
    venue,
)
from voice_pipeline.agent import (
    KEEPALIVE_S,
    SAY_DELAY_SLOW,
    SAY_DELAY_TODAY,
    Agent,
    build_llm,
    close_llm_sessions,
)
from voice_pipeline.agent_tools import (
    FAILURE,
    IDLE_DELEGATE_TOOLS,
    MODEL_HIDDEN,
    PI_URL,
    TRANSIENT,
    build_tools,
    configure,
    has_delegate_jobs,
    standard_servers,
    with_builtin_search,
    without_tools,
)
from voice_pipeline.ask_documents import (
    DocResult,
    JsonRequest,
    ask_documents,
    gemini_json_request,
)
from voice_pipeline.chain_state import (
    SIDE_TOOLS,
    ChainState,
    is_answer,
    is_chain_tool,
    is_followup,
    is_side_question,
    wants_cancel,
)
from voice_pipeline.device_say import device_sentence
from voice_pipeline.doc_answer import (
    DOC_ANSWER_TIMEOUT_S,
    DocAnswer,
    Request,
    decode_rows,
    gemini_request,
    norm,
    passages_with_meta,
)
from voice_pipeline.doc_answer import answer as doc_answer
from voice_pipeline.general import answer_general
from voice_pipeline.hedge import HedgedAgent, HedgeGate, gated_tools
from voice_pipeline.notices import NoticeQueue
from voice_pipeline.payload_guard import PayloadFilter, SourceFilter, strip_sources
from voice_pipeline.router import (
    HISTORY_PAIRS,
    NO_SLOT_SAY,
    SLOT_SAY,
    TIMEOUT_S,
    LocalRouter,
    Routed,
    spec,
)
from voice_pipeline.scratchpad import PROFILE_RULE, Scratchpad, install_profile
from voice_pipeline.spoken_filter import SentenceFilter, is_apology, is_backchannel
from voice_pipeline.tools.smart_home import load_phonebook

ROOT = Path(__file__).resolve().parents[2]
_logger = logging.getLogger(__name__)
# A web lookup the router runs can take 0.8-2.9 s (grounded Gemini: 1.3 s with no search, 2.1 s with
# one). If it is still running this long after it started, the client is told a tool call is under
# way and plays that tool's pre-recorded line ("Let me look that up."), as for a model tool call.
# Simulated on 65 measured lookups: first sound 1.8 s -> 1.2 s and answer start 2.8 s -> 2.2 s for
# the questions that need a search; quick ones (median 0.8 s) finish before it would fire.
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
LOOKUP_ACK_S = 1.0
ACKED_LOOKUPS = frozenset({"web_answer"})
# Spoken at once when the router sends a web lookup (routed turns have no model `say`); listed in
# config/fast_phrases.json so the client plays it from memory.
OFFLINE_SAY = "I can't reach the internet right now, but I can still do home and device things."
LOOKUP_FAIL_SAY = "I couldn't get that from the internet right now."
OFFLINE_LOOKUP_S = 1.0  # offline, a lookup gets this long (a cached forecast answers in ~0.1 s)
LOOKUP_GIVE_S = 4.0  # a routed lookup that failed after this long is not handed to Gemini again
LOOKUP_SAY = "Let me look that up."
MUSIC_SAY = (
    "Finding some music."  # a routed PLAY_MUSIC still running after MUSIC_SAY_S (a music search)
)
MUSIC_SAY_S = 0.7
NOTICE_POLL_S = 0.25  # how often the notices file is read while a client listens for them
DOCS_WARMUP_QUERY = "warm up"
DOCS_WARMUP_TIMEOUT_S = 90.0
DOCS_TURN_BUDGET_S = 6.0

ACTION_CALLS = frozenset(
    {"weather", "get_current_time", "smart_home", "set_volume", "get_volume", "device_status"}
    | set(spec.HOME_INTENTS)
)
NO_ANSWER_POLICY = True
DOCS_DONT_SAY = "The documents don't say."
DOCS_NO_COVER_PREFIX = "Your documents don't cover that. "
DOCS_UNCONFIRMED_PREFIX = "I couldn't confirm that in your documents. "
# chain phases in which the next utterance is the answer to the chain's own question ("yes",
# "approve", the name): the chain keeps such a turn, even when it reads like a document question
_CHAIN_MID_PROTOCOL = ("awaiting_name", "awaiting_fact_approval", "awaiting_publish_confirm")

# The docs verdict for this turn already came from ask_documents (a miss); the Gemini turn that
# follows must not be offered the document tools, or it can re-answer from an unchecked
# search_docs (ticket 244). Web tools stay. Hidden for that turn only (see _docs_miss_turn).
DOC_MISS_HIDDEN_TOOLS = frozenset(
    {"search_docs", "file_context", "docs_catalog", "docs_enumerate", "docs_status"}
)


def _strip_repeated_first_sentence(text: str) -> str:
    """Collapse a first sentence repeated verbatim right after itself."""
    s = (text or "").strip()
    m = re.match(r"^(.+?[.!?])\s+(.+?[.!?])", s)
    if m and m.group(1) == m.group(2):
        return (m.group(1) + " " + s[m.end() :].lstrip()).strip()
    return s or text


def _join_miss_prefix(prefix: str, text: str) -> str:
    """Put the miss prefix on a reply once (ticket 244): when the model already leads with it,
    do not say it twice, and collapse a first sentence repeated verbatim right after itself."""
    text = (text or "").strip()
    if not text:
        return prefix
    if text.startswith(prefix) or text.startswith(prefix.rstrip()):
        return _strip_repeated_first_sentence(text)
    return prefix + _strip_repeated_first_sentence(text)


def _docs_rejected_brief(rejected: dict[str, str] | None) -> dict[str, str] | None:
    """Log-safe docs rejection: the failed check plus a 120-char excerpt (file text is data)."""
    if not rejected:
        return None
    text = rejected.get("quote") or rejected.get("answer") or ""
    return {"failed": rejected.get("failed") or "", "excerpt": text[:120]}


def _docs_miss_fields(doc_result: DocResult, doc_s: float | None) -> dict[str, Any]:
    """Turn-log fields for a docs-first fall-through (None values dropped)."""
    fields = {
        "docs": doc_result.status,
        "doc_s": doc_s,
        "doc_calls": doc_result.calls,
        "fallback": doc_result.fallback,
        "sources": doc_result.sources,
        "rejected": _docs_rejected_brief(doc_result.rejected),
    }
    return {k: v for k, v in fields.items() if v is not None}


def token_guard(token: str) -> Callable[[ServerConnection, HttpRequest], HttpResponse | None]:
    """A `process_request` hook: the WebSocket handshake needs `Authorization: Bearer <token>`.
    Anything else gets HTTP 401 before a turn can start (the delegate sandbox shares the laptop's
    loopback, so the port alone is not a boundary). The token is never logged."""
    expected = token.encode()

    def check(connection: ServerConnection, request: HttpRequest) -> HttpResponse | None:
        header = request.headers.get("Authorization") or ""
        scheme, _, given = header.partition(" ")
        if scheme.lower() == "bearer" and hmac.compare_digest(given.strip().encode(), expected):
            return None
        _logger.warning(
            "rejected a connection without a valid token from %s", connection.remote_address
        )
        return connection.respond(401, "Unauthorized\n")

    return check


class OnlineProbe:
    """Whether Gemini's API is reachable: a TCP connect every `every` s; offline after two
    failures in a row, online again after one success. A turn that needs Gemini while offline is
    answered at once instead of waiting out the LLM timeout (15 s with the retry)."""

    def __init__(self, host: str = "generativelanguage.googleapis.com", every: float = 5.0) -> None:
        self.host, self.every, self.online, self._fails = host, every, True, 0

    async def check(self) -> bool:
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection(self.host, 443), 2.0)
        except (TimeoutError, OSError):
            return False
        writer.close()
        return True

    async def run(self) -> None:
        while True:
            ok = await self.check()
            self._fails = 0 if ok else self._fails + 1
            self.online = ok or self._fails < 2
            await asyncio.sleep(self.every)


class AgentServer:
    """Serves `agent` turns over WebSocket and logs each one."""

    def __init__(
        self,
        agent: Any,
        record: list[dict[str, Any]],
        log: Path,
        tools: Any = None,
        router: LocalRouter | None = None,
        notices: NoticeQueue | None = None,
        doc_request: Request | None = None,
        jobs_file: Path | None = None,
        chain: ChainState | None = None,
        case_f_block: str = "",
        docs_first: bool = False,
        doc_json_request: JsonRequest | None = None,
    ) -> None:
        self.chain = chain
        self.case_f_block = case_f_block
        self.jobs_file = jobs_file
        self.doc_request = doc_request
        self.router = router
        self.notices = notices
        self.docs_first = docs_first
        self.doc_json_request = doc_json_request
        self._doc_cache: dict[str, DocResult] = {}
        self._sources: dict[str, list[str]] = {}
        # notices read from the file, not yet played to the end (or cut)
        self.unacked: dict[int, dict[str, str]] = {}
        self.notice_ids = 0
        self.checks: set[asyncio.Task[None]] = set()  # second opinions running behind a reply
        self.test_delay: tuple[str, float] | None = None  # (word, seconds): --test-delay-word
        # the "[Now: ...]" line every Gemini request ends with; only `run` turns it on
        self.stamp: Callable[[], str] | None = None
        self.probe: OnlineProbe | None = None  # set by run(); None: always online (tests)
        self.router_lines: dict[str, str] = {}  # reply we spoke -> the router lines behind it
        self.agent = agent
        self.record = record
        self.log = log
        self.run_id = getattr(log, "stem", "")  # tests pass no log
        self.turns = 0
        self.route = ""  # which branch of `handle` took the current turn
        self.tail: list[str] = []  # what steer/stamped appended to the last user message
        self.ledger = search_count.LEDGER
        self.lock = asyncio.Lock()
        # with the agent's tools, common home commands skip the LLM (fast_path.py)
        self.tools = {t.name: t for t in tools.standard_tools} if tools is not None else {}
        self._checked_sources: list[str] = []  # files checked for the last docs turn
        self._docs_checked = False  # this turn's docs verdict came from ask_documents (a miss)
        self._compound_attempted = False  # this turn's docs miss already paid for the split (247)
        self._sufficiency_attempted = False  # this turn already paid for the check (250)

    async def fast(self, found: fast_path.Match, on_text: Any) -> dict[str, Any]:
        """Run the tool the LLM would have called, then say the fixed confirmation."""
        started = time.perf_counter()
        t0 = trace.elapsed()
        if found.tool == "clock":  # the clock answers; no tool runs
            entry: dict[str, Any] = {"tool": "clock", "args": dict(found.args)}
            self.record.append(entry)
            trace.annotate(entry, by="fast-path", t0=t0, status="ok", results=[])
            text = fast_path.local_answer(found, datetime.now(ZoneInfo(fast_path.TIMEZONE)))
            await on_text(text)
            elapsed = time.perf_counter() - started
            return {"status": "ok", "text": text, "first_call_s": 0.0, "first_text_s": elapsed,
                    "end_s": elapsed, "fast_path": True}  # fmt: skip
        results: list[Any] = []

        async def done(result: Any, *args: Any, **kwargs: Any) -> None:
            results.append(result)

        params = SimpleNamespace(
            function_name=found.tool,
            tool_call_id="fast-path",
            arguments=dict(found.args),
            result_callback=done,
        )
        mark = len(self.record)
        await self.tools[found.tool].handler(params)
        trace.stamp_new(self.record, mark, "fast-path", t0, "ok")
        text = fast_path.spoken_result(found, results[0] if results else {"ok": False})
        await on_text(text)
        elapsed = time.perf_counter() - started
        return {"status": "ok", "text": text, "first_call_s": 0.0, "first_text_s": elapsed,
                "end_s": elapsed, "fast_path": True}  # fmt: skip

    def history(self, messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """The last earlier (user, router lines) pair(s); a reply the router did not write
        (Gemini's) stands in as a plain `none|` line."""
        pairs: list[tuple[str, str]] = []
        said: str | None = None
        for m in messages[:-1]:
            if m.get("role") == "user":
                said = str(m.get("content", ""))
            elif m.get("role") == "assistant" and said is not None:
                reply = str(m.get("content", ""))
                pairs.append(
                    (said, self.router_lines.get(reply) or "none|" + " ".join(reply.split())[:100])
                )
                said = None
        return pairs[-HISTORY_PAIRS:]

    def steer(self, messages: list[dict[str, Any]], *, case_f: bool) -> list[dict[str, Any]]:
        """The messages for Gemini: the chain's state, and when `case_f` the Case F chain
        protocol, appended to the last user message (P2). The protocol is not in the base
        prompt: it goes with a turn while a chain is active, when the router escalates a
        turn Gemini cannot see coming, and when there is no router to filter."""
        if not messages:
            return messages
        last = messages[-1]
        if last.get("role") != "user" or not isinstance(last.get("content"), str):
            return messages
        suffix = ""
        parts: list[str] = []
        if self.chain is not None and self.chain.active():
            suffix += "\n" + self.chain.block()
            parts.append("chain")
        if case_f and self.case_f_block:
            suffix += "\n" + self.case_f_block
            parts.append("case_f")
        if call_policy.changeable_question(str(last["content"])):
            note = (
                call_policy.CHANGEABLE_NOTE_DOCS
                if ("ask_documents" in self.tools or "file_context" in self.tools)
                else call_policy.CHANGEABLE_NOTE
            )
            suffix += "\n" + note
            parts.append("note")
        if self.stamp is not None:
            suffix += "\n" + self.stamp()  # last: the volatile part goes after the stable part
            parts.append("now")
        self.tail = parts
        if not suffix:
            return messages
        return [*messages[:-1], {**last, "content": last["content"] + suffix}]

    def stamped(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """`messages` with the changeable note (when the question can have changed) and the
        [Now: ...] line added, in `steer`'s order (the hedged requests start before the router
        decides, so they carry nothing else)."""
        if not messages:
            return messages
        last = messages[-1]
        if last.get("role") != "user" or not isinstance(last.get("content"), str):
            return messages
        suffix = ""
        parts: list[str] = []
        if call_policy.changeable_question(str(last["content"])):
            note = (
                call_policy.CHANGEABLE_NOTE_DOCS
                if ("ask_documents" in self.tools or "file_context" in self.tools)
                else call_policy.CHANGEABLE_NOTE
            )
            suffix += "\n" + note
            parts.append("note")
        if self.stamp is not None:
            suffix += "\n" + self.stamp()  # last: the volatile part goes after the stable part
            parts.append("now")
        if not suffix:
            return messages
        self.tail = parts
        return [*messages[:-1], {**last, "content": last["content"] + suffix}]

    def refers_to_setting(self, said: str, messages: list[dict[str, Any]], before: int) -> bool:
        """ "Turn it up" right after a temperature or brightness command is about that setting,
        not the volume: the model gets it (the prompt says to ask for a number). Needs an earlier
        reply in this conversation, so a stale record from another session cannot trigger it."""
        if fast_path.normalize(said) not in fast_path.CONTEXT_DEPENDENT or not before:
            return False
        if not any(m.get("role") == "assistant" for m in messages[:-1]):
            return False
        return str(self.record[before - 1].get("intent", "")) in ("TEMPERATURE", "BRIGHTNESS")

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Run one of the agent's tools directly and return its result."""
        results: list[Any] = []

        async def done(result: Any, *rest: Any, **kwargs: Any) -> None:
            results.append(result)

        params = SimpleNamespace(
            function_name=name, tool_call_id="router", arguments=args, result_callback=done
        )
        mark, t0 = len(self.record), trace.elapsed()
        try:
            await self.tools[name].handler(params)
        except asyncio.CancelledError:
            trace.stamp_new(self.record, mark, "router", t0, "cancelled")
            raise
        except Exception:
            trace.stamp_new(self.record, mark, "router", t0, "failed")
            raise
        trace.stamp_new(self.record, mark, "router", t0, "ok")
        return results[0] if results else {"ok": False}

    async def acked_call(
        self, tool: str, args: dict[str, Any], on_tool: Any, home: bool, said: bool = False
    ) -> Any:
        """`call_tool`, telling the client after LOOKUP_ACK_S that a slow lookup is under way
        (`said`: a sentence for it was already spoken, so the client plays no filler line)."""
        if not home and self.offline():
            # offline: a cached forecast still answers at once; anything else fails fast
            try:
                return await asyncio.wait_for(self.call_tool(tool, args), OFFLINE_LOOKUP_S)
            except TimeoutError:
                return "The service did not answer."
        if home or on_tool is None or tool not in ACKED_LOOKUPS:
            return await self.call_tool(tool, args)

        async def announce() -> None:
            await asyncio.sleep(LOOKUP_ACK_S)
            await on_tool(tool, said)

        ack = asyncio.create_task(announce())
        try:
            return await self.call_tool(tool, args)
        finally:
            ack.cancel()

    async def said_if_slow(self, tool: str, args: dict[str, Any], on_say: Any) -> Any:
        """`call_tool`; if it is still running after MUSIC_SAY_S, MUSIC_SAY is spoken first."""
        call = asyncio.ensure_future(self.call_tool(tool, args))
        done, _ = await asyncio.wait({call}, timeout=MUSIC_SAY_S)
        if not done:
            await on_say(MUSIC_SAY + " ")
        return await call

    async def lookup(self, tool: str, args: dict[str, Any]) -> Any:
        """The web lookup, its blip retried once; None when it failed."""
        result = await self.call_tool(tool, args)
        if TRANSIENT.search(str(result)):  # lookups are read-only
            result = await self.call_tool(tool, args)
        return None if FAILURE.search(str(result)) else result

    async def _checked_docs(self, user: str) -> tuple[DocAnswer, list[dict[str, Any]]]:
        """Search documents and run doc_answer with cached file_names; shared by doc_race and
        _try_docs_first."""
        got = DocAnswer("timeout")
        passages_list: list[dict[str, Any]] = []

        async def ask() -> DocAnswer:
            nonlocal passages_list
            try:
                passages_list = passages_with_meta(
                    await self.call_tool("search_docs", {"query": user})
                )
            except Exception:
                return DocAnswer("error")
            assert self.doc_request is not None
            stamp = self.stamp() if self.stamp is not None else None
            return await doc_answer(
                self.doc_request,
                passages_list,
                user,
                stamp,
                file_names=doc_route.cached_files(self),
            )

        try:
            got = await asyncio.wait_for(ask(), DOC_ANSWER_TIMEOUT_S)
        except TimeoutError:
            pass
        except Exception:
            got = DocAnswer("error")
        return got, passages_list

    async def doc_race(self, user: str) -> tuple[str | None, dict[str, Any]]:
        """Passages of the documents and one Gemini request, checked (doc_answer.py). Never raises:
        (the answer or None, the `race` fields)."""
        started = time.perf_counter()
        got, passages_list = await self._checked_docs(user)
        fields: dict[str, Any] = {
            "docs": got.status,
            "doc_s": round(time.perf_counter() - started, 3),
        }
        if got.rejected:
            fields["rejected"] = got.rejected  # kept so a failed check can be diagnosed later
        if got.status == "accepted" and got.text is not None:
            fields["citation_metadata"] = doc_route.citation_metadata(got.passage, got.quote or "")
            fields["sources"] = [str(p.get("source", "")) for p in passages_list if p.get("source")]
            return doc_route.format_citation(
                got.text,
                got.quote or "",
                got.passage,
                doc_route.cached_files(self),
            ), fields
        return None, fields

    async def raced_lookup(
        self,
        tool: str,
        args: dict[str, Any],
        user: str,
        on_tool: Any,
        race: dict[str, Any],
        said: bool = False,
    ) -> str | None:
        """The web lookup and the documents' answer at the same time: the document answer if it
        passed its checks, else the web result; None if neither is usable. Nothing is spoken here;
        if neither has decided LOOKUP_ACK_S after the start the client is told a lookup is under
        way. `race` is filled in. When the turn's docs verdict already came from ask_documents
        (`_docs_checked`) the race is skipped: the legacy doc_race could speak the decoy that
        ask_documents just rejected, so the plain web lookup stands alone."""

        async def announce() -> None:
            await asyncio.sleep(LOOKUP_ACK_S)
            await on_tool(tool, said)

        web = asyncio.create_task(self.lookup(tool, args))
        ack = (
            asyncio.create_task(announce())
            if on_tool is not None and tool in ACKED_LOOKUPS
            else None
        )
        try:
            if not self._docs_checked:
                text, fields = await self.doc_race(user)
                race.update(fields)
                if text is not None:
                    race["winner"] = "docs"
                    self._checked_sources = fields.get("sources", [])
                    return text
            try:
                result = await web
            except Exception:
                result = None
            race["winner"] = "web" if result is not None else None
            return None if result is None else str(result)
        finally:
            for task in (ack, web):
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task

    async def run_routed(
        self,
        routed: Routed,
        started: float,
        on_text: Any,
        claim: Callable[[], bool] = lambda: True,
        on_tool: Any = None,
        on_say: Any = None,
    ) -> dict[str, Any] | None:
        """Run the router's calls with the agent's own tools and say the result. `claim` is asked
        before the first home command and before speaking: False means Gemini already answered
        (nothing is run or said). None: Gemini should take the turn, because the claim failed or a
        lookup failed before anything changed (it retries with another topic, as it does without
        the router)."""
        spoken: list[str] = []
        claimed = False
        announced = False

        def take() -> bool:
            nonlocal claimed
            claimed = claimed or claim()
            return claimed

        steps = call_policy.plan(routed.calls, routed.user, set(self.tools))
        for step in steps:
            call = step.call
            name = call.name
            if name == "none":
                spoken.append(call.args[0].strip())
            elif name == "get_current_time":
                zone = spec.to_tool_args(call).get("timezone", fast_path.TIMEZONE)
                now = datetime.now(ZoneInfo(zone))
                what = fast_path.clock_question(routed.user)
                text = fast_path.local_answer(fast_path.Match("clock", {"what": what}, None), now)
                spoken.append(
                    text
                    if zone == fast_path.TIMEZONE
                    else f"{text[:-1]} in {zone.split('/')[-1].replace('_', ' ')}."
                )
            else:
                home, tool, args = step.home, step.tool, step.args
                if (
                    on_say is not None
                    and not home
                    and not announced
                    and tool in call_policy.DOCUMENT_LOOKUPS
                ):
                    announced = True
                    await on_say(LOOKUP_SAY + " ")
                if (
                    not home
                    and len(steps) == 1
                    and tool in call_policy.DOCUMENT_LOOKUPS
                    and self.doc_request is not None
                    and "search_docs" in self.tools
                ):
                    routed.race = {}
                    asked = args.get("query") or args.get("topic") or routed.user
                    said = await self.raced_lookup(
                        tool, args, asked, on_tool, routed.race, announced
                    )
                    if said is None:
                        return None
                    spoken.append(said)
                    continue
                if home and not take():
                    return None
                if home and name == "PLAY_MUSIC" and on_say is not None and not announced:
                    announced = True
                    result = await self.said_if_slow(tool, args, on_say)
                else:
                    result = await self.acked_call(tool, args, on_tool, home, announced)
                if (
                    not home
                    and TRANSIENT.search(str(result))
                    and not claimed
                    and not self.offline()
                ):
                    result = await self.call_tool(
                        tool, args
                    )  # lookups are read-only: retry a blip once
                if not home and FAILURE.search(str(result)) and not claimed:
                    return None
                if home and isinstance(result, str):  # the music tool's own sentence
                    spoken.append(result)
                    continue
                if home:
                    ok = isinstance(result, dict) and result.get("ok") is not False
                    if ok:
                        spoken.append(
                            NO_SLOT_SAY.get(name)
                            or SLOT_SAY[name].format(call.args[0] if call.args else "")
                        )
                    else:
                        spoken.append(
                            fast_path.spoken_result(fast_path.Match(tool, args, None), result)
                        )
                        break
                elif tool == "device_status":
                    said = device_sentence(result, routed.user)
                    if said is None:
                        return None
                    spoken.append(said)
                else:
                    spoken.append(str(result))
        if not take():
            return None
        text = " ".join(spoken)
        await on_text(text)
        self.router_lines[text] = routed.lines
        while len(self.router_lines) > 64:
            self.router_lines.pop(next(iter(self.router_lines)))
        elapsed = time.perf_counter() - started
        return {
            "status": "ok",
            "text": text,
            "first_call_s": routed.seconds,
            "first_text_s": elapsed,
            "end_s": elapsed,
            "routed": True,
            "router_s": routed.seconds,
            "router_lines": routed.lines,
            **({"race": routed.race} if routed.race else {}),
        }

    def offline(self) -> bool:
        return self.probe is not None and not self.probe.online

    async def _try_docs_first(
        self,
        user: str,
        on_text: Any,
        claim: Callable[[], bool] = lambda: True,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Try docs answer for escalate/none; (result, passage sources) or (None, [])."""
        if (
            not doc_route.has_enough_content_words(user)
            or "search_docs" not in self.tools
            or self.doc_request is None
        ):
            return None, []
        started = time.perf_counter()
        got, passages_list = await self._checked_docs(user)
        sources = [str(p.get("source", "")) for p in passages_list if p.get("source")]
        if got.status == "accepted" and got.text is not None:
            if not claim():
                return None, []
            body = doc_route.format_citation(
                got.text, got.quote or "", got.passage, doc_route.cached_files(self)
            )
            doc_s = round(time.perf_counter() - started, 3)
            meta = doc_route.citation_metadata(got.passage, got.quote or "")
            return await self.spoken(
                body,
                on_text,
                docs="accepted",
                doc_s=doc_s,
                route="docs",
                citation_metadata=meta,
            ), sources
        return None, []

    async def _ask_docs(self, said: str, session: str) -> DocResult:
        """Run ask_documents with per-turn cache (docs-first path)."""
        key = norm(said)
        if key in self._doc_cache:
            return self._doc_cache[key]

        async def _ctx(q: str) -> list[dict[str, Any]]:
            raw = await self.call_tool("file_context", {"query": q})
            return passages_with_meta(raw)

        async def _web(q: str) -> str | None:
            r = await self.lookup("web_answer", {"query": q})
            return str(r) if r is not None else None

        async def _catalog() -> list[str]:
            status = await self.call_tool("docs_status", {})
            names, _ = doc_route.parse_docs_status(status)
            return names

        async def _file_has(token: str, rel_key: str) -> bool:
            # fail open: False means "the name is absent" and rejects a valid answer, so an
            # error, a list cut at the limit, or an undecodable result cannot prove absence
            try:
                raw = await self.call_tool("docs_enumerate", {"query": token, "limit": 100})
            except Exception:
                return True
            if isinstance(raw, str) and "[truncated" in raw:
                return True
            rows = decode_rows(raw)
            if not rows and isinstance(raw, str) and raw.strip():
                return True  # undecodable: absence cannot be proven
            if len(rows) >= 100:  # cut at the limit: the file may sit past the end
                return True
            return any(r.get("rel_key") == rel_key for r in rows)

        assert self.doc_json_request is not None
        result = await ask_documents(
            said,
            context=_ctx,
            request=self.doc_json_request,
            web=_web,
            catalog=_catalog,
            file_names=doc_route.cached_files(self),
            budget_s=DOCS_TURN_BUDGET_S,
            file_has=_file_has,
        )
        self._doc_cache[key] = result
        self._sources[session] = list(dict.fromkeys(result.sources))
        return result

    async def _maybe_supplement(
        self,
        said: str,
        doc_result: DocResult,
        speak: Awaitable[dict[str, Any]],
        on_text: Any,
    ) -> dict[str, Any]:
        """The sufficiency check after an accepted document answer (ticket 250): the verdict
        starts as a task BEFORE the document answer is spoken (first audio is never delayed),
        the document answer is spoken as today, then the verdict is awaited (2.5 s). A stale
        or missing_half verdict with a web question gets ONE supplement sentence from the web
        lookup (worded as an addition, never a retraction; the ledger, hedge and directness
        check all apply); a wrong_question is logged, never spoken, and never paid for (no
        web lookup: an unspoken reply must not reach the turn's lookup log, or it would draw
        a stray "Source:" citation and a web_verify against an answer no one heard). When the
        document lane already ran the web for this turn (a status "web" result carries the
        web question and text), nothing is said or looked up: the answer is already whole,
        and the internal "(already looked up this turn)" marker is never spoken. No web
        lookup when the server is offline (R2-4), and the lookup is ONE non-retried call
        under a tight cap. Off unless sufficiency.SUFFICIENCY; the cheap gate and the
        per-turn flag limit it; any failure speaks nothing more (the old behaviour,
        byte-for-byte). A cancelled or failing speak cancels the verdict task with it."""
        if doc_result.web_text or doc_result.web_question:
            # R1-2: the document lane already called the web for this turn (status "web")
            # and appended the reply to the spoken answer: there is nothing to add, no
            # second lookup, and no internal "(already looked up this turn)" marker to speak
            return await speak
        task: asyncio.Task[sufficiency.Verdict | None] | None = None
        if (
            sufficiency.SUFFICIENCY
            and self.doc_json_request is not None
            and not self._sufficiency_attempted
        ):
            source_year = temporal.source_year(None, doc_result.file or "", doc_result.file or "")
            if sufficiency.should_check(said, source_year, doc_result.spoken or ""):
                self._sufficiency_attempted = True
                task = asyncio.create_task(
                    sufficiency.verdict(
                        self.doc_json_request,
                        said,
                        doc_result.spoken or "",
                        source_year,
                        sufficiency.today_line(),
                    )
                )
        try:
            result = await speak  # the document answer stands and speaks, as today
        except BaseException:
            # R1-4: the speak was cancelled (barging in, disconnect) or raised: the running
            # verdict task must not leak past the turn
            if task is not None:
                task.cancel()
            raise
        if task is None:
            return result
        started = time.perf_counter()
        try:
            got = await asyncio.wait_for(task, sufficiency.AWAIT_TIMEOUT_S)
        except (TimeoutError, Exception):
            got = None
        web_spoken = False
        if got is not None and got.verdict == "wrong_question":
            # R1-3: the answer is about something else than what was asked: log the verdict,
            # never speak a fix, and do not pay for a web lookup whose reply could not be
            # spoken (it would sit in the turn's lookup log, drawing a stray citation and a
            # web_verify against an answer no one heard)
            _logger.info(
                "sufficiency wrong_question (not spoken): verdict=%s reason=%s",
                got.verdict,
                got.reason,
            )
        elif (
            got is not None
            and got.verdict in ("stale", "missing_half")
            and got.web_question
            and not self.offline()  # R2-4: offline, the lookup would only stall the turn
        ):
            web: Any = None
            try:
                # R2-4: ONE non-retried lookup under a tight cap: the supplement is an
                # addition, so a transient blip must not pay for a second trip
                reply = await asyncio.wait_for(
                    self.call_tool("web_answer", {"query": got.web_question}),
                    sufficiency.WEB_TIMEOUT_S,
                )
                if not FAILURE.search(str(reply)):
                    web = reply
            except (TimeoutError, Exception):
                web = None
            text = "" if web is None else str(web)
            if (
                text
                and sufficiency.HEDGE_TAIL not in text
                and agent_tools.REPEAT_PREFIX not in text  # R1-2: the marker is never spoken
            ):
                # an addition, never a retraction: the document answer stays spoken as is
                lead = (
                    "The web has a newer figure: " if got.verdict == "stale" else "From the web: "
                )
                await on_text(lead + text)
                result["text"] = f"{result['text']} {lead}{text}"
                web_spoken = True
        result.update(
            {
                "sufficiency": got.verdict if got is not None else "none",
                "sufficiency_web": web_spoken,
                "sufficiency_s": round(time.perf_counter() - started, 3),
            }
        )
        return result

    async def _ask_general(self, said: str, session: str) -> DocResult:
        """Run answer_general with the per-turn cache (GENERAL lane, ticket 221b)."""
        key = norm(said)
        if key in self._doc_cache:
            return self._doc_cache[key]

        async def _catalog() -> list[dict[str, Any]]:
            return decode_rows(await self.call_tool("docs_catalog", {"max_chars": 6000}))

        async def _enumerate_(q: str) -> list[dict[str, Any]]:
            return decode_rows(await self.call_tool("docs_enumerate", {"query": q, "limit": 50}))

        assert self.doc_json_request is not None
        result = await answer_general(
            said,
            catalog=_catalog,
            enumerate_=_enumerate_,
            request=self.doc_json_request,
            budget_s=DOCS_TURN_BUDGET_S,
        )
        self._doc_cache[key] = result
        self._sources[session] = list(dict.fromkeys(result.sources))
        return result

    async def _docs_general_turn(
        self, said: str, session: str | None, on_text: Any, turned_at: float
    ) -> dict[str, Any]:
        """GENERAL lane (ticket 221b): the catalogue answer, shared by both branches so they
        cannot drift."""
        gen_result = await self._ask_general(said, session or "")
        doc_s = round(time.perf_counter() - turned_at, 3)
        self.route = "docs_general"
        if gen_result.status == "none":
            # a catalogue 'none' is not proof the folder is empty: check the actual file count
            names: list[str] = []
            with contextlib.suppress(TimeoutError, Exception):
                status = await asyncio.wait_for(
                    self.call_tool("docs_status", {}),
                    doc_route.DOCS_STATUS_TIMEOUT_S,
                )
                names, _ = doc_route.parse_docs_status(status)
            if names:
                return await self.spoken(
                    f"I have {len(names)} files; ask me about a topic or a folder.",
                    on_text,
                    route="docs_general",
                )
            return await self.spoken(doc_route.EMPTY_FOLDER_REPLY, on_text, route="docs_general")
        return await self.spoken(
            gen_result.spoken or "",
            on_text,
            docs=gen_result.status,
            doc_s=doc_s,
            route="docs_general",
            doc_calls=gen_result.calls,
            fallback=gen_result.fallback,
            sources=gen_result.sources,
        )

    async def _docs_miss_turn(
        self,
        doc_result: DocResult,
        doc_s: float,
        said: str,
        messages: list[dict[str, Any]],
        on_text: Any,
        on_tool: Any,
        on_say: Any,
        web_fallback: Callable[[str], Awaitable[dict[str, Any] | None]] | None = None,
        session: str = "",
    ) -> dict[str, Any]:
        """The no-answer policy for a docs miss (ticket 217d), shared by both branches: a
        doc-scoped 'none' refuses; the router's web/wiki calls run only when they are not
        refused (web_fallback, which speaks the miss prefix before the answer); otherwise
        Gemini with the prefix. The Gemini turn that follows a miss is not offered the document
        tools (ticket 244): it already has an unchecked miss, so it must not re-answer from
        search_docs. The document tools are hidden for this turn only and the base tool set is
        restored when the turn is done."""
        self.route = "docs_miss"
        base = (
            IDLE_DELEGATE_TOOLS
            if self.jobs_file is not None and not has_delegate_jobs(self.jobs_file)
            else frozenset()
        )
        try:
            self.agent.hide(DOC_MISS_HIDDEN_TOOLS)
            return await self._docs_miss_answer(
                doc_result, doc_s, said, messages, on_text, on_tool, on_say, web_fallback, session
            )
        finally:
            self.agent.hide(base)

    async def _docs_miss_answer(
        self,
        doc_result: DocResult,
        doc_s: float,
        said: str,
        messages: list[dict[str, Any]],
        on_text: Any,
        on_tool: Any,
        on_say: Any,
        web_fallback: Callable[[str], Awaitable[dict[str, Any] | None]] | None = None,
        session: str = "",
    ) -> dict[str, Any]:
        """The docs-miss answer, run with the document tools hidden (see _docs_miss_turn)."""
        doc_fields = _docs_miss_fields(doc_result, doc_s)
        # the miss prefix is spoken only when the question itself is about documents (243):
        # a non-document question that misses the documents is answered without it
        prefix = ""
        if (
            NO_ANSWER_POLICY
            and doc_result.status in ("none", "check_failed")
            and doc_route.names_documents(said, doc_route.cached_files(self))
        ):
            prefix = (
                DOCS_NO_COVER_PREFIX if doc_result.status == "none" else DOCS_UNCONFIRMED_PREFIX
            )
        # 247: before the plain miss line, a compound question (a document fact plus a web
        # fact) is split and answered per lane. The split is paid for only by a miss that
        # references the documents, at most once per turn; a None result falls through to
        # the existing answer unchanged.
        if (
            compound.COMPOUND
            and not self._compound_attempted
            and doc_result.status in ("none", "check_failed")
            and doc_route.names_documents(said, doc_route.cached_files(self))
        ):
            self._compound_attempted = True
            if (res := await compound.answer(self, said, doc_s, session, on_text)) is not None:
                return res
        if (
            NO_ANSWER_POLICY
            and doc_result.status == "none"
            and doc_route.document_cue(said, doc_route.cached_files(self))
        ):
            return await self.spoken(DOCS_DONT_SAY, on_text, **doc_fields)
        if web_fallback is not None and (web := await web_fallback(prefix)) is not None:
            web.update(doc_fields)
            return web
        msgs = self.steer(messages, case_f=True)
        if msgs and msgs[-1].get("role") == "user" and isinstance(msgs[-1].get("content"), str):
            msgs[-1]["content"] += " The user's documents were checked and do not answer this."

        async def _miss_gemini(
            m: list[dict[str, Any]] = msgs,
            p: str = prefix,
            f: dict[str, Any] = doc_fields,
        ) -> dict[str, Any]:
            if p:
                await on_text(p)
            res = await self.gemini(m, on_text, on_tool, on_say)
            if p:
                res["text"] = _join_miss_prefix(p, res.get("text") or "")
            res.update(f)
            return res

        return await _miss_gemini()

    async def spoken(self, text: str, on_text: Any, **info: Any) -> dict[str, Any]:
        await on_text(text)
        return {"status": "ok", "text": text, **info}

    async def gemini(
        self, messages: list[dict[str, Any]], on_text: Any, on_tool: Any, on_say: Any
    ) -> dict[str, Any]:
        if self.offline():
            return await self.spoken(OFFLINE_SAY, on_text, offline=True)
        result: dict[str, Any] = await self.agent.ask(
            messages, reset=True, on_text=on_text, on_tool=on_tool, on_say=on_say
        )
        return result

    async def routed_turn(
        self, messages: list[dict[str, Any]], on_text: Any, on_tool: Any, on_say: Any
    ) -> dict[str, Any]:
        """The local router and Gemini answer the turn at the same time when the agent is hedged
        (the first to answer wins, hedge.py); otherwise the router goes first and Gemini only
        takes what it cannot run."""
        assert self.router is not None
        router = self.router
        started = time.perf_counter()
        user = str(messages[-1].get("content", ""))
        info: dict[str, Any] = {}
        hedged = isinstance(self.agent, HedgedAgent)  # Gemini may win: say nothing before the claim

        async def local(claim: Callable[[], bool]) -> dict[str, Any] | None:
            routed = await router.route(self.history(messages), user, set(self.tools))
            info.update(router_s=routed.seconds, router_lines=routed.lines)
            has_history = any(m.get("role") == "assistant" for m in messages[:-1])
            if (
                routed.reason is None
                and has_history
                and call_policy.lookup_needs_context(routed.calls, user)
            ):
                routed.reason = "the question refers back to the conversation"
            if self.offline() and re.match(r"\s*(web_answer|wiki_summary)\|", routed.lines):
                info["router_reason"] = "offline lookup"  # no cache: fail at once
                return None
            if routed.reason is None:
                done = await self.run_routed(
                    routed, started, on_text, claim, on_tool, None if hedged else on_say
                )
                if done is not None:
                    return done
                routed.reason = "lookup failed or Gemini first"
                if routed.race:
                    info["race"] = routed.race
            # Docs-first for escalate/none/unusable: try documents before Gemini
            if routed.calls and all(c.name in ("escalate", "none") for c in routed.calls):
                docs_result, sources = await self._try_docs_first(user, on_text, claim)
                if docs_result is not None:
                    self._checked_sources = sources
                    return docs_result
            info["router_reason"] = routed.reason
            return None

        if isinstance(self.agent, HedgedAgent):
            # the requests start before the router decides, so the base prompt only; the Case F
            # protocol joins the turns from the moment a chain is active (the handle above)
            result = await self.agent.ask(
                self.stamped(messages),
                reset=True,
                on_text=on_text,
                on_tool=on_tool,
                on_say=on_say,
                local=local,
            )
            return result if result.get("routed") else {**result, **info}
        done = await local(lambda: True)
        if done is not None:
            return done
        looked_up = re.match(
            r"\s*(web_answer|wiki_summary|weather)\|", str(info.get("router_lines", ""))
        )
        if looked_up and (self.offline() or time.perf_counter() - started > LOOKUP_GIVE_S):
            # the link is slow or down: Gemini would only repeat the same slow trip
            return await self.spoken(LOOKUP_FAIL_SAY, on_text, **info)
        # the router could not answer: the escalated turn gets the full Case F protocol
        result = await self.gemini(self.steer(messages, case_f=True), on_text, on_tool, on_say)
        return {**result, **info}

    def check_in_background(self, question: str, claim: str, turn: str = "") -> None:
        """Ask the grounded second-opinion tool about a reply that was just spoken (web_verify:
        it skips what does not change and stops at its allowance). A different answer comes back
        as a notice the client says when it is idle; anything else is only logged, in
        checks/<run>.jsonl next to the turn log (not in it: the turn file is one row per turn)."""
        if "web_verify" not in self.tools or not claim:
            return

        async def check() -> None:
            started = time.perf_counter()
            ledger_from = trace.ledger_size(self.ledger)
            try:
                async with asyncio.timeout(20):
                    out = str(
                        await self.call_tool("web_verify", {"question": question, "claim": claim})
                    )
            except Exception as exc:  # a check that fails changes nothing
                _logger.info("second opinion failed: %s", exc)
                return
            first = out.splitlines()[0] if out else ""
            checks = self.log.parent / "checks" / f"{self.run_id}.jsonl"
            checks.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "time": time.strftime("%H:%M:%S"),
                "run": self.run_id,
                "turn": turn,
                "question": question,
                "verdict": first[:60],
                "s": round(time.perf_counter() - started, 3),
                "ledger": trace.ledger_slice(
                    self.ledger, ledger_from, trace.ledger_size(self.ledger)
                ),
            }
            with checks.open("a") as sink:
                sink.write(json.dumps(row) + "\n")
            fix = re.search(r"CORRECTION:\s*(.+)", out)
            if first.startswith("VERDICT: DIFFERS") and fix:
                # two grounded looks disagree and neither is known to be right: say both
                self.notice_ids += 1
                text = f"A second check found something different. {fix.group(1)}"
                self.unacked[self.notice_ids] = {"text": text, "job": ""}

        task = asyncio.create_task(check())
        self.checks.add(task)
        task.add_done_callback(self.checks.discard)

    def collect_notices(self) -> None:
        """Move the delegate server's new notices into `unacked` (each gets an id)."""
        for item in self.notices.pop_items() if self.notices is not None else []:
            self.notice_ids += 1
            self.unacked[self.notice_ids] = item
            self.note_chain_job(item.get("job", ""))

    def note_chain_job(self, job_id: str) -> None:
        """The chain's job ended: remember what it made. TODO(ticket 04): move to
        awaiting_publish_confirm when the user asked to publish."""
        chain = self.chain
        if chain is None or not job_id or chain.data["job_id"] != job_id or not chain.active():
            return
        try:
            jobs = json.loads(chain.path.with_name("jobs.json").read_text())["jobs"]
            row = next(j for j in jobs if j["id"] == job_id)
        except (OSError, ValueError, KeyError, StopIteration):
            return
        files = row.get("files") or []
        main = agent_tools.main_file(files)
        if row.get("status") == "done" and main:
            chain.update(artifact_path=main)
            manifest = agent_tools.read_manifest(main)
            if manifest is not None:  # what the deliverable holds and lacks
                chain.update(items=manifest["items"], gaps=manifest["gaps"])
        if row.get("status") == "done":
            chain.mark_done()

    async def drop_chain(self, on_text: Any) -> dict[str, Any]:
        """The user said never mind: end the chain, and stop its job if it still runs."""
        chain = self.chain
        assert chain is not None
        job = str(chain.data["job_id"])
        chain.update("idle")
        if job and "delegate_cancel" in self.tools:
            with contextlib.suppress(Exception):
                await self.call_tool("delegate_cancel", {"job": job})
        return await self.spoken("Okay, dropped.", on_text)

    def _chain_yields_to_docs(self, said: str) -> bool:
        """An open chain hands this turn to the checked document path (240): docs-first is on
        and the utterance names a file, or it is a question while the chain is not mid-protocol
        (in the awaiting_* phases a turn answers the chain's own question: the chain keeps it).
        A yielding turn is not cancelled out of the chain; it counts as one unrelated turn."""
        if not (
            self.docs_first and self.doc_json_request is not None and "file_context" in self.tools
        ):
            return False
        chain = self.chain
        if chain is None:
            return False
        if doc_route.document_cue(said, doc_route.cached_files(self)):
            return True
        return doc_route.wh_question_start(said) and chain.data["phase"] not in _CHAIN_MID_PROTOCOL

    async def notice_channel(self, ws: ServerConnection) -> None:
        """A push channel for finished jobs (R1): every notice goes out the moment it appears in
        the file, and again on a new connection until the client acks it as played to the end.
        The client decides when the user is idle; this side only keeps the notices."""
        sent: set[int] = set()

        async def acks() -> None:
            async for raw in ws:
                with contextlib.suppress(ValueError, AttributeError, TypeError):
                    self.unacked.pop(int(json.loads(raw)["ack"]), None)

        reader = asyncio.create_task(acks())
        try:
            while not reader.done():
                self.collect_notices()
                for nid, item in list(self.unacked.items()):
                    if nid not in sent:
                        sent.add(nid)
                        note = {"id": nid, "text": item["text"], "job": item.get("job", "")}
                        await ws.send(json.dumps({"notice": note}))
                await asyncio.sleep(NOTICE_POLL_S)
        except ConnectionClosed:
            pass
        finally:
            reader.cancel()

    async def delayed(self, work: Any, user: str) -> dict[str, Any]:
        """`work`, after the test-only pause when the user's words contain --test-delay-word (to
        hear the client's wait and give-up lines on demand); the pause is cancelled like a turn."""
        if self.test_delay is not None and self.test_delay[0] in user.lower():
            try:
                await asyncio.sleep(self.test_delay[1])
            except asyncio.CancelledError:
                work.close()  # the turn was dropped before the work coroutine ran: no GC warning
                raise
        result: dict[str, Any] = await work
        return result

    async def sync_music_paused(self) -> None:
        """The client paused the music itself (the user spoke over it): the music tool's state
        follows, silently, so a bare "play" resumes it. A no-op when nothing is playing."""
        if "music" not in self.tools:
            return
        with contextlib.suppress(Exception):
            await self.call_tool("music", {"action": "pause", "query": ""})

    async def handle(self, ws: ServerConnection) -> None:
        request = json.loads(await ws.recv())
        if request.get("notices"):
            await self.notice_channel(ws)
            return
        session = trace.clean_session(request.get("session"))
        messages = [m for m in request["messages"] if m.get("role") != "system"]
        if request.get("music") == "paused":
            await self.sync_music_paused()

        payloads = PayloadFilter()
        sources = SourceFilter()
        sentences = SentenceFilter(
            lambda: any(c.get("tool") == "delegate" for c in self.record[before:])
        )

        async def send_event(event: dict[str, Any]) -> None:
            await ws.send(json.dumps(event))

        async def on_text(text: str) -> None:
            if text := sentences.feed(sources.feed(payloads.feed(text))):
                with contextlib.suppress(ConnectionClosed):
                    await ws.send(json.dumps({"text": text}))

        async def on_say(text: str) -> None:
            # its own flagged message: the client speaks it now instead of holding it back
            if text := sources.feed(payloads.feed(text)):
                sentences.note(text)  # what the model says next must not restate it
                with contextlib.suppress(ConnectionClosed):
                    await ws.send(json.dumps({"text": text, "flush": True}))

        async def on_tool(name: str, said: bool) -> None:
            with contextlib.suppress(ConnectionClosed):
                await ws.send(json.dumps({"event": "tool", "name": name, "said": said}))

        async with self.lock:
            agent_tools.set_event_sink(send_event)
            if self.jobs_file is not None:
                self.agent.hide(
                    frozenset() if has_delegate_jobs(self.jobs_file) else IDLE_DELEGATE_TOOLS
                )
            before = len(self.record)
            self.turns += 1
            turn_id = trace.turn_id(self.run_id, self.turns)
            trace.begin_turn()
            self.route, self.tail = "", []
            self._docs_checked = False  # per-turn: only this turn's docs miss skips doc_race
            self._compound_attempted = False  # per-turn: the compound split runs at most once (247)
            self._sufficiency_attempted = False  # per-turn: sufficiency runs at most once (250)
            ledger_from = trace.ledger_size(self.ledger)
            tokens_before = trace.usage_total(self.agent)
            last = messages[-1] if messages else {}
            question.current = str(last.get("content", "")) if last.get("role") == "user" else ""
            found = (
                fast_path.match(str(last.get("content", "")))
                if last.get("role") == "user"
                else None
            )
            agent_tools.reset_turn_lookups()
            chain = self.chain
            to_model = found is not None and self.refers_to_setting(
                question.current, messages, before
            )
            if to_model:
                found = None  # not the volume: the router would say the same, so it is skipped too
            if (
                found is not None
                and found.args.get("what") in ("place", "barangay")
                and chain is not None
                and chain.active()
            ):
                found = None  # "where are we" during a task is about the task: Gemini sees it
            said = question.current
            if (
                chain is not None
                and chain.active()
                and chain.data["done"]
                and not wants_cancel(said)
                and not is_followup(said)
                and not is_side_question(said)
            ):
                chain.update("idle")  # the finished task is over; its facts go with it
            chain_yield = (
                chain is not None
                and chain.active()
                and last.get("role") == "user"
                and not (chain.data["done"] and is_side_question(said))
                and self._chain_yields_to_docs(said)
            )
            if chain_yield:
                # a document question slips past the open chain: the documents answer it, the
                # chain is not cancelled, the turn just counts as unrelated (three expire it)
                assert chain is not None  # chain_yield is true only while a chain is open
                chain.touch_unrelated()
            work: Any
            if chain is not None and chain.active() and wants_cancel(said):
                self.route = "cancel_chain"
                work = self.drop_chain(on_text)
            elif (
                last.get("role") == "user"
                and (is_backchannel(said) or is_apology(said))
                and not (
                    chain is not None and chain.active() and chain.data["phase"].startswith("await")
                )
                and not (
                    len(messages) > 1
                    and call_policy.OFFER.search(str(messages[-2].get("content", "")))
                )
            ):
                self.route = "backchannel"
                work = self.spoken("", on_text)  # "mm-hmm", "I'm sorry": no request, no reply
            elif found is not None and (found.tool == "clock" or found.tool in self.tools):
                self.route = "fast"
                work = self.fast(found, on_text)
            elif (
                chain is not None
                and chain.active()
                and last.get("role") == "user"
                and not (chain.data["done"] and is_side_question(said))
                and not chain_yield
            ):
                # a research/build task is under way: Gemini sees its state and protocol
                self.route = "chain"
                work = self.gemini(self.steer(messages, case_f=True), on_text, on_tool, on_say)
            else:
                if (
                    self.docs_first
                    and self.doc_json_request is not None
                    and "file_context" in self.tools
                    and last.get("role") == "user"
                    and not call_policy.answers_an_offer(messages, self.router_lines)
                ):
                    self._doc_cache.clear()
                    turned_at = time.perf_counter()
                    has_assistant = any(m.get("role") == "assistant" for m in messages[:-1])
                    if doc_route.refers_to_own_sources(said) and has_assistant:
                        self.route = "sources"
                        sess_sources = self._sources.get(session or "", [])
                        if sess_sources:
                            work = self.spoken(
                                doc_route.checked_sources_sentence(sess_sources, 3),
                                on_text,
                                route="sources",
                            )
                        else:
                            work = self.spoken(
                                "I haven't looked anything up yet.",
                                on_text,
                                route="sources",
                            )
                    elif (
                        self.router is None
                        and not to_model
                        and not (
                            doc_route.non_document_intent(said)
                            and not doc_route.document_cue(said, doc_route.cached_files(self))
                        )
                        and doc_route.is_general_question(said)
                    ):
                        work = self._docs_general_turn(said, session, on_text, turned_at)
                    elif (
                        self.router is None
                        and not to_model
                        and not (
                            doc_route.non_document_intent(said)
                            and not doc_route.document_cue(said, doc_route.cached_files(self))
                        )
                        and doc_route.question_like(said)
                    ):
                        # no local router (the Pi): every question goes to the checked document
                        # path first; commands and chit-chat fall through to Gemini below
                        doc_result = await self._ask_docs(said, session or "")
                        doc_s = round(time.perf_counter() - turned_at, 3)
                        if doc_result.status == "accepted":
                            self.route = "docs"
                            work = self._maybe_supplement(
                                said,
                                doc_result,
                                self.spoken(
                                    doc_result.spoken or "",
                                    on_text,
                                    docs=doc_result.status,
                                    doc_s=doc_s,
                                    route="docs",
                                    doc_calls=doc_result.calls,
                                    fallback=doc_result.fallback,
                                    sources=doc_result.sources,
                                ),
                                on_text,
                            )
                        else:
                            work = self._docs_miss_turn(
                                doc_result,
                                doc_s,
                                said,
                                messages,
                                on_text,
                                on_tool,
                                on_say,
                                session=session or "",
                            )
                    elif self.router is not None and not to_model:

                        async def routed_or_agent(r: Routed) -> dict[str, Any]:
                            """The router's calls; Gemini takes the turn when they cannot run."""
                            done = await self.run_routed(r, turned_at, on_text)
                            if done is not None:
                                return done
                            return await self.gemini(
                                self.steer(messages, case_f=True), on_text, on_tool, on_say
                            )

                        async def tagged(w: Any, r: Routed) -> dict[str, Any]:
                            """The turn's result, with what the router said kept in the trace."""
                            res: dict[str, Any] = await w
                            res.setdefault("router_lines", r.lines)
                            res.setdefault("router_reason", r.reason or "")
                            res.setdefault("router_s", r.seconds)
                            return res

                        routed = await self.router.route(
                            self.history(messages),
                            said,
                            set(self.tools),
                        )
                        if (
                            routed.calls
                            and all(c.name in ACTION_CALLS for c in routed.calls)
                            and not doc_route.document_cue(said, doc_route.cached_files(self))
                        ):
                            # a command is never a document question, even one that cannot run
                            self.route = "routed"
                            if routed.reason is None:
                                work = routed_or_agent(routed)
                            else:
                                work = self.gemini(
                                    self.steer(messages, case_f=True), on_text, on_tool, on_say
                                )
                        else:
                            # a closed-list pleasantry with no document cue ("Thanks!",
                            # "okay", "goodbye") is not a document question - the router's
                            # own reply answers, with no docs call and no prefix. Everything
                            # else goes to the documents, exactly as before 235: 241 inverts
                            # the old gate (a question_like skip dropped real questions with
                            # spoken lead-ins, "Got it, what's the office Wi-Fi password?")
                            if doc_route.is_pleasantry(said) and not doc_route.document_cue(
                                said, doc_route.cached_files(self)
                            ):
                                self.route = "routed"
                                if routed.reason is None:
                                    work = routed_or_agent(routed)
                                else:
                                    work = self.gemini(
                                        self.steer(messages, case_f=True), on_text, on_tool, on_say
                                    )
                            elif doc_route.is_general_question(said):
                                work = self._docs_general_turn(said, session, on_text, turned_at)
                            else:
                                doc_result = await self._ask_docs(said, session or "")
                                doc_s = round(time.perf_counter() - turned_at, 3)
                                if doc_result.status == "accepted":
                                    self.route = "docs"
                                    work = self._maybe_supplement(
                                        said,
                                        doc_result,
                                        self.spoken(
                                            doc_result.spoken or "",
                                            on_text,
                                            docs=doc_result.status,
                                            doc_s=doc_s,
                                            route="docs",
                                            doc_calls=doc_result.calls,
                                            fallback=doc_result.fallback,
                                            sources=doc_result.sources,
                                        ),
                                        on_text,
                                    )
                                else:
                                    # the docs verdict is a miss: in the web fallback the legacy
                                    # doc_race must not re-answer it (it could speak the decoy
                                    # ask_documents just rejected)
                                    self._docs_checked = True

                                    async def _web_or_none(prefix: str) -> dict[str, Any] | None:
                                        if (
                                            routed.reason is not None
                                            or not routed.calls
                                            or not all(
                                                c.name in {"web_answer", "wiki_summary"}
                                                for c in routed.calls
                                            )
                                        ):
                                            return None
                                        self.route = "routed"
                                        if prefix:  # the miss prefix goes out before the answer
                                            await on_text(prefix)
                                        result = await routed_or_agent(routed)
                                        if prefix:
                                            result["text"] = _join_miss_prefix(
                                                prefix, result.get("text") or ""
                                            )
                                        return result

                                    work = self._docs_miss_turn(
                                        doc_result,
                                        doc_s,
                                        said,
                                        messages,
                                        on_text,
                                        on_tool,
                                        on_say,
                                        web_fallback=_web_or_none,
                                        session=session or "",
                                    )
                        work = tagged(work, routed)
                    else:
                        prefetched_status: Any = None
                        can_docs = (
                            "search_docs" in self.tools
                            and self.doc_request is not None
                            and last.get("role") == "user"
                        )
                        is_docs = can_docs and doc_route.refers_to_documents(
                            said, doc_route.cached_files(self)
                        )
                        if (
                            can_docs
                            and not is_docs
                            and "docs_status" in self.tools
                            and doc_route.might_name_a_file(said)
                        ):
                            try:
                                prefetched_status = await asyncio.wait_for(
                                    self.call_tool("docs_status", {}),
                                    doc_route.DOCS_STATUS_TIMEOUT_S,
                                )
                                names, _ = doc_route.parse_docs_status(prefetched_status)
                                if names:
                                    doc_route.set_cached_files(self, names)
                                is_docs = doc_route.refers_to_documents(
                                    said, doc_route.cached_files(self)
                                )
                            except (TimeoutError, Exception):
                                pass

                        def normal_turn() -> tuple[str, Any]:
                            if self.router is not None and not to_model:
                                return "routed", self.routed_turn(
                                    messages, on_text, on_tool, on_say
                                )
                            return "gemini", self.gemini(
                                self.steer(messages, case_f=self.router is None),
                                on_text,
                                on_tool,
                                on_say,
                            )

                        if doc_route.refers_to_sources(said) and last.get("role") == "user":
                            self.route = "sources"
                            if self._checked_sources:
                                work = self.spoken(
                                    doc_route.checked_sources_sentence(self._checked_sources, 5),
                                    on_text,
                                    route="sources",
                                )
                            else:
                                work = self.spoken(
                                    "I haven't looked anything up yet.",
                                    on_text,
                                    route="sources",
                                )
                        elif is_docs:
                            self.route = "docs"
                            work = doc_route.handle_docs_turn(
                                self,
                                said,
                                on_text,
                                prefetched_status=prefetched_status,
                            )
                        elif can_docs and doc_route.has_enough_content_words(said):
                            work = doc_route.handle_implicit_docs_turn(
                                self,
                                said,
                                on_text,
                                normal_turn,
                                prefetched_status=prefetched_status,
                            )
                        else:
                            self.route, work = normal_turn()
                else:
                    prefetched_status_2: Any = None
                    can_docs = (
                        "search_docs" in self.tools
                        and self.doc_request is not None
                        and last.get("role") == "user"
                        and not call_policy.answers_an_offer(messages, self.router_lines)
                    )
                    is_docs = can_docs and doc_route.refers_to_documents(
                        said, doc_route.cached_files(self)
                    )
                    if (
                        can_docs
                        and not is_docs
                        and "docs_status" in self.tools
                        and doc_route.might_name_a_file(said)
                    ):
                        try:
                            prefetched_status_2 = await asyncio.wait_for(
                                self.call_tool("docs_status", {}),
                                doc_route.DOCS_STATUS_TIMEOUT_S,
                            )
                            names, _ = doc_route.parse_docs_status(prefetched_status_2)
                            if names:
                                doc_route.set_cached_files(self, names)
                            is_docs = doc_route.refers_to_documents(
                                said, doc_route.cached_files(self)
                            )
                        except (TimeoutError, Exception):
                            pass

                    def normal_turn() -> tuple[str, Any]:
                        if (
                            self.router is not None
                            and last.get("role") == "user"
                            and not to_model
                            and not call_policy.answers_an_offer(messages, self.router_lines)
                        ):
                            return "routed", self.routed_turn(messages, on_text, on_tool, on_say)
                        return "gemini", self.gemini(
                            self.steer(messages, case_f=self.router is None),
                            on_text,
                            on_tool,
                            on_say,
                        )

                    if doc_route.refers_to_sources(said) and last.get("role") == "user":
                        self.route = "sources"
                        if self._checked_sources:
                            work = self.spoken(
                                doc_route.checked_sources_sentence(self._checked_sources, 5),
                                on_text,
                                route="sources",
                            )
                        else:
                            work = self.spoken(
                                "I haven't looked anything up yet.",
                                on_text,
                                route="sources",
                            )
                    elif is_docs:
                        self.route = "docs"
                        work = doc_route.handle_docs_turn(
                            self,
                            said,
                            on_text,
                            prefetched_status=prefetched_status_2,
                        )
                    elif can_docs and doc_route.has_enough_content_words(said):
                        work = doc_route.handle_implicit_docs_turn(
                            self,
                            said,
                            on_text,
                            normal_turn,
                            prefetched_status=prefetched_status_2,
                        )
                    else:
                        self.route, work = normal_turn()
            turn = asyncio.create_task(self.delayed(work, question.current))
            closed = asyncio.create_task(ws.wait_closed())
            await asyncio.wait({turn, closed}, return_when=asyncio.FIRST_COMPLETED)
            cancelled = not turn.done()
            if cancelled:
                # R1-4: the connection is gone: stop the turn here. Left alone, the
                # sufficiency supplement would keep running its verdict and web lookup
                # (several more seconds), stalling self.lock for the next turn — even a fast
                # ME3 command — and paying for calls the user will not hear.
                await self.agent.cancel()
                turn.cancel()
            try:
                result = await turn
            except asyncio.CancelledError:
                if not turn.cancelled():
                    raise  # handle itself was cancelled, not this turn
                result = {"status": "cancelled", "text": ""}
            ledger_to = trace.ledger_size(self.ledger)
            tokens = trace.usage_delta(tokens_before, trace.usage_total(self.agent))
            agent_tools.set_event_sink(None)
            closed.cancel()
            for held in [*payloads.finish(), *sources.finish()]:
                await on_text(held)
            if rest := sentences.finish():
                with contextlib.suppress(ConnectionClosed):
                    await ws.send(json.dumps({"text": rest}))
            if isinstance(result.get("text"), str):
                result["text"] = strip_sources(result["text"])
                for gone in sentences.dropped:  # the log holds what was spoken
                    result["text"] = result["text"].replace(gone, "", 1)
                result["text"] = re.sub(r"\s{2,}", " ", result["text"]).strip()
            if (
                not cancelled
                and result["status"] == "ok"
                and isinstance(result.get("text"), str)
                and (cite := agent_tools.turn_source_sentence(result["text"]))
            ):
                # the citation is the code's to guarantee, not the model's to remember
                with contextlib.suppress(ConnectionClosed):
                    await ws.send(json.dumps({"text": cite}))
                result["text"] = f"{result['text']} {cite.strip()}"
                result["cited_by_server"] = True
            if not cancelled:
                reply = (
                    {"done": True, "turn": turn_id}
                    if result["status"] == "ok"
                    else {"error": "timeout"}
                )
                try:
                    await ws.send(json.dumps(reply))
                except ConnectionClosed:
                    cancelled = True
            if not cancelled and result["status"] == "ok" and (claim := agent_tools.turn_claim()):
                self.check_in_background(said, claim, turn_id)
            row = {
                "time": time.strftime("%H:%M:%S"),
                "question": messages[-1]["content"] if messages else "",
                "calls": self.record[before:],
                "cancelled": cancelled,
                "run": self.run_id,
                "turn": turn_id,
                **({"session": session} if session else {}),
                "route": self.route,
                "tail": self.tail,
                "tokens": tokens,
                "ledger": trace.ledger_slice(self.ledger, ledger_from, ledger_to),
                **(
                    {"unchecked": True}  # a changeable fact answered with no lookup: audit these
                    if call_policy.changeable_question(said)
                    and result.get("text")
                    and not any(
                        c.get("tool") in call_policy.LOOKUP_ANSWERS or "intent" in c
                        for c in self.record[before:]
                    )
                    else {}
                ),
                **({"dropped_payload": payloads.dropped} if payloads.dropped else {}),
                **({"dropped_sentences": sentences.dropped} if sentences.dropped else {}),
                **result,
            }
            with self.log.open("a") as sink:
                sink.write(json.dumps(row) + "\n")
            if chain is not None and not cancelled and chain.active() and not chain_yield:
                # lights, music, the clock: turns the router or the fast path handle make no chain
                # call, so they never counted; five of them left a research chain alive until a
                # website request picked its notes up (2026-11-11). A plain web answer is not chain
                # work either (240): the chain may still call web_answer, but the turn does not
                # keep the chain alive; chain_yield turns were already counted above.
                used = [str(c.get("tool", "")) for c in self.record[before:]]
                if not any(is_chain_tool(t) for t in used if t != "web_answer") and not is_answer(
                    said
                ):
                    chain.touch_unrelated(side=all(t in SIDE_TOOLS for t in used))


def now_line(now: datetime | None = None) -> str:
    """The line every Gemini request ends with: the date, the time, the time zone and the place
    (the model has no clock, and "here" must not be guessed)."""
    now = now or datetime.now(ZoneInfo(fast_path.TIMEZONE))
    return f"[Now: {now:%A, %B %-d, %Y, %-I:%M %p}, {fast_path.TIMEZONE}. Place: {venue.here()}.]"


async def warm_docs(tools: Any) -> None:
    """Warm the docs index once at server start; never fail startup."""
    standard = getattr(tools, "standard_tools", ()) or ()
    schema = next((t for t in standard if t.name == "search_docs"), None)
    if schema is None or schema.handler is None:
        return

    async def discard(result: Any, *rest: Any, **kwargs: Any) -> None:
        pass

    params = SimpleNamespace(
        function_name="search_docs",
        tool_call_id="warm-up",
        arguments={"query": DOCS_WARMUP_QUERY},
        result_callback=discard,
    )
    try:
        await asyncio.wait_for(schema.handler(params), DOCS_WARMUP_TIMEOUT_S)
    except Exception as exc:
        print(f"docs warm-up skipped: {type(exc).__name__}", flush=True)


async def run(
    port: int,
    live: str | None,
    system_file: Path,
    pi_url: str | None,
    fast: bool = True,
    hedge: float | None = 0.0,
    keepalive: float = KEEPALIVE_S,
    ipv4_only: bool = False,
    router_url: str | None = None,
    router_timeout: float = TIMEOUT_S,
    router_hold: float = 0.0,
    notices: Path | None = None,
    builtin_search: bool = False,
    token: str | None = None,
    hide_idle_delegate: Path | None = None,
    say_delays: dict[str, float] | None = None,
    test_delay: tuple[str, float] | None = None,
    host: str = "127.0.0.1",
    docs_dir: str | Path | None = None,
    docs_retriever: str | None = None,
    docs_first: bool = False,
    docs_db: str | Path | None = None,
) -> int:
    record: list[dict[str, Any]] = []
    state_dir = (
        notices.parent
        if notices
        else Path(agent_tools.DELEGATE_STATE_DIR or "~/.local/state/voice_agent/delegate")
    ).expanduser()
    chain = ChainState(state_dir)  # an expired state is reset to idle here
    scratch: Scratchpad | None = None
    with contextlib.suppress(OSError, KeyError, ValueError):
        pad = Scratchpad(agent_tools.delegate_workspace() / "scratch" / "NOTES.md")
        chain.on_change = lambda event: pad.reset() if event == "end" else None
        scratch = pad
    resolved_docs = agent_tools.resolve_docs_dir(docs_dir)
    docs_args: dict[str, Any] = {"docs_dir": resolved_docs}
    if docs_retriever is not None:
        docs_args["docs_retriever"] = docs_retriever
    if docs_db is not None:
        docs_args["docs_db"] = docs_db
    servers, optional = standard_servers(pi_url, **docs_args)
    tools = await build_tools(
        record, live=live, servers=servers, optional=optional, chain=chain, scratchpad=scratch
    )
    system = system_file.read_text()
    selfinfo.load(agent_tools.REPO_ROOT / "config" / "profile.md")  # answers about the owner, local
    with contextlib.suppress(OSError, KeyError, ValueError):
        profile = agent_tools.REPO_ROOT / "config" / "profile.md"
        if install_profile(profile, agent_tools.delegate_workspace() / "scratch") and (
            system_file.name == "system_agent.txt"
        ):
            system += "\n\n" + PROFILE_RULE  # only when the owner filled one in
    if (who := selfinfo.owner_name()) and system_file.name == "system_agent.txt":
        system += f"\n\nThe user's name is {who}: use it when they ask to be greeted or addressed."
    with contextlib.suppress(Exception):  # once, about 1 s; offline the answer names the city only
        await asyncio.wait_for(venue.resolve(), venue.TIMEOUT_S + 1)
    if (line := venue.prompt_line()) and system_file.name == "system_agent.txt":
        system += "\n\n" + line
    if agent_tools.publish_configured() and system_file.name == "system_agent.txt":
        system += "\n\n" + agent_tools.PUBLISH_RULE  # the tool exists only when a repo is set
    case_f_block = ""  # P2: the Case F chain protocol, kept out of the base prompt
    if not builtin_search and system_file.name == "system_agent.txt":
        with contextlib.suppress(OSError):
            case_f_block = system_file.with_name("system_agent_case_f.txt").read_text()
    seen_by_model = with_builtin_search if builtin_search else (lambda t: t)

    def llm_tools(t: Any) -> Any:  # what the model sees: no music tool (smart_home reaches it)
        return without_tools(seen_by_model(t), MODEL_HIDDEN)

    if builtin_search:  # the model has to know the date to know what can have changed
        system = system.replace(
            "{today}", f"{datetime.now(ZoneInfo(fast_path.TIMEZONE)):%A, %B %-d, %Y}"
        )

    def llm() -> Any:
        return build_llm(
            system, keepalive_s=keepalive, ipv4_only=ipv4_only, builtin_search=builtin_search
        )

    agent: Agent | HedgedAgent
    if hedge is None:
        agent = Agent(llm(), llm_tools(tools), usage=True, say_delays=say_delays)
        await agent.start()
        await agent.ask([{"role": "user", "content": "Say OK."}], reset=True)  # warm-up
    else:  # R4: two requests per turn, the first to answer wins (hedge.py)
        gate = HedgeGate()
        agent = HedgedAgent(
            [
                Agent(
                    llm(), llm_tools(gated_tools(tools, gate, i)), usage=True, say_delays=say_delays
                )
                for i in range(2)
            ],
            gate,
            hedge,
            hold_s=router_hold,
        )
        await agent.start()
        await agent.warm_up()
    log = ROOT / "out/agent_server" / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    router = None
    if router_url:
        book = {c["name"]: c["relation"] for c in load_phonebook()}
        router = LocalRouter(router_url, book, router_timeout, api_key=token)
        await router.warm_up()
    queue = NoticeQueue(notices) if notices else None
    doc_request = None
    doc_json_request = None
    if any(t.name == "search_docs" for t in tools.standard_tools):
        doc_request = await gemini_request(keepalive, ipv4_only)
        if docs_first:
            doc_json_request = await gemini_json_request(keepalive_s=keepalive, ipv4_only=ipv4_only)
        await warm_docs(tools)
    server = AgentServer(
        agent,
        record,
        log,
        tools if fast or router else None,
        router,
        queue,
        doc_request,
        hide_idle_delegate,
        chain,
        case_f_block=case_f_block,
        docs_first=docs_first,
        doc_json_request=doc_json_request,
    )
    server.test_delay = test_delay
    server.stamp = now_line
    net.prefer_ipv4()  # a broken IPv6 route made the probe call the Pi offline (8 s connects)
    server.probe = OnlineProbe()
    probe_task = asyncio.create_task(server.probe.run())
    try:
        guard = token_guard(token) if token else None
        async with serve(server.handle, host, port, process_request=guard):
            note = "" if token else ", NO TOKEN (--no-token)"
            print(f"agent server on ws://{host}:{port}{note}, log {log}", flush=True)
            await asyncio.Future()
    finally:
        probe_task.cancel()
        await close_llm_sessions()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--host",
        default=os.environ.get("AGENT_HOST", "127.0.0.1"),
        help="address to listen on ($AGENT_HOST): 127.0.0.1 (the laptop stack) or the Pi's cable "
        "address (HARNESS=pi); anything but loopback needs the session token",
    )
    parser.add_argument("--port", type=int, default=8781)
    parser.add_argument(
        "--apps-root",
        default=None,
        help="the web apps folder (default ~/agent_sandbox/workspace/apps; $APPS_ROOT)",
    )
    parser.add_argument(
        "--apps-host",
        default=os.environ.get("APPS_PUBLIC_HOST"),
        help="open_app sends the client http://<this host>:8795/<name>/ instead of opening a "
        "browser here (HARNESS=pi: the Pi's cable address; $APPS_PUBLIC_HOST)",
    )
    parser.add_argument("--delegate-state-dir", default=None, help="jobs.json, notices.jsonl")
    parser.add_argument("--delegate-config", default=None, help="a delegate.toml (paths, limits)")
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="documents drop folder (default $VOICE_AGENT_DOCS_DIR or ~/agent_sandbox/docs)",
    )
    parser.add_argument(
        "--docs-retriever",
        choices=("bm25", "fts5", "auto"),
        default=None,
        help="documents retriever mode (default $VOICE_AGENT_RAG_RETRIEVAL or auto)",
    )
    parser.add_argument(
        "--docs-db",
        default=os.environ.get("VOICE_AGENT_DOCS_DB") or None,
        help="pre-built ingest DB (DB + DB.fts) for the docs server; --docs-dir is then the "
        "folder of the raw files (default $VOICE_AGENT_DOCS_DB)",
    )
    parser.add_argument(
        "--docs-first",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("VOICE_AGENT_DOCS_FIRST") == "1",
        help="router first, then the checked document answer (ask_documents) before any "
        "Gemini leg (default $VOICE_AGENT_DOCS_FIRST=1)",
    )
    parser.add_argument(
        "--no-token",
        action="store_true",
        help="serve without LAN_SPEECH_TOKEN (tests only: any local process, the delegate "
        "sandbox included, could then drive the agent's tools)",
    )
    parser.add_argument("--live", default=None, help="ME2 dashboard URL; default records only")
    parser.add_argument("--system", type=Path, default=None, help="the system prompt file")
    parser.add_argument(
        "--builtin-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="prototype: the model searches the web itself (Gemini's Google Search in the same "
        "request as the function tools, in place of web_answer); uses system_agent_search.txt. "
        "With --hedge both requests would search: use --hedge -1",
    )
    parser.add_argument("--pi-url", default=PI_URL, help="the Pi's device tools; '' for none")
    parser.add_argument(
        "--fast-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="answer common home commands without the LLM (fast_path.py)",
    )
    parser.add_argument(
        "--hedge",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="R4: send every LLM request twice and keep the first to answer; SECONDS > 0 sends "
        "the second only if the first has said nothing by then; -1 turns it off",
    )
    parser.add_argument(
        "--keepalive",
        type=float,
        default=KEEPALIVE_S,
        metavar="SECONDS",
        help="keep idle connections to Gemini this long (0 = the SDK default of 15 s, which lets "
        "them go cold between turns)",
    )
    parser.add_argument(
        "--ipv4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="connect to Gemini over IPv4 only (the IPv6 path to Google resets connections on this "
        "network; docs/OPTIMIZATION-RESEARCH.md)",
    )
    parser.add_argument(
        "--router-url",
        default=os.environ.get("ROUTER_URL") or None,
        help="llama-server of the fine-tuned router (router.py): it answers simple turns itself "
        "and hands the rest to Gemini; off by default",
    )
    parser.add_argument("--router-timeout", type=float, default=TIMEOUT_S, metavar="SECONDS")
    parser.add_argument(
        "--router-hold",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="0: a plain race, whichever of the router and Gemini answers first wins (Gemini when "
        "the network is good, the router when it is slow or down); SECONDS > 0 keeps Gemini from "
        "claiming the turn that long so the router's complete answer is not beaten by Gemini's "
        "short filler sentence (about +0.3 s on turns the router defers)",
    )
    parser.add_argument(
        "--say-delays",
        choices=["default", "today", "slow"],
        default="default",
        help="how long the model's `say` is held per tool: today = weather only, slow = also "
        "web_answer, search_docs and delegate",
    )
    parser.add_argument(
        "--hide-idle-delegate",
        action="store_true",
        help="offer delegate_status and delegate_cancel only once the delegate server's job table "
        "(jobs.json next to --notices) lists a job: fewer schema tokens per request",
    )
    parser.add_argument(
        "--notices",
        default=str(Path.home() / ".local/state/voice_agent/delegate/notices.jsonl"),
        help="where the delegate server writes the spoken notices of jobs that ended; '' for none",
    )
    parser.add_argument(
        "--test-delay-word",
        default="",
        help="TEST ONLY: a turn whose words contain this waits --test-delay-s before it is "
        "answered, so the client's wait lines and give-up line can be heard on demand",
    )
    parser.add_argument("--test-delay-s", type=float, default=12.0)
    args = parser.parse_args(argv)
    token = None if args.no_token else os.environ.get("LAN_SPEECH_TOKEN")
    if not token and not args.no_token:
        parser.error("LAN_SPEECH_TOKEN is not set (pass --no-token only for tests)")
    if args.no_token and args.host not in LOOPBACK:
        parser.error("--no-token is only for loopback (--host 127.0.0.1)")
    configure(
        apps_root=args.apps_root,
        apps_host=args.apps_host,
        delegate_state_dir=args.delegate_state_dir,
        delegate_config=args.delegate_config,
    )
    try:
        return asyncio.run(
            run(
                args.port,
                args.live,
                args.system
                or ROOT
                / (
                    "config/eval/system_agent_search.txt"
                    if args.builtin_search
                    else "config/eval/system_agent.txt"
                ),
                args.pi_url or None,
                args.fast_path,
                None if args.hedge < 0 else args.hedge,
                args.keepalive,
                args.ipv4,
                args.router_url,
                args.router_timeout,
                args.router_hold,
                Path(args.notices) if args.notices else None,
                args.builtin_search,
                token,
                Path(args.notices).with_name("jobs.json")
                if args.hide_idle_delegate and args.notices
                else None,
                {"today": SAY_DELAY_TODAY, "slow": SAY_DELAY_SLOW}.get(args.say_delays),
                (args.test_delay_word.lower(), args.test_delay_s) if args.test_delay_word else None,
                args.host,
                docs_dir=args.docs_dir,
                docs_retriever=args.docs_retriever,
                docs_first=args.docs_first,
                docs_db=args.docs_db,
            )
        )
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
