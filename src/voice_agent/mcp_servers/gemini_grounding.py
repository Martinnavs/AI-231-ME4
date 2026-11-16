"""One spoken answer from Gemini with Google Search grounding (the first choice of `web_answer`).

Gemini 3.1 flash-lite with the `google_search` tool answers a factual question in one request and
searches only when the answer could have changed (office holders, results, prices, "latest"):
settled facts come back in about 0.8 s, searched ones in 1.6-2.0 s, against 1.3 s for Tavily on
every question (25 questions, Manila, 2026-11-03: median 0.79 s against 1.33 s). Told today's date
and to search for anything that changes, it searched 5 of 8 changeable questions and none of the
settled ones; without that, "last NBA championship" and "next solar eclipse" came back stale.

Cost: Google bills each search query ($14 per 1,000 beyond 5,000 free a month, shared by Gemini 3
models, paid projects only). Every call is appended to a ledger and, when the month's searches
reach `MONTHLY_SEARCH_CAP`, the caller is told to use another backend. The key is read from
$GEMINI_API_KEY or ~/.secrets/google-ai-studio/api-key.txt, sent only as a header, never logged.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from voice_agent import gate

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent"
)
KEY_FILE = Path(
    os.environ.get("GEMINI_KEY_FILE", Path.home() / ".secrets/google-ai-studio/api-key.txt")
)
LEDGER = Path(
    os.environ.get(
        "WEB_ANSWER_LEDGER", Path.home() / ".local/state/voice_agent/web_answer_ledger.jsonl"
    )
)
TIMEZONE = "Asia/Manila"
MONTHLY_SEARCH_CAP = 4500  # of the 5,000 free; a few hundred are left for other Gemini 3 uses
DEFAULT_PLACE = "Cebu City"
SYSTEM = (
    "You are a voice assistant. Today is {today}. The user is in {place}; 'here' means there, "
    "and otherwise ignore it. "
    "Your answer is spoken aloud: at most two short sentences, plain words, no lists, no "
    "markdown. Anything that can change needs a search result, not memory: who holds a job or "
    "office, prices and rates, versions, rankings, records, results, anything described as "
    "latest, last, next, current or new. Settled facts (capitals, authors, historical dates) "
    "need no search."
)
SYSTEM_COMPARE = (
    "You compare two spoken answers to the same question. Reply with exactly one word: SAME if "
    "they give the same key fact (the same person, number, date or result; wording and extra "
    "detail do not matter), DIFFERENT if the key facts conflict."
)
FORCE_SEARCH = " Search the web before you answer this question; do not answer from memory."
SYSTEM_RESEARCH = (
    "You are researching for a voice assistant. Today is {today}. The user is in {place}: use it "
    "only when the question depends on where they are. "
    "Search the web, then answer in at most six short sentences of plain words, no lists, no "
    "markdown, naming the sources "
    "(site or publication) the facts come from. Say plainly what you could not find."
)


@dataclass
class Grounded:
    text: str
    searches: int  # Google Search queries Gemini ran (the billed unit)
    titles: list[str]  # titles of the pages it used, for checking the answer
    tokens: int | None
    seconds: float
    sources: list[tuple[str, str]] = field(default_factory=list)  # (title, uri) of each page
    lead: int = 0  # index in `sources` of the page behind the answer's first claim


def load_key() -> str | None:
    """The API key from the environment or the key file, or None (never printed or logged)."""
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()
    try:
        return "".join(KEY_FILE.read_text().split()) or None
    except OSError:
        return None


def plain(text: str) -> str:
    """Spoken text from a model reply: markdown marks and line breaks gone."""
    return " ".join(re.sub(r"[*_`#]+", "", text).split())


def parse(data: dict[str, Any]) -> tuple[str, int, list[str], int | None]:
    """(text, searches, page titles, tokens) of a generateContent response."""
    candidate = (data.get("candidates") or [{}])[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = plain("".join(str(p.get("text", "")) for p in parts))
    grounding = candidate.get("groundingMetadata") or {}
    titles = [
        str(c.get("web", {}).get("title", "")) for c in grounding.get("groundingChunks") or []
    ]
    tokens = (data.get("usageMetadata") or {}).get("totalTokenCount")
    return text, len(grounding.get("webSearchQueries") or []), titles, tokens


def parse_sources(data: dict[str, Any]) -> tuple[list[tuple[str, str]], int]:
    """([(title, uri)], index of the page behind the first supported claim) of a response."""
    grounding = ((data.get("candidates") or [{}])[0].get("groundingMetadata")) or {}
    sources = [
        (str(c.get("web", {}).get("title", "")), str(c.get("web", {}).get("uri", "")))
        for c in grounding.get("groundingChunks") or []
    ]
    for support in grounding.get("groundingSupports") or []:
        indices = support.get("groundingChunkIndices") or []
        if indices and 0 <= indices[0] < len(sources):
            return sources, int(indices[0])
    return sources, 0


CHANGEABLE = re.compile(
    r"\b(ceo|president|chair(?:man|woman|person)?|head|mayor|governor|coach|minister|senator|"
    r"secretary|director|dean|chancellor|leader|captain|owner|champions?|winner|"
    r"current(?:ly)?|now|latest|last|next|newest|new|recent(?:ly)?|today|this (?:year|week|month)|"
    r"price|rate|version|release|ranking|record|score|result)\b",
    re.IGNORECASE,
)


def changeable_regex(question: str) -> bool:
    """The word-list guess: roles, "latest", prices, results (0.55 of FreshQA changing)."""
    return bool(CHANGEABLE.search(question))


def changeable(question: str) -> bool:
    """The question's answer can have changed since a model was trained (a role, a result,
    anything current). The EverGreen classifier when its model is installed (voice_agent.gate),
    else the regex. A miss costs only a missing check, never a wrong answer."""
    verdict = gate.mutable(question)
    return changeable_regex(question) if verdict is None else verdict


GENERIC_LABELS = frozenset(["www", "com", "org", "net", "gov", "edu", "info", "io", "co", "ph"])
GENERIC_LABELS |= {"uk", "us", "au", "ca", "de", "fr", "jp", "news"}
# how a site is said when its domain label is not its name ("theguardian" is The Guardian)
ALIASES = {
    "theguardian": "The Guardian",
    "nytimes": "The New York Times",
    "washingtonpost": "The Washington Post",
    "wsj": "The Wall Street Journal",
    "gmanetwork": "GMA News",
    "taiwannews": "Taiwan News",
    "channelnewsasia": "CNA",
    "techpowerup": "TechPowerUp",
    "hothardware": "HotHardware",
    "youtube": "YouTube",
    "upd": "UP Diliman",
    "up": "UP",
    "cnbc": "CNBC",
    "cnn": "CNN",
    "bbc": "BBC",
    "ft": "FT",
    "sec": "SEC",
    "npr": "NPR",
    "pna": "PNA",
    "scmp": "SCMP",
    "inquirer": "Inquirer",
    "philstar": "Philstar",
    "rappler": "Rappler",
    "wikipedia": "Wikipedia",
    "britannica": "Britannica",
    "reuters": "Reuters",
}
_DOMAIN = re.compile(r"^[\w-]+(\.[\w-]+)+$")
_TITLE_SPLIT = re.compile(r"\s[-|\u2013\u2014]\s")


def publisher(title: str) -> str | None:
    """The publisher's spoken name from a source title ("nvidia.com" gives "Nvidia",
    "up.edu.ph" gives "UP", "Jensen Huang - Forbes" gives "Forbes"), or None. Web text: only
    letters, digits, spaces and "&", at most 40 characters."""
    title = title.strip()
    if _DOMAIN.match(title):
        labels = [x for x in title.lower().split(".") if x]
        if len(labels) > 1 and len(labels[-1]) == 2:
            labels.pop()  # a country-code ending (.ai, .ph, .uk), once: "up" in up.edu.ph stays
        while len(labels) > 1 and labels[-1] in GENERIC_LABELS:
            labels.pop()
        name = labels[-1]
        if name in ALIASES:
            name = ALIASES[name]
        elif "-" in name and all(len(part) <= 4 for part in name.split("-")):
            name = name.upper()  # abs-cbn
        else:
            name = name.upper() if len(name) <= 3 else name.title()
    else:
        last = _TITLE_SPLIT.split(title)[-1].strip() if _TITLE_SPLIT.search(title) else ""
        name = last if 0 < len(last.split()) <= 4 else ""
    name = re.sub(r"[^\w &-]", "", name).strip()[:40]
    return name or None


def source_names(answer: Grounded, limit: int = 2) -> list[str]:
    """The distinct publishers behind an answer, the page behind its first claim first."""
    if not answer.sources:
        return []
    lead = min(answer.lead, len(answer.sources) - 1)
    order = [answer.sources[lead], *answer.sources[:lead], *answer.sources[lead + 1 :]]
    return list(dict.fromkeys(n for title, _ in order if (n := publisher(title))))[:limit]


# words that start a sentence without being a name: lowercased when "As per X," goes in front
_PLAIN_START = frozenset(
    [
        "the",
        "a",
        "an",
        "it",
        "he",
        "she",
        "they",
        "there",
        "this",
        "these",
        "that",
        "his",
        "her",
        "their",
        "its",
    ]
)


def attribute(text: str, names: list[str], only_one: bool = False) -> str:
    """ "As per Nvidia and CNN, <answer>": the source goes first, where a listener (or a reader
    skimming a screen) takes it in before the claim. `only_one`: a fact that changes with a
    single site behind it says so, a double check that failed is told, not hidden."""
    if not names or not text:
        return text
    who = " and ".join(names[:2])
    if len(names) == 1 and only_one:
        who += " (the only source I found)"
    first, _, rest = text.partition(" ")
    if first.lower() in _PLAIN_START:
        text = first.lower() + (" " + rest if rest else "")
    return f"As per {who}, {text}"


def source_sentence(answer: Grounded, changeable_fact: bool = False) -> str:
    """The trailing form: " Source: Nvidia." for the page behind the answer, " Sources: Nvidia
    and Forbes." for two sites, and for a fact that changes with one site " Source: Nvidia. Only
    one source found.". "" with no usable name. Research results carry it as a last line (the
    server speaks it when the reply names no source); `attribute` is the spoken lead-in."""
    names = source_names(answer)
    if not names:
        return ""
    if len(names) >= 2:
        return f" Sources: {names[0]} and {names[1]}."
    return f" Source: {names[0]}." + (" Only one source found." if changeable_fact else "")


def parse_same(text: str) -> bool | None:
    """True for SAME, False for DIFFERENT, None for anything else (an unreadable comparison
    never overrides an answer)."""
    word = re.search(r"\b(SAME|DIFFERENT)\b", text.upper())
    return None if word is None else word.group(1) == "SAME"


async def compare(http: httpx.AsyncClient, first: str, second: str, key: str) -> bool | None:
    """Do two answers to one question give the same key fact? One small request, no search tool."""
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_COMPARE}]},
        "contents": [{"parts": [{"text": f"Answer 1: {first}\nAnswer 2: {second}"}]}],
        "generationConfig": {"thinkingConfig": {"thinkingLevel": "minimal"}},
    }
    response = await http.post(GEMINI_URL, json=body, headers={"x-goog-api-key": key})
    response.raise_for_status()
    return parse_same(parse(response.json())[0])


class VerifyBudget:
    """How many second-opinion checks may still run: a count kept in a file, so restarting the
    server does not hand out a fresh allowance. Delete the file to start over."""

    def __init__(self, path: Path | None, maximum: int) -> None:
        self.path, self.maximum = path, maximum

    def used(self) -> int:
        try:
            return int(json.loads(self.path.read_text())["used"]) if self.path else 0
        except (OSError, ValueError, KeyError, TypeError):
            return 0

    def take(self) -> bool:
        """Spend one check; False when none is left (or checks are off)."""
        used = self.used()
        if self.maximum <= 0 or used >= self.maximum:
            return False
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps({"used": used + 1, "max": self.maximum}))
            except OSError:
                return False  # a count that cannot be kept must not become unlimited
        return True


VERIFY_USED = Path(
    os.environ.get("WEB_VERIFY_USED", Path.home() / ".local/state/voice_agent/verify_used.json")
)
GATE_OUTCOMES = Path(
    os.environ.get(
        "WEB_GATE_OUTCOMES", Path.home() / ".local/state/voice_agent/gate_outcomes.jsonl"
    )
)

# said when a changeable question was answered without a search and nothing could check it. It
# carries no guess: an unverified answer from memory is the thing to avoid, with or without a
# label (docs/SEARCH-SPEC.md, E2). The pipeline treats a reply with "couldn't find", "can't say"
# or "error" as a failed lookup and asks again, so none of those words may appear here or in a
# source sentence
HEDGE = "I couldn't confirm that with a search, so I won't guess."


def request_body(
    query: str,
    today: str,
    place: str,
    research: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """The generateContent request: one place, so the capture script sends what `ask` sends."""
    system = (SYSTEM_RESEARCH if research else SYSTEM).format(today=today, place=place)
    return {
        "systemInstruction": {"parts": [{"text": system + (FORCE_SEARCH if force else "")}]},
        "contents": [{"parts": [{"text": query}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"thinkingConfig": {"thinkingLevel": "minimal"}},
    }


async def ask(
    http: httpx.AsyncClient,
    query: str,
    key: str,
    now: Callable[[], datetime] = lambda: datetime.now(ZoneInfo(TIMEZONE)),
    research: bool = False,
    place: str = DEFAULT_PLACE,
    force: bool = False,
) -> Grounded | None:
    """Gemini's grounded answer to `query`, or None when it gave no text. `force` adds the
    instruction to search first (the retry of a changeable question that was answered from
    memory). Network and HTTP errors are raised (httpx.HTTPError) for the caller to fall back on."""
    body = request_body(query, f"{now():%A, %B %-d, %Y}", place, research, force)
    started = time.perf_counter()
    response = await http.post(GEMINI_URL, json=body, headers={"x-goog-api-key": key})
    response.raise_for_status()
    data = response.json()
    text, searches, titles, tokens = parse(data)
    if not text:
        return None
    sources, lead = parse_sources(data)
    return Grounded(text, searches, titles, tokens, time.perf_counter() - started, sources, lead)


class Ledger:
    """What the grounded calls cost: one JSON line per call, and the month's search count."""

    def __init__(self, path: Path | None, cap: int = MONTHLY_SEARCH_CAP, month: str = "") -> None:
        self.path, self.cap = path, cap
        self.month = month or time.strftime("%Y-%m")
        self.searches = self._read()

    def _read(self) -> int:
        if self.path is None:
            return 0
        try:
            rows = [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]
        except (OSError, ValueError):
            return 0
        return sum(int(r.get("searches", 0)) for r in rows if str(r.get("month")) == self.month)

    @property
    def exhausted(self) -> bool:
        return self.searches >= self.cap

    def add(
        self, grounded: Grounded, query: str = "", forced: bool = False, check: str = ""
    ) -> None:
        self.searches += grounded.searches
        extra: dict[str, Any] = {"query": query[:200]} if query else {}
        if check:
            extra["check"] = check
        if forced:
            extra["forced"] = True
        if grounded.sources:
            extra["sources"] = [t[:60] for t, _ in grounded.sources[:3]]
        self._write(grounded.searches, grounded.tokens or 0, grounded.seconds, extra=extra)

    def add_failure(self, seconds: float, kind: str) -> None:
        """A grounded call that timed out or failed (`kind`): a row with no cost, so the
        latency tail can be counted."""
        self._write(0, 0, seconds, kind)

    def _write(
        self,
        searches: int,
        tokens: int,
        seconds: float,
        failed: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.path is None:
            return
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "month": self.month,
               "searches": searches, "tokens": tokens,
               "seconds": round(seconds, 2)}  # fmt: skip
        row |= extra or {}
        if failed:
            row["failed"] = failed  # type: ignore[assignment]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as sink:
                sink.write(json.dumps(row) + "\n")
        except OSError:
            pass  # a ledger that cannot be written must not lose the answer
