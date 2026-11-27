"""Sufficiency check after an accepted document answer (ticket 250).

An accepted document answer was spoken as is, with no check of whether it really answers
the question that was asked. This module makes ONE call to the document lane's own cheap
JSON model and asks one of four verdicts about the answer that was just spoken:
`sufficient` (the answer states what was asked and nothing time-sensitive is missing),
`stale` (the question is about the present or the next period but the source is from an
earlier period), `missing_half` (a normal question quietly needs a fact outside the
documents and the answer covers only part of it), `wrong_question` (the answer is about
something else than what was asked).

Fail-open everywhere: off unless SUFFICIENCY is turned on; the cheap gate limits the call
to time-sensitive questions, sources older than the current period, and short answers
(the server calls it only from the accepted document routes, never for pleasantries or
non-document turns); it runs at most once per turn (the server's per-turn flag); and any
error, timeout or unparsable reply says nothing extra (the old behaviour, byte-for-byte).
The verdict starts as a task BEFORE the document answer is spoken (no first-audio
penalty) and the document answer always stands as spoken today: the check can add ONE
supplement sentence after it, worded as an addition, never a retraction.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from voice_pipeline.ask_documents import JsonRequest

SUFFICIENCY = False  # off by default; the experiment flips it, the tests monkeypatch it on

# the verdict call: the document lane's own cheap JSON model, no retrieval
VERDICT_TIMEOUT_S = 1.5
# after the document answer is spoken, the already-running verdict task gets this long
AWAIT_TIMEOUT_S = 2.5
# the optional web lookup that supplies what is missing or current: one non-retried trip
# with a tight cap (R2-4) — the supplement is an addition, never a reason to stall the turn
WEB_TIMEOUT_S = 2.5
# a document answer shorter than this many words is always checked (the gate's third leg)
SHORT_ANSWER_WORDS = 6

VERDICTS = frozenset({"sufficient", "stale", "missing_half", "wrong_question"})

# the verdict's reply shape, enforced by the transport's structured output (R1-1): the
# document lane's own REPLY_SCHEMA has no `verdict` key, so the verdict call must ask for
# this one, or the reply can never name the verdict
SUFFICIENCY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": sorted(VERDICTS)},
        "web_question": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "web_question", "reason"],
}

# the web lookup's honest hedge (copied from evaluators.HEDGE, which copies the web
# server's own line; test_evaluators.py pins that copy to gemini_grounding.py via AST and
# test_predeploy_250.py pins it to this one). A lookup that comes back this way has not
# actually checked the web.
HEDGE_TAIL = "I couldn't confirm that with a search, so I won't guess."

# the gate's time-sensitivity when the root package is not importable (another package and
# venv, so a local copy): present/next/latest/current/now/this year/month/today/still,
# word-bounded, case-insensitive
_TIME_SENSITIVE = re.compile(
    r"\b(current(?:ly)?|present(?:ly)?|latest|newest|now|today|still|"
    r"this (?:year|month)|next (?:year|month|week|fiscal year))\b",
    re.IGNORECASE,
)
_FISCAL = re.compile(r"\bfy(\d{2,4})\b", re.IGNORECASE)
_YEAR = re.compile(r"(20\d{2})")

try:  # the root project's classifier when importable without network; a miss is the regex
    from voice_agent.mcp_servers import gemini_grounding as _grounding
except Exception:
    _grounding = None


def time_sensitive(question: str) -> bool:
    """The gate's first leg: the question is about the present or the next period.
    `gemini_grounding.changeable` when the root package is importable (and answers); the
    local regex otherwise."""
    if _grounding is not None:
        try:
            return bool(_grounding.changeable(question))
        except Exception:
            pass
    return bool(_TIME_SENSITIVE.search(question or ""))


def _fiscal_year(now: datetime) -> int:
    """The fiscal year `now` falls in (a fiscal year starts in July): FY27 in 2026-11."""
    return now.year + 1 if now.month >= 7 else now.year


def source_earlier(source_year: str | None, now: datetime | None = None) -> bool:
    """The source is from an earlier period than `now` (default the real clock): its year
    is below `now`'s, or it names a fiscal year (FY26) below the fiscal year `now` falls
    in. No year known, or a current one: False."""
    text = (source_year or "").strip()
    if not text:
        return False
    now = now or datetime.now()
    m = _FISCAL.search(text)
    if m:
        fy = int(m.group(1))
        if fy < 100:
            fy += 2000
        return fy < _fiscal_year(now)
    m = _YEAR.search(text)
    if m:
        return int(m.group(1)) < now.year
    return False


def should_check(
    question: str, source_year: str | None, answer: str, now: datetime | None = None
) -> bool:
    """The cheap gate (over-triggering and latency are the main risks): the call is paid
    for only when the question is time-sensitive, the source is from an earlier period,
    or the answer is shorter than SHORT_ANSWER_WORDS words. The caller is responsible for
    asking only on document turns (never pleasantries or non-document routes)."""
    if time_sensitive(question):
        return True
    if len((answer or "").split()) < SHORT_ANSWER_WORDS:
        return True
    return source_earlier(source_year, now)


@dataclass
class Verdict:
    """The sufficiency verdict. `web_question` is empty when the web adds nothing (or the
    verdict is wrong_question, which is logged, not spoken)."""

    verdict: str
    web_question: str
    reason: str


# Generic English only; the examples use invented entities. No test-corpus words.
SYSTEM = (
    "You do not answer questions; you judge whether an answer that was just spoken is "
    "enough. You see only the question, the spoken answer (from the user's own "
    "documents), the year of the source if known, and today's date; use nothing else, "
    "never your own knowledge. Reply with JSON only, an object with exactly the keys "
    "verdict, web_question and reason. verdict is 'sufficient' when the answer states "
    "what was asked and nothing time-sensitive is missing (the normal case: when in "
    "doubt, sufficient); 'stale' when the question is about the present or the next "
    "period but the source is from an earlier period; 'missing_half' when the question, "
    "as ordinarily worded, quietly needs a fact outside the documents and the answer "
    "covers only part of it; 'wrong_question' when the answer is about something else "
    "than what was asked. web_question is a standalone question that a web search would "
    "answer to supply what is missing or current, empty when the verdict is sufficient "
    "or wrong_question. reason is at most 12 words. Examples: "
    "'What is the Zorblat index right now?' answered 'As per the index sheet: 12.5' "
    "(source year 2041, today 2044-03-02) -> "
    '{"verdict": "stale", "web_question": "What is the current Zorblat index?", '
    '"reason": "the source is three years old"} '
    "'Who runs the Zorblat bureau, and where is its office?' answered 'The file says "
    "Keira Moss runs the bureau' (source year 2044, today 2044-03-02) -> "
    '{"verdict": "missing_half", "web_question": "Where is the office of the Zorblat '
    'bureau?", "reason": "the office location is outside the file"} '
    "'How deep is the Quenth river?' answered 'The file says Keira Moss runs the "
    "bureau' (source year unknown, today 2044-03-02) -> "
    '{"verdict": "wrong_question", "web_question": "", "reason": "the answer is about '
    'the bureau, not the river"}'
)


def today_line() -> str:
    """Today's date for the prompt (the model has no clock)."""
    return datetime.now().strftime("%Y-%m-%d")


async def verdict(
    request: JsonRequest | None,
    question: str,
    answer: str,
    source_year: str | None,
    today: str,
) -> Verdict | None:
    """ONE call to the document lane's JSON model: the sufficiency verdict, or None. Any
    error, timeout or unparsable reply is None (fail-open: nothing extra is said)."""
    if request is None:
        return None
    user = (
        f"Question: {question}\n"
        f"Spoken answer: {answer}\n"
        f"Source year: {source_year or 'unknown'}\n"
        f"Today: {today}"
    )
    try:
        raw = await asyncio.wait_for(
            request(SYSTEM, user, SUFFICIENCY_SCHEMA), timeout=VERDICT_TIMEOUT_S
        )
    except Exception:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    v = str(data.get("verdict") or "").strip().lower()
    if v not in VERDICTS:
        return None
    return Verdict(
        verdict=v,
        web_question=str(data.get("web_question") or "").strip(),
        reason=str(data.get("reason") or "").strip(),
    )
