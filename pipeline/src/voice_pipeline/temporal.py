"""T3 temporal year tag: cue detection and citation-year resolution for accepted answers.

Ticket 263 adds the date-based period machinery the sufficiency check uses to decide
staleness deterministically (in code, not from prompt wording): the period a document's
answer covers (dates stated in the answer text, then the file card's date range, then a
period label in the file name or title) and the period a question targets (present, the
next period, a fixed past period, or no explicit period). The gate's time-sensitivity
cue list and these target-period cues are built from the same word groups below, in this
one place, so the gate and the period decision cannot drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# --- the shared cue word groups (the ONE source; the gate's cue list and the
# target-period cues below are all built from these) ---

PRESENT_WORDS = (
    r"current(?:ly)?|present(?:ly)?|latest|newest|most recent|now|today|still"
    r"|recently|at present|at the moment"
)
_YEAR_QUALIFIERS = r"(?:fiscal |calendar |academic |tax )?"
# one non-capturing group, word-bounded when composed into a cue: a bare unit word must
# keep its "this/next/last" prefix, and a unit must not match inside another word
# ("headquartered" is not a time cue)
_PERIOD_UNITS = r"(?:month|week|term|semester|quarter|spring|summer|fall|winter)"
_MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)

# The shared time-sensitivity cues (ticket 259, R2-3): the ONE cue list for both the
# sufficiency gate (present/next-period questions are checked) and the (year) citation tag.
# Present words plus this/next/last periods; a fixed past period ("in 2025", "last year") is
# NOT a cue here - it is the gate's `pinned_past` case in sufficiency.py.
TIME_CUE_RE = re.compile(
    rf"\b({PRESENT_WORDS}|(?:this|next)\s+{_YEAR_QUALIFIERS}year|"
    rf"(?:this|next|last)\s+(?:fiscal\s+)?{_PERIOD_UNITS}|next\s+(?:{_MONTH_NAMES}))\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
# an explicit fiscal-year label (FY26 / FY2026): a fiscal year can end in the previous
# calendar year, so the label beats a plain year found in the same text (ticket 259, R2-2)
_FY_RE = re.compile(r"(?<![A-Za-z0-9])(FY)(\d{2,4})(?!\d)", re.IGNORECASE)
# any plausible 4-digit year (19xx/20xx), digit-bounded so "3005" (a room number) and
# "19934" never read as years
_ANY_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")

_MONTH_BY_NAME = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH_ABBR = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _month_num(name: str) -> int:
    """The month number of a month name or its three-letter abbreviation."""
    return _MONTH_ABBR[name.lower()[:3]]


def has_temporal_cue(question: str) -> bool:
    """True when the question asks for a current/latest fact (the shared cue list)."""
    return bool(TIME_CUE_RE.search(question or ""))


def source_year(date_max: str | None, file_path: str, title: str) -> str | None:
    """Citation year: an explicit fiscal-year label (FY26 / FY2026) wins, since a fiscal
    year can end in the previous calendar year (FY2026 ended in September 2026); else the
    card date_max's year, else a 2000-2099 year in the file path, else in the file title,
    else None (tag omitted). Fiscal years keep their label ("FY2026"), so the sufficiency
    gate's fiscal comparison sees them (ticket 259, R2-2)."""
    texts = (date_max or "", file_path or "", title or "")
    for text in texts:
        m = _FY_RE.search(text)
        if m:
            return f"FY{m.group(2)}"
    for text in texts:
        m = _YEAR_RE.search(text)
        if m:
            return m.group(1)
    return None


# ===========================================================================
# Ticket 263: date-based period derivation (source period vs target period)
# ===========================================================================


def _month_span(year: int, month: int) -> tuple[date, date]:
    last_day = (date(year, month, 1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    return date(year, month, 1), last_day


def _quarter_span(year: int, quarter: int) -> tuple[date, date]:
    first = (quarter - 1) * 3 + 1
    return _month_span(year, first)[0], _month_span(year, first + 2)[1]


def _season_span(name: str, year: int) -> tuple[date, date]:
    if name == "spring":
        return date(year, 1, 1), date(year, 5, 31)
    if name == "summer":
        return date(year, 6, 1), date(year, 8, 31)
    if name == "fall":
        return date(year, 9, 1), date(year, 11, 30)
    # winter: December through the end of February the next year
    return date(year, 12, 1), date(year + 1, 3, 1) - timedelta(days=1)


# a two-semester academic approximation: the fall term runs September through December,
# the spring term January through May (June through August is the break). `name` is the
# season word ("fall"/"spring"), with or without a trailing "term"
def _term_span(name: str, year: int) -> tuple[date, date]:
    if "fall" in name:
        return date(year, 9, 1), date(year, 12, 31)
    return date(year, 1, 1), date(year, 5, 31)


def _occurrences(unit: str, today: date) -> list[tuple[str, int, date, date]]:
    """The (label, year, start, end) spans of the unit around `today`, for the
    this/next/last resolution. A bare month walks the adjacent calendar months; every
    other unit lists its occurrences in the current year and the two neighbours (weeks
    resolve directly in `_resolve`)."""
    if unit == "month":
        out: list[tuple[str, int, date, date]] = []
        for k in range(-14, 15):
            total = today.month - 1 + k
            year = today.year + total // 12
            month = total % 12 + 1
            out.append((unit, year, *_month_span(year, month)))
        return out
    out = []
    for year in (today.year - 1, today.year, today.year + 1):
        if unit == "year":
            out.append((unit, year, date(year, 1, 1), date(year, 12, 31)))
        elif unit in _MONTH_BY_NAME:
            out.append((unit, year, *_month_span(year, _MONTH_BY_NAME[unit])))
        elif unit == "quarter":
            out.extend((f"Q{q}", year, *_quarter_span(year, q)) for q in (1, 2, 3, 4))
        elif unit in ("spring", "summer", "fall", "winter"):
            out.append((unit, year, *_season_span(unit, year)))
        elif unit in ("term", "semester"):
            out.extend((f"{n} term", year, *_term_span(n, year)) for n in ("fall", "spring"))
    return out


def _resolve(unit: str, today: date, which: str) -> tuple[date, date] | None:
    """The span of the unit referenced by this/next/last from `today`."""
    if unit == "week":
        monday = today - timedelta(days=today.weekday())
        if which == "this":
            return monday, monday + timedelta(days=6)
        if which == "next":
            return monday + timedelta(weeks=1), monday + timedelta(weeks=1, days=6)
        return monday - timedelta(weeks=1), monday - timedelta(days=1)
    occurrences = _occurrences(unit, today)
    if not occurrences:
        return None
    if which == "this":
        for _, _, start, end in occurrences:
            if start <= today <= end:
                return start, end
        return None
    if which == "next":
        later = [(start, end) for _, _, start, end in occurrences if start > today]
        return min(later, key=lambda se: se[0]) if later else None
    before = [(start, end) for _, _, start, end in occurrences if end < today]
    return max(before, key=lambda se: se[1]) if before else None


def _unit_of(word: str) -> str:
    """The period unit of a matched cue word: 'year', a unit name, or a month name."""
    w = re.sub(r"^(?:this|next|last)\s+", "", word.lower())
    w = re.sub(r"^(?:fiscal|calendar|academic|tax)\s+", "", w)
    return w


@dataclass(frozen=True)
class TargetPeriod:
    """The period a question targets (ticket 263). `kind`: 'present', 'next', 'past'
    (a fixed period the question names), or 'unspecified' (no explicit period; a
    time-varying fact the gate still flags). `span`: the resolved (start, end) when the
    unit resolves deterministically; `match`: the character span of the cue word in the
    question (the fast path's web-question restatement replaces it)."""

    kind: str
    unit: str | None = None
    span: tuple[date, date] | None = None
    label: str | None = None
    match: tuple[int, int] | None = None


def _fiscal_span(n: int) -> tuple[date, date]:
    """A fiscal year labelled N runs October 1 of the previous year through September
    30 of N (the federal convention; under every common convention a fiscal year has
    ended by September 30 of its label, ticket 263)."""
    return date(n - 1, 10, 1), date(n, 9, 30)


def _fiscal_label(now: datetime, which: str) -> int:
    """The fiscal-year label of the period this/next/last points at: the fiscal year in
    progress in July-December is labelled the next calendar year (FY2027 in 2026-11)."""
    today = now.date()
    current = today.year + 1 if today.month >= 7 else today.year
    return {"this": current, "next": current + 1, "last": current - 1}[which]


def _refine_year_with_season(
    target: TargetPeriod, question: str, span: tuple[date, date]
) -> TargetPeriod:
    """A year cue narrowed by a season word in the question ('fall semester next
    year'): the span becomes that season (or that fall/spring term, when the question
    says semester/term) of the named year - a figure covering the whole year does not
    answer the asked season."""
    m = re.search(r"\b(spring|summer|fall|winter)\b", question, re.IGNORECASE)
    if m is None:
        return target
    season = m.group(1).lower()
    year = span[0].year
    season_span = (
        _term_span(season, year)
        if re.search(r"\b(?:semester|term)\b", question, re.IGNORECASE)
        else _season_span(season, year)
    )
    return TargetPeriod(target.kind, "season", season_span, target.label, target.match)


# the target-period cues: the same word groups as the gate's TIME_CUE_RE, tagged by
# which period the question points at (the shared cue list, one place)
TARGET_PERIOD_RE = re.compile(
    rf"(?P<next>next\s+{_YEAR_QUALIFIERS}year|next\s+(?:fiscal\s+)?{_PERIOD_UNITS}"
    rf"|next\s+(?:{_MONTH_NAMES}))"
    rf"|(?P<present>this\s+{_YEAR_QUALIFIERS}year|this\s+(?:fiscal\s+)?{_PERIOD_UNITS}"
    rf"|this\s+(?:{_MONTH_NAMES}))"
    rf"|(?P<last>last\s+{_YEAR_QUALIFIERS}year|last\s+(?:fiscal\s+)?{_PERIOD_UNITS})"
    rf"|(?P<presentword>{PRESENT_WORDS})",
    re.IGNORECASE,
)


def question_target_period(question: str, now: datetime | None = None) -> TargetPeriod:
    """The period the question targets, deterministically: an explicit next-period cue
    (next year / next month / next quarter / next <season> / next <month>), a present cue
    (now / today / latest / recently / at the moment / this month / ...), or a fixed past
    period (`last <period>` or a year before the current one). No cue: 'unspecified' —
    the question names no period (a time-varying fact the gate may still check)."""
    now = now or datetime.now()
    today = now.date()
    m = TARGET_PERIOD_RE.search(question or "")
    if m is not None:
        if m.lastgroup == "presentword":
            return TargetPeriod("present", None, None, m.group(0), (m.start(), m.end()))
        which = {"next": "next", "present": "this", "last": "last"}[m.lastgroup or "present"]
        kind = {"this": "present", "next": "next", "last": "past"}[which]
        cue = m.group(0)
        if "fiscal" in cue.lower():
            # a fiscal year has its own calendar (October 1 through September 30)
            fiscal_span = _fiscal_span(_fiscal_label(now, which))
            return TargetPeriod(kind, "fiscal", fiscal_span, cue, (m.start(), m.end()))
        unit = _unit_of(cue)
        span = _resolve(unit, today, which)
        target = TargetPeriod(kind, unit, span, cue, (m.start(), m.end()))
        if unit == "year" and span is not None:
            target = _refine_year_with_season(target, question, span)
        return target
    m = _ANY_YEAR_RE.search(question or "")
    if m is not None:
        y = int(m.group(1))
        if y < today.year:
            return TargetPeriod(
                "past", "year", (date(y, 1, 1), date(y, 12, 31)), m.group(1), (m.start(1), m.end(1))
            )
        if y == today.year:
            # a question about the (in-progress) current year targets the present: an
            # earlier-year source does not cover it
            return TargetPeriod(
                "present",
                "year",
                (date(y, 1, 1), date(y, 12, 31)),
                m.group(1),
                (m.start(1), m.end(1)),
            )
    return TargetPeriod("unspecified")


# --- the source period: the dates a document's answer covers ---

# period labels in a text, in priority order (a label that names the document's period
# beats a single reading's date): an explicit fiscal-year label (FY26 / FY2026 /
# "fiscal year 26" / "fiscal 2026"), a full date, a "Month YYYY" / "YYYY-MM" month, a
# quarter, an academic year (2025-26 / 2025-2026), and finally a bare calendar year.
_FY_LABEL_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:fy|f/y|fiscal\s+year|fiscal)\s*:?\s*(20\d{2}|\d{2})(?!\d)",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b((?:19|20)\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b")
_MONTH_YEAR_RE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
# a month range with one year ("July-September 2026", "November through February 2026")
_MONTH_RANGE_RE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"(?:\s*(?:-|\u2013|to|through)\s*)"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_YEAR_MONTH_RE = re.compile(r"\b((?:19|20)\d{2})[-/](0[1-9]|1[0-2])\b")
_QUARTER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:q([1-4])\b[^\d]{0,3}((?:19|20)\d{2})|((?:19|20)\d{2})\s*q\s*([1-4])(?!\d))",
    re.IGNORECASE,
)
_ACADEMIC_RE = re.compile(r"\b(20\d{2})\s*[-\u2013]\s*(\d{2}|20\d{2})\b")


@dataclass(frozen=True)
class SourcePeriod:
    """The period a document's answer covers (ticket 263): its (start, end) dates, the
    label it was read from, its shape (`kind`: fiscal / date / month / quarter /
    academic / year / range), its confidence (high: an unambiguous period label — a
    fiscal-year label, a full date, a month, a quarter, an academic year; medium: a bare
    year or the file card's date range), and where it came from (the answer text, the
    file card, or the file name/title)."""

    start: date
    end: date
    label: str
    kind: str
    confidence: str
    basis: str


# the web-hedge tail of a gold/ideal answer ("... so verify live"): that sentence points
# at the period to check, it does not state the period the figure covers, so it never
# reads as the source period
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WEB_HEDGE_RE = re.compile(
    r"\b(?:verify|confirm|check)\s+(?:it\s+)?(?:live|online|on the web)\b", re.IGNORECASE
)
# "2026 ... through August" in the same sentence: the year's reading stops at that month
# (a year-to-date total is not the whole year)
_THROUGH_MONTH_RE = re.compile(
    r"\bthrough\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b", re.IGNORECASE
)


def _strip_web_hedges(text: str) -> str:
    """A web-hedge clause ("... so verify live") points at the period to check, not the
    period the figure covers. First pass: drop the hedge sentences entirely (in an ideal
    answer the hedge is its own sentence). If that leaves nothing, second pass: cut each
    sentence at the clause that leads into the hedge, keeping the figure's own period."""
    sentences = _SENTENCE_SPLIT_RE.split(text)
    kept = [s for s in sentences if not _WEB_HEDGE_RE.search(s)]
    if any(s.strip() for s in kept):
        return " ".join(s for s in kept if s.strip()).strip()
    out: list[str] = []
    for sentence in sentences:
        m = _WEB_HEDGE_RE.search(sentence)
        if m is None:
            out.append(sentence)
            continue
        lead = list(re.finditer(r"\b(?:so|but|and)\b\s+", sentence[: m.start()]))
        if lead:
            out.append(sentence[: lead[-1].start()].rstrip(" ,;:").strip())
        else:
            out.append((sentence[: m.start()] + " " + sentence[m.end() :]).strip())
    return " ".join(s for s in out if s).strip()


def _parse_period_text(text: str, basis: str) -> SourcePeriod | None:
    if not text:
        return None
    m = _FY_LABEL_RE.search(text)
    if m is not None:
        n = int(m.group(1))
        if n < 100:
            n += 2000
        # a fiscal year labelled N has ended by September 30 of N under every common
        # convention (ticket 263): October 1 of the previous year through September of N
        return SourcePeriod(*_fiscal_span(n), f"FY{n}", "fiscal", "high", basis)
    m = _ISO_DATE_RE.search(text)
    if m is not None:
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return SourcePeriod(d, d, d.isoformat(), "date", "high", basis)
    m = _MONTH_RANGE_RE.search(text)
    if m is not None:
        y = int(m.group(3))
        start, end = _month_span(y, _month_num(m.group(1)))
        start2, end2 = _month_span(y, _month_num(m.group(2)))
        return SourcePeriod(min(start, start2), max(end, end2), m.group(0), "month", "high", basis)
    m = _MONTH_YEAR_RE.search(text)
    if m is not None:
        month = _month_num(m.group(1))
        y = int(m.group(2))
        start, end = _month_span(y, month)
        return SourcePeriod(start, end, m.group(0), "month", "high", basis)
    m = _YEAR_MONTH_RE.search(text)
    if m is not None:
        y, month = int(m.group(1)), int(m.group(2))
        start, end = _month_span(y, month)
        return SourcePeriod(start, end, m.group(0), "month", "high", basis)
    m = _QUARTER_RE.search(text)
    if m is not None:
        if m.group(1):
            q, y = int(m.group(1)), int(m.group(2))
        else:
            y, q = int(m.group(3)), int(m.group(4))
        start, end = _quarter_span(y, q)
        return SourcePeriod(start, end, m.group(0), "quarter", "high", basis)
    m = _ACADEMIC_RE.search(text)
    if m is not None:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if y2 < 100:
            y2 += 2000
        if y2 == y1 + 1:
            # an academic year runs August through July (ticket 263)
            return SourcePeriod(
                date(y1, 8, 1), date(y2, 7, 31), m.group(0), "academic", "high", basis
            )
    m = _ANY_YEAR_RE.search(text)
    if m is not None:
        # the newest year stated in the text is the period the figure belongs to
        # ("... for 2025 and ... for 2026" reads the 2026 figure)
        y = max(int(v) for v in _ANY_YEAR_RE.findall(text))
        label = str(y)
        start, end = date(y, 1, 1), date(y, 12, 31)
        sentence = next((s for s in _SENTENCE_SPLIT_RE.split(text) if label in s), "")
        tm = _THROUGH_MONTH_RE.search(sentence)
        if tm is not None:
            end = _month_span(y, _month_num(tm.group(1)))[1]
        return SourcePeriod(start, end, label, "year", "medium", basis)
    return None


def _parse_iso(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def source_period(
    answer_text: str,
    date_min: str | None,
    date_max: str | None,
    file_name: str,
    title: str,
) -> SourcePeriod | None:
    """The period a document's answer covers, derived deterministically in priority
    order (ticket 263): dates stated in the answer text, the file card's
    date_max/date_min, a period label in the file name or title. None when no period
    can be read (a timeless source)."""
    p = _parse_period_text(_strip_web_hedges(answer_text or ""), "answer")
    if p is not None:
        return p
    lo, hi = _parse_iso(date_min), _parse_iso(date_max)
    if lo is not None or hi is not None:
        lo = lo or (hi or date(1, 1, 1))
        hi = hi or lo
        return SourcePeriod(
            lo, hi, f"{lo.isoformat()}..{hi.isoformat()}", "range", "medium", "card"
        )
    p = _parse_period_text(file_name or "", "name")
    if p is not None:
        return p
    return _parse_period_text(title or "", "name")


@dataclass(frozen=True)
class StaleDecision:
    """The code-side staleness decision (ticket 263): the source period ended before
    today (or before the period the question names) while the question targets the
    present or the next period — or a fixed past period that matches the source is
    never stale. `confident` (a high-confidence source period) is what lets the gate
    always run the check and the fast path skip the LLM; `fast_path` is the confident
    ended-source present/next case. `hint` is the deterministic one-line fact the
    verdict prompt and the fast-path reason carry."""

    stale: bool
    confident: bool
    covers: bool
    fast_path: bool
    hint: str


def stale_decision(
    source: SourcePeriod | None, target: TargetPeriod, now: datetime | None = None
) -> StaleDecision:
    """Stale when the source period ended before today (or before the period the
    question names) and the question targets the present or the next period; a fiscal
    year labelled N is treated as possibly still running until September 30 of N; a
    fixed past period the source covers is never stale; a source that covers today, or
    an unspecified target, never makes a confident stale call (the over-trigger guard)."""
    now = now or datetime.now()
    today = now.date()
    if source is None or target.kind == "unspecified":
        return StaleDecision(False, False, True, False, "")
    s, e = source.start, source.end
    high = source.confidence == "high"
    if target.kind == "present":
        span_ok = target.span is not None and (
            source.start <= target.span[0] and source.end >= target.span[1]
        )
        covers_today = s <= today <= e
        if span_ok or covers_today:
            if source.kind == "fiscal" and covers_today:
                hint = (
                    f"the source covers today; fiscal year {source.label} may still be "
                    f"running (it ends {e.isoformat()})"
                )
            elif span_ok and not covers_today:
                ts, te = target.span or (s, e)
                hint = (
                    f"the source covers the period the question names "
                    f"({ts.isoformat()} to {te.isoformat()})"
                )
            else:
                hint = f"the source covers {s.isoformat()} to {e.isoformat()}; today is inside it"
            return StaleDecision(False, high, True, False, hint)
        if e < today:
            # the source ended and the question asks for the present: stale. The fast
            # path needs a high-confidence source that ends before the asked span (or an
            # unspanned present question): a source that ends inside the asked span (a
            # mid-month reading for a month question) only pays for the check
            fast = high and (target.span is None or e < target.span[0])
            hint = (
                f"the source covers {s.isoformat()} to {e.isoformat()}; "
                f"it ended before today {today.isoformat()}"
            )
            return StaleDecision(True, high, False, fast, hint)
        return StaleDecision(
            False, False, False, False,
            f"the source covers {s.isoformat()} to {e.isoformat()}; "
            f"the question targets the present",
        )
    if target.span is None:
        return StaleDecision(False, False, True, False, "")
    t_start, t_end = target.span
    if e < t_start:
        if target.kind == "next":
            hint = (
                f"the source covers {s.isoformat()} to {e.isoformat()}; it ended before the "
                f"period asked (from {t_start.isoformat()})"
            )
        else:  # a fixed past period the source does not reach
            hint = (
                f"the source covers {s.isoformat()} to {e.isoformat()}; it precedes the "
                f"period asked (from {t_start.isoformat()})"
            )
        return StaleDecision(True, high, False, high, hint)
    if s <= t_end:
        # it overlaps the named period (fully or partially): the source is not from an
        # earlier period - a fixed past period the source matches is never stale
        covers = s <= t_start and e >= t_end
        word = "covers" if covers else "overlaps"
        return StaleDecision(
            False, high, True, False,
            f"the source {word} the period asked ({t_start.isoformat()} to {t_end.isoformat()})",
        )
    return StaleDecision(
        False, False, False, False,
        f"the source covers {s.isoformat()} to {e.isoformat()}; the question asks about "
        f"{t_start.isoformat()} to {t_end.isoformat()}",
    )


@dataclass(frozen=True)
class PeriodFacts:
    """The structured date facts the sufficiency check works from (ticket 263): the
    source period, the target period, and the code-side decision. `evidence_line`
    renders them as the structured line the verdict prompt receives, so the LLM only
    confirms the period judgement instead of making it."""

    source: SourcePeriod | None
    target: TargetPeriod
    decision: StaleDecision

    def evidence_line(self, today: date | None = None) -> str:
        today = today or date.today()
        words = {
            "present": "the present",
            "next": "the next period",
            "past": "a fixed past period",
            "unspecified": "no explicit period (a time-varying fact)",
        }
        parts: list[str] = []
        if self.source is not None:
            line = (
                f"the source covers {self.source.start.isoformat()} to "
                f"{self.source.end.isoformat()}"
            )
            if self.source.label:
                line += f" (label {self.source.label})"
            parts.append(line)
        parts.append(f"today is {today.isoformat()}")
        parts.append(f"the question targets {words.get(self.target.kind, 'no explicit period')}")
        if self.decision.stale:
            state = "the source period has ended"
        elif self.source is not None and self.source.end < today:
            state = "the source period ended before today"
        elif self.source is not None:
            state = "the source period has not ended"
        else:
            state = "no source period could be read"
        parts.append(state)
        out = "Period facts: " + "; ".join(parts) + "."
        if self.decision.hint:
            out += f" Note: {self.decision.hint}."
        return out


def period_facts(
    source: SourcePeriod | None, target: TargetPeriod, now: datetime | None = None
) -> PeriodFacts:
    now = now or datetime.now()
    return PeriodFacts(source, target, stale_decision(source, target, now))


def present_web_question(question: str, target: TargetPeriod) -> str:
    """The fast path's web question (ticket 263): the question restated for the present
    period, using only the question's own words (nothing private is added): the
    next-period expression is replaced by 'now'; a present or unspecified question is
    already about the present and is returned unchanged."""
    q = (question or "").strip()
    if target.kind == "next" and target.match is not None:
        s, e = target.match
        q = re.sub(r"\s{2,}", " ", f"{q[:s]}now{q[e:]}").strip()
    return q
