"""T3 temporal year tag: cue detection and citation-year resolution for accepted answers."""

from __future__ import annotations

import re

# The shared time-sensitivity cues (ticket 259, R2-3): the ONE cue list for both the
# sufficiency gate (present/next-period questions are checked) and the (year) citation tag.
# Present words plus this/next/last periods; a fixed past period ("in 2025", "last year") is
# NOT a cue here - it is the gate's `pinned_past` case in sufficiency.py.
TIME_CUE_RE = re.compile(
    r"\b(current(?:ly)?|present(?:ly)?|latest|newest|most recent|now|today|still|"
    r"(?:this|next) (?:fiscal |calendar |academic |tax )?year|"
    r"(?:this|next|last) (?:fiscal )?(?:month|week|term|semester|quarter|"
    r"spring|summer|fall|winter)|"
    r"next (?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december))\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
# an explicit fiscal-year label (FY26 / FY2026): a fiscal year can end in the previous
# calendar year, so the label beats a plain year found in the same text (ticket 259, R2-2)
_FY_RE = re.compile(r"(?<![A-Za-z0-9])(FY)(\d{2,4})(?!\d)", re.IGNORECASE)


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
