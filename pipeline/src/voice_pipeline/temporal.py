"""T3 temporal year tag: cue detection and citation-year resolution for accepted answers."""

from __future__ import annotations

import re

_TEMPORAL_RE = re.compile(
    r"\b(current|currently|latest|now|today|this year|next year|newest|most recent)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def has_temporal_cue(question: str) -> bool:
    """True when the question asks for a current/latest fact (closed cue list)."""
    return bool(_TEMPORAL_RE.search(question or ""))


def source_year(date_max: str | None, file_path: str, title: str) -> str | None:
    """Citation year, in order: the card date_max's year, else a 2000-2099 year in the file
    path, else in the file title, else None (tag omitted)."""
    for text in (date_max or "", file_path or "", title or ""):
        m = _YEAR_RE.search(text)
        if m:
            return m.group(1)
    return None
