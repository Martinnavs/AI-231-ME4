"""The reply text in the form a voice should say it, applied only on the voice/TTS side
(ticket 231): the 'As per <file>:' lead-in and any file token or path become what a person
would say, money and percent get a unit word, markdown marks are stripped. The agent
server's text (text mode, the offline evals) is untouched. Idempotent; the digits inside
numbers are never changed, only the marks around them."""

from __future__ import annotations

import re

_EXT = r"(?:md|txt|csv|tsv|pdf|docx|doc|xlsx|xls|pptx|ppt|html|htm|json|log|ini|toml)"
# 'As per gs_salary_table_2026.pdf: $45,980' -> 'gs salary table 2026: 45,980 dollars'
_AS_PER = re.compile(r"\bAs per\s+((?:[\w-]+/)*[\w-]+\." + _EXT + r")\s*:\s*", re.IGNORECASE)
# a file token anywhere: 'a/b.pdf' -> 'b', 'notes.md' -> 'notes'
_FILE = re.compile(r"\b(?:[\w-]+/)*([\w-]+)\." + _EXT + r"\b")
_BULLET = re.compile(r"(?m)^\s*- ")
_HEADER = re.compile(r"(?m)^\s*#{1,6}\s*")
_DOLLAR = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
_PERCENT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s?%")
_YEAR = re.compile(r"\(((?:19|20)\d{2})\)")


def _base(token: str) -> str:
    """Base name without extension, underscores and hyphens as spaces ('a/b.pdf' -> 'b')."""
    return re.sub(r"[_-]+", " ", token.rsplit("/", 1)[-1])


def spoken_form(text: str) -> str:
    """The text as a voice should say it (ticket 231); plain prose comes back unchanged."""
    text = _HEADER.sub("", _BULLET.sub("", text))
    text = text.replace("|", "").replace("*", "").replace("`", "")
    text = _AS_PER.sub(lambda m: _base(m[1]) + ": ", text)
    text = _FILE.sub(lambda m: _base(m[1]), text)
    text = text.replace("/", " ")
    text = _DOLLAR.sub(r"\1 dollars", text)
    text = _PERCENT.sub(r"\1 percent", text)

    def _year(m: re.Match[str]) -> str:
        rest = text[: m.start()] + text[m.end() :]
        if re.search(rf"(?<!\d){m[1]}(?!\d)", rest):  # the same year is spoken elsewhere
            return " "
        return f"from {m[1]}"

    text = _YEAR.sub(_year, text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # a dropped '(2026)' may orphan the mark
    return re.sub(r"[ \t]{2,}", " ", text).strip()
