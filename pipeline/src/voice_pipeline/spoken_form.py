"""The reply text in the form a voice should say it, applied only on the voice/TTS side
(ticket 231, ticket 242): the 'As per <file>:' docs citation lead becomes 'From your
<title words> document:' (the card title passed as `doc_title` when the caller has one,
else the humanized stem - never the filename), a bare negative number becomes 'minus N',
any remaining file token or path becomes what a person would say, money and percent get a
unit word, markdown marks are stripped. The web lead 'As per Wikipedia and CNN, ...' names
publishers, has no file extension, and is untouched. The agent server's text (text mode,
the offline evals) is untouched. Idempotent; the digits inside numbers are never changed,
only the marks around them."""

from __future__ import annotations

import re

_EXT = r"(?:md|txt|csv|tsv|pdf|docx|doc|xlsx|xls|pptx|ppt|html|htm|json|log|ini|toml)"
# 'As per gs_salary_table_2026.pdf: $45,980'
#   -> 'From your gs salary table 2026 document: 45,980 dollars'
# the token may carry spaces ('As per plain notes.md: X'); the optional '(YYYY)' is the
# temporal year tag (ask_documents) and stays in the sentence
_AS_PER = re.compile(
    r"\bAs per\s+((?:[\w-]+/)*[\w-]+(?: [\w-]+)*\." + _EXT + r")((?:\s*\(\d{4}\))?)\s*:\s*",
    re.IGNORECASE,
)
# the citation's own unknown-file fallback: 'As per your documents: X' -> 'From your documents: X'
_DOCS_FALLBACK = re.compile(r"\bAs per\s+your documents((?:\s*\(\d{4}\))?)\s*:\s*", re.IGNORECASE)
# a file token anywhere: 'a/b.pdf' -> 'b', 'notes.md' -> 'notes'
_FILE = re.compile(r"\b(?:[\w-]+/)*([\w-]+)\." + _EXT + r"\b")
# a bare negative number: '-215' -> 'minus 215'; not a range ('5-10'), a hyphenated word
# ('well-known'), a date ('2026-01'), or a dollar amount ('$-215')
_NEGATIVE = re.compile(r"(?<![\w$])-(\d[\d,]*(?:\.\d+)?)")
_BULLET = re.compile(r"(?m)^\s*- ")
_HEADER = re.compile(r"(?m)^\s*#{1,6}\s*")
_DOLLAR = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
_PERCENT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s?%")
_YEAR = re.compile(r"\(((?:19|20)\d{2})\)")
# a card title that is an auto-generated table description, not a document name
_TABLE_LIKE = re.compile(r"\btable\b|\brow headers?\b", re.IGNORECASE)


def _base(token: str) -> str:
    """Base name without extension, underscores and hyphens as spaces ('a/b.pdf' -> 'b')."""
    return re.sub(r"[_-]+", " ", token.rsplit("/", 1)[-1])


def _doc_words(file_token: str) -> str:
    """The file's stem in words (mirrors doc_route.spoken_title): no extension, no folder,
    separators as spaces, a leading date/number-only run ('2024 01') dropped."""
    stem = re.sub(r"\." + _EXT + r"$", "", file_token.rsplit("/", 1)[-1], flags=re.IGNORECASE)
    words = " ".join(re.sub(r"[_\-.]+", " ", stem).split())
    tokens = words.split()
    while tokens and tokens[0].isdigit():
        tokens.pop(0)
    return " ".join(tokens) if tokens else words


def _title_for(doc_title: str | None) -> str | None:
    """The card title for the 'From your document on <title>:' lead: the cleaned title (as in
    doc_route.spoken_title) when it is 60 chars or fewer and not a table description; else
    None (the humanized stem)."""
    if not doc_title:
        return None
    title = " ".join(str(doc_title).split())
    title = re.sub(r"\." + _EXT + r"$", "", title, flags=re.IGNORECASE).rstrip(" .,:;!?-")
    if len(title) > 60 or _TABLE_LIKE.search(title):
        return None
    return title


def spoken_form(text: str, doc_title: str | None = None) -> str:
    """The text as a voice should say it (tickets 231/242); plain prose comes back unchanged.
    `doc_title` is the cited document's card title, when the caller has one: the citation
    lead is then spoken as 'From your document on <title>:' instead of the stem words."""
    text = _HEADER.sub("", _BULLET.sub("", text))
    text = text.replace("|", "").replace("*", "").replace("`", "")
    title = _title_for(doc_title)

    def _lead(m: re.Match[str]) -> str:
        year = m.group(2) or ""
        if title is not None:
            return f"From your document on {title}{year}: "
        words = _doc_words(m[1])
        if not words:  # an all-number stem ('2031.pdf') has no words to say
            return f"From your document{year}: "
        return f"From your {words} document{year}: "

    text = _AS_PER.sub(_lead, text)
    text = _DOCS_FALLBACK.sub(lambda m: f"From your documents{m.group(1) or ''}: ", text)
    text = _FILE.sub(lambda m: _base(m[1]), text)
    text = text.replace("/", " ")
    text = _DOLLAR.sub(r"\1 dollars", text)
    text = _PERCENT.sub(r"\1 percent", text)
    text = _NEGATIVE.sub(lambda m: f"minus {m[1]}", text)

    def _year(m: re.Match[str]) -> str:
        rest = text[: m.start()] + text[m.end() :]
        if re.search(rf"(?<!\d){m[1]}(?!\d)", rest):  # the same year is spoken elsewhere
            return " "
        return f"from {m[1]}"

    text = _YEAR.sub(_year, text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # a dropped '(2026)' may orphan the mark
    return re.sub(r"[ \t]{2,}", " ", text).strip()
