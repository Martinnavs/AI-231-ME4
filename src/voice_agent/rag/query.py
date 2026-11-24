"""Selectable query/selection variants for FTS5 retrieval.

Variants: ``baseline`` (default, byte-identical), ``q1`` (cue/stop strip),
``q2`` (IDF term prune), ``q3`` (tiered AND-then-OR), ``q4`` (bigram phrases).
Combinations: ``q1+q2`` applies left to right.  ``q5`` is ``files=`` on search.
Active variant from ``VOICE_AGENT_RAG_QUERY`` (default ``baseline``).
"""

from __future__ import annotations

import os
import re
import string
from collections.abc import Sequence
from pathlib import Path

from doc_chat.corpus import tokenize

MAX_QUERY_TERMS = 64
DEFAULT_DF_MAX = 0.20
_CUE_PHRASES = ["according to", "tell me"]
_CUE_STOP = frozenset(
    [
        "documents",
        "document",
        "docs",
        "doc",
        "files",
        "file",
        "notes",
        "folder",
        "says",
        "say",
        "mentions",
        "mention",
        "states",
        "state",
        "my",
        "our",
        "these",
        "those",
        "please",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "how",
        "does",
        "do",
        "did",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "about",
    ]
)
_FILE_EXT = frozenset(["md", "txt", "csv", "pdf", "docx", "pptx", "xlsx", "html"])
_FILE_TYPE_SET = frozenset(
    [
        "log",
        "logs",
        "file",
        "document",
        "doc",
        "report",
        "csv",
        "spreadsheet",
        "sheet",
        "workbook",
        "pdf",
        "deck",
        "slides",
        "presentation",
        "notes",
        "readme",
        "page",
        "article",
        "manual",
        "guide",
        "syllabus",
        "schedule",
        "table",
        "data",
    ]
)


def _active_variant() -> str:
    return os.environ.get("VOICE_AGENT_RAG_QUERY", "baseline").strip().lower()


def _quote_term(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def _build_match_query(query: str) -> str:
    """Baseline: safe OR-joined FTS5 MATCH string of at most 64 unique quoted terms."""
    terms = list(dict.fromkeys(tokenize(query)))[:MAX_QUERY_TERMS]
    return " OR ".join(_quote_term(t) for t in terms) if terms else ""


class QueryStep:
    __slots__ = ("match", "is_and")

    def __init__(self, match: str, *, is_and: bool = False) -> None:
        self.match = match
        self.is_and = is_and


def _extract_file_names(text: str, corpus_files: Sequence[str]) -> set[str]:
    if not text or not corpus_files:
        return set()
    qwords = [w.strip(string.punctuation) for w in text.lower().split()]
    out: set[str] = set()
    for fn in corpus_files:
        stem = Path(fn).stem.lower()
        if stem in {"docs", "file", "files", "document", "documents", "data"}:
            continue
        parts = [p for p in re.split(r"[^a-zA-Z0-9]+", stem) if p]
        for t in parts:
            if len(t) < 4 or not t[0].isalpha():
                continue
            for i in range(len(qwords)):
                if qwords[i] != t:
                    continue
                j = i + 1
                if j < len(qwords) and qwords[j] in _FILE_TYPE_SET:
                    out.add(fn)
                    break
                if j + 1 < len(qwords) and qwords[j + 1] in _FILE_TYPE_SET:
                    out.add(fn)
                    break
        if fn not in out:
            fname = Path(fn).name.lower()
            if fname in qwords:
                out.add(fn)
    return out


def get_scoped_files(query: str, corpus_files: Sequence[str]) -> set[str]:
    return _extract_file_names(query, corpus_files)


def _q1_terms(query: str) -> list[str]:
    raw = tokenize(query)
    stop = set(_CUE_STOP)
    qlower = query.lower()
    for phrase in _CUE_PHRASES:
        if phrase in qlower:
            stop.update(w.lower() for w in phrase.split())
    for word in query.lower().split():
        w = word.strip(string.punctuation)
        ext = Path(w).suffix.lstrip(".")
        if ext in _FILE_EXT:
            stop.add(Path(w).stem.lower())
            stop.add(w)
    kept = [t for t in raw if t.lower() not in stop]
    if len(kept) < 2:
        return list(dict.fromkeys(raw))[:MAX_QUERY_TERMS]
    return list(dict.fromkeys(kept))[:MAX_QUERY_TERMS]


def _q2_terms(terms: list[str], df_table: dict[str, float] | None) -> list[str]:
    if df_table is None:
        return terms
    ranked = sorted(terms, key=lambda t: df_table.get(t.lower(), 0.0))
    threshold = DEFAULT_DF_MAX * len(ranked)
    kept = [t for t in ranked if df_table.get(t.lower(), 0.0) <= threshold]
    if len(kept) < 2:
        kept = ranked[: max(2, len(ranked))]
    return kept[:MAX_QUERY_TERMS]


def _q4_terms(query: str) -> list[str]:
    stop = set(_CUE_STOP)
    for phrase in _CUE_PHRASES:
        stop.update(w.lower() for w in phrase.split())
    return list(dict.fromkeys(t for t in tokenize(query) if t.lower() not in stop))[
        :MAX_QUERY_TERMS
    ]


def _build_q3_steps(terms: list[str]) -> list[QueryStep]:
    if len(terms) < 3:
        match = " OR ".join(_quote_term(t) for t in terms)
        return [QueryStep(match)] if match else []
    ranked = sorted(terms, key=lambda t: (len(t), t))
    and_t, or_t = ranked[:3], ranked[3:]
    and_m = " AND ".join(_quote_term(t) for t in and_t)
    or_m = " OR ".join(_quote_term(t) for t in or_t) if or_t else and_m
    return [QueryStep(and_m, is_and=True), QueryStep(or_m)]


def build_query(
    query: str,
    variant: str | None = None,
    *,
    df_table: dict[str, float] | None = None,
) -> list[QueryStep]:
    """Build query steps for a named variant."""
    if variant is None:
        variant = _active_variant()
    if variant == "baseline":
        m = _build_match_query(query)
        return [QueryStep(m)] if m else []
    parts = [p.strip() for p in variant.split("+") if p.strip()]
    raw = list(dict.fromkeys(tokenize(query)))[:MAX_QUERY_TERMS]
    if not raw:
        return []
    terms, has_q3 = list(raw), False
    for p in parts:
        if p == "q1":
            terms = _q1_terms(query)
        elif p == "q2":
            terms = _q2_terms(terms, df_table)
        elif p == "q3":
            has_q3 = True
        elif p == "q4":
            terms = _q4_terms(query)
    if has_q3:
        return _build_q3_steps(terms)
    if len(parts) == 1 and parts[0] == "q4":
        inf = _q4_terms(query)
        bigrams = [_quote_term(f"{inf[i]} {inf[i + 1]}") for i in range(len(inf) - 1)]
        m = " OR ".join([_quote_term(t) for t in inf] + bigrams)
        return [QueryStep(m)] if m else []
    if len(parts) == 1 and parts[0] == "q1":
        m = " OR ".join(_quote_term(t) for t in terms[:MAX_QUERY_TERMS])
        return [QueryStep(m)] if m else []
    if len(parts) == 1 and parts[0] == "q2":
        m = " OR ".join(_quote_term(t) for t in terms)
        return [QueryStep(m)] if m else []
    m = " OR ".join(_quote_term(t) for t in terms[:MAX_QUERY_TERMS])
    return [QueryStep(m)] if m else []
