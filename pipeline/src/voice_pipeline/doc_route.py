"""Deterministic document route: explicit document references, citations, and ingest notices."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from voice_pipeline.doc_answer import (
    DOC_ANSWER_TIMEOUT_S,
    DOC_TURN_TIMEOUT_S,
    DocAnswer,
    _content_words,
    passages_with_meta,
)
from voice_pipeline.doc_answer import (
    answer as doc_answer,
)

NOT_FOUND_OFFER = "The documents don't say. Want me to look it up on the web?"
TRUNCATED_NOT_FOUND = (
    "I only read the first {amount} of {name}, so the answer may be further down. "
    "Want me to look it up on the web?"
)
EMPTY_FOLDER_REPLY = "Your documents folder is empty, so there is nothing to search."
STILL_READING_REPLY = "I'm still reading your documents. Ask me again in a moment."
STILL_INDEXING_REPLY = "I'm still reading your documents. Ask me again in a moment."
DOCS_STATUS_TIMEOUT_S = 0.5
IMPLICIT_MIN_CONTENT_WORDS = 3
# best(top3) >= 0.34: 25/29 (86.2%) answerable, 0 off_topic, 0/34 benchmark_v1 triggers
IMPLICIT_COVERAGE_THRESHOLD = 0.34

_EXPLICIT_DOC = re.compile(
    r"\b(?:my|the|our|these|those)\s+(?:documents?|docs|notes)\b"
    r"|\bmy\s+files?\b"
    r"|\bwhat\s+(?:do|does)\s+(?:my|the)\s+(?:documents?|docs|notes|files?)\s+say\b"
    r"|\b(?:the|my)\s+files?\s+(?:say|says|mention|mentions|state|states|show|shows)\b"
    r"|\b(?:according\s+to|in|from|check|search|look\s+in)\s+(?:my|the)\s+"
    r"(?:documents?|docs|notes)\b",
    re.IGNORECASE,
)
_THE_FILES = re.compile(r"\bthe\s+files\b", re.IGNORECASE)
_OWN_WORK_CUE = re.compile(
    r"\b(?:game|games|app|apps|clone|the fix|the bug|the code"
    r"|you (?:made|built|wrote|created|coded)|in the game)\b",
    re.IGNORECASE,
)
_FILE_PREFETCH_CUE = re.compile(
    r"\baccording\s+to\b|\.(?:md|txt|csv|pdf|docx)\b",
    re.IGNORECASE,
)
_STEM_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "do",
        "for",
        "from",
        "go",
        "he",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "no",
        "of",
        "on",
        "or",
        "so",
        "the",
        "to",
        "up",
        "us",
        "we",
    }
)

# Extra stop words for the implicit gate: common words that appear frequently in document
# headers/body (timestamps, section titles, generic terms) and cause false positive matches
# on non-document turns. Kept separate from _OVERLAP_STOP so the answer overlap check stays stricter.
_IMPLICIT_GATE_STOP = frozenset(
    _STEM_STOP
    | {
        "about",
        "above",
        "after",
        "again",
        "also",
        "am",
        "another",
        "around",
        "away",
        "back",
        "bad",
        "before",
        "best",
        "big",
        "bit",
        "call",
        "can",
        "cold",
        "come",
        "date",
        "day",
        "degree",
        "did",
        "different",
        "do",
        "does",
        "don",
        "down",
        "early",
        "even",
        "get",
        "got",
        "good",
        "got",
        "going",
        "got",
        "got",
        "go",
        "got",
        "had",
        "has",
        "have",
        "here",
        "hot",
        "how",
        "huh",
        "hmm",
        "hmm",
        "hmm",
        "hmm",
        "im",
        "just",
        "know",
        "late",
        "last",
        "let",
        "light",
        "lights",
        "line",
        "long",
        "look",
        "lot",
        "make",
        "many",
        "may",
        "might",
        "minute",
        "minutes",
        "much",
        "must",
        "need",
        "never",
        "next",
        "night",
        "none",
        "now",
        "one",
        "only",
        "out",
        "own",
        "place",
        "right",
        "run",
        "say",
        "set",
        "shall",
        "she",
        "should",
        "sick",
        "something",
        "still",
        "take",
        "tell",
        "then",
        "there",
        "thing",
        "things",
        "think",
        "three",
        "time",
        "timer",
        "times",
        "today",
        "tonight",
        "took",
        "turn",
        "two",
        "up",
        "use",
        "used",
        "version",
        "very",
        "want",
        "way",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whole",
        "will",
        "with",
        "won",
        "would",
        "yeah",
        "year",
        "yesterday",
        "yet",
        "you",
        "your",
    }
)


def _matches_file_name(text: str, file_names: Sequence[str]) -> bool:
    for raw_name in file_names:
        name = Path(raw_name.strip()).name if raw_name.strip() else ""
        if not name:
            continue
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE):
            return True
        stem = Path(name).stem
        if (
            len(stem) >= 2
            and stem.lower() not in _STEM_STOP
            and (
                re.search(
                    rf"\b(?:according\s+to|in|from|per|check)\s+(?:(?:the|my)\s+)?{re.escape(stem)}(?!\w)",
                    text,
                    re.IGNORECASE,
                )
                or re.search(
                    rf"(?<!\w){re.escape(stem)}\s+(?:says?|mentions?|states?)\b",
                    text,
                    re.IGNORECASE,
                )
            )
        ):
            return True
    return False


def refers_to_documents(text: str, file_names: Sequence[str] = ()) -> bool:
    """True when the user explicitly asks about their documents, notes, docs, or a named file."""
    if not text or not text.strip():
        return False
    if _EXPLICIT_DOC.search(text) or _matches_file_name(text, file_names):
        return True
    return bool(_THE_FILES.search(text) and not _OWN_WORK_CUE.search(text))


def might_name_a_file(text: str) -> bool:
    """Cheap pre-check before calling `docs_status` on turn 1 when file names are not cached yet."""
    return bool(_FILE_PREFETCH_CUE.search(text))


def _resolve_source_name(source: str, all_sources: Sequence[str] | None = None) -> str:
    """Return basename, or shortest unambiguous suffix when the basename is not unique."""
    base = Path(source).name if source else ""
    if not base:
        return "your documents"
    if all_sources is None:
        return base
    dupes = [s for s in all_sources if Path(s).name == base]
    if len(dupes) <= 1:
        return base
    # Deduplicate: find shortest suffix (parent/basename) that is unique.
    for suffix_len in range(1, 5):
        candidates = ["/".join(Path(s).parts[-suffix_len:]) for s in dupes]
        if len(set(candidates)) == len(candidates):
            return "/".join(Path(source).parts[-suffix_len:])
    return base


def format_citation(
    answer_text: str,
    quote: str,
    meta: Mapping[str, Any] | None,
    all_sources: Sequence[str] | None = None,
) -> str:
    """Return terse ``As per <file basename>: <answer>`` for speech."""
    info = meta or {}
    file_path = str(info.get("file") or "").strip()
    return f"As per {_resolve_source_name(file_path, all_sources)}: {answer_text}"


def citation_metadata(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return full citation metadata dict for the log."""
    info = meta or {}
    return {
        "file": str(info.get("file") or ""),
        "section": str(info.get("section") or ""),
        "page": info.get("page"),
        "date": str(info.get("date") or ""),
        "quote": str(info.get("quote") or ""),
    }


def _decode_status(result: Any) -> dict[str, Any] | None:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        with contextlib.suppress(ValueError):
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                return parsed
    return None


def parse_docs_status(result: Any) -> tuple[list[str], list[dict[str, Any]]]:
    """Return ``(file_names, failures_with_file_meta)`` from a T2 `docs_status` payload."""
    data = _decode_status(result)
    if data is None:
        return [], []
    raw_files = data.get("files")
    files_raw: list[Any] = raw_files if isinstance(raw_files, list) else []
    by_name: dict[str, dict[str, Any]] = {}
    names: list[str] = []
    for entry in files_raw:
        if isinstance(entry, dict) and str(entry.get("name") or "").strip():
            name = str(entry["name"])
            names.append(name)
            by_name[name] = entry

    raw_failures = data.get("failures")
    failures_raw: list[Any] = raw_failures if isinstance(raw_failures, list) else []
    failures: list[dict[str, Any]] = []
    for item in failures_raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        file_meta = by_name.get(name, {})
        failures.append(
            {
                "name": name,
                "status": str(item.get("status") or file_meta.get("status") or "unreadable"),
                "reason": str(item.get("reason") or ""),
                "mtime": file_meta.get("mtime"),
                "size": file_meta.get("size"),
            }
        )
    return names, failures


def docs_indexing_state(status_result: Any) -> tuple[bool, bool, str | None]:
    """Return ``(indexing, ready, parsing)`` from a decoded docs_status payload.

    Defaults to ``(False, True, None)`` when the payload is missing or from an old server
    that does not include these fields.
    """
    data = _decode_status(status_result)
    if data is None:
        return False, True, None
    return (
        bool(data.get("indexing", False)),
        bool(data.get("ready", True)),
        str(data.get("parsing")) if data.get("parsing") is not None else None,
    )


def new_failures_prefix(
    status_result: Any, seen: dict[str, tuple[Any, ...]]
) -> tuple[str, list[str]]:
    """Return ``(spoken_prefix, file_names)`` naming each newly failed file once per change."""
    names, failures = parse_docs_status(status_result)
    current: dict[str, tuple[tuple[Any, ...], str]] = {}
    for entry in failures:
        name = entry["name"]
        sig = (entry["mtime"], entry["size"], entry["status"], entry["reason"])
        why = (entry["reason"].strip() or entry["status"].strip() or "unreadable").rstrip(".")
        current[name] = (sig, f"I couldn't read {name}: {why}.")

    for old_name in list(seen):
        if old_name not in current:
            del seen[old_name]

    notices: list[str] = []
    for name, (sig, sentence) in current.items():
        if seen.get(name) != sig:
            seen[name] = sig
            notices.append(sentence)
    return " ".join(notices), names


def has_enough_content_words(text: str) -> bool:
    """True when ``text`` has at least ``IMPLICIT_MIN_CONTENT_WORDS`` content words."""
    return len(_content_words(text)) >= IMPLICIT_MIN_CONTENT_WORDS


def _gate_content_words(text: str) -> set[str]:
    """Content words for the implicit gate (broader stop list than _content_words)."""
    from voice_pipeline.doc_answer import norm

    words: set[str] = set()
    for raw in re.findall(r"[a-z0-9']+", norm(text)):
        word = raw.removesuffix("'s").strip("'")
        if word and word not in _IMPLICIT_GATE_STOP:
            words.add(word)
    return words


def implicit_gate_passes(
    text: str,
    passages: Sequence[Mapping[str, Any]],
    *,
    theta: float = IMPLICIT_COVERAGE_THRESHOLD,
) -> bool:
    """True when >=2 distinct gate-content words from the query match
    the text of a single top-3 passage.
    ``theta`` is kept for API compatibility but the decision is word-count based."""
    if not has_enough_content_words(text) or not passages:
        return False
    qwords = _gate_content_words(text)
    if not qwords:
        return False
    for p in passages[:3]:
        pwords = _gate_content_words(p.get("text", ""))
        if len(qwords & pwords) >= 2:
            return True
    return False


def cached_files(server: Any) -> list[str]:
    files: list[str] | None = getattr(server, "_doc_files", None)
    if files is None:
        files = []
        server._doc_files = files
    return files


def set_cached_files(server: Any, names: list[str]) -> None:
    server._doc_files = list(names)


async def _collect_failure_prefix(server: Any, prefetched_status: Any = None) -> str:
    seen_failures: dict[str, tuple[Any, ...]] | None = getattr(server, "_doc_failures", None)
    if seen_failures is None:
        seen_failures = {}
        server._doc_failures = seen_failures

    cached_files(server)

    status_result = prefetched_status
    if status_result is None and "docs_status" in server.tools:
        with contextlib.suppress(Exception):
            status_result = await server.call_tool("docs_status", {})

    if status_result is None:
        return ""
    failure_prefix, names = new_failures_prefix(status_result, seen_failures)
    if names:
        set_cached_files(server, names)
    return failure_prefix


async def _search_passages(server: Any, user: str) -> list[dict[str, Any]]:
    raw = await server.call_tool("search_docs", {"query": user})
    passages = passages_with_meta(raw)
    files = cached_files(server)
    for passage in passages:
        src = passage.get("source")
        if src and src not in files:
            files.append(src)
    return passages


def folder_is_empty(status_result: Any) -> bool:
    """True when a valid `docs_status` payload explicitly reports an empty `files` list."""
    data = _decode_status(status_result)
    if data is None:
        return False
    files = data.get("files")
    return isinstance(files, list) and len(files) == 0


_TRUNCATED_AMOUNT_RE = re.compile(r"first (\d+) (rows|MB)")


def truncated_searched_file(status_result: Any, sources: Sequence[str]) -> tuple[str, str] | None:
    """Return ``(file_name, amount_text)`` when a searched source was truncated.

    ``amount_text`` is the human-readable span, e.g. ``"50000 rows"`` or ``"64 MB"``.
    """
    data = _decode_status(status_result)
    if data is None:
        return None
    files = data.get("files")
    if not isinstance(files, list):
        return None
    for entry in files:
        if not isinstance(entry, dict):
            continue
        note = entry.get("note")
        if not isinstance(note, str):
            continue
        m = _TRUNCATED_AMOUNT_RE.search(note)
        if not m:
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        for source in sources:
            if name == source or source.endswith("/" + name):
                return (name, f"{m.group(1)} {m.group(2)}")
    return None


async def _fetch_docs_status(server: Any, prefetched_status: Any = None) -> Any:
    if prefetched_status is not None:
        return prefetched_status
    if "docs_status" not in server.tools:
        return None
    try:
        return await asyncio.wait_for(server.call_tool("docs_status", {}), DOCS_STATUS_TIMEOUT_S)
    except (TimeoutError, Exception):
        return None


async def handle_docs_turn(
    server: Any,
    user: str,
    on_text: Any,
    *,
    prefetched_status: Any = None,
) -> dict[str, Any]:
    """Execute the deterministic `route="docs"` turn on ``server``."""
    started = time.perf_counter()
    status_result = await _fetch_docs_status(server, prefetched_status)
    if status_result is not None:
        failure_prefix = await _collect_failure_prefix(server, status_result)
    else:
        cached_files(server)
        failure_prefix = ""

    if folder_is_empty(status_result):
        got = DocAnswer("none")
        body = EMPTY_FOLDER_REPLY
    else:
        got = DocAnswer("timeout")
        search_done = False
        searched: list[str] = []

        async def ask() -> DocAnswer:
            nonlocal search_done, searched
            try:
                passages = await _search_passages(server, user)
            except Exception:
                search_done = True
                return DocAnswer("error")
            search_done = True
            searched = [str(p["source"]) for p in passages]
            return await doc_answer(
                server.doc_request, passages, user, file_names=cached_files(server)
            )

        timed_out_in_search = False
        try:
            got = await asyncio.wait_for(ask(), DOC_TURN_TIMEOUT_S)
        except TimeoutError:
            timed_out_in_search = not search_done
        except Exception:
            got = DocAnswer("error")

        if got.status == "accepted" and got.text is not None:
            body = format_citation(
                got.text, got.quote or "", got.passage, all_sources=cached_files(server)
            )
        elif timed_out_in_search:
            body = STILL_READING_REPLY
        elif (
            status_result is not None
            and (doc_state := docs_indexing_state(status_result))
            and doc_state[0]
            and not doc_state[1]
        ):
            body = STILL_INDEXING_REPLY
        else:
            _trunc = truncated_searched_file(status_result, searched)
            if _trunc is not None:
                body = TRUNCATED_NOT_FOUND.format(amount=_trunc[1], name=_trunc[0])
            else:
                body = NOT_FOUND_OFFER
    text = f"{failure_prefix} {body}".strip() if failure_prefix else body

    extra: dict[str, Any] = {
        "docs": got.status,
        "doc_s": round(time.perf_counter() - started, 3),
    }
    if got.rejected:
        extra["rejected"] = got.rejected
    if got.status == "accepted" and got.passage:
        extra["citation"] = citation_metadata({**(got.passage or {}), "quote": got.quote or ""})
    result: dict[str, Any] = await server.spoken(text, on_text, **extra)
    return result


async def handle_implicit_docs_turn(
    server: Any,
    user: str,
    on_text: Any,
    fallback: Callable[[], tuple[str, Any]],
    *,
    prefetched_status: Any = None,
    theta: float = IMPLICIT_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    """Gate an implicit turn via local ``search_docs``; fall through silently on miss or fail."""
    if not has_enough_content_words(user):
        route_name, next_work = fallback()
        server.route = route_name
        result: dict[str, Any] = await next_work
        return result

    started = time.perf_counter()
    got = DocAnswer("none")

    async def try_docs() -> DocAnswer:
        passages = await _search_passages(server, user)
        if not implicit_gate_passes(user, passages, theta=theta):
            return DocAnswer("none")
        return await doc_answer(server.doc_request, passages, user, file_names=cached_files(server))

    try:
        got = await asyncio.wait_for(try_docs(), DOC_ANSWER_TIMEOUT_S)
    except (TimeoutError, Exception):
        got = DocAnswer("error")

    doc_s = round(time.perf_counter() - started, 3)
    if got.status == "accepted" and got.text is not None:
        failure_prefix = await _collect_failure_prefix(server, prefetched_status)
        body = format_citation(
            got.text, got.quote or "", got.passage, all_sources=cached_files(server)
        )
        text = f"{failure_prefix} {body}".strip() if failure_prefix else body
        server.route = "docs"
        kw: dict[str, Any] = {"docs": got.status, "doc_s": doc_s}
        if got.passage:
            kw["citation"] = citation_metadata({**(got.passage or {}), "quote": got.quote or ""})
        spoken_result: dict[str, Any] = await server.spoken(text, on_text, **kw)
        return spoken_result

    route_name, next_work = fallback()
    server.route = route_name
    fallback_result: dict[str, Any] = await next_work
    fallback_result.setdefault("doc_s", doc_s)
    return fallback_result
