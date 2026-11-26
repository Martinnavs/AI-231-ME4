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
# Calibrated offline on the T4 snapshot (rag_passages.json): clears lex-2 (0.571), num-3 (0.600),
# id-1 (0.625) for 100% lex/num/id trigger rate with 0/3 off_topic and 0/34 benchmark_v1 triggers.
IMPLICIT_COVERAGE_THRESHOLD = 0.55

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


_SOURCES_RE = re.compile(
    r"(?:which sources|what files|sources checked|files used"
    r"|where did.*come from|what did you use)",
    re.IGNORECASE,
)


def refers_to_sources(text: str) -> bool:
    """True when the user asks which sources/files were checked (old path)."""
    return bool(_SOURCES_RE.search(text))


_OWN_SOURCES_RE = re.compile(
    r"\b(?:which|what)\s+(?:sources?|files?|documents?|sites?)\s+did\s+you\b"
    r"|\bwhere\s+did\s+(?:you|that|this|the\s+answer)\s+(?:get|come|find)\b"
    r"|\bwhat\s+did\s+you\s+(?:check|use|look\s+at)\b"
    r"|\b(?:your|the)\s+sources?\s*\?\s*$"
    r"|\bwhich\s+(?:file|document|source)s?\s+did\s+(?:that|this|it)\s+come\s+from\b"
    r"|\bwhat\s+(?:file|document)\s+was\s+that\s+from\b"
    r"|\bwhich\s+(?:file|document)\s+was\s+that\s+from\b"
    r"|\bthat\s+came\s+from\s+which\s+(?:file|document)\b",
    re.I,
)


def refers_to_own_sources(text: str) -> bool:
    """True when the user asks which sources the assistant checked (docs-first path)."""
    return bool(_OWN_SOURCES_RE.search(text))


_NON_DOC_INTENT = re.compile(
    r"\b(?:weather|forecast|temperature)\b.*\b(?:today|tonight|tomorrow|now|outside|right\s+now|this\s+(?:week|weekend|morning|afternoon|evening))\b"
    r"|\b(?:today|tomorrow|tonight)'?s?\s+(?:weather|forecast)\b"
    r"|\b(?:is\s+it\s+(?:going\s+to\s+)?|will\s+it\s+)rain(?:ing)?\b"
    r"|\b(?:any|the\s+latest|today'?s)\s+news\b"
    r"|\bwhat(?:'s|\s+is)\s+the\s+time\b|\bwhat\s+time\s+is\s+it\b|\bwhat(?:'s|\s+is)\s+(?:today'?s\s+date|the\s+date\s+today)\b"
    r"|\b(?:set|start|cancel)\s+(?:a\s+|an\s+)?(?:timer|alarm)\b|\bremind\s+me\b"
    r"|^\s*(?:please\s+)?play\b|\b(?:turn|set)\s+(?:up|down|the)\s+(?:the\s+)?volume\b"
    r"|\bturn\s+(?:on|off)\s+(?:the\s+)?lights?\b"
    r"|^\s*(?:what(?:'s|\s+is)|how(?:'s|\s+is))\s+the\s+weather(?:\s+like)?\s*\??\s*$",
    re.IGNORECASE,
)


def non_document_intent(text: str) -> bool:
    """True for weather/time/timer/media/news intents: skip docs, go straight to Gemini."""
    return bool(_NON_DOC_INTENT.search(text or ""))


def document_cue(text: str, file_names: Sequence[str]) -> bool:
    """True when an action-phrased text still concerns a document."""
    doc_words = frozenset(
        [
            "document",
            "file",
            "datasheet",
            "spreadsheet",
            "syllabus",
            "handbook",
            "guide",
            "table",
            "sheet",
            "calendar",
        ]
    )
    tokens = re.findall(r"[a-z]{4,}", text.lower())
    if any(w in doc_words for w in tokens):
        return True
    stems = set()
    for fn in file_names:
        base = Path(fn).stem
        for piece in re.split(r"[/_.-]", base):
            if len(piece) >= 4:
                stems.add(piece.lower())
    return bool(stems and any(w in stems for w in tokens))


def _resolve_name(file_path: str, all_files: Sequence[str]) -> str:
    """Return basename, or shortest unambiguous path suffix when duplicates exist."""
    from pathlib import Path as _Path

    basename = _Path(file_path).name
    matches = [str(f) for f in all_files if _Path(str(f)).name == basename]
    if len(matches) <= 1:
        return basename
    # Disambiguate: find shortest relative suffix that separates this file
    target = str(file_path)
    # Try progressively longer suffixes (basename is already tried, go up one dir)
    for depth in range(1, 10):
        parts = _Path(target).parts[-depth - 1 :]
        candidate = str(_Path(*parts)) if parts else basename
        other_parts = [_Path(str(m)).parts[-depth - 1 :] for m in matches if m != target]
        others = [str(_Path(*p)) if p else basename for p in other_parts]
        if candidate not in others:
            return candidate
    return basename


def format_citation(
    answer_text: str, quote: str, meta: Mapping[str, Any] | None, all_files: Sequence[str] = ()
) -> str:
    """Format `As per <file basename>: <answer>`.

    When the same basename appears multiple times in *all_files* the shortest unambiguous
    path suffix is used instead. No section, page, date or quote in speech.
    """
    from pathlib import Path as _Path

    file_path = str((meta or {}).get("file") or "").strip() or "your documents"
    if all_files:
        display = _resolve_name(file_path, all_files)
    else:
        display = _Path(file_path).name if file_path else "your documents"
    return f"As per {display}: {answer_text}"


def citation_metadata(meta: Mapping[str, Any] | None, quote: str) -> dict[str, Any]:
    """Return full source metadata for logging: ``{file, path, section, page, date, quote}``."""
    info = meta or {}
    return {
        "file": str(info.get("file") or ""),
        "path": str(info.get("file") or ""),
        "section": str(info.get("section") or ""),
        "page": info.get("page"),
        "date": str(info.get("date") or ""),
        "quote": quote,
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


MAX_FAILURE_NOTICES = 2


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
    capped = notices[:MAX_FAILURE_NOTICES]
    extra = len(notices) - MAX_FAILURE_NOTICES
    if extra > 0:
        capped.append(f"And {extra} more files I couldn't read.")
    return " ".join(capped), names


def has_enough_content_words(text: str) -> bool:
    """True when ``text`` has at least ``IMPLICIT_MIN_CONTENT_WORDS`` content words."""
    return len(_content_words(text)) >= IMPLICIT_MIN_CONTENT_WORDS


def implicit_gate_passes(
    text: str,
    passages: Sequence[Mapping[str, Any]],
    *,
    theta: float = IMPLICIT_COVERAGE_THRESHOLD,
) -> bool:
    """True when an implicit turn has enough content words and top passage coverage >= ``theta``."""
    if not has_enough_content_words(text) or not passages:
        return False
    return float(passages[0].get("coverage") or 0.0) >= theta


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
            body = format_citation(got.text, got.quote or "", got.passage, cached_files(server))
            server._checked_sources = searched
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
    if got.status == "accepted" and got.text is not None:
        extra["citation_metadata"] = citation_metadata(got.passage, got.quote or "")
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
    passages_ref: list[dict[str, Any]] = []

    async def try_docs() -> DocAnswer:
        nonlocal passages_ref
        passages_ref = await _search_passages(server, user)
        if not implicit_gate_passes(user, passages_ref, theta=theta):
            return DocAnswer("none")
        return await doc_answer(
            server.doc_request, passages_ref, user, file_names=cached_files(server)
        )

    try:
        got = await asyncio.wait_for(try_docs(), DOC_ANSWER_TIMEOUT_S)
    except (TimeoutError, Exception):
        got = DocAnswer("error")

    doc_s = round(time.perf_counter() - started, 3)
    if got.status == "accepted" and got.text is not None:
        failure_prefix = await _collect_failure_prefix(server, prefetched_status)
        body = format_citation(got.text, got.quote or "", got.passage, cached_files(server))
        text = f"{failure_prefix} {body}".strip() if failure_prefix else body
        server.route = "docs"
        sources = [str(p.get("source", "")) for p in passages_ref if p.get("source")]
        server._checked_sources = sources
        meta = citation_metadata(got.passage, got.quote or "")
        spoken_result: dict[str, Any] = await server.spoken(
            text, on_text, docs=got.status, doc_s=doc_s, citation_metadata=meta
        )
        return spoken_result

    route_name, next_work = fallback()
    server.route = route_name
    fallback_result: dict[str, Any] = await next_work
    fallback_result.setdefault("doc_s", doc_s)
    return fallback_result


# --- pleasantries: the closed list the router-branch docs gate skips (ticket 241) ---

_PLEASANTRY_MAX_WORDS = 6
# Normalized (lowercased, punctuation stripped) clauses; "Hello, good evening." is the
# clauses "hello" + "good evening". Combinations of list entries are pleasantries.
_PLEASANTRY_PIECES = frozenset(
    {
        "hello",
        "hi",
        "hey",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "good day",
        "thanks",
        "thank you",
        "thank you very much",
        "thanks a lot",
        "thanks a bunch",
        "thanks very much",
        "thanks so much",
        "okay",
        "ok",
        "alright",
        "all right",
        "okay thanks",
        "ok thanks",
        "got it",
        "got it thanks",
        "understood",
        "sure",
        "sure thing",
        "sounds good",
        "yes",
        "yeah",
        "yep",
        "no",
        "nope",
        "nah",
        "right",
        "cool",
        "great",
        "nice",
        "good",
        "fine",
        "goodbye",
        "bye",
        "bye bye",
        "good bye",
        "see you",
        "see ya",
        "see you later",
        "see you soon",
        "later",
        "cheers",
        "take care",
        "never mind",
        "say ok",
        "sorry",
        "so sorry",
        "my bad",
        "oops",
        "thats all",
        "that is all",
        "thats all for now",
        "nothing else",
        "nothing more",
        "youre welcome",
        "no problem",
        "no worries",
        "anytime",
        "you too",
        "have a nice day",
        "have a good one",
        "all the best",
        "stop",
        "hmm",
        "hmmmm",
        "mm",
        "mmm",
        "uh",
        "uhh",
    }
)
_PLEASANTRY_WH = re.compile(
    r"\b(?:what(?:'s)?|who(?:'s)?|whom|whose|when|where|why|how|which)\b", re.IGNORECASE
)


def is_pleasantry(said: str) -> bool:
    """True for a closed-list pleasantry or acknowledgement ("Hello, good evening.",
    "Got it. Goodbye.", "Say OK."): the docs gate skips these and lets the router's own
    reply answer. A pleasantry that carries a question word, a digit, or a document cue
    is False - it may be a real question wearing a greeting."""
    text = (said or "").strip().lower()
    if not text:
        return False
    if _PLEASANTRY_WH.search(text) or re.search(r"\d", text):
        return False
    if document_cue(text, ()):
        return False
    pieces: list[str] = []
    for sentence in re.split(r"[.!?]+", text):
        for clause in sentence.split(","):
            norm = " ".join(
                re.sub(r"[^a-z0-9]+", " ", clause.replace("'", "").replace("’", "")).split()
            )
            if norm:
                pieces.append(norm)
    if not pieces or sum(len(p.split()) for p in pieces) > _PLEASANTRY_MAX_WORDS:
        return False
    return all(p in _PLEASANTRY_PIECES for p in pieces)


# --- question_like: what goes to the document path first (ticket 241 inverted gate) ---

# Spoken lead-ins, stripped (repeated, comma-separated) before the start check.
_LEADING_FILLER = re.compile(
    r"^\s*(?:sorry\s+i\s+mean|okay\s+then|broadly\s+speaking|can\s+you\s+tell\s+me"
    r"|could\s+you\s+tell\s+me|got\s+it|understood|all\s+right|alright|tell\s+me|sorry|i\s+mean"
    r"|actually|right|okay|ok|hey|please|well|now|so|and)\b\s*,?\s*",
    re.IGNORECASE,
)


def _strip_leading_fillers(said: str) -> str:
    text = said
    while True:
        stripped = _LEADING_FILLER.sub("", text, count=1)
        if stripped == text:
            return text
        text = stripped


_QUESTION_START = re.compile(
    r"^\s*(?:on\s+which|in\s+which|during\s+which|for\s+which|at\s+which|to\s+whom|how\s+come"
    r"|i\s+want\s+to\s+know|i'd\s+like\s+to\s+know|do\s+you\s+know|tell\s+me"
    r"|what|who|whom|whose|when|where|which|why|how|is|are|was|were|does|do|did|can|could|"
    r"should|will|would|has|have|had|may|might|according|in\s+the|per\s+the)\b",
    re.IGNORECASE,
)
_COMMAND_START = re.compile(
    r"^\s*(?:please\s+)?(?:turn|switch|set|play|pause|resume|stop|skip|call|text|send|open|close|"
    r"dim|brighten|mute|unmute|remind|start|cancel|make|build|create|delete|add|put)\b",
    re.IGNORECASE,
)
# A wh-word or an auxiliary: "The ATSDR module mentions a law from 2010. Who signed it?" is
# question-like because "Who" starts a clause after the period - and so is a garbled
# question start ("This was California's commercial electricity price in August 2026.")
# wherever the auxiliary sits, since ASR can mangle the real question word.
_WH_OR_AUX = (
    r"what|who|whom|whose|when|where|why|how|which|is|are|was|were|does|do|did|can|could"
    r"|should|will|would|has|have|had|may|might|shall"
)
_CLAUSE_QUESTION = re.compile(rf"[.,]\s*(?:{_WH_OR_AUX})\b", re.IGNORECASE)
_QUESTION_WORD_ANYWHERE = re.compile(rf"\b(?:{_WH_OR_AUX})\b", re.IGNORECASE)


def question_like(said: str) -> bool:
    """A question about something (not a home command): worth a document lookup first.

    Leading fillers ("Got it, ", "Sorry, I mean ") are stripped before the start check, a
    wh-word or auxiliary starting a clause after a comma/period counts, and so does one
    anywhere in the utterance (a garbled question start). Home commands and closed-list
    pleasantries are never questions."""
    if _COMMAND_START.match(said):
        return False
    if is_pleasantry(said):
        return False
    if _QUESTION_START.match(_strip_leading_fillers(said)):
        return True
    return bool(_CLAUSE_QUESTION.search(said) or _QUESTION_WORD_ANYWHERE.search(said))


_WH_START = re.compile(
    r"^\s*(?:what|who|whom|whose|when|where|which|why|how|according)\b", re.IGNORECASE
)


def wh_question_start(said: str) -> bool:
    """A narrow question test: after leading fillers, it starts with a wh-word. Used where a
    wrong yes costs more than a miss (an open chain handing a turn to the documents)."""
    if _COMMAND_START.match(said) or is_pleasantry(said):
        return False
    return bool(_WH_START.match(_strip_leading_fillers(said)))


_GENERAL_QUESTION_PATTERNS = (
    re.compile(r"\bwhat\s+are\s+(?:all\s+)?the\s+(?:files?|documents?)\s+about\b", re.IGNORECASE),
    re.compile(
        r"\bwhat\s+(?:do\s+you\s+have|can\s+i\s+ask|files\s+are\s+there|is\s+in\s+here)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\boverview\s+of\s+(?:my|the|these|those)\s+(?:files?|documents?)\b", re.IGNORECASE
    ),
    re.compile(
        r"\bsummar(?:y|ise|ize)\s+(?:of\s+)?(?:my|the|these|those)\s+(?:files?|documents?|folder)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\blist\s+(?:all|every)\s+(?:of\s+)?(?:my|the|these|those)\s+(?:files?|documents?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bhow\s+many\s+files?\b", re.IGNORECASE),
    re.compile(r"\bmain\s+(?:topics?|themes?)\b", re.IGNORECASE),
    re.compile(
        r"\bwhich\s+(?:other\s+)?files?\s+(?:mention|mentions|are|is|contain|contains|come|came)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bhow\s+do\s+(?:the\s+)?two\b.*\bfiles?\b.*\brelate\b", re.IGNORECASE),
)


def is_general_question(text: str) -> bool:
    """True for questions about the file set as a whole (GENERAL lane, ticket 221b)."""
    if not text or not text.strip():
        return False
    return any(p.search(text) for p in _GENERAL_QUESTION_PATTERNS)
