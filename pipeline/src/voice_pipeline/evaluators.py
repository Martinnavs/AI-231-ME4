"""Single-purpose code evaluators over T1 turn rows (docs/TURN-TRACE.md).

Each evaluator is a pure function ``evaluate_<name>(row, expect) -> Verdict`` returning
``pass``, ``fail``, or ``n/a``. No function here computes or prints an aggregate score.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from voice_pipeline import agent_tools, call_policy, doc_answer

Outcome = Literal["pass", "fail", "n/a"]

# Copied from src/voice_agent/mcp_servers/gemini_grounding.py (read-only across packages;
# test_evaluators.py pins both strings together via AST).
HEDGE = "I couldn't confirm that with a search, so I won't guess."

_ABSTAIN_PHRASE = re.compile(
    r"(?:couldn'?t|could not)\s+confirm|documents?\s+(?:don'?t|do not)\s+say",
    re.IGNORECASE,
)
_CLOSEST_RESULT = re.compile(r"closest result says\s*:", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
_ABBREV_DOTS = re.compile(r"(?<=\b[A-Z])\.(?=[A-Z]\b)")
_CONTRACTION_TAIL = re.compile(r"'(?:s|m|re|ve|ll|d)$")

# Words skipped wherever they appear (articles/conjunctions inside titles and "I" pronouns).
_ALWAYS_COMMON = frozenset(
    {
        "i",
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "from",
        "by",
        "is",
        "was",
        "ok",
        "okay",
        "mr",
        "mrs",
        "ms",
        "dr",
    }
)

# Ordinary words that are capitalised only because they open a sentence (never proper names).
_SENTENCE_STARTERS = _ALWAYS_COMMON | frozenset(
    {
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "some",
        "any",
        "no",
        "none",
        "all",
        "both",
        "each",
        "every",
        "either",
        "neither",
        "many",
        "much",
        "more",
        "most",
        "few",
        "several",
        "other",
        "another",
        "such",
        "but",
        "nor",
        "so",
        "yet",
        "as",
        "into",
        "onto",
        "out",
        "over",
        "per",
        "under",
        "until",
        "upon",
        "within",
        "without",
        "about",
        "above",
        "across",
        "after",
        "against",
        "along",
        "among",
        "around",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "between",
        "beyond",
        "despite",
        "during",
        "except",
        "inside",
        "near",
        "outside",
        "since",
        "through",
        "throughout",
        "toward",
        "towards",
        "via",
        "while",
        "although",
        "though",
        "because",
        "if",
        "unless",
        "whether",
        "what",
        "who",
        "whom",
        "whose",
        "which",
        "when",
        "where",
        "why",
        "how",
        "whatever",
        "whoever",
        "yes",
        "yeah",
        "yep",
        "nope",
        "well",
        "sure",
        "sorry",
        "please",
        "thanks",
        "thank",
        "hello",
        "hi",
        "hey",
        "note",
        "source",
        "sources",
        "only",
        "according",
        "based",
        "currently",
        "recently",
        "now",
        "today",
        "yesterday",
        "tomorrow",
        "tonight",
        "later",
        "earlier",
        "previously",
        "formerly",
        "originally",
        "initially",
        "eventually",
        "finally",
        "meanwhile",
        "additionally",
        "furthermore",
        "moreover",
        "however",
        "nevertheless",
        "nonetheless",
        "otherwise",
        "instead",
        "therefore",
        "thus",
        "hence",
        "consequently",
        "overall",
        "generally",
        "typically",
        "usually",
        "normally",
        "mostly",
        "mainly",
        "largely",
        "roughly",
        "approximately",
        "nearly",
        "almost",
        "least",
        "even",
        "still",
        "just",
        "also",
        "too",
        "indeed",
        "actually",
        "basically",
        "essentially",
        "notably",
        "specifically",
        "particularly",
        "especially",
        "unfortunately",
        "fortunately",
        "thankfully",
        "sadly",
        "surprisingly",
        "reportedly",
        "allegedly",
        "apparently",
        "seemingly",
        "presumably",
        "probably",
        "possibly",
        "perhaps",
        "maybe",
        "certainly",
        "definitely",
        "clearly",
        "obviously",
        "are",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "might",
        "must",
        "shall",
        "should",
        "would",
        "let",
        "look",
        "see",
        "check",
        "try",
        "keep",
        "make",
        "take",
        "give",
        "get",
        "go",
        "come",
        "say",
        "tell",
        "ask",
        "find",
        "use",
        "want",
    }
)


@dataclass(frozen=True)
class Verdict:
    """Result of one single-purpose evaluator on one turn row."""

    name: str
    outcome: Outcome
    reason: str = ""


Evaluator = Callable[[Mapping[str, Any], Mapping[str, Any]], Verdict]


def _calls(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = row.get("calls")
    return [c for c in raw if isinstance(c, Mapping)] if isinstance(raw, list) else []


def _present_tools(row: Mapping[str, Any]) -> set[str]:
    """Tool names (plus `smart_home` / intent names) from non-cancelled calls in `row`."""
    present: set[str] = set()
    for call in _calls(row):
        if call.get("status") == "cancelled":
            continue
        if isinstance(call.get("tool"), str) and call["tool"]:
            present.add(call["tool"])
        elif isinstance(call.get("intent"), str) and call["intent"]:
            present.add("smart_home")
            present.add(call["intent"])
    return present


def _tool_list(expect: Mapping[str, Any], key: str, fallback: str) -> list[str]:
    raw = expect.get(key)
    if raw is None:
        raw = expect.get(fallback)
    if not isinstance(raw, list | tuple | set | frozenset):
        return []
    return [str(item) for item in raw if str(item)]


def _is_changeable(row: Mapping[str, Any], expect: Mapping[str, Any]) -> bool:
    if expect.get("changeable") is not None:
        return bool(expect["changeable"])
    return call_policy.changeable_question(str(row.get("question", "")))


def evaluate_lookup_required(row: Mapping[str, Any], expect: Mapping[str, Any]) -> Verdict:
    """Pass when at least one required tool (non-cancelled) ran in the turn."""
    required = _tool_list(expect, "required_tools", "required")
    if not required:
        return Verdict("lookup_required", "n/a", "no required tools")
    present = _present_tools(row)
    matched = [t for t in required if t in present]
    if matched:
        return Verdict("lookup_required", "pass", f"found: {', '.join(matched)}")
    return Verdict(
        "lookup_required",
        "fail",
        f"missing required tool from {required}; got {sorted(present)}",
    )


def evaluate_forbidden_absent(row: Mapping[str, Any], expect: Mapping[str, Any]) -> Verdict:
    """Pass when no forbidden tool (non-cancelled) ran in the turn."""
    forbidden = _tool_list(expect, "forbidden_tools", "forbidden")
    if not forbidden:
        return Verdict("forbidden_absent", "n/a", "no forbidden tools")
    present = _present_tools(row)
    hit = [t for t in forbidden if t in present]
    if hit:
        return Verdict("forbidden_absent", "fail", f"forbidden tool(s) called: {', '.join(hit)}")
    return Verdict("forbidden_absent", "pass", "no forbidden tools called")


def evaluate_searched(row: Mapping[str, Any], expect: Mapping[str, Any]) -> Verdict:
    """Pass when a changeable web-lookup turn has a non-`check` ledger row with searches >= 1."""
    if not _is_changeable(row, expect):
        return Verdict("searched", "n/a", "not a changeable question")

    route = str(row.get("route") or "")
    if route in ("fast", "backchannel", "cancel_chain"):
        return Verdict("searched", "n/a", f"non-lookup route={route}")

    required = set(_tool_list(expect, "required_tools", "required"))
    if required and not (required & agent_tools.LOOKUP_TOOLS):
        return Verdict("searched", "n/a", "required tools are not web lookups")

    forbidden = set(_tool_list(expect, "forbidden_tools", "forbidden"))
    calls = _calls(row)
    called_lookup = any(
        c.get("tool") in agent_tools.LOOKUP_TOOLS and c.get("status") != "cancelled" for c in calls
    )
    if (forbidden >= agent_tools.LOOKUP_TOOLS or "web_answer" in forbidden) and not called_lookup:
        return Verdict("searched", "n/a", "web lookup forbidden")

    non_web_ran = any(
        (c.get("tool") not in agent_tools.LOOKUP_TOOLS if "tool" in c else "intent" in c)
        and c.get("status") != "cancelled"
        for c in calls
    )
    if non_web_ran and not called_lookup and not (required & agent_tools.LOOKUP_TOOLS):
        return Verdict("searched", "n/a", "non-web tool turn")

    ledger = row.get("ledger")
    rows = ledger.get("rows") if isinstance(ledger, Mapping) else None
    ledger_rows = [r for r in rows if isinstance(r, Mapping)] if isinstance(rows, list) else []
    own_rows = [r for r in ledger_rows if "check" not in r]
    searches = sum(int(r.get("searches", 0)) for r in own_rows if int(r.get("searches", 0)) > 0)
    if any(int(r.get("searches", 0)) >= 1 for r in own_rows):
        return Verdict("searched", "pass", f"searches={searches}")
    return Verdict("searched", "fail", "no non-check ledger row with searches >= 1")


def evaluate_duplicate_lookups(row: Mapping[str, Any], expect: Mapping[str, Any]) -> Verdict:
    """Fail when more than one non-joined lookup execution occurred in the turn (F3 metric)."""
    del expect
    lookups = [c for c in _calls(row) if c.get("tool") in agent_tools.LOOKUP_TOOLS]
    if not lookups:
        return Verdict("duplicate_lookups", "n/a", "no lookup calls")
    non_joined = [c for c in lookups if c.get("status") != "joined" and not c.get("cached")]
    if len(non_joined) > 1:
        counts = Counter(
            (str(c.get("by") or "unknown"), str(c.get("status") or "unknown")) for c in non_joined
        )
        breakdown = ", ".join(f"{by}: {n} {status}" for (by, status), n in counts.items())
        return Verdict(
            "duplicate_lookups", "fail", f"{len(non_joined)} non-joined lookups ({breakdown})"
        )
    return Verdict("duplicate_lookups", "pass", f"{len(non_joined)} non-joined lookup")


def evaluate_note_reached(row: Mapping[str, Any], expect: Mapping[str, Any]) -> Verdict:
    """Pass when a changeable turn answered by Gemini carries `note` in `tail` (F2 metric)."""
    if not _is_changeable(row, expect):
        return Verdict("note_reached", "n/a", "not a changeable question")
    route = str(row.get("route") or "")
    went_to_gemini = not row.get("offline") and (
        route in ("gemini", "chain") or (route == "routed" and not row.get("routed"))
    )
    if not went_to_gemini:
        return Verdict("note_reached", "n/a", f"route={route or 'none'} did not use Gemini")
    tail = row.get("tail")
    tail_parts = list(tail) if isinstance(tail, list) else []
    if "note" in tail_parts:
        return Verdict("note_reached", "pass", "note in tail")
    return Verdict("note_reached", "fail", f"note missing from tail={tail_parts!r}")


def _parse_number(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


def _extract_numbers(text: str) -> list[tuple[str, Decimal]]:
    out: list[tuple[str, Decimal]] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(0)
        parsed = _parse_number(raw)
        if parsed is not None:
            out.append((raw, parsed))
    return out


def _fold(text: str) -> str:
    """NFKD-decompose, drop combining marks, and casefold for comparison."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def _normalize_for_words(text: str) -> str:
    collapsed = _ABBREV_DOTS.sub("", text).replace("’", "'")
    return _fold(collapsed)


def _evidence_words(texts: list[str]) -> set[str]:
    words: set[str] = set()
    for text in texts:
        normed = _normalize_for_words(text)
        for token in _WORD_RE.findall(normed):
            words.add(token)
            stripped = _CONTRACTION_TAIL.sub("", token).rstrip("'")
            if stripped:
                words.add(stripped)
    return words


def _doc_items(raw: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    decoder, pos = json.JSONDecoder(), 0
    while pos < len(raw):
        if raw[pos].isspace():
            pos += 1
            continue
        try:
            value, pos = decoder.raw_decode(raw, pos)
        except ValueError:
            break
        batch = value if isinstance(value, list) else [value]
        for item in batch:
            if isinstance(item, dict):
                items.append(item)
    return items


def _doc_evidence(raw: str) -> list[str]:
    texts = doc_answer.passages_of(raw)
    if not texts:
        return [raw]
    meta: list[str] = []
    for item in _doc_items(raw):
        for key in ("source", "file_date", "section"):
            val = item.get(key)
            if val:
                meta.append(str(val))
        page = item.get("page")
        if page is not None and str(page).strip():
            meta.append(str(page))
    return texts + meta


def _evidence_texts(row: Mapping[str, Any], expect: Mapping[str, Any]) -> tuple[list[str], bool]:
    """(all evidence strings including question, whether lookup/passage evidence exists)."""
    lookup_texts: list[str] = []
    for call in _calls(row):
        if call.get("status") == "cancelled":
            continue
        raw_text = call.get("result_text")
        if isinstance(raw_text, str) and raw_text:
            if call.get("tool") == "search_docs":
                lookup_texts.extend(_doc_evidence(raw_text))
            else:
                lookup_texts.append(raw_text)
        sources = call.get("sources")
        if isinstance(sources, list):
            lookup_texts.extend(str(s) for s in sources if s)

    raw_passages = expect.get("passages")
    if isinstance(raw_passages, str) and raw_passages:
        lookup_texts.append(raw_passages)
    elif isinstance(raw_passages, list):
        lookup_texts.extend(str(p) for p in raw_passages if p)

    if not lookup_texts:
        return [], False
    q = str(row.get("question") or "")
    return ([q, *lookup_texts] if q else lookup_texts), True


def _unsupported_name_tokens(reply: str, allowed_words: set[str]) -> list[str]:
    offending: list[str] = []
    for line in reply.splitlines():
        at_start = True
        for chunk in line.split():
            core_chunk = _ABBREV_DOTS.sub("", chunk).replace("’", "'")
            ended = core_chunk.rstrip("\"'”’)]").endswith((".", "!", "?", ":"))
            cleaned = core_chunk.strip("\"'“”‘’()[]{},;:.!?")
            for sub in cleaned.split("-"):
                token = sub.strip("'")
                if not token or not any(ch.isupper() for ch in token):
                    continue
                folded = _fold(token)
                skip_set = _SENTENCE_STARTERS if at_start else _ALWAYS_COMMON
                words = _WORD_RE.findall(folded) or [folded]
                for word in words:
                    base = _CONTRACTION_TAIL.sub("", word).rstrip("'")
                    if word in skip_set or base in skip_set:
                        continue
                    if word not in allowed_words and base not in allowed_words:
                        offending.append(token)
                        break
            at_start = ended
    return offending


def evaluate_support(row: Mapping[str, Any], expect: Mapping[str, Any]) -> Verdict:
    """S10/S11: every number, date and capitalised name in reply appears in evidence."""
    text = str(row.get("text") or "").strip()
    if not text or row.get("cancelled"):
        return Verdict("support", "n/a", "no completed reply")
    evidence, has_lookup = _evidence_texts(row, expect)
    if not has_lookup:
        return Verdict("support", "n/a", "no lookup result or passages")

    allowed_numbers = {num for item in evidence for _, num in _extract_numbers(item)}
    allowed_words = _evidence_words(evidence)

    offending: list[str] = []
    seen: set[str] = set()
    for raw_num, parsed_num in _extract_numbers(text):
        if parsed_num not in allowed_numbers and raw_num not in seen:
            seen.add(raw_num)
            offending.append(raw_num)
    for token in _unsupported_name_tokens(text, allowed_words):
        if token not in seen:
            seen.add(token)
            offending.append(token)

    if offending:
        return Verdict("support", "fail", f"unsupported: {', '.join(offending)}")
    return Verdict("support", "pass", "all numbers, dates and names supported")


def _source_named_in_reply(source: str, reply: str, question: str) -> bool:
    escaped = re.escape(source)
    if source.lower() in question.lower():
        pattern = (
            rf"(?:according to|per|from|sources?:?)\s+{escaped}\b"
            rf"|\b{escaped}\s+(?:says|said|reports|reported)\b"
        )
    else:
        pattern = rf"\b{escaped}\b"
    return re.search(pattern, reply, re.IGNORECASE) is not None


def evaluate_source_coverage(row: Mapping[str, Any], expect: Mapping[str, Any]) -> Verdict:
    """Pass when a lookup with sources is attributed in the reply or via `cited_by_server`."""
    del expect
    if row.get("cancelled"):
        return Verdict("source_coverage", "n/a", "turn cancelled")
    sources: list[str] = []
    for call in _calls(row):
        if call.get("status") == "cancelled":
            continue
        call_sources = call.get("sources")
        if isinstance(call_sources, list):
            for s in call_sources:
                if isinstance(s, str) and s and s not in sources:
                    sources.append(s)
    if not sources:
        return Verdict("source_coverage", "n/a", "no lookup with sources")
    if row.get("cited_by_server") is True:
        return Verdict("source_coverage", "pass", "cited_by_server")
    reply = str(row.get("text") or "")
    question = str(row.get("question") or "")
    matched = [s for s in sources if _source_named_in_reply(s, reply, question)]
    if matched:
        return Verdict("source_coverage", "pass", f"named: {', '.join(matched)}")
    return Verdict("source_coverage", "fail", f"reply names none of {sources}")


def _abstained(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(HEDGE):
        return True
    if _CLOSEST_RESULT.search(stripped):
        return False
    return _ABSTAIN_PHRASE.search(stripped) is not None


def evaluate_abstention(row: Mapping[str, Any], expect: Mapping[str, Any]) -> Verdict:
    """Pass when expected-abstain turns abstain and expected-answer turns do not."""
    want = expect.get("abstain")
    if want is None or row.get("cancelled"):
        return Verdict("abstention", "n/a", "abstain not specified")
    text = str(row.get("text") or "").strip()
    if not text and not want:
        return Verdict("abstention", "n/a", "empty reply")
    got = _abstained(text)
    if bool(want):
        if got:
            return Verdict("abstention", "pass", "abstained as expected")
        return Verdict("abstention", "fail", "expected abstention, got answer")
    if got:
        return Verdict("abstention", "fail", "unexpected abstention")
    return Verdict("abstention", "pass", "answered without abstaining")


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _doc_citation_matches(
    source: str, file_date: str, reply: str, question: str, meta: Mapping[str, Any] | None = None
) -> tuple[bool, bool]:
    file_ok = bool(source) and _source_named_in_reply(_fold(source), _fold(reply), _fold(question))
    if meta is not None:
        meta_date = str(meta.get("date") or "")
        date_ok = bool(_DATE_RE.match(file_date) and meta_date == file_date)
        meta_file = str(meta.get("file") or "")
        file_ok = file_ok or (
            bool(source)
            and _source_named_in_reply(_fold(source), _fold(meta_file), _fold(question))
        )
    else:
        date_ok = bool(
            _DATE_RE.match(file_date)
            and re.search(rf"(?<!\d){re.escape(file_date)}(?!\d)", reply) is not None
        )
    return file_ok, date_ok


def evaluate_doc_citation(row: Mapping[str, Any], expect: Mapping[str, Any]) -> Verdict:
    """Pass when a non-abstaining docs-route reply cites a passage's file and YYYY-MM-DD date."""
    del expect
    if row.get("cancelled"):
        return Verdict("doc_citation", "n/a", "turn cancelled")
    route = str(row.get("route") or "")
    if route != "docs":
        return Verdict("doc_citation", "n/a", f"route={route or 'none'}")
    reply = str(row.get("text") or "").strip()
    if _abstained(reply):
        return Verdict("doc_citation", "n/a", "abstention reply")

    meta = row.get("citation_metadata")
    passages: list[tuple[str, str]] = []
    for call in _calls(row):
        if call.get("status") == "cancelled" or call.get("tool") != "search_docs":
            continue
        raw_text = call.get("result_text")
        if not isinstance(raw_text, str) or not raw_text:
            continue
        for item in _doc_items(raw_text):
            source = str(item.get("source") or "").strip()
            file_date = str(item.get("file_date") or "").strip()
            if source or file_date:
                passages.append((source, file_date))

    if not passages:
        return Verdict("doc_citation", "fail", "no search_docs passages to cite")

    question = str(row.get("question") or "")
    any_file = False
    any_date = False
    for source, file_date in passages:
        file_ok, date_ok = _doc_citation_matches(source, file_date, reply, question, meta)
        any_file = any_file or file_ok
        any_date = any_date or date_ok
        if file_ok and date_ok:
            return Verdict("doc_citation", "pass", f"cited {source} ({file_date})")

    if any_file:
        return Verdict("doc_citation", "fail", "reply missing matching modified date")
    if any_date:
        return Verdict("doc_citation", "fail", "reply missing matching source filename")
    return Verdict("doc_citation", "fail", "reply names neither source file nor modified date")


EVALUATORS: dict[str, Evaluator] = {
    "lookup_required": evaluate_lookup_required,
    "forbidden_absent": evaluate_forbidden_absent,
    "searched": evaluate_searched,
    "duplicate_lookups": evaluate_duplicate_lookups,
    "note_reached": evaluate_note_reached,
    "support": evaluate_support,
    "source_coverage": evaluate_source_coverage,
    "abstention": evaluate_abstention,
    "doc_citation": evaluate_doc_citation,
}


def evaluate_all(row: Mapping[str, Any], expect: Mapping[str, Any]) -> list[Verdict]:
    """Run every registered evaluator on one turn row; returns individual verdicts only."""
    return [fn(row, expect) for fn in EVALUATORS.values()]
