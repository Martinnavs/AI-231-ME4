"""File cards: per-file metadata summary built at ingest time."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from voice_agent.rag.docindex import Chunk

MAX_SECTIONS = 20
MAX_ENTITIES = 25
MAX_CARD_BYTES = 2048

_TABULAR_KINDS = frozenset({"csv", "tsv", "xlsx", "xls"})

_DATE_RE1 = re.compile(r"\b(19|20)\d{2}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b")
_DATE_RE2 = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
    re.I,
)
_ENTITY_RE = re.compile(r"(?<!^)[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*|[A-Z]{3,}")
_MONTH_MAP = {
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


@dataclass
class FileCard:
    rel_key: str
    title: str
    kind: str
    sections: list[str] = field(default_factory=list)
    table_columns: list[str] = field(default_factory=list)
    top_entities: list[str] = field(default_factory=list)
    date_min: str | None = None
    date_max: str | None = None
    row_count: int = 0
    chunk_count: int = 0


def build_card(
    rel_key: str,
    chunks: list[Chunk],
    *,
    kind: str,
) -> FileCard:
    """Build a FileCard from parsed chunks (static text analysis only)."""
    if not chunks:
        return FileCard(rel_key=rel_key, title="", kind=kind)

    # Title
    title = _extract_title(chunks, kind, rel_key)

    # Sections
    sections = _extract_sections(chunks)

    # Table columns
    table_columns = _extract_table_columns(chunks, kind)

    # Entities
    top_entities = _extract_entities(chunks)

    # Date range
    date_min, date_max = _extract_dates(chunks)

    # Row count
    row_count = _count_rows(chunks, kind)

    # Chunk count
    chunk_count = len(chunks)

    return FileCard(
        rel_key=rel_key,
        title=title,
        kind=kind,
        sections=sections,
        table_columns=table_columns,
        top_entities=top_entities,
        date_min=date_min,
        date_max=date_max,
        row_count=row_count,
        chunk_count=chunk_count,
    )


def card_to_json(card: FileCard) -> str:
    """Compact JSON, capped at MAX_CARD_BYTES (truncate sections if needed)."""
    data = {
        "rel_key": card.rel_key,
        "title": card.title,
        "kind": card.kind,
        "sections": card.sections,
        "table_columns": card.table_columns,
        "top_entities": card.top_entities,
        "date_min": card.date_min,
        "date_max": card.date_max,
        "row_count": card.row_count,
        "chunk_count": card.chunk_count,
    }
    text = json.dumps(data, separators=(",", ":"))
    if len(text.encode("utf-8")) <= MAX_CARD_BYTES:
        return text
    # Truncate sections to fit
    half = len(card.sections)
    while half > 0 and len(text.encode("utf-8")) > MAX_CARD_BYTES:
        half = max(0, half // 2)
        data["sections"] = card.sections[:half]
        text = json.dumps(data, separators=(",", ":"))
    return text


# ---- private helpers ----


def _extract_title(chunks: list[Chunk], kind: str, rel_key: str) -> str:
    # Tabular: use filename
    if kind in _TABULAR_KINDS:
        base = rel_key.rsplit(".", 1)[0] if "." in rel_key else rel_key
        return base.rsplit("/", 1)[-1]

    # Presentation: first slide title text
    if kind in ("pptx", "ppt"):
        for c in chunks:
            if c.text.strip():
                return c.text.strip().split("\n")[0]
        return rel_key

    # Document: first heading
    for c in chunks:
        section = (c.section or "").strip()
        if section:
            return section
        header = (c.header or "").strip()
        if header:
            return header
        text = c.text.strip()
        if text.startswith("#"):
            return text.lstrip("#").strip()

    return rel_key


def _extract_sections(chunks: list[Chunk]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for c in chunks:
        s = (c.section or "").strip()
        if s and s not in seen:
            seen.add(s)
            result.append(s)
            if len(result) >= MAX_SECTIONS:
                break
    return result


def _extract_table_columns(chunks: list[Chunk], kind: str) -> list[str]:
    if kind not in _TABULAR_KINDS or not chunks:
        return []
    text = chunks[0].text.strip()
    if not text:
        return []
    # Try to parse header row
    sep = "," if kind in ("csv",) else "\t" if kind == "tsv" else ","
    cols = [c.strip() for c in text.split(sep)]
    return [c for c in cols if c][:20]


def _extract_entities(chunks: list[Chunk]) -> list[str]:
    from collections import Counter

    counter: Counter[str] = Counter()
    sample = chunks[:5]
    for c in sample:
        text = c.text
        if not text:
            continue
        # Process line by line to handle ^ anchor
        for line in text.split("\n"):
            entities = _ENTITY_RE.findall(line)
            for e in entities:
                e = e.strip()
                if len(e) >= 3:
                    counter[e] += 1
    return [e for e, _ in counter.most_common(MAX_ENTITIES)]


def _extract_dates(
    chunks: list[Chunk],
) -> tuple[str | None, str | None]:
    from datetime import date

    dates: list[date] = []
    for c in chunks:
        text = c.text
        if not text:
            continue
        for m in _DATE_RE1.finditer(text):
            try:
                y, mo, d = m.group(0).replace("/", "-").split("-")
                dates.append(date(int(y), int(mo), int(d)))
            except (ValueError, OverflowError):
                pass
        for m in _DATE_RE2.finditer(text):
            try:
                parts = m.group(0)
                month_str = parts[0:3].lower()
                month = _MONTH_MAP.get(month_str)
                if month is None:
                    continue
                nums = re.findall(r"\d+", parts)
                if len(nums) >= 2:
                    dates.append(date(int(nums[1]), month, int(nums[0])))
            except (ValueError, OverflowError, IndexError):
                pass

    if not dates:
        return None, None
    mn = min(dates)
    mx = max(dates)
    return mn.isoformat(), mx.isoformat()


def _count_rows(chunks: list[Chunk], kind: str) -> int:
    if kind in _TABULAR_KINDS:
        # Data chunks = all except the header chunk (index 0)
        return max(0, len(chunks) - 1)
    # Count distinct pages
    pages: set[int | None] = set()
    for c in chunks:
        if c.page is not None:
            pages.add(c.page)
    return len(pages)
