"""Document index with structure-aware chunking, BM25 retrieval, and stat-based refresh."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import itertools
import json
import os
import re
import shutil
import sys
import threading
import warnings
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

from doc_chat.corpus import tokenize
from voice_agent.rag import canonical, canonical_cache, convert, zipguard

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_CSV_ROWS = 50_000
MAX_CSV_PREAMBLE_ROWS = 5  # title/preamble rows looked past when finding a csv header
PREFIX_KINDS = frozenset({"txt", "md", "csv", "tsv", "log"})
PREFIX_TRUNCATED_NOTE = f"[file truncated: only the first {MAX_FILE_BYTES // 2**20} MB are indexed]"
MAX_PDF_PAGES = 1000
LEGACY_PARSE_VERSION = 3  # 3: a csv title line is not the header (254)
PDF_ENGINE_ENV = "VOICE_AGENT_RAG_PDF_ENGINE"
MAX_DOCX_MEMBER_BYTES = MAX_FILE_BYTES
MAX_DOCX_TOTAL_BYTES = zipguard.MAX_ZIP_TOTAL_BYTES
MAX_DOCX_COMPRESSION_RATIO = zipguard.MAX_ZIP_COMPRESSION_RATIO
MAX_DOCX_MEMBERS = zipguard.MAX_ZIP_MEMBERS
MAX_DOCX_XML_DEPTH = zipguard.MAX_DOCX_XML_DEPTH
MAX_FOLDER_DEPTH = 20
MAX_HEADING_LINE_CHARS = 300
MAX_HEADING_TITLE_CHARS = 120
MAX_SECTION_PATH_CHARS = 300
CHUNK_SIZE = 900
CHUNK_OVERLAP = int(CHUNK_SIZE * 0.15)
MAX_REPLACEMENT_RATIO = 0.30
MAX_CHUNKS_PER_FILE = 2  # Soft per-file chunk cap during initial selection.
HARD_CHUNKS_PER_FILE = 4  # Hard per-file chunk cap when chunks clear the score gate.
OVERFLOW_SCORE_RATIO = 0.75  # Min fraction of best score needed to pass the soft cap.
DEDUP_JACCARD = 0.75  # Cross-file token Jaccard threshold to skip near-duplicate chunks.
HEADER_WEIGHT = 1.5
LEGACY_FORMATS = frozenset({"md", "txt", "csv", "pdf", "docx"})
_VALID_CONVERTERS_BY_FORMAT: dict[str, frozenset[str]] = {
    "md": frozenset({"legacy", "identity"}),
    "txt": frozenset({"legacy", "identity"}),
    "csv": frozenset({"legacy", "identity"}),
    "pdf": frozenset({"legacy", "pypdf", "markitdown"}),
    "docx": frozenset({"legacy", "markitdown"}),
    "pptx": frozenset({"markitdown", "direct"}),
    "xlsx": frozenset({"markitdown", "direct"}),
    "html": frozenset({"markitdown", "direct"}),
    "epub": frozenset({"markitdown", "direct"}),
    "rtf": frozenset({"direct"}),
    "xls": frozenset({"direct"}),
    "doc": frozenset({"direct"}),
    "ppt": frozenset({"direct"}),
    "tsv": frozenset({"legacy", "identity"}),
    "log": frozenset({"legacy", "identity"}),
    "ini": frozenset({"legacy", "identity"}),
    "toml": frozenset({"legacy", "identity"}),
    "yaml": frozenset({"legacy", "identity"}),
    "cfg": frozenset({"legacy", "identity"}),
    "conf": frozenset({"legacy", "identity"}),
}
CANONICAL_CONVERTERS: dict[str, str] = {
    "md": "legacy",
    "txt": "legacy",
    "csv": "legacy",
    "pdf": "legacy",
    "docx": "legacy",
    "pptx": "direct",
    "xlsx": "direct",
    "html": "direct",
    "epub": "direct",
    "rtf": "direct",
    "xls": "direct",
    "doc": "direct",
    "ppt": "direct",
    "tsv": "legacy",
    "log": "legacy",
    "ini": "legacy",
    "toml": "legacy",
    "yaml": "legacy",
    "cfg": "legacy",
    "conf": "legacy",
}
SUPPORTED_EXTENSIONS = frozenset(
    {
        ".md",
        ".txt",
        ".csv",
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".html",
        ".htm",
        ".epub",
        ".rtf",
        ".xls",
        ".tsv",
        ".log",
        ".ini",
        ".toml",
        ".yaml",
        ".cfg",
        ".conf",
        ".doc",
        ".ppt",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }
)
_SENTINEL = object()


def plain_fallback_enabled() -> bool:
    return os.environ.get("VOICE_AGENT_RAG_PLAIN_FALLBACK") == "1"


def looks_like_text(path: Path, probe: int = 8192) -> bool:
    try:
        head = path.read_bytes()[:probe]
    except OSError:
        return False
    if not head or b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
        return True
    except UnicodeDecodeError as exc:
        return exc.start >= len(head) - 3


class _DefaultCacheDirType:
    """Sentinel type distinguishing omitted ``canonical_cache_dir`` from ``None``."""


_DEFAULT_CACHE_DIR = _DefaultCacheDirType()

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_MONTH_MAP: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_MONTH_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?\s+"
    r"(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
    re.IGNORECASE,
)
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*)$")
_DOCX_HEADING_RE = re.compile(r"^Heading\s*([1-9])$", re.IGNORECASE)
_DOCX_MAX_HEADING_LEVEL = 6
_DOCX_MAX_TABLE_CELLS = 20_000
_DOCX_MAX_TABLE_ROWS = 500
_DOCX_MAX_GRID_SPAN = 64
_DOCX_MAX_HEADER_ROWS = 4
_DOCX_MAX_STYLE_CHAIN = 20
MAX_DOCX_HEADER_CHARS = 120
MAX_DOCX_CELL_CHARS = 4_000
DOCX_OUTPUT_BUDGET_CHARS = 10_000_000
_DOCX_MAX_TOTAL_TABLE_ROWS = 20_000
_DOCX_OUTPUT_TOO_LARGE = "docx text exceeds the size limit"
_DOCX_LINE_START_ESCAPE_RE = re.compile(r"^(?:#+[ \t]|`{3,}|~{3,}|<!-- rag:|<!-- Slide number:)")
_DOCX_SKIP_TAGS = frozenset({"del", "instrText", "fldData"})
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_LINE_SPLIT_RE = re.compile(r"\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_SPLIT_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9'\u2019]+")
_LEADING_NUM_PREFIX_RE = re.compile(r"^\d{1,3}[-_.\s]+")


@dataclass
class FileRecord:
    """Metadata and ingest status for one file in the indexed folder."""

    path: str
    name: str
    mtime: float
    ingested_at: str
    sha256: str
    size: int
    kind: str
    dates_in_text: list[str]
    status: str
    error: str | None = None


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage derived from an indexed file."""

    id: str
    file_sha256: str
    chunk_sha256: str
    position: int
    page: int | None
    section: str
    file_date: str
    source: str
    header: str
    text: str
    fmt: str = ""
    slide: int | None = None
    sheet: str | None = None
    canonical_sha256: str = ""
    converter: str = ""
    ocr_confidence: float | None = None


@dataclass
class RefreshReport:
    """Summary of changes and failures observed during a refresh scan."""

    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failures: list[tuple[str, str, str]] = field(default_factory=list)
    unchanged: int = 0


class Retriever(Protocol):
    """Pluggable ranking backend over the index's chunk list."""

    def build(self, chunks: Sequence[Chunk]) -> None: ...

    def rank(self, query: str, k: int) -> list[tuple[int, float]]: ...


def _tokenize_header(header: str) -> list[str]:
    """Split file stem and section titles on non-alphanumerics, dropping numeric prefixes."""
    file_part, _, section_part = header.partition(" > ")
    stem = Path(file_part.strip()).stem if file_part.strip() else ""
    stem = _LEADING_NUM_PREFIX_RE.sub("", stem)
    parts = [part for part in _NON_ALNUM_RE.split(stem) if part]
    if section_part:
        parts.extend(part for part in _NON_ALNUM_RE.split(section_part) if part)
    return tokenize(" ".join(parts))


class Bm25Retriever:
    """BM25Okapi retriever with a boosted header field over ``chunk.header`` and ``chunk.text``."""

    def __init__(self) -> None:
        self._bm25: Any = None
        self._header_bm25: Any = None
        self._doc_token_sets: list[frozenset[str]] = []

    def build(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            self._bm25 = None
            self._header_bm25 = None
            self._doc_token_sets = []
            return

        from rank_bm25 import BM25Okapi

        header_tokenized = [_tokenize_header(chunk.header) for chunk in chunks]
        tokenized = [
            header_tokens + tokenize(chunk.text)
            for header_tokens, chunk in zip(header_tokenized, chunks, strict=True)
        ]
        self._doc_token_sets = [frozenset(tokens) for tokens in tokenized]
        if not any(tokenized):
            self._bm25 = None
            self._header_bm25 = None
            return
        self._bm25 = BM25Okapi(tokenized)
        self._header_bm25 = BM25Okapi(header_tokenized) if any(header_tokenized) else None

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        if k <= 0 or self._bm25 is None:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        scores = self._bm25.get_scores(q_tokens)
        header_scores = (
            self._header_bm25.get_scores(q_tokens) if self._header_bm25 is not None else None
        )
        candidates: list[tuple[int, float]] = []
        for idx, doc_tokens in enumerate(self._doc_token_sets):
            if not (q_set & doc_tokens):
                continue
            score = float(scores[idx])
            if header_scores is not None:
                score += HEADER_WEIGHT * float(header_scores[idx])
            candidates.append((idx, score))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates[:k]


@dataclass
class _IndexedEntry:
    record: FileRecord
    size: int
    mtime_ns: int
    chunks: list[Chunk]
    ctime_ns: int = 0
    canonical_key: str = ""


@dataclass(frozen=True)
class LoadedFile:
    """Result of loading and chunking a single file."""

    status: str
    error: str | None
    kind: str
    size: int
    mtime: float
    mtime_ns: int
    ctime_ns: int
    sha256: str
    chunks: list[Chunk]
    dates: list[str]
    canonical_key: str
    unchanged: bool
    note: str | None


def _make_lf(
    *,
    status: str,
    error: str | None,
    kind: str,
    size: int,
    mtime: float,
    mtime_ns: int,
    ctime_ns: int,
    sha256: str,
    chunks: list[Chunk],
    dates: list[str],
    canonical_key: str,
    unchanged: bool,
    note: str | None,
) -> LoadedFile:
    return LoadedFile(
        status=status,
        error=error,
        kind=kind,
        size=size,
        mtime=mtime,
        mtime_ns=mtime_ns,
        ctime_ns=ctime_ns,
        sha256=sha256,
        chunks=chunks,
        dates=dates,
        canonical_key=canonical_key,
        unchanged=unchanged,
        note=note,
    )


def load_file(
    file_path: Path,
    rel_key: str,
    *,
    converters: Mapping[str, str],
    canonical_cache: canonical_cache.CanonicalCache | Any,
    prev: _IndexedEntry | None,
    now_iso: str,
) -> LoadedFile:
    """Load and chunk a single file, returning structured result data."""
    kind = _normalize_format(file_path.suffix.lower()) or "unknown"
    size = mtime_ns = ctime_ns = 0
    mtime = 0.0
    sha256 = ""

    def _fail(status: str, error: str) -> LoadedFile:
        return _make_lf(
            status=status,
            error=error,
            kind=kind,
            size=size,
            mtime=mtime,
            mtime_ns=mtime_ns,
            ctime_ns=ctime_ns,
            sha256=sha256,
            chunks=[],
            dates=[],
            canonical_key="",
            unchanged=False,
            note=None,
        )

    try:
        stat_res = file_path.stat()
        size = stat_res.st_size
        mtime_ns = stat_res.st_mtime_ns
        ctime_ns = stat_res.st_ctime_ns
        mtime, file_date = _safe_file_date(stat_res.st_mtime)

        ext = file_path.suffix.lower()
        # Handle image files: status ok, 0 chunks, image waiting for OCR
        if ext in frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}):
            from voice_agent.rag.images import image_file_ref  # noqa: PLC0415

            ref = image_file_ref(file_path, rel_key)
            if ref is not None:
                return _make_lf(
                    status="ok",
                    error=None,
                    kind="image",
                    size=size,
                    mtime=mtime,
                    mtime_ns=mtime_ns,
                    ctime_ns=ctime_ns,
                    sha256=sha256,
                    chunks=[],
                    dates=[],
                    canonical_key="",
                    unchanged=False,
                    note="image waiting for OCR",
                )
            return _fail("empty", "image too small to index")
        mode = converters.get(kind)
        if mode is None and plain_fallback_enabled() and looks_like_text(file_path):
            kind = "txt"
            mode = converters.get("txt")
        if mode is None:
            return _fail("unsupported", f"unsupported file extension: {ext or '(none)'}")

        raw_bytes: bytes | None = None
        if size > MAX_FILE_BYTES:
            if kind in PREFIX_KINDS:
                with open(file_path, "rb") as _fh:
                    prefix_raw = _fh.read(MAX_FILE_BYTES)
                    last_nl = prefix_raw.rfind(b"\n")
                    if last_nl > 0:
                        prefix_raw = prefix_raw[: last_nl + 1]
                raw_bytes = prefix_raw
            else:
                return _fail(
                    "too_large", f"file size {size} bytes exceeds limit of {MAX_FILE_BYTES} bytes"
                )

        if raw_bytes is None:
            raw_bytes = file_path.read_bytes()
        sha256 = hash_bytes(raw_bytes)

        if prev is not None and prev.record.status == "ok" and prev.record.sha256 == sha256:
            return _make_lf(
                status="ok",
                error=prev.record.error,
                kind=kind,
                size=size,
                mtime=mtime,
                mtime_ns=mtime_ns,
                ctime_ns=ctime_ns,
                sha256=sha256,
                chunks=prev.chunks,
                dates=prev.record.dates_in_text,
                canonical_key=prev.canonical_key,
                unchanged=True,
                note=prev.record.error,
            )

        file_chunks, dates, status, error, canonical_key = _load_chunks_for_mode(
            raw_bytes,
            kind=kind,
            mode=mode,
            source=rel_key,
            file_sha256=sha256,
            file_date=file_date,
            cache=canonical_cache,
            file_path=file_path,
        )
        if status == "scanned":
            return _make_lf(
                status="ok",
                error=None,
                kind=kind,
                size=size,
                mtime=mtime,
                mtime_ns=mtime_ns,
                ctime_ns=ctime_ns,
                sha256=sha256,
                chunks=[],
                dates=[],
                canonical_key="",
                unchanged=False,
                note=error or "scanned: pages waiting for OCR",
            )
        if status != "ok":
            return _fail(status=status, error=error or "")
        if not file_chunks:
            return _fail("empty", "no text to index")

        truncated = kind in ("csv", "xlsx", "xls") and any(
            canonical.TABLE_TRUNCATED_NOTE in chunk.text for chunk in file_chunks
        )
        prefix_cut = size > MAX_FILE_BYTES and kind in PREFIX_KINDS
        note = (
            canonical.TABLE_TRUNCATED_NOTE
            if truncated
            else PREFIX_TRUNCATED_NOTE
            if prefix_cut
            else None
        )
        return _make_lf(
            status="ok",
            error=None,
            kind=kind,
            size=size,
            mtime=mtime,
            mtime_ns=mtime_ns,
            ctime_ns=ctime_ns,
            sha256=sha256,
            chunks=file_chunks,
            dates=dates,
            canonical_key=canonical_key,
            unchanged=False,
            note=note,
        )
    except Exception as exc:
        return _make_lf(
            status="unreadable",
            error=f"could not be read ({type(exc).__name__})",
            kind=kind,
            size=size,
            mtime=mtime,
            mtime_ns=mtime_ns,
            ctime_ns=ctime_ns,
            sha256=sha256,
            chunks=[],
            dates=[],
            canonical_key="",
            unchanged=False,
            note=None,
        )


def extract_dates(text: str) -> list[str]:
    """Extract ISO-8601 and 'Month D, YYYY' dates, normalized to YYYY-MM-DD."""
    matches: list[tuple[int, str]] = []
    for match in _ISO_DATE_RE.finditer(text):
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if not (1000 <= year <= 2999):
            continue
        try:
            iso = date(year, month, day).isoformat()
        except ValueError:
            continue
        matches.append((match.start(), iso))

    for match in _MONTH_DATE_RE.finditer(text):
        month_num = _MONTH_MAP.get(match.group(1).lower())
        if month_num is None:
            continue
        day, year = int(match.group(2)), int(match.group(3))
        if not (1000 <= year <= 2999):
            continue
        try:
            iso = date(year, month_num, day).isoformat()
        except ValueError:
            continue
        matches.append((match.start(), iso))

    matches.sort(key=lambda item: item[0])
    return list(dict.fromkeys(iso for _, iso in matches))


def hash_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of raw file bytes."""
    return hashlib.sha256(data).hexdigest()


def parse_csv_text(text: str, *, delimiter: str = ",", max_rows: int = MAX_CSV_ROWS) -> str:
    """Format CSV rows as ``col: value`` lines using the first row as headers."""
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    head = list(itertools.islice(reader, MAX_CSV_PREAMBLE_ROWS + 1))
    if not head:
        return ""

    def _filled(row: Sequence[str]) -> int:
        return sum(1 for cell in row if cell.strip())

    # Title or preamble lines above the header (at most one cell filled, a table below): each is
    # kept as its own line and the header is the first row with several cells, so the column
    # names reach every data line instead of col_2, col_3, ... A file whose first row already
    # has several cells (or that has no such row near the top) is read exactly as before.
    header_at = 0
    if _filled(head[0]) < 2:
        header_at = next((i for i, row in enumerate(head) if _filled(row) >= 2), 0)
    headers = head[header_at]
    lines: list[str] = [
        next(cell.strip() for cell in row if cell.strip())
        for row in head[:header_at]
        if _filled(row) == 1
    ]
    reader = itertools.chain(head[header_at + 1 :], reader)
    clean_headers = [col.strip() or f"col_{idx + 1}" for idx, col in enumerate(headers)]

    data_rows = 0
    for row in reader:
        if data_rows >= max_rows:
            lines.append(canonical.TABLE_TRUNCATED_NOTE)
            break
        data_rows += 1
        pairs: list[tuple[str, str]] = []
        for idx, val in enumerate(row):
            stripped = val.strip()
            if not stripped:
                continue
            col = clean_headers[idx] if idx < len(clean_headers) else f"col_{idx + 1}"
            pairs.append((col, stripped))
        lines.extend(canonical.table_lines(pairs))
    return "\n".join(lines)


def _xml_local(tag: str) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def _xml_attr(elem: ET.Element, attr_name: str) -> str | None:
    for key, value in elem.attrib.items():
        if _xml_local(key) == attr_name:
            return value
    return None


def _has_forbidden_xml_decl(xml_bytes: bytes) -> bool:
    return zipguard.has_forbidden_xml_decl(xml_bytes)


def _read_zip_member_bounded(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    max_bytes: int,
    *,
    keep: bool,
) -> bytes:
    return zipguard.read_member_bounded(zf, info, max_bytes, keep=keep, label="docx")


@dataclass(frozen=True)
class _DocxCell:
    """One parsed ``<w:tc>`` cell with horizontal span and vertical merge metadata."""

    text: str
    grid_span: int
    vmerge: str | None


@dataclass(frozen=True)
class _DocxRawRow:
    """One parsed ``<w:tr>`` row before grid expansion."""

    is_tbl_header: bool
    grid_before: int
    cells: list[_DocxCell]


def _parse_docx_nonneg_int(raw_val: str | None) -> int | None:
    """Parse an ASCII non-negative integer attribute, returning None if invalid."""
    if raw_val is None:
        return None
    stripped = raw_val.strip()
    if not stripped or not stripped.isascii() or not stripped.isdigit():
        return None
    return int(stripped)


def _parse_docx_outline_lvl(raw_val: str | None) -> int | None:
    """Map OOXML ``w:outlineLvl`` (0..8) to Markdown heading level (1..6), else None."""
    lvl = _parse_docx_nonneg_int(raw_val)
    if lvl is not None and 0 <= lvl <= 8:
        return min(lvl + 1, _DOCX_MAX_HEADING_LEVEL)
    return None


def _heading_level_from_style_name(name: str) -> int | None:
    """Map OOXML style names ``Heading 1..6``, ``Title``, and ``Subtitle`` to heading levels."""
    stripped = name.strip()
    if not stripped:
        return None
    match = _DOCX_HEADING_RE.match(stripped)
    if match:
        return min(int(match.group(1)), _DOCX_MAX_HEADING_LEVEL)
    lower = stripped.lower()
    if lower == "title":
        return 1
    if lower == "subtitle":
        return 2
    return None


def _select_alternate_content_branch(node: ET.Element) -> ET.Element | None:
    """Return the first ``mc:Choice`` child of ``mc:AlternateContent``, else ``mc:Fallback``."""
    fallback: ET.Element | None = None
    for child in node:
        local = _xml_local(child.tag)
        if local == "Choice":
            return child
        if local == "Fallback" and fallback is None:
            fallback = child
    return fallback


def _extract_docx_p_props(p_node: ET.Element) -> tuple[str, bool, int | None]:
    """Extract ``(style_val, has_direct_outline, direct_outline_lvl)`` from ``w:pPr``."""
    style_val = ""
    has_direct_outline = False
    direct_outline: int | None = None
    for child in p_node:
        if _xml_local(child.tag) != "pPr":
            continue
        for prop in child:
            local = _xml_local(prop.tag)
            if local == "pStyle" and not style_val:
                style_val = (_xml_attr(prop, "val") or "").strip()
            elif local == "outlineLvl" and not has_direct_outline:
                has_direct_outline = True
                direct_outline = _parse_docx_outline_lvl(_xml_attr(prop, "val"))
    return style_val, has_direct_outline, direct_outline


def _parse_docx_styles(styles_root: ET.Element) -> dict[str, int | None]:
    """Build a ``styleId -> heading_level`` map from ``word/styles.xml``."""
    raw_styles: dict[str, tuple[bool, int | None, str]] = {}
    for child in styles_root:
        if _xml_local(child.tag) != "style":
            continue
        stype = _xml_attr(child, "type")
        if stype is not None and stype.strip().lower() != "paragraph":
            continue
        style_id = (_xml_attr(child, "styleId") or "").strip()
        if not style_id:
            continue

        name_val = ""
        based_on = ""
        has_outline = False
        outline_lvl: int | None = None
        for sub in child:
            local = _xml_local(sub.tag)
            if local == "name" and not name_val:
                name_val = (_xml_attr(sub, "val") or "").strip()
            elif local == "basedOn" and not based_on:
                based_on = (_xml_attr(sub, "val") or "").strip()
            elif local == "pPr" and not has_outline:
                for prop in sub:
                    if _xml_local(prop.tag) == "outlineLvl":
                        has_outline = True
                        outline_lvl = _parse_docx_outline_lvl(_xml_attr(prop, "val"))
                        break

        if has_outline:
            raw_styles[style_id] = (True, outline_lvl, based_on)
        else:
            name_lvl = _heading_level_from_style_name(name_val) or _heading_level_from_style_name(
                style_id
            )
            raw_styles[style_id] = (name_lvl is not None, name_lvl, based_on)

    style_map: dict[str, int | None] = {}
    for sid in raw_styles:
        if sid in style_map:
            continue
        seen: set[str] = set()
        cur = sid
        resolved: int | None = None
        for _ in range(_DOCX_MAX_STYLE_CHAIN):
            if cur in seen or cur not in raw_styles:
                break
            seen.add(cur)
            has_explicit, explicit_lvl, parent_id = raw_styles[cur]
            if has_explicit:
                resolved = explicit_lvl
                break
            if not parent_id:
                break
            cur = parent_id
        style_map[sid] = resolved
    return style_map


def _resolve_docx_heading_level(
    style_val: str,
    has_direct_outline: bool,
    direct_outline: int | None,
    style_map: Mapping[str, int | None] | None,
) -> int | None:
    """Resolve paragraph heading level: direct ``outlineLvl``, else style map, else name."""
    if has_direct_outline:
        return direct_outline
    if not style_val:
        return None
    if style_map is not None and style_val in style_map:
        return style_map[style_val]
    return _heading_level_from_style_name(style_val)


def _extract_docx_paragraph(
    p_node: ET.Element,
    *,
    style_map: Mapping[str, int | None] | None,
    allow_heading: bool,
    seen_fn_set: set[str],
    seen_fn_order: list[str],
) -> str | None:
    """Extract text for one ``<w:p>``, skipping deletions/field code and resolving Choice."""
    style_val, has_direct_outline, direct_outline = _extract_docx_p_props(p_node)
    text_parts: list[str] = []
    stack: list[ET.Element] = list(reversed(list(p_node)))
    while stack:
        elem = stack.pop()
        local = _xml_local(elem.tag)
        if local == "pPr" or local in _DOCX_SKIP_TAGS:
            continue
        if local == "AlternateContent":
            branch = _select_alternate_content_branch(elem)
            if branch is not None:
                stack.append(branch)
            continue
        if local == "t":
            if elem.text:
                text_parts.append(elem.text)
            continue
        if local == "tab":
            text_parts.append(" ")
            continue
        if local in {"br", "cr"}:
            text_parts.append("\n")
            continue
        if local == "footnoteReference":
            fn_id = (_xml_attr(elem, "id") or "").strip()
            if fn_id:
                prefix = (
                    " " if text_parts and not text_parts[-1].endswith((" ", "\t", "\n")) else ""
                )
                text_parts.append(f"{prefix}[fn {fn_id}]")
                if fn_id not in seen_fn_set:
                    seen_fn_set.add(fn_id)
                    seen_fn_order.append(fn_id)
            continue
        stack.extend(reversed(list(elem)))

    para_text = "".join(text_parts).strip()
    if not para_text:
        return None

    if allow_heading:
        level = _resolve_docx_heading_level(
            style_val, has_direct_outline, direct_outline, style_map
        )
        if level is not None:
            single_line = " ".join(para_text.split())
            if single_line:
                return f"{'#' * level} {single_line}"
    return para_text


def _extract_docx_footnotes(footnotes_root: ET.Element) -> dict[str, str]:
    """Extract ``footnote_id -> single-line text`` from ``word/footnotes.xml``."""
    footnotes: dict[str, str] = {}
    dummy_set: set[str] = set()
    dummy_order: list[str] = []
    for child in footnotes_root:
        if _xml_local(child.tag) != "footnote":
            continue
        ftype = (_xml_attr(child, "type") or "").strip().lower()
        if ftype in {"separator", "continuationseparator"}:
            continue
        fn_id = (_xml_attr(child, "id") or "").strip()
        if not fn_id:
            continue
        para_texts: list[str] = []
        stack: list[ET.Element] = list(reversed(list(child)))
        while stack:
            elem = stack.pop()
            local = _xml_local(elem.tag)
            if local in _DOCX_SKIP_TAGS:
                continue
            if local == "AlternateContent":
                branch = _select_alternate_content_branch(elem)
                if branch is not None:
                    stack.append(branch)
                continue
            if local == "p":
                p_text = _extract_docx_paragraph(
                    elem,
                    style_map=None,
                    allow_heading=False,
                    seen_fn_set=dummy_set,
                    seen_fn_order=dummy_order,
                )
                if p_text:
                    collapsed = " ".join(p_text.split())
                    if collapsed:
                        para_texts.append(collapsed)
                continue
            stack.extend(reversed(list(elem)))
        joined = " ".join(para_texts).strip()
        if joined:
            footnotes[fn_id] = joined
    return footnotes


def _collect_docx_table_rows(tbl_node: ET.Element) -> list[ET.Element]:
    """Collect ``<w:tr>`` rows of ``tbl_node`` iteratively without entering nested tables."""
    tr_nodes: list[ET.Element] = []
    stack: list[ET.Element] = list(reversed(list(tbl_node)))
    while stack:
        elem = stack.pop()
        local = _xml_local(elem.tag)
        if local in {"tblPr", "tblGrid", "tbl"} or local in _DOCX_SKIP_TAGS:
            continue
        if local == "AlternateContent":
            branch = _select_alternate_content_branch(elem)
            if branch is not None:
                stack.append(branch)
            continue
        if local == "tr":
            tr_nodes.append(elem)
            continue
        stack.extend(reversed(list(elem)))
    return tr_nodes


def _collect_docx_row_cells(tr_node: ET.Element) -> tuple[bool, int, list[ET.Element]]:
    """Return ``(is_tbl_header, grid_before, tc_nodes)`` for one ``<w:tr>``."""
    is_tbl_header = False
    grid_before = 0
    for child in tr_node:
        if _xml_local(child.tag) != "trPr":
            continue
        for prop in child:
            local = _xml_local(prop.tag)
            if local == "tblHeader":
                val = (_xml_attr(prop, "val") or "").strip().lower()
                is_tbl_header = val not in {"0", "false", "off"}
            elif local == "gridBefore":
                parsed = _parse_docx_nonneg_int(_xml_attr(prop, "val"))
                if parsed is not None:
                    grid_before = min(parsed, _DOCX_MAX_GRID_SPAN)

    tc_nodes: list[ET.Element] = []
    stack: list[ET.Element] = list(reversed(list(tr_node)))
    while stack:
        elem = stack.pop()
        local = _xml_local(elem.tag)
        if local == "trPr" or local in _DOCX_SKIP_TAGS:
            continue
        if local == "AlternateContent":
            branch = _select_alternate_content_branch(elem)
            if branch is not None:
                stack.append(branch)
            continue
        if local == "tc":
            tc_nodes.append(elem)
            continue
        stack.extend(reversed(list(elem)))
    return is_tbl_header, grid_before, tc_nodes


def _parse_docx_tc_props(tc_node: ET.Element) -> tuple[int, str | None]:
    """Return ``(grid_span, vmerge)`` from ``<w:tcPr>``."""
    grid_span = 1
    vmerge: str | None = None
    for child in tc_node:
        if _xml_local(child.tag) != "tcPr":
            continue
        for prop in child:
            local = _xml_local(prop.tag)
            if local == "gridSpan":
                parsed = _parse_docx_nonneg_int(_xml_attr(prop, "val"))
                if parsed is not None and parsed >= 1:
                    grid_span = min(parsed, _DOCX_MAX_GRID_SPAN)
            elif local == "vMerge":
                val = (_xml_attr(prop, "val") or "").strip().lower()
                vmerge = "restart" if val == "restart" else "continue"
    return grid_span, vmerge


def _extract_docx_cell_text(
    tc_node: ET.Element,
    *,
    seen_fn_set: set[str],
    seen_fn_order: list[str],
) -> str:
    """Extract single-line text across all paragraphs inside a table cell."""
    para_texts: list[str] = []
    stack: list[ET.Element] = list(reversed(list(tc_node)))
    while stack:
        elem = stack.pop()
        local = _xml_local(elem.tag)
        if local == "tcPr" or local in _DOCX_SKIP_TAGS:
            continue
        if local == "AlternateContent":
            branch = _select_alternate_content_branch(elem)
            if branch is not None:
                stack.append(branch)
            continue
        if local == "p":
            p_text = _extract_docx_paragraph(
                elem,
                style_map=None,
                allow_heading=False,
                seen_fn_set=seen_fn_set,
                seen_fn_order=seen_fn_order,
            )
            if p_text:
                collapsed = " ".join(p_text.split())
                if collapsed:
                    para_texts.append(collapsed)
            continue
        stack.extend(reversed(list(elem)))
    return " ".join(para_texts).strip()


class _DocxOutputTooLargeError(Exception):
    """Raised when the running docx output exceeds ``DOCX_OUTPUT_BUDGET_CHARS``."""


class _DocxOutputBudget:
    def __init__(self, max_chars: int, max_rows: int) -> None:
        self.remaining_chars = max_chars
        self.remaining_rows = max_rows

    def charge(self, text: str) -> None:
        self.remaining_chars -= len(text) + 2
        if self.remaining_chars < 0:
            raise _DocxOutputTooLargeError


def _escape_docx_line_start(line: str) -> str:
    if _DOCX_LINE_START_ESCAPE_RE.match(line):
        return f"\\{line}"
    return line


def _forward_fill_header_row(row: Sequence[str]) -> list[str]:
    """Forward-fill blank cells in an upper header row left to right."""
    filled: list[str] = []
    current = ""
    for cell in row:
        if cell:
            current = cell
        filled.append(current)
    return filled


def _format_docx_table(
    tbl_node: ET.Element,
    *,
    doc_cell_count: int,
    seen_fn_set: set[str],
    seen_fn_order: list[str],
    budget: _DocxOutputBudget,
) -> tuple[list[str], int, ET.Element | None]:
    """Build a merged grid for ``<w:tbl>`` and format rows via ``canonical.format_table_row``.

    A 1x1 layout table returns ``([], doc_cell_count, tc_node)`` so the caller unwraps it.
    Tables exceeding ``_DOCX_MAX_TABLE_ROWS`` (500) or ``_DOCX_MAX_TABLE_CELLS`` (20_000)
    are truncated silently because ``load_docx`` returns a fixed 3-tuple without a flag.
    """
    tr_nodes = _collect_docx_table_rows(tbl_node)
    if not tr_nodes:
        return [], doc_cell_count, None

    if len(tr_nodes) == 1:
        _, grid_before, first_tcs = _collect_docx_row_cells(tr_nodes[0])
        if grid_before == 0 and len(first_tcs) == 1:
            first_span, _ = _parse_docx_tc_props(first_tcs[0])
            if first_span == 1:
                return [], doc_cell_count, first_tcs[0]

    raw_rows: list[_DocxRawRow] = []
    for tr_node in tr_nodes[:_DOCX_MAX_TABLE_ROWS]:
        if doc_cell_count >= _DOCX_MAX_TABLE_CELLS:
            break
        is_tbl_header, grid_before, tc_nodes = _collect_docx_row_cells(tr_node)
        parsed_cells: list[_DocxCell] = []
        for tc_node in tc_nodes:
            if doc_cell_count >= _DOCX_MAX_TABLE_CELLS:
                break
            grid_span, vmerge = _parse_docx_tc_props(tc_node)
            doc_cell_count += grid_span
            cell_text = _extract_docx_cell_text(
                tc_node,
                seen_fn_set=seen_fn_set,
                seen_fn_order=seen_fn_order,
            )
            parsed_cells.append(
                _DocxCell(text=cell_text[:MAX_DOCX_CELL_CHARS], grid_span=grid_span, vmerge=vmerge)
            )
        raw_rows.append(
            _DocxRawRow(is_tbl_header=is_tbl_header, grid_before=grid_before, cells=parsed_cells)
        )

    if not raw_rows:
        return [], doc_cell_count, None

    explicit_hdr_count = 0
    for raw_row in raw_rows:
        if raw_row.is_tbl_header and explicit_hdr_count < _DOCX_MAX_HEADER_ROWS:
            explicit_hdr_count += 1
        else:
            break

    expanded_rows: list[list[str]] = []
    vmerge_state: list[str] = []
    for r_idx, raw_row in enumerate(raw_rows):
        is_header = r_idx < explicit_hdr_count
        row_cells = [""] * raw_row.grid_before
        for col in range(min(raw_row.grid_before, len(vmerge_state))):
            vmerge_state[col] = ""

        for cell in raw_row.cells:
            col_start = len(row_cells)
            needed = col_start + cell.grid_span
            if len(vmerge_state) < needed:
                vmerge_state.extend([""] * (needed - len(vmerge_state)))
            for offset in range(cell.grid_span):
                col = col_start + offset
                if cell.vmerge == "continue":
                    if vmerge_state[col]:
                        val = vmerge_state[col][:MAX_DOCX_HEADER_CHARS]
                    else:
                        val = cell.text if (offset == 0 or is_header) else ""
                else:
                    val = cell.text if (offset == 0 or is_header) else ""
                    vmerge_state[col] = val if cell.vmerge == "restart" else ""
                row_cells.append(val)
        expanded_rows.append(row_cells)

    ncols = max((len(row) for row in expanded_rows), default=0)
    if ncols == 0:
        return [], doc_cell_count, None

    for row in expanded_rows:
        if len(row) < ncols:
            row.extend([""] * (ncols - len(row)))

    if explicit_hdr_count > 0:
        header_grid = [list(row) for row in expanded_rows[:explicit_hdr_count]]
        for r_idx in range(len(header_grid) - 1):
            header_grid[r_idx] = _forward_fill_header_row(header_grid[r_idx])
        headers: list[str] = []
        for col in range(ncols):
            parts: list[str] = []
            for r_idx in range(len(header_grid)):
                part = header_grid[r_idx][col]
                if part and (not parts or parts[-1] != part):
                    parts.append(part)
            headers.append(" / ".join(parts) if parts else f"col_{col + 1}")
        data_start = explicit_hdr_count
    elif len(expanded_rows) >= 2 and all(cell != "" for cell in expanded_rows[0]):
        headers = expanded_rows[0]
        data_start = 1
    else:
        headers = [f"col_{col + 1}" for col in range(ncols)]
        data_start = 0

    headers = [h[:MAX_DOCX_HEADER_CHARS] for h in headers]
    table_lines: list[str] = []
    for row_cells in expanded_rows[data_start:]:
        if budget.remaining_rows <= 0:
            break
        formatted = canonical.format_table_row(headers, row_cells)
        if formatted:
            formatted = _escape_docx_line_start(formatted)
            budget.remaining_rows -= 1
            budget.charge(formatted)
            table_lines.append(formatted)
    return table_lines, doc_cell_count, None


def _extract_docx_paragraphs(
    root: ET.Element,
    *,
    style_map: Mapping[str, int | None] | None = None,
    footnote_map: Mapping[str, str] | None = None,
) -> list[str]:
    """Walk DOCX XML in reading order, emitting paragraphs, headings, tables, and footnotes."""
    blocks: list[str] = []
    seen_fn_set: set[str] = set()
    seen_fn_order: list[str] = []
    doc_cell_count = 0
    budget = _DocxOutputBudget(DOCX_OUTPUT_BUDGET_CHARS, _DOCX_MAX_TOTAL_TABLE_ROWS)

    stack: list[ET.Element] = [root]
    while stack:
        node = stack.pop()
        local = _xml_local(node.tag)
        if local in _DOCX_SKIP_TAGS:
            continue
        if local == "AlternateContent":
            branch = _select_alternate_content_branch(node)
            if branch is not None:
                stack.append(branch)
            continue
        if local == "p":
            para_text = _extract_docx_paragraph(
                node,
                style_map=style_map,
                allow_heading=True,
                seen_fn_set=seen_fn_set,
                seen_fn_order=seen_fn_order,
            )
            if para_text:
                budget.charge(para_text)
                blocks.append(para_text)
            continue
        if local == "tbl":
            table_lines, doc_cell_count, unwrapped_tc = _format_docx_table(
                node,
                doc_cell_count=doc_cell_count,
                seen_fn_set=seen_fn_set,
                seen_fn_order=seen_fn_order,
                budget=budget,
            )
            if unwrapped_tc is not None:
                stack.extend(reversed(list(unwrapped_tc)))
            elif table_lines:
                blocks.append("\n".join(table_lines))
            continue
        stack.extend(reversed(list(node)))

    if footnote_map and seen_fn_order:
        fn_lines: list[str] = []
        for fn_id in seen_fn_order:
            fn_body = footnote_map.get(fn_id)
            if fn_body:
                fn_line = _escape_docx_line_start(f"[fn {fn_id}] {fn_body}")
                budget.charge(fn_line)
                fn_lines.append(fn_line)
        if fn_lines:
            blocks.append("\n".join(fn_lines))
    return blocks


def _parse_docx_xml_bounded(
    doc_bytes: bytes,
    max_depth: int,
) -> tuple[ET.Element | None, str | None]:
    """Parse docx XML with iterparse, aborting as soon as element depth exceeds ``max_depth``."""
    return zipguard.parse_xml_bounded(doc_bytes, max_depth, label="docx")


def _parse_docx_xml_member(xml_bytes: bytes) -> tuple[ET.Element | None, str | None]:
    """Apply DOCTYPE/ENTITY and depth guards to one DOCX XML member."""
    if _has_forbidden_xml_decl(xml_bytes):
        return None, "docx XML contains forbidden DOCTYPE or ENTITY declaration"
    try:
        root, depth_error = _parse_docx_xml_bounded(xml_bytes, MAX_DOCX_XML_DEPTH)
    except ET.ParseError as exc:
        return None, f"docx XML parse error: {exc}"
    if depth_error is not None or root is None:
        return None, depth_error or "docx XML parse error"
    return root, None


def _scan_docx_refs(doc_root: ET.Element) -> tuple[bool, bool]:
    """Return ``(has_pstyle, has_footnote_ref)`` for lazy reading of auxiliary DOCX parts."""
    has_pstyle = False
    has_fn_ref = False
    for elem in doc_root.iter():
        local = _xml_local(elem.tag)
        if local == "pStyle":
            has_pstyle = True
        elif local == "footnoteReference":
            has_fn_ref = True
        if has_pstyle and has_fn_ref:
            break
    return has_pstyle, has_fn_ref


def load_docx(raw_bytes: bytes) -> tuple[str, str, str | None]:
    """Extract markdown-structured text from a ``.docx`` archive with zip/XML guards."""
    preflight_err = zipguard.preflight_zip(
        raw_bytes,
        label="docx",
        scan_xml_members=False,
        max_members=MAX_DOCX_MEMBERS,
        max_member_bytes=MAX_DOCX_MEMBER_BYTES,
        max_total_bytes=MAX_DOCX_TOTAL_BYTES,
        max_compression_ratio=MAX_DOCX_COMPRESSION_RATIO,
        max_xml_depth=MAX_DOCX_XML_DEPTH,
    )
    if preflight_err is not None:
        return "", "unreadable", preflight_err

    doc_root: ET.Element | None = None
    styles_root: ET.Element | None = None
    footnotes_root: ET.Element | None = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                doc_info: zipfile.ZipInfo | None = None
                styles_info: zipfile.ZipInfo | None = None
                footnotes_info: zipfile.ZipInfo | None = None
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    if info.filename == "word/document.xml":
                        doc_info = info
                    elif info.filename == "word/styles.xml":
                        styles_info = info
                    elif info.filename == "word/footnotes.xml":
                        footnotes_info = info

                if doc_info is None:
                    return "", "unreadable", "missing word/document.xml"

                doc_bytes = _read_zip_member_bounded(zf, doc_info, MAX_DOCX_MEMBER_BYTES, keep=True)
                total_read = len(doc_bytes)
                doc_root, member_err = _parse_docx_xml_member(doc_bytes)
                if member_err is not None or doc_root is None:
                    return "", "unreadable", member_err or "docx XML parse error"

                has_pstyle, has_fn_ref = _scan_docx_refs(doc_root)
                if has_pstyle and styles_info is not None:
                    styles_bytes = _read_zip_member_bounded(
                        zf, styles_info, MAX_DOCX_MEMBER_BYTES, keep=True
                    )
                    total_read += len(styles_bytes)
                    if total_read > MAX_DOCX_TOTAL_BYTES:
                        return (
                            "",
                            "unreadable",
                            f"docx total uncompressed size exceeds {MAX_DOCX_TOTAL_BYTES} bytes",
                        )
                    styles_root, member_err = _parse_docx_xml_member(styles_bytes)
                    if member_err is not None:
                        return "", "unreadable", member_err

                if has_fn_ref and footnotes_info is not None:
                    footnotes_bytes = _read_zip_member_bounded(
                        zf, footnotes_info, MAX_DOCX_MEMBER_BYTES, keep=True
                    )
                    total_read += len(footnotes_bytes)
                    if total_read > MAX_DOCX_TOTAL_BYTES:
                        return (
                            "",
                            "unreadable",
                            f"docx total uncompressed size exceeds {MAX_DOCX_TOTAL_BYTES} bytes",
                        )
                    footnotes_root, member_err = _parse_docx_xml_member(footnotes_bytes)
                    if member_err is not None:
                        return "", "unreadable", member_err
    except (
        UserWarning,
        zipfile.BadZipFile,
        ValueError,
        zipfile.LargeZipFile,
        OSError,
        EOFError,
        zlib.error,
        RuntimeError,
    ) as exc:
        return "", "unreadable", zipguard.format_zip_error(exc, label="docx")

    style_map = _parse_docx_styles(styles_root) if styles_root is not None else None
    footnote_map = _extract_docx_footnotes(footnotes_root) if footnotes_root is not None else None
    try:
        paragraphs = _extract_docx_paragraphs(
            doc_root,
            style_map=style_map,
            footnote_map=footnote_map,
        )
    except _DocxOutputTooLargeError:
        return "", "unreadable", _DOCX_OUTPUT_TOO_LARGE
    if not paragraphs:
        return "", "empty", "docx contains no text"
    return "\n\n".join(paragraphs), "ok", None


def _sparse_pdf_result(
    pages: list[tuple[int, str]], sparse_pages: list[int]
) -> tuple[list[tuple[int, str]], str, str | None]:
    """Final verdict for a PDF once its pages have been extracted.

    Text pages are always kept. ``scanned`` is returned only when NO page has
    text; otherwise the text pages go back with status ``ok`` and a note
    listing the sparse page numbers (capped at 20) so a later OCR phase can
    pick them up.
    """
    if sparse_pages and not pages:
        return pages, "scanned", f"scanned: {len(sparse_pages)} pages without text"
    if sparse_pages:
        listed = ", ".join(str(n) for n in sparse_pages[:20])
        return pages, "ok", f"ok ({len(sparse_pages)} pages without text: {listed})"
    return pages, "ok", None


def _load_pdf_pages_pypdf(raw_bytes: bytes) -> tuple[list[tuple[int, str]], str, str | None]:
    """pypdf engine for :func:`load_pdf_pages`."""
    try:
        import pypdf
    except ImportError:
        return [], "unsupported", "pypdf not installed"

    try:
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
        if reader.is_encrypted:
            return [], "unreadable", "pdf is encrypted"
        num_pages = len(reader.pages)
        if num_pages > MAX_PDF_PAGES:
            return (
                [],
                "too_large",
                f"pdf has {num_pages} pages (exceeds limit of {MAX_PDF_PAGES})",
            )
        pages: list[tuple[int, str]] = []
        sparse_pages: list[int] = []
        for page_num, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            stripped = extracted.strip()
            if not stripped or len(stripped) < 20:
                sparse_pages.append(page_num)
            else:
                pages.append((page_num, stripped))
    except Exception as exc:
        msg = str(exc).strip() or type(exc).__name__
        return [], "unreadable", f"pdf parse error: {msg}"

    return _sparse_pdf_result(pages, sparse_pages)


def _pdfium_module() -> Any | None:
    try:
        import pypdfium2  # type: ignore[import-untyped]
    except ImportError:
        return None
    return pypdfium2


def _load_pdf_pages_pdfium(
    pdfium: Any, raw_bytes: bytes
) -> tuple[list[tuple[int, str]], str, str | None]:
    """pypdfium2 engine for :func:`load_pdf_pages`."""
    pages: list[tuple[int, str]] = []
    sparse_pages: list[int] = []
    try:
        doc = pdfium.PdfDocument(raw_bytes)
        try:
            num_pages = len(doc)
            if num_pages > MAX_PDF_PAGES:
                return (
                    [],
                    "too_large",
                    f"pdf has {num_pages} pages (exceeds limit of {MAX_PDF_PAGES})",
                )
            for page_num in range(1, num_pages + 1):
                page = doc[page_num - 1]
                try:
                    textpage = page.get_textpage()
                    try:
                        extracted = textpage.get_text_range() or ""
                    finally:
                        textpage.close()
                finally:
                    page.close()
                stripped = (
                    extracted.replace("\ufffe", "")
                    .replace("\r\n", "\n")
                    .replace("\r", "\n")
                    .strip()
                )
                if not stripped or len(stripped) < 20:
                    sparse_pages.append(page_num)
                else:
                    pages.append((page_num, stripped))
        finally:
            doc.close()
    except Exception as exc:
        msg = str(exc).strip() or type(exc).__name__
        if "password" in msg.lower():
            return [], "unreadable", "pdf is encrypted"
        return [], "unreadable", f"pdf parse error: {msg}"

    return _sparse_pdf_result(pages, sparse_pages)


def load_pdf_pages(raw_bytes: bytes) -> tuple[list[tuple[int, str]], str, str | None]:
    """Extract 1-based ``(page_num, text)`` units from PDF bytes.

    Engine is chosen by ``VOICE_AGENT_RAG_PDF_ENGINE`` (pypdf | pdfium | auto).
    The default is ``auto``: pdfium when importable, else pypdf; an unknown value means pypdf.

    Pages with fewer than 20 characters of native text are not returned. When
    any such pages exist, the note lists how many there are and their page
    numbers (capped at 20); status is ``scanned`` only when NO page has text.
    """
    engine = os.environ.get(PDF_ENGINE_ENV, "auto").strip().lower()
    if engine in ("pdfium", "auto"):
        pdfium = _pdfium_module()
        if pdfium is not None:
            return _load_pdf_pages_pdfium(pdfium, raw_bytes)
    return _load_pdf_pages_pypdf(raw_bytes)


def decode_and_load(raw_bytes: bytes, kind: str) -> tuple[str, str, str | None]:
    """Decode bytes and apply format-specific loading.

    Returns ``(text, status, error)``.
    """
    if kind == "docx":
        return load_docx(raw_bytes)

    decoded = raw_bytes.decode("utf-8-sig", errors="replace")
    if not decoded.strip():
        return "", "empty", "file is empty"

    replacement_ratio = decoded.count("\ufffd") / len(decoded)
    if replacement_ratio > MAX_REPLACEMENT_RATIO:
        return (
            "",
            "unreadable",
            f"unreadable file ({replacement_ratio:.0%} replacement characters)",
        )

    if kind == "tsv":
        try:
            loaded = parse_csv_text(decoded, delimiter="\t")
        except csv.Error as exc:
            return "", "unreadable", f"csv parse error: {exc}"
        if not loaded.strip():
            return "", "empty", "csv contains no data rows"
        return loaded, "ok", None

    if kind == "csv":
        try:
            loaded = parse_csv_text(decoded)
        except csv.Error as exc:
            return "", "unreadable", f"csv parse error: {exc}"
        if not loaded.strip():
            return "", "empty", "csv contains no data rows"
        return loaded, "ok", None

    return decoded, "ok", None


def split_markdown_sections(
    text: str,
    *,
    max_heading_level: int = 6,
) -> list[tuple[str, str]]:
    """Split markdown text into ``(section_path, body)`` pairs."""
    heading_re = (
        _MD_HEADING_RE
        if max_heading_level == 6
        else re.compile(rf"^(#{{1,{max_heading_level}}})\s+(\S.*)$")
    )
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    fence_char: str | None = None
    fence_len = 0

    def _current_section() -> str:
        return " > ".join(title for _, title in stack)[:MAX_SECTION_PATH_CHARS]

    def _flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((_current_section(), body))
        current_lines.clear()

    for line in text.splitlines():
        stripped = line.lstrip()
        fence_match = _FENCE_RE.match(stripped)
        if fence_match:
            marker = fence_match.group(1)
            if fence_char is None:
                fence_char = marker[0]
                fence_len = len(marker)
            elif (
                marker[0] == fence_char
                and len(marker) >= fence_len
                and not stripped[len(marker) :].strip()
            ):
                fence_char = None
                fence_len = 0
            current_lines.append(line)
            continue

        if fence_char is None and len(line) <= MAX_HEADING_LINE_CHARS:
            heading_match = heading_re.match(line)
            if heading_match:
                title = heading_match.group(2).rstrip().rstrip("#").rstrip()
                if title:
                    _flush()
                    level = len(heading_match.group(1))
                    title = title[:MAX_HEADING_TITLE_CHARS]
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, title))
                    continue

        current_lines.append(line)

    _flush()
    return sections


def _split_into_atoms(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, str]]:
    """Break text hierarchically (paragraphs -> lines -> sentences -> words) into atoms."""
    atoms: list[tuple[str, str]] = []

    def _slice_hard(piece: str, initial_sep: str) -> None:
        step = max(1, chunk_size - chunk_overlap)
        pos = 0
        first = True
        while pos < len(piece):
            sep = initial_sep if first else ""
            atoms.append((sep, piece[pos : pos + chunk_size]))
            first = False
            if pos + chunk_size >= len(piece):
                break
            pos += step

    def _split_words(sentence: str, initial_sep: str) -> None:
        words = [w for w in _WORD_SPLIT_RE.split(sentence.strip()) if w]
        for idx, word in enumerate(words):
            sep = initial_sep if idx == 0 else " "
            if len(word) <= chunk_size:
                atoms.append((sep, word))
            else:
                _slice_hard(word, sep)

    def _split_sentences(line: str, initial_sep: str) -> None:
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(line.strip()) if s]
        for idx, sentence in enumerate(sentences):
            sep = initial_sep if idx == 0 else " "
            if len(sentence) <= chunk_size:
                atoms.append((sep, sentence))
            else:
                _split_words(sentence, sep)

    def _split_lines(para: str, initial_sep: str) -> None:
        lines = [ln.strip() for ln in _LINE_SPLIT_RE.split(para.strip()) if ln.strip()]
        for idx, line in enumerate(lines):
            sep = initial_sep if idx == 0 else "\n"
            if len(line) <= chunk_size:
                atoms.append((sep, line))
            else:
                _split_sentences(line, sep)

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text.strip()) if p.strip()]
    for idx, para in enumerate(paragraphs):
        sep = "" if idx == 0 else "\n\n"
        if len(para) <= chunk_size:
            atoms.append((sep, para))
        else:
            _split_lines(para, sep)
    return atoms


def split_section(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split a section into overlapping chunks, keeping every non-empty body."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    atoms = _split_into_atoms(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not atoms:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(atoms):
        buf = [atoms[start][1]]
        cur_len = len(atoms[start][1])
        j = start + 1
        while j < len(atoms):
            sep, seg = atoms[j]
            add_len = len(sep) + len(seg)
            if cur_len + add_len <= chunk_size:
                buf.append(sep + seg)
                cur_len += add_len
                j += 1
            else:
                break

        chunk = "".join(buf).strip()
        if chunk:
            chunks.append(chunk)

        if j >= len(atoms):
            break

        next_start = j
        overlap_with_next = len(atoms[j][1])
        k = j - 1
        while k > start:
            cand_sep = atoms[k + 1][0]
            cand_add = len(atoms[k][1]) + len(cand_sep)
            if overlap_with_next + cand_add > chunk_size:
                break
            overlap_with_next += cand_add
            next_start = k
            if overlap_with_next - len(atoms[j][1]) >= chunk_overlap:
                break
            k -= 1
        start = next_start

    return chunks


def _chunk_header(source: str, section: str, page: int | None) -> str:
    if section:
        return f"{source} > {section}"
    if page is not None:
        return f"{source} > p. {page}"
    return source


def _build_chunks_from_specs(
    specs: Sequence[tuple[str, int | None, int | None, str | None, str]],
    *,
    fmt: str,
    source: str,
    file_sha256: str,
    file_date: str,
    canonical_sha256: str,
    converter: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    position = 0
    for section, page, slide, sheet, body in specs:
        header = _chunk_header(source, section, page)
        for body_chunk in split_section(
            body,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            chunks.append(
                Chunk(
                    id=f"{source}#{position}",
                    file_sha256=file_sha256,
                    chunk_sha256=hash_bytes(body_chunk.encode("utf-8")),
                    position=position,
                    page=page,
                    section=section,
                    file_date=file_date,
                    source=source,
                    header=header,
                    text=body_chunk,
                    fmt=fmt,
                    slide=slide,
                    sheet=sheet,
                    canonical_sha256=canonical_sha256,
                    converter=converter,
                )
            )
            position += 1
    return chunks


def chunk_text(
    text: str,
    *,
    kind: str,
    source: str,
    file_sha256: str,
    file_date: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    pages: Sequence[tuple[int, str]] | None = None,
    fmt: str | None = None,
    canonical_sha256: str = "",
    converter: str = "legacy",
) -> list[Chunk]:
    """Split loaded text into structure-aware ``Chunk`` records."""
    specs: list[tuple[str, int | None, int | None, str | None, str]]
    if kind == "pdf" and pages is not None:
        specs = [("", page_num, None, None, body) for page_num, body in pages if body.strip()]
    elif kind == "md":
        specs = [
            (section, None, None, None, body) for section, body in split_markdown_sections(text)
        ]
    elif kind == "docx":
        specs = [
            (section, None, None, None, body)
            for section, body in split_markdown_sections(text, max_heading_level=9)
        ]
    else:
        stripped = text.strip()
        specs = [("", None, None, None, stripped)] if stripped else []

    return _build_chunks_from_specs(
        specs,
        fmt=fmt if fmt is not None else kind,
        source=source,
        file_sha256=file_sha256,
        file_date=file_date,
        canonical_sha256=canonical_sha256,
        converter=converter,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def chunk_canonical(
    canonical_text: str,
    *,
    kind: str,
    source: str,
    file_sha256: str,
    file_date: str,
    canonical_sha256: str,
    converter: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split canonical Markdown text into structure-aware ``Chunk`` records."""
    units, _ = canonical.split_units(canonical_text)
    specs: list[tuple[str, int | None, int | None, str | None, str]] = []
    for unit in units:
        if kind == "pdf":
            specs.append(("", unit.page, None, None, unit.body))
        elif kind == "pptx":
            slide_prefix = f"Slide {unit.slide}" if unit.slide is not None else ""
            md_sections = split_markdown_sections(unit.body)
            if not md_sections and unit.body.strip():
                md_sections = [("", unit.body.strip())]
            for sub_section, body in md_sections:
                if slide_prefix and sub_section:
                    section = f"{slide_prefix} > {sub_section}"[:MAX_SECTION_PATH_CHARS]
                else:
                    section = slide_prefix or sub_section
                specs.append((section, unit.page, unit.slide, None, body))
        elif kind in ("xlsx", "xls"):
            section = f"Sheet {unit.sheet}"[:MAX_SECTION_PATH_CHARS] if unit.sheet else ""
            specs.append((section, unit.page, None, unit.sheet, unit.body))
        elif kind in ("md", "docx", "html", "epub", "rtf"):
            max_level = 9 if kind == "docx" else 6
            for section, body in split_markdown_sections(unit.body, max_heading_level=max_level):
                specs.append((section, unit.page, unit.slide, unit.sheet, body))
        else:
            stripped = unit.body.strip()
            if stripped:
                specs.append(("", unit.page, unit.slide, unit.sheet, stripped))

    return _build_chunks_from_specs(
        specs,
        fmt=kind,
        source=source,
        file_sha256=file_sha256,
        file_date=file_date,
        canonical_sha256=canonical_sha256,
        converter=converter,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _is_skipped_name(name: str) -> bool:
    return name.startswith(".") or name.startswith("~$") or name.startswith(".~lock")


def _walk_folder(root: Path) -> tuple[list[Path], list[Path]]:
    """Walk ``root`` iteratively with a depth cap, skipping hidden files and symlink escapes."""
    if not root.exists() or not root.is_dir():
        return [], []

    root_resolved = root.resolve()
    seen_dirs: set[Path] = {root_resolved}
    collected: list[Path] = []
    too_deep_dirs: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]

    while stack:
        current, depth = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except OSError:
            continue

        subdirs_to_push: list[tuple[Path, int]] = []
        for entry in entries:
            if _is_skipped_name(entry.name):
                continue
            if entry.is_symlink():
                try:
                    resolved = entry.resolve()
                except OSError:
                    continue
                if not resolved.is_relative_to(root_resolved):
                    continue
                if not entry.exists():
                    continue
            if entry.is_dir():
                try:
                    dir_resolved = entry.resolve()
                except OSError:
                    continue
                if not dir_resolved.is_relative_to(root_resolved) or dir_resolved in seen_dirs:
                    continue
                seen_dirs.add(dir_resolved)
                next_depth = depth + 1
                if next_depth > MAX_FOLDER_DEPTH:
                    too_deep_dirs.append(entry)
                else:
                    subdirs_to_push.append((entry, next_depth))
            elif entry.is_file():
                collected.append(entry)

        stack.extend(reversed(subdirs_to_push))

    collected.sort(key=lambda p: p.relative_to(root).as_posix())
    too_deep_dirs.sort(key=lambda p: p.relative_to(root).as_posix())
    return collected, too_deep_dirs


def _safe_file_date(mtime: float) -> tuple[float, str]:
    try:
        return mtime, datetime.fromtimestamp(mtime, tz=UTC).date().isoformat()
    except (ValueError, OverflowError, OSError):
        return 0.0, ""


def _normalize_format(fmt: str) -> str:
    norm = fmt.strip().lower().lstrip(".")
    if norm == "htm":
        return "html"
    return norm


def _resolve_converters(converters: Mapping[str, str] | None) -> dict[str, str]:
    resolved: dict[str, str] = {fmt: "legacy" for fmt in LEGACY_FORMATS}
    source = CANONICAL_CONVERTERS if converters is None else converters
    for raw_fmt, mode in source.items():
        fmt = _normalize_format(raw_fmt)
        allowed = _VALID_CONVERTERS_BY_FORMAT.get(fmt)
        if allowed is None:
            raise ValueError(f"unsupported converter format: {raw_fmt!r}")
        if mode not in allowed:
            raise ValueError(f"unsupported converter mode {mode!r} for format {fmt!r}")
        resolved[fmt] = mode
    return resolved


def _format_converter_label(res: convert.ConvertResult) -> str:
    base = res.converter or "unknown"
    if base != "legacy":
        base = f"{base}+pp{canonical.POSTPROC_VERSION}"
    if res.fallback:
        return f"{base}/fallback={res.fallback}"
    return base


def _markitdown_import_blocked() -> bool:
    return (
        sys.modules.get("markitdown", _SENTINEL) is None
        or sys.modules.get("markitdown.converters", _SENTINEL) is None
    )


INPROCESS_MAX_BYTES = 16 * 1024 * 1024  # 16 MiB cap for in-process conversion


def _should_use_inprocess(kind: str, raw_bytes: bytes) -> bool:
    """Return True when in-process conversion is preferred for *kind*."""
    return (
        kind in {"pptx", "xlsx", "html"}
        and len(raw_bytes) <= INPROCESS_MAX_BYTES
        and os.environ.get("VOICE_AGENT_RAG_CONVERT_ISOLATION") != "always"
    )


def _direct_import_blocked(kind: str) -> str | None:
    if kind == "xlsx" and sys.modules.get("openpyxl", _SENTINEL) is None:
        return "openpyxl not installed"
    if kind == "pptx" and sys.modules.get("pptx", _SENTINEL) is None:
        return "python-pptx not installed"
    if kind in ("html", "epub") and (
        sys.modules.get("lxml", _SENTINEL) is None
        or sys.modules.get("lxml.html", _SENTINEL) is None
    ):
        return "lxml not installed"
    if kind == "doc" and shutil.which("catdoc") is None:
        return "doc (catdoc not installed)"
    if kind == "ppt" and shutil.which("catppt") is None:
        return "ppt (catppt not installed)"
    return None


def _convert_canonical(raw_bytes: bytes, kind: str, mode: str) -> convert.ConvertResult:
    if mode == "pypdf":
        return convert.convert_pdf(raw_bytes, mode="pypdf")
    if mode == "markitdown":
        if _markitdown_import_blocked():
            return convert.ConvertResult(
                text="",
                status="unsupported",
                error="markitdown not installed",
            )
        return convert.convert_isolated(raw_bytes, kind)
    if mode == "direct":
        blocked_err = _direct_import_blocked(kind)
        if blocked_err is not None:
            return convert.ConvertResult(
                text="",
                status="unsupported",
                error=blocked_err,
            )
        if _should_use_inprocess(kind, raw_bytes):
            return convert.convert_inprocess(raw_bytes, kind)
        return convert.convert_isolated(raw_bytes, kind, engine="direct")
    raise ValueError(f"unexpected canonical mode: {mode!r}")


def _package_version(pkg_name: str) -> str:
    mod = sys.modules.get(pkg_name, _SENTINEL)
    if mod is None:
        return "unavailable"
    if mod is not _SENTINEL:
        ver = getattr(mod, "__version__", None)
        if isinstance(ver, str) and ver:
            return ver
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _canonical_cache_key_for(file_sha256: str, kind: str, mode: str) -> str:
    from voice_agent.rag import readers  # type: ignore[import-not-used]

    if mode == "identity":
        converter_id, converter_version = "identity", "1"
    elif mode == "pypdf":
        converter_id, converter_version = "pypdf", _package_version("pypdf")
    elif mode == "markitdown":
        converter_id, converter_version = "markitdown", _package_version("markitdown")
    elif mode == "direct":
        if kind == "xlsx":
            converter_id, converter_version = "direct-openpyxl", _package_version("openpyxl")
        elif kind == "pptx":
            converter_id, converter_version = (
                "direct-python-pptx",
                _package_version("python-pptx"),
            )
        elif kind in ("html", "epub"):
            converter_id, converter_version = "direct-lxml", _package_version("lxml")
        elif kind == "rtf":
            converter_id, converter_version = "direct-striprtf", _package_version("striprtf")
        elif kind == "xls":
            converter_id, converter_version = "direct-xlrd", _package_version("xlrd")
        else:
            raise ValueError(f"unexpected direct format: {kind!r}")
        if kind in ("xlsx", "xls") and readers._table_headers_mode() == "stack":
            converter_id += "+stack1"
    else:
        raise ValueError(f"unexpected canonical mode: {mode!r}")
    return canonical_cache.cache_key(
        file_sha256,
        kind,
        converter_id,
        converter_version,
        canonical.POSTPROC_VERSION,
    )


def _format_cached_canonical(converter_label: str, canonical_text: str) -> str:
    """Prepend the converter header marker line to canonical Markdown for disk caching."""
    return f"<!-- rag:converter={converter_label} -->\n{canonical_text}"


_DIRECT_LABEL_PREFIX_BY_KIND: dict[str, str] = {
    "xlsx": "direct-openpyxl-",
    "pptx": "direct-python-pptx-",
    "html": "direct-lxml-",
    "epub": "direct-lxml-",
    "rtf": "direct-striprtf-",
    "xls": "direct-xlrd-",
}


def _is_valid_converter_label(label: str, mode: str, *, kind: str = "") -> bool:
    if not label or len(label) > 120 or any(ch.isspace() for ch in label):
        return False
    if "<" in label or ">" in label or "-->" in label:
        return False
    if mode == "identity":
        return label == "identity"
    pp_suffix = f"+pp{canonical.POSTPROC_VERSION}"
    if mode == "pypdf":
        return label.startswith("pypdf-") and label.endswith(pp_suffix)
    if mode == "markitdown":
        return label.startswith(("markitdown-", "pypdf-", "legacy/fallback="))
    if mode == "direct":
        expected_prefix = _DIRECT_LABEL_PREFIX_BY_KIND.get(kind)
        if expected_prefix is None:
            return False
        base_label, fb_sep, fb_val = label.partition("/fallback=")
        if fb_sep and fb_val != "truncated":
            return False
        return (
            base_label.startswith(expected_prefix)
            and base_label.endswith(pp_suffix)
            and len(base_label) > len(expected_prefix) + len(pp_suffix)
        )
    return False


def _has_valid_cache_markers(body: str, *, kind: str) -> bool:
    normalized = body.replace("\r\n", "\n").replace("\r", "\n")
    for line in normalized.split("\n"):
        if not line.lstrip().startswith("<!-- rag:"):
            continue
        if line != line.lstrip():
            return False
        parsed = canonical._parse_marker_line(line)
        if parsed is None:
            return False
        key, val = parsed
        if kind == "pdf":
            if key != "page" or int(val) < 1:
                return False
        elif kind == "pptx":
            if key != "slide" or int(val) < 1:
                return False
        elif kind in ("xlsx", "xls"):
            if key != "sheet" or not val or val != canonical.clean_sheet_name(val):
                return False
        else:
            return False
    return True


def _parse_cached_canonical(
    cached_payload: str,
    *,
    kind: str,
    mode: str,
) -> tuple[str, str] | None:
    """Validate a cached canonical payload and return ``(canonical_text, converter_label)``."""
    first_line, sep, body = cached_payload.partition("\n")
    if not sep or not body.strip():
        return None
    if len(body) > canonical.MAX_CANONICAL_CHARS:
        return None
    parsed = canonical._parse_marker_line(first_line)
    if parsed is None:
        return None
    header_key, converter_label = parsed
    if header_key != "converter" or not _is_valid_converter_label(converter_label, mode, kind=kind):
        return None
    if not _has_valid_cache_markers(body, kind=kind):
        return None
    units, _ = canonical.split_units(body)
    if not units:
        return None
    return body, converter_label


def _build_canonical_chunks(
    canonical_text: str,
    *,
    kind: str,
    source: str,
    file_sha256: str,
    file_date: str,
    converter_label: str,
) -> tuple[list[Chunk], list[str]]:
    canonical_sha256 = hash_bytes(canonical_text.encode("utf-8"))
    dates = extract_dates(canonical.strip_markers(canonical_text))
    file_chunks = chunk_canonical(
        canonical_text,
        kind=kind,
        source=source,
        file_sha256=file_sha256,
        file_date=file_date,
        canonical_sha256=canonical_sha256,
        converter=converter_label,
    )
    return file_chunks, dates


def _legacy_parser_version(kind: str) -> str:
    if kind == "pdf":
        engine = os.environ.get(PDF_ENGINE_ENV, "auto").strip().lower()
        if engine in ("pdfium", "auto") and _pdfium_module() is not None:
            return "pdfium-" + _package_version("pypdfium2")
        return "pypdf-" + _package_version("pypdf")
    return "docx" if kind == "docx" else "text"


def _parse_legacy_payload(
    payload: str, kind: str
) -> tuple[str, list[tuple[int, str]] | None] | None:
    try:
        obj = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(obj, dict) or obj.get("v") != LEGACY_PARSE_VERSION:
        return None
    if kind == "pdf":
        raw_pages = obj.get("pages")
        if (
            set(obj) != {"v", "pages"}
            or not isinstance(raw_pages, list)
            or not raw_pages
            or len(raw_pages) > MAX_PDF_PAGES
        ):
            return None
        pages: list[tuple[int, str]] = []
        for item in raw_pages:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or type(item[0]) is not int
                or item[0] < 1
                or not isinstance(item[1], str)
            ):
                return None
            pages.append((item[0], item[1]))
        return "\n\n".join(t for _, t in pages), pages
    text = obj.get("text")
    if set(obj) != {"v", "text"} or not isinstance(text, str) or not text:
        return None
    return text, None


def _load_legacy(
    raw_bytes: bytes,
    *,
    kind: str,
    source: str,
    file_sha256: str,
    file_date: str,
    cache: canonical_cache.CanonicalCache | None,
) -> tuple[list[Chunk], list[str], str, str | None, str]:
    use_cache = cache is not None and cache.enabled
    key = ""
    loaded_text = ""
    pdf_pages: list[tuple[int, str]] | None = None
    hit = False
    if use_cache:
        assert cache is not None
        key = canonical_cache.cache_key(
            file_sha256,
            kind,
            "legacy-parse",
            _legacy_parser_version(kind),
            LEGACY_PARSE_VERSION,
        )
        cached_payload = cache.get(key)
        if cached_payload is not None:
            parsed = _parse_legacy_payload(cached_payload, kind)
            if parsed is not None:
                loaded_text, pdf_pages = parsed
                hit = True
    if not hit:
        if kind == "pdf":
            pdf_pages, status, error = load_pdf_pages(raw_bytes)
            loaded_text = "\n\n".join(page_text for _, page_text in pdf_pages)
        else:
            loaded_text, status, error = decode_and_load(raw_bytes, kind)
        if status != "ok":
            return [], [], status, error, ""
    dates = extract_dates(loaded_text)
    file_chunks = chunk_text(
        loaded_text,
        kind=kind,
        source=source,
        file_sha256=file_sha256,
        file_date=file_date,
        pages=pdf_pages,
        fmt=kind,
        canonical_sha256="",
        converter="legacy",
    )
    if not use_cache:
        return file_chunks, dates, "ok", None, ""
    if hit and file_chunks:
        return file_chunks, dates, "ok", None, key
    if not hit and file_chunks:
        assert cache is not None
        payload_obj: dict[str, Any] = (
            {"v": LEGACY_PARSE_VERSION, "pages": [[n, t] for n, t in pdf_pages]}
            if pdf_pages is not None
            else {"v": LEGACY_PARSE_VERSION, "text": loaded_text}
        )
        cache.put(key, json.dumps(payload_obj, ensure_ascii=False, separators=(",", ":")))
        return file_chunks, dates, "ok", None, key
    return file_chunks, dates, "ok", None, ""


def _load_chunks_for_mode(
    raw_bytes: bytes,
    *,
    kind: str,
    mode: str,
    source: str,
    file_sha256: str,
    file_date: str,
    cache: canonical_cache.CanonicalCache | None = None,
    file_path: Path | None = None,
) -> tuple[list[Chunk], list[str], str, str | None, str]:
    if mode == "direct" and kind in ("doc", "ppt") and file_path is not None:
        from voice_agent.rag import readers

        reader_func = readers.read_doc if kind == "doc" else readers.read_ppt
        res = reader_func(file_path)
        if res.status != "ok":
            return [], [], res.status, res.error, ""
        canonical_text = res.text
        converter_label = _format_converter_label(res)
        file_chunks, dates = _build_canonical_chunks(
            canonical_text,
            kind=kind,
            source=source,
            file_sha256=file_sha256,
            file_date=file_date,
            converter_label=converter_label,
        )
        if not file_chunks:
            return [], [], "empty", "no text to index", ""
        key = f"catdoc:{file_sha256}"
        if cache is not None and cache.enabled:
            cache.put(key, _format_cached_canonical(converter_label, canonical_text))
        return file_chunks, dates, "ok", None, key

    if mode == "legacy":
        return _load_legacy(
            raw_bytes,
            kind=kind,
            source=source,
            file_sha256=file_sha256,
            file_date=file_date,
            cache=cache,
        )

    if mode == "markitdown" and _markitdown_import_blocked():
        return [], [], "unsupported", "markitdown not installed", ""
    if mode == "direct":
        blocked_err = _direct_import_blocked(kind)
        if blocked_err is not None:
            return [], [], "unsupported", blocked_err, ""

    key = _canonical_cache_key_for(file_sha256, kind, mode)
    if cache is not None and cache.enabled:
        cached_payload = cache.get(key)
        if cached_payload is not None:
            parsed_cache = _parse_cached_canonical(cached_payload, kind=kind, mode=mode)
            if parsed_cache is not None:
                cached_text, cached_converter = parsed_cache
                file_chunks, dates = _build_canonical_chunks(
                    cached_text,
                    kind=kind,
                    source=source,
                    file_sha256=file_sha256,
                    file_date=file_date,
                    converter_label=cached_converter,
                )
                if file_chunks:
                    return file_chunks, dates, "ok", None, key

    if mode == "identity":
        loaded_text, status, error = decode_and_load(raw_bytes, kind)
        if status != "ok":
            return [], [], status, error, ""
        canonical_text = canonical.neutralize_markers(loaded_text)
        converter_label = "identity"
    else:
        res = _convert_canonical(raw_bytes, kind, mode)
        if res.status != "ok":
            return [], [], res.status, res.error, ""
        canonical_text = res.text
        converter_label = _format_converter_label(res)
        if (
            len(canonical_text) > canonical.MAX_CANONICAL_CHARS
            or not _is_valid_converter_label(converter_label, mode, kind=kind)
            or not _has_valid_cache_markers(canonical_text, kind=kind)
        ):
            return [], [], "unreadable", "isolated converter returned invalid result", ""

    file_chunks, dates = _build_canonical_chunks(
        canonical_text,
        kind=kind,
        source=source,
        file_sha256=file_sha256,
        file_date=file_date,
        converter_label=converter_label,
    )
    if not file_chunks:
        return [], [], "empty", "no text to index", ""

    if cache is not None and cache.enabled:
        cache.put(key, _format_cached_canonical(converter_label, canonical_text))
    return file_chunks, dates, "ok", None, key


class DocIndex:
    """Stat-cached, incremental document index over a local folder."""

    def __init__(
        self,
        folder: Path | str,
        retriever_factory: Callable[[], Retriever] = Bm25Retriever,
        *,
        converters: Mapping[str, str] | None = None,
        canonical_cache_dir: Path | str | None | _DefaultCacheDirType = _DEFAULT_CACHE_DIR,
    ) -> None:
        self.folder = Path(folder)
        self._retriever_factory = retriever_factory
        self._retriever: Retriever = retriever_factory()
        self._converters = _resolve_converters(converters)
        resolved_cache_dir: Path | str | None
        if isinstance(canonical_cache_dir, _DefaultCacheDirType):
            resolved_cache_dir = self.folder / ".rag_cache" / "canonical"
        else:
            resolved_cache_dir = canonical_cache_dir
        self._canonical_cache = canonical_cache.CanonicalCache(resolved_cache_dir, self.folder)
        self._entries: dict[str, _IndexedEntry] = {}
        self._chunks: list[Chunk] = []
        # Serialises refresh() callers: a second caller blocks until the first finishes.
        self._refresh_lock = threading.Lock()
        # Guards the published state (_entries, _chunks, _retriever) only; never held while parsing.
        self._lock = threading.RLock()
        self._parsing: str | None = None

    @property
    def converters(self) -> dict[str, str]:
        """Configured per-format converter mapping for this index."""
        return dict(self._converters)

    @property
    def canonical_cache(self) -> canonical_cache.CanonicalCache:
        """On-disk canonical Markdown cache for this index."""
        return self._canonical_cache

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        """Current indexed chunks in deterministic file/position order."""
        with self._lock:
            return tuple(self._chunks)

    @property
    def retriever(self) -> Retriever:
        """Active retrieval backend for this index."""
        with self._lock:
            return self._retriever

    def parsing(self) -> str | None:
        """Relative name of the file an in-flight ``refresh()`` is reading, else ``None``."""
        return self._parsing

    def records(self) -> list[FileRecord]:
        """Return metadata records for all tracked files in deterministic order."""
        with self._lock:
            entries = self._entries
        return [entries[key].record for key in sorted(entries)]

    def failures(self) -> list[tuple[str, str, str]]:
        """Return ``(name, status, error)`` for all currently failing files."""
        with self._lock:
            entries = self._entries
        return [
            (entry.record.name, entry.record.status, entry.record.error or "")
            for key in sorted(entries)
            if (entry := entries[key]).record.status != "ok"
        ]

    def refresh(self) -> RefreshReport:
        """Scan ``folder`` via stat, re-hashing and re-chunking only modified files.

        Concurrent callers are serialised; readers keep seeing the last published snapshot.
        """
        with self._refresh_lock:
            try:
                return self._refresh_locked()
            finally:
                self._parsing = None

    def _refresh_locked(self) -> RefreshReport:
        report = RefreshReport()
        files, too_deep_dirs = _walk_folder(self.folder)
        seen_keys: set[str] = set()
        new_entries: dict[str, _IndexedEntry] = {}
        index_dirty = False
        now_iso = datetime.now(UTC).isoformat(timespec="seconds")

        def _record_failure(
            *,
            rel_key: str,
            path_str: str,
            name: str,
            kind: str,
            status: str,
            error: str,
            prev: _IndexedEntry | None,
            mtime: float = 0.0,
            mtime_ns: int = 0,
            ctime_ns: int = 0,
            size: int = 0,
            sha256: str = "",
        ) -> None:
            nonlocal index_dirty
            if prev is not None and prev.record.status == "ok":
                report.removed.append(name)
                index_dirty = True
            rec = FileRecord(
                path=path_str,
                name=name,
                mtime=mtime,
                ingested_at=now_iso,
                sha256=sha256,
                size=size,
                kind=kind,
                dates_in_text=[],
                status=status,
                error=error,
            )
            new_entries[rel_key] = _IndexedEntry(
                rec, size=size, mtime_ns=mtime_ns, ctime_ns=ctime_ns, chunks=[]
            )
            report.failures.append((name, status, error))

        for dir_path in too_deep_dirs:
            dir_name = f"{dir_path.relative_to(self.folder).as_posix()}/"
            seen_keys.add(dir_name)
            _record_failure(
                rel_key=dir_name,
                path_str=str(dir_path),
                name=dir_name,
                kind="dir",
                status="unreadable",
                error=f"folder nested deeper than {MAX_FOLDER_DEPTH} levels, not scanned",
                prev=self._entries.get(dir_name),
            )

        for file_path in files:
            rel_key = file_path.relative_to(self.folder).as_posix()
            seen_keys.add(rel_key)
            name = rel_key
            ext = file_path.suffix.lower()
            kind = _normalize_format(ext) or "unknown"
            prev = self._entries.get(rel_key)

            # Stat-based unchanged shortcut
            try:
                stat_res = file_path.stat()
                size = stat_res.st_size
                mtime_ns = stat_res.st_mtime_ns
                ctime_ns = stat_res.st_ctime_ns
                mtime, _ = _safe_file_date(stat_res.st_mtime)
            except OSError as exc:
                _record_failure(
                    rel_key=rel_key,
                    path_str=str(file_path),
                    name=name,
                    kind=kind,
                    status="unreadable",
                    error=f"could not be read ({type(exc).__name__})",
                    prev=prev,
                )
                continue

            if (
                prev is not None
                and prev.size == size
                and prev.mtime_ns == mtime_ns
                and prev.ctime_ns == ctime_ns
            ):
                new_entries[rel_key] = prev
                if prev.record.status == "ok":
                    report.unchanged += 1
                else:
                    report.failures.append(
                        (prev.record.name, prev.record.status, prev.record.error or "")
                    )
                continue

            self._parsing = name
            lf = load_file(
                file_path,
                rel_key,
                converters=self._converters,
                canonical_cache=self._canonical_cache,
                prev=prev,
                now_iso=now_iso,
            )

            if lf.unchanged:
                assert prev is not None
                new_entries[rel_key] = _IndexedEntry(
                    FileRecord(
                        path=str(file_path),
                        name=name,
                        mtime=lf.mtime,
                        ingested_at=now_iso,
                        sha256=lf.sha256,
                        size=lf.size,
                        kind=kind,
                        dates_in_text=prev.record.dates_in_text,
                        status="ok",
                        error=prev.record.error,
                    ),
                    size=lf.size,
                    mtime_ns=lf.mtime_ns,
                    ctime_ns=lf.ctime_ns,
                    chunks=lf.chunks,
                    canonical_key=lf.canonical_key,
                )
                report.unchanged += 1
                continue

            if lf.status != "ok":
                _record_failure(
                    rel_key=rel_key,
                    path_str=str(file_path),
                    name=name,
                    kind=kind,
                    status=lf.status,
                    error=lf.error or "",
                    prev=prev,
                    mtime=lf.mtime,
                    mtime_ns=lf.mtime_ns,
                    ctime_ns=lf.ctime_ns,
                    size=lf.size,
                    sha256=lf.sha256,
                )
                continue

            new_entries[rel_key] = _IndexedEntry(
                FileRecord(
                    path=str(file_path),
                    name=name,
                    mtime=lf.mtime,
                    ingested_at=now_iso,
                    sha256=lf.sha256,
                    size=lf.size,
                    kind=kind,
                    dates_in_text=lf.dates,
                    status="ok",
                    error=lf.note,
                ),
                size=lf.size,
                mtime_ns=lf.mtime_ns,
                ctime_ns=lf.ctime_ns,
                chunks=lf.chunks,
                canonical_key=lf.canonical_key,
            )
            if prev is not None and prev.record.status == "ok":
                report.changed.append(name)
            else:
                report.added.append(name)
            index_dirty = True

        for old_key in sorted(self._entries):
            if old_key not in seen_keys:
                old_entry = self._entries[old_key]
                if old_entry.record.status == "ok":
                    report.removed.append(old_entry.record.name)
                    index_dirty = True

        self._parsing = None
        with self._lock:
            self._entries = new_entries
            if index_dirty:
                self._chunks = [
                    chunk for key in sorted(self._entries) for chunk in self._entries[key].chunks
                ]
                self._retriever = self._retriever_factory()
                self._retriever.build(self._chunks)
                if self._canonical_cache.enabled:
                    keep_keys = {
                        entry.canonical_key
                        for entry in self._entries.values()
                        if entry.record.status == "ok" and entry.canonical_key
                    }
                    self._canonical_cache.prune(keep_keys)

        return report

    def search(
        self,
        query: str,
        k: int = 4,
        *,
        per_file_cap: int | None = None,
        adaptive_cap: bool = False,
        whole_files: int = 0,
        whole_max_chunks: int = 40,
    ) -> list[dict[str, Any]]:
        """Search indexed chunks and return at most ``k`` dicts best-first."""
        with self._lock:
            return self._search_locked(
                query,
                k,
                per_file_cap=per_file_cap,
                adaptive_cap=adaptive_cap,
                whole_files=whole_files,
                whole_max_chunks=whole_max_chunks,
            )

    def _search_locked(
        self,
        query: str,
        k: int,
        *,
        per_file_cap: int | None = None,
        adaptive_cap: bool = False,
        whole_files: int = 0,
        whole_max_chunks: int = 40,
    ) -> list[dict[str, Any]]:
        """Rank under ``_lock``: fts5/hybrid factories share one instance that build mutates."""
        if k <= 0 or not self._chunks or not query.strip():
            return []

        q_tokens = set(tokenize(query))
        ranked = self._retriever.rank(query, len(self._chunks))
        valid_ranked = [
            (self._chunks[idx], score) for idx, score in ranked if 0 <= idx < len(self._chunks)
        ]
        if not valid_ranked:
            return []

        results, _ = select_results(
            valid_ranked,
            k,
            q_tokens,
            per_file_cap=per_file_cap,
            adaptive_cap=adaptive_cap,
            whole_files=whole_files,
            whole_max_chunks=whole_max_chunks,
        )
        return results


def select_results(
    ranked: Sequence[tuple[Chunk, float]],
    k: int,
    q_tokens: set[str],
    *,
    per_file_cap: int | None = None,
    adaptive_cap: bool = False,
    whole_files: int = 0,
    whole_max_chunks: int = 40,
) -> tuple[list[dict[str, Any]], bool]:
    """Select top-k chunks, applying per-file caps, dedup, and overflow.

    Returns ``(results, filled)`` where ``filled`` is True iff k chunks were selected
    without needing the overflow fallback (the ``len(selected) >= k`` break was hit).

    Options (all default to existing behaviour):
    - ``per_file_cap``: override MAX/HARD per-file cap for this call.
    - ``adaptive_cap``: a file may hold up to ``k-2`` chunks when it owns the top-ranked
      chunk AND at least 3 of the raw top-10; otherwise standard caps apply.
    - ``whole_files``: after selection, add every chunk of the top-N files (ranked by
      best chunk) that have <= ``whole_max_chunks`` chunks, in document order,
      deduplicated; result may exceed k.
    """
    if k <= 0 or not ranked:
        return [], True

    if per_file_cap is not None:
        effective_soft = per_file_cap
        effective_hard = per_file_cap
    else:
        effective_soft = MAX_CHUNKS_PER_FILE
        effective_hard = HARD_CHUNKS_PER_FILE

    # Adaptive cap: determine which file(s) may be expanded
    adaptive_source: str | None = None
    if adaptive_cap and len(ranked) >= 10:
        top10 = ranked[:10]
        top10_sources: dict[str, int] = {}
        for chunk, _ in top10:
            src = chunk.source
            top10_sources[src] = top10_sources.get(src, 0) + 1
        best_source = top10[0][0].source
        if top10_sources.get(best_source, 0) >= 3:
            adaptive_source = best_source

    best_score = max(score for _, score in ranked)
    selected: list[tuple[Chunk, float]] = []
    overflow: list[tuple[Chunk, float]] = []
    per_source: dict[str, int] = {}
    tokens_by_chunk: dict[str, set[str]] = {}

    for chunk, score in ranked:
        source = chunk.source

        # Determine effective cap for this source
        if adaptive_source == source:
            cur_soft = k - 2
            cur_hard = k - 2
        else:
            cur_soft = effective_soft
            cur_hard = effective_hard

        count = per_source.get(source, 0)
        beyond_soft = count >= cur_soft
        can_overflow = (
            count < cur_hard and best_score > 0 and score >= OVERFLOW_SCORE_RATIO * best_score
        )
        if beyond_soft and not can_overflow:
            overflow.append((chunk, score))
            continue

        chunk_key = chunk.id
        chunk_tokens = tokens_by_chunk.get(chunk_key)
        if chunk_tokens is None:
            chunk_tokens = set(tokenize(f"{chunk.header}\n{chunk.text}"))
            tokens_by_chunk[chunk_key] = chunk_tokens

        is_cross_dup = False
        for sel_chunk, _ in selected:
            if sel_chunk.source == source:
                continue
            sel_tokens = tokens_by_chunk[sel_chunk.id]
            union_len = len(chunk_tokens | sel_tokens)
            if union_len > 0 and (len(chunk_tokens & sel_tokens) / union_len) >= DEDUP_JACCARD:
                is_cross_dup = True
                break
        if is_cross_dup:
            continue

        selected.append((chunk, score))
        per_source[source] = count + 1
        if len(selected) >= k:
            break

    filled = len(selected) >= k
    if not filled and overflow:
        selected.extend(overflow[: k - len(selected)])

    results: list[dict[str, Any]] = []
    for chunk, score in selected[:k]:
        if q_tokens:
            chunk_key = chunk.id
            chunk_tokens = tokens_by_chunk.get(chunk_key)
            if chunk_tokens is None:
                chunk_tokens = set(tokenize(f"{chunk.header}\n{chunk.text}"))
                tokens_by_chunk[chunk_key] = chunk_tokens
            coverage = len(q_tokens & chunk_tokens) / len(q_tokens)
        else:
            coverage = 0.0
        results.append(
            {
                "source": chunk.source,
                "id": chunk.id,
                "section": chunk.section,
                "page": chunk.page,
                "file_date": chunk.file_date,
                "text": chunk.text,
                "score": float(score),
                "coverage": float(coverage),
            }
        )

    # whole_files: add every chunk of the top-N files (by best chunk rank) that have
    # <= whole_max_chunks total chunks, in document order, deduplicated
    if whole_files > 0:
        all_chunks = [chunk for chunk, _ in ranked]
        source_chunks: dict[str, list[Chunk]] = {}
        for chunk in all_chunks:
            source_chunks.setdefault(chunk.source, []).append(chunk)

        # Determine file order by best (highest-score) chunk rank in the full ranking
        best_rank_by_source: dict[str, int] = {}
        for rank_idx, (chunk, _) in enumerate(ranked):
            src = chunk.source
            if src not in best_rank_by_source:
                best_rank_by_source[src] = rank_idx
        file_ranking = sorted(best_rank_by_source, key=lambda s: best_rank_by_source[s])

        already_returned_ids = {r["id"] for r in results}

        for i, source in enumerate(file_ranking):
            if i >= whole_files:
                break
            chunks_for_file = source_chunks.get(source, [])
            if len(chunks_for_file) > whole_max_chunks:
                continue
            # Add in document (position) order, deduplicated
            for chunk in sorted(chunks_for_file, key=lambda c: c.position):
                if chunk.id not in already_returned_ids:
                    coverage = 0.0
                    if q_tokens:
                        c_tokens = tokens_by_chunk.get(chunk.id)
                        if c_tokens is None:
                            c_tokens = set(tokenize(f"{chunk.header}\n{chunk.text}"))
                            tokens_by_chunk[chunk.id] = c_tokens
                        coverage = len(q_tokens & c_tokens) / len(q_tokens)
                    results.append(
                        {
                            "source": chunk.source,
                            "id": chunk.id,
                            "section": chunk.section,
                            "page": chunk.page,
                            "file_date": chunk.file_date,
                            "text": chunk.text,
                            "score": float(best_score),
                            "coverage": float(coverage),
                        }
                    )
                    already_returned_ids.add(chunk.id)

    return results, filled
