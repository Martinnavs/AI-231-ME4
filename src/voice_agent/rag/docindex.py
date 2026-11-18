"""Document index with structure-aware chunking, BM25 retrieval, and stat-based refresh."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import warnings
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

from doc_chat.corpus import tokenize

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_CSV_ROWS = 2000
MAX_PDF_PAGES = 500
MAX_DOCX_MEMBER_BYTES = MAX_FILE_BYTES
MAX_DOCX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_DOCX_COMPRESSION_RATIO = 100.0
MAX_DOCX_MEMBERS = 1000
MAX_DOCX_XML_DEPTH = 100
MAX_FOLDER_DEPTH = 20
MAX_HEADING_LINE_CHARS = 300
MAX_HEADING_TITLE_CHARS = 120
MAX_SECTION_PATH_CHARS = 300
CHUNK_SIZE = 900
CHUNK_OVERLAP = int(CHUNK_SIZE * 0.15)
MAX_REPLACEMENT_RATIO = 0.30
MAX_CHUNKS_PER_FILE = 2
HEADER_WEIGHT = 1.5
SUPPORTED_EXTENSIONS = frozenset({".md", ".txt", ".csv", ".pdf", ".docx"})

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
_XML_FORBIDDEN_RE = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
_XML_FORBIDDEN_BYTES_RE = re.compile(rb"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
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


def parse_csv_text(text: str, max_rows: int = MAX_CSV_ROWS) -> str:
    """Format CSV rows as ``col: value`` lines using the first row as headers."""
    reader = csv.reader(io.StringIO(text))
    headers = next(reader, None)
    if headers is None:
        return ""
    clean_headers = [col.strip() or f"col_{idx + 1}" for idx, col in enumerate(headers)]

    lines: list[str] = []
    data_rows = 0
    for row in reader:
        if data_rows >= max_rows:
            break
        data_rows += 1
        pairs: list[str] = []
        for idx, val in enumerate(row):
            stripped = val.strip()
            if not stripped:
                continue
            col = clean_headers[idx] if idx < len(clean_headers) else f"col_{idx + 1}"
            pairs.append(f"{col}: {stripped}")
        if pairs:
            lines.append("; ".join(pairs))
    return "\n".join(lines)


def _xml_local(tag: str) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def _xml_attr(elem: ET.Element, attr_name: str) -> str | None:
    for key, value in elem.attrib.items():
        if _xml_local(key) == attr_name:
            return value
    return None


def _has_forbidden_xml_decl(xml_bytes: bytes) -> bool:
    if _XML_FORBIDDEN_BYTES_RE.search(xml_bytes):
        return True
    if b"\x00" in xml_bytes and _XML_FORBIDDEN_BYTES_RE.search(xml_bytes.replace(b"\x00", b"")):
        return True
    if xml_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = xml_bytes.decode("utf-16", errors="replace")
    else:
        text = xml_bytes.decode("utf-8", errors="replace")
    return bool(_XML_FORBIDDEN_RE.search(text))


def _read_zip_member_bounded(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    max_bytes: int,
    *,
    keep: bool,
) -> bytes:
    total = 0
    chunks: list[bytes] = []
    with zf.open(info, "r") as member:
        while True:
            to_read = min(65536, (max_bytes + 1) - total)
            chunk = member.read(to_read)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(
                    f"docx member {info.filename!r} exceeds {max_bytes} uncompressed bytes"
                )
            if keep:
                chunks.append(chunk)
    return b"".join(chunks) if keep else b""


def _extract_docx_paragraphs(root: ET.Element) -> list[str]:
    lines: list[str] = []

    def _walk(node: ET.Element) -> None:
        if _xml_local(node.tag) == "p":
            style_val = ""
            for child in node:
                if _xml_local(child.tag) == "pPr":
                    for prop in child:
                        if _xml_local(prop.tag) == "pStyle":
                            style_val = (_xml_attr(prop, "val") or "").strip()
                            break
            text_parts: list[str] = []
            for sub in node.iter():
                local = _xml_local(sub.tag)
                if local == "t" and sub.text:
                    text_parts.append(sub.text)
                elif local == "tab":
                    text_parts.append(" ")
                elif local in {"br", "cr"}:
                    text_parts.append("\n")
            para_text = "".join(text_parts).strip()
            if para_text:
                heading_match = _DOCX_HEADING_RE.match(style_val)
                if heading_match:
                    level = int(heading_match.group(1))
                    single_line = " ".join(para_text.split())
                    if single_line:
                        lines.append(f"{'#' * level} {single_line}")
                else:
                    lines.append(para_text)
            return
        for child in node:
            _walk(child)

    _walk(root)
    return lines


def _parse_docx_xml_bounded(
    doc_bytes: bytes,
    max_depth: int,
) -> tuple[ET.Element | None, str | None]:
    """Parse docx XML with iterparse, aborting as soon as element depth exceeds ``max_depth``."""
    root: ET.Element | None = None
    depth = 0
    for event, elem in ET.iterparse(io.BytesIO(doc_bytes), events=("start", "end")):
        if event == "start":
            depth += 1
            if root is None:
                root = elem
            if depth > max_depth:
                return None, f"docx XML is nested deeper than {max_depth} levels"
        else:
            depth -= 1
    if root is None:
        return None, "docx XML parse error: empty document"
    return root, None


def load_docx(raw_bytes: bytes) -> tuple[str, str, str | None]:
    """Extract markdown-structured text from a ``.docx`` archive with zip/XML guards."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                infos = zf.infolist()
                if len(infos) > MAX_DOCX_MEMBERS:
                    return (
                        "",
                        "unreadable",
                        f"docx has too many zip members ({len(infos)} > {MAX_DOCX_MEMBERS})",
                    )

                seen_names: set[str] = set()
                total_declared_bytes = 0
                doc_info: zipfile.ZipInfo | None = None
                for info in infos:
                    if info.filename in seen_names:
                        return (
                            "",
                            "unreadable",
                            f"duplicate zip member name: {info.filename!r}",
                        )
                    seen_names.add(info.filename)
                    if info.file_size > 0 and (
                        info.compress_size == 0
                        or (info.file_size / info.compress_size) > MAX_DOCX_COMPRESSION_RATIO
                    ):
                        return (
                            "",
                            "unreadable",
                            f"suspicious compression ratio in {info.filename!r}",
                        )
                    if info.file_size > MAX_DOCX_MEMBER_BYTES:
                        return (
                            "",
                            "unreadable",
                            f"docx member {info.filename!r} exceeds {MAX_DOCX_MEMBER_BYTES} "
                            "uncompressed bytes",
                        )
                    total_declared_bytes += max(0, info.file_size)
                    if total_declared_bytes > MAX_DOCX_TOTAL_BYTES:
                        return (
                            "",
                            "unreadable",
                            f"docx total uncompressed size exceeds {MAX_DOCX_TOTAL_BYTES} bytes",
                        )
                    if not info.is_dir() and info.filename == "word/document.xml":
                        doc_info = info

                if doc_info is None:
                    return "", "unreadable", "missing word/document.xml"

                doc_bytes = _read_zip_member_bounded(zf, doc_info, MAX_DOCX_MEMBER_BYTES, keep=True)
    except UserWarning as exc:
        if "overlap" in str(exc).lower():
            return "", "unreadable", "docx has overlapping zip entries"
        return "", "unreadable", f"invalid docx archive: {exc}"
    except zipfile.BadZipFile as exc:
        if "overlap" in str(exc).lower():
            return "", "unreadable", "docx has overlapping zip entries"
        return "", "unreadable", f"invalid docx archive: {exc}"
    except ValueError as exc:
        return "", "unreadable", str(exc)
    except (
        zipfile.LargeZipFile,
        OSError,
        EOFError,
        zlib.error,
        RuntimeError,
    ) as exc:
        return "", "unreadable", f"invalid docx archive: {exc}"

    if _has_forbidden_xml_decl(doc_bytes):
        return "", "unreadable", "docx XML contains forbidden DOCTYPE or ENTITY declaration"

    try:
        root, depth_error = _parse_docx_xml_bounded(doc_bytes, MAX_DOCX_XML_DEPTH)
    except ET.ParseError as exc:
        return "", "unreadable", f"docx XML parse error: {exc}"
    if depth_error is not None or root is None:
        return "", "unreadable", depth_error or "docx XML parse error"

    paragraphs = _extract_docx_paragraphs(root)
    if not paragraphs:
        return "", "empty", "docx contains no text"
    return "\n\n".join(paragraphs), "ok", None


def load_pdf_pages(raw_bytes: bytes) -> tuple[list[tuple[int, str]], str, str | None]:
    """Extract 1-based ``(page_num, text)`` units from PDF bytes via ``pypdf``."""
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
        for page_num, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            stripped = extracted.strip()
            if stripped:
                pages.append((page_num, stripped))
    except Exception as exc:
        msg = str(exc).strip() or type(exc).__name__
        return [], "unreadable", f"pdf parse error: {msg}"

    if not pages:
        return [], "unreadable", "pdf has no text layer"
    return pages, "ok", None


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
) -> list[Chunk]:
    """Split loaded text into structure-aware ``Chunk`` records."""
    units: list[tuple[str, int | None, str]]
    if kind == "pdf" and pages is not None:
        units = [("", page_num, body) for page_num, body in pages if body.strip()]
    elif kind == "md":
        units = [(section, None, body) for section, body in split_markdown_sections(text)]
    elif kind == "docx":
        units = [
            (section, None, body)
            for section, body in split_markdown_sections(text, max_heading_level=9)
        ]
    else:
        stripped = text.strip()
        units = [("", None, stripped)] if stripped else []

    chunks: list[Chunk] = []
    position = 0
    for section, page, body in units:
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
                )
            )
            position += 1
    return chunks


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


class DocIndex:
    """Stat-cached, incremental document index over a local folder."""

    def __init__(
        self,
        folder: Path | str,
        retriever_factory: Callable[[], Retriever] = Bm25Retriever,
    ) -> None:
        self.folder = Path(folder)
        self._retriever_factory = retriever_factory
        self._retriever: Retriever = retriever_factory()
        self._entries: dict[str, _IndexedEntry] = {}
        self._chunks: list[Chunk] = []

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        """Current indexed chunks in deterministic file/position order."""
        return tuple(self._chunks)

    @property
    def retriever(self) -> Retriever:
        """Active retrieval backend for this index."""
        return self._retriever

    def records(self) -> list[FileRecord]:
        """Return metadata records for all tracked files in deterministic order."""
        return [self._entries[key].record for key in sorted(self._entries)]

    def failures(self) -> list[tuple[str, str, str]]:
        """Return ``(name, status, error)`` for all currently failing files."""
        return [
            (entry.record.name, entry.record.status, entry.record.error or "")
            for key in sorted(self._entries)
            if (entry := self._entries[key]).record.status != "ok"
        ]

    def refresh(self) -> RefreshReport:
        """Scan ``folder`` via stat, re-hashing and re-chunking only modified files."""
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
            kind = ext.lstrip(".") or "unknown"
            prev = self._entries.get(rel_key)

            size = 0
            mtime_ns = 0
            ctime_ns = 0
            mtime = 0.0
            sha256 = ""
            try:
                stat_res = file_path.stat()
                size = stat_res.st_size
                mtime_ns = stat_res.st_mtime_ns
                ctime_ns = stat_res.st_ctime_ns
                mtime, file_date = _safe_file_date(stat_res.st_mtime)

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

                if ext not in SUPPORTED_EXTENSIONS:
                    _record_failure(
                        rel_key=rel_key,
                        path_str=str(file_path),
                        name=name,
                        kind=kind,
                        status="unsupported",
                        error=f"unsupported file extension: {ext or '(none)'}",
                        prev=prev,
                        mtime=mtime,
                        mtime_ns=mtime_ns,
                        ctime_ns=ctime_ns,
                        size=size,
                    )
                    continue

                if size > MAX_FILE_BYTES:
                    _record_failure(
                        rel_key=rel_key,
                        path_str=str(file_path),
                        name=name,
                        kind=kind,
                        status="too_large",
                        error=f"file size {size} bytes exceeds limit of {MAX_FILE_BYTES} bytes",
                        prev=prev,
                        mtime=mtime,
                        mtime_ns=mtime_ns,
                        ctime_ns=ctime_ns,
                        size=size,
                    )
                    continue

                raw_bytes = file_path.read_bytes()
                sha256 = hash_bytes(raw_bytes)
                if prev is not None and prev.record.status == "ok" and prev.record.sha256 == sha256:
                    updated_rec = FileRecord(
                        path=str(file_path),
                        name=name,
                        mtime=mtime,
                        ingested_at=now_iso,
                        sha256=sha256,
                        size=size,
                        kind=kind,
                        dates_in_text=prev.record.dates_in_text,
                        status="ok",
                        error=None,
                    )
                    new_entries[rel_key] = _IndexedEntry(
                        updated_rec,
                        size=size,
                        mtime_ns=mtime_ns,
                        ctime_ns=ctime_ns,
                        chunks=prev.chunks,
                    )
                    report.unchanged += 1
                    continue

                pdf_pages: list[tuple[int, str]] | None = None
                if kind == "pdf":
                    pdf_pages, status, error = load_pdf_pages(raw_bytes)
                    loaded_text = "\n\n".join(page_text for _, page_text in pdf_pages)
                else:
                    loaded_text, status, error = decode_and_load(raw_bytes, kind)
                if status != "ok":
                    _record_failure(
                        rel_key=rel_key,
                        path_str=str(file_path),
                        name=name,
                        kind=kind,
                        status=status,
                        error=error or "",
                        prev=prev,
                        mtime=mtime,
                        mtime_ns=mtime_ns,
                        ctime_ns=ctime_ns,
                        size=size,
                        sha256=sha256,
                    )
                    continue

                dates = extract_dates(loaded_text)
                file_chunks = chunk_text(
                    loaded_text,
                    kind=kind,
                    source=name,
                    file_sha256=sha256,
                    file_date=file_date,
                    pages=pdf_pages,
                )
                if not file_chunks:
                    _record_failure(
                        rel_key=rel_key,
                        path_str=str(file_path),
                        name=name,
                        kind=kind,
                        status="empty",
                        error="no text to index",
                        prev=prev,
                        mtime=mtime,
                        mtime_ns=mtime_ns,
                        ctime_ns=ctime_ns,
                        size=size,
                        sha256=sha256,
                    )
                    continue
            except Exception as exc:
                _record_failure(
                    rel_key=rel_key,
                    path_str=str(file_path),
                    name=name,
                    kind=kind,
                    status="unreadable",
                    error=f"could not be read ({type(exc).__name__})",
                    prev=prev,
                    mtime=mtime,
                    mtime_ns=mtime_ns,
                    ctime_ns=ctime_ns,
                    size=size,
                    sha256=sha256,
                )
                continue

            rec = FileRecord(
                path=str(file_path),
                name=name,
                mtime=mtime,
                ingested_at=now_iso,
                sha256=sha256,
                size=size,
                kind=kind,
                dates_in_text=dates,
                status="ok",
                error=None,
            )
            new_entries[rel_key] = _IndexedEntry(
                rec, size=size, mtime_ns=mtime_ns, ctime_ns=ctime_ns, chunks=file_chunks
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

        self._entries = new_entries
        if index_dirty:
            self._chunks = [
                chunk for key in sorted(self._entries) for chunk in self._entries[key].chunks
            ]
            self._retriever = self._retriever_factory()
            self._retriever.build(self._chunks)

        return report

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """Search indexed chunks and return at most ``k`` dicts best-first."""
        if k <= 0 or not self._chunks or not query.strip():
            return []

        q_tokens = set(tokenize(query))
        ranked = self._retriever.rank(query, len(self._chunks))
        selected: list[tuple[int, float]] = []
        overflow: list[tuple[int, float]] = []
        per_source: dict[str, int] = {}
        for idx, score in ranked:
            if idx < 0 or idx >= len(self._chunks):
                continue
            source = self._chunks[idx].source
            count = per_source.get(source, 0)
            if count < MAX_CHUNKS_PER_FILE:
                selected.append((idx, score))
                per_source[source] = count + 1
                if len(selected) >= k:
                    break
            else:
                overflow.append((idx, score))

        if len(selected) < k and overflow:
            selected.extend(overflow[: k - len(selected)])

        results: list[dict[str, Any]] = []
        for idx, score in selected[:k]:
            chunk = self._chunks[idx]
            if q_tokens:
                chunk_tokens = set(tokenize(f"{chunk.header}\n{chunk.text}"))
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
        return results
