"""Canonical Markdown normalization and locator-marker post-processing.

Pure functions with no I/O and no dependency on ``docindex``. All text scans
use ``str`` methods rather than unanchored regular expressions so untrusted
inputs cannot trigger catastrophic backtracking.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

POSTPROC_VERSION = 2
MAX_CANONICAL_CHARS = 24_000_000
# Mirrors docindex.MAX_CSV_ROWS without importing docindex.
MAX_CSV_ROWS = 50_000
TABLE_LINE_MAX = 800
TABLE_TRUNCATED_NOTE = f"[table truncated: only the first {MAX_CSV_ROWS} rows are indexed]"
MAX_SHEET_NAME_CHARS = 120
MAX_MARKER_NUMBER = 100_000
_MAX_MARKER_DIGITS = 7

_MARKER_PREFIX = "<!-- rag:"
_MARKER_SUFFIX = " -->"
_PPTX_SLIDE_PREFIX = "<!-- Slide number: "
_LIGATURE_TABLE: dict[int, str] = {
    cp: unicodedata.normalize("NFKC", chr(cp)) for cp in range(0xFB00, 0xFB07)
}
_VALID_MARKER_KEYS = frozenset({"page", "slide", "sheet", "converter"})


@dataclass(frozen=True)
class PostprocessResult:
    """Result of canonical post-processing and any dropped header/footer lines."""

    text: str
    dropped_lines: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Unit:
    """One logical unit (page, slide, sheet, or markerless body) split from canonical text."""

    page: int | None = None
    slide: int | None = None
    sheet: str | None = None
    body: str = ""


def clean_sheet_name(name: str) -> str:
    """Sanitize a spreadsheet name for a ``<!-- rag:sheet=... -->`` marker."""
    cleaned = (
        name.replace("\r", "")
        .replace("\n", "")
        .replace("-->", "")
        .replace("<", "")
        .replace(">", "")
        .strip()
    )
    return cleaned[:MAX_SHEET_NAME_CHARS].strip()


def neutralize_markers(text: str) -> str:
    """Drop any lines whose left-stripped content starts with ``<!-- rag:``."""
    lines = text.split("\n")
    kept = [line for line in lines if not line.lstrip().startswith(_MARKER_PREFIX)]
    if len(kept) == len(lines):
        return text
    return "\n".join(kept)


def _trim_empty_lines(lines: list[str]) -> list[str]:
    """Return ``lines`` with leading and trailing empty strings removed in linear time."""
    start = 0
    end = len(lines)
    while start < end and lines[start] == "":
        start += 1
    while end > start and lines[end - 1] == "":
        end -= 1
    return lines[start:end]


def _parse_bounded_positive_int(val: str) -> int | None:
    """Parse an ASCII digit string in ``1..MAX_MARKER_NUMBER`` without ``ValueError``."""
    if not val or len(val) > _MAX_MARKER_DIGITS:
        return None
    if not (val.isascii() and val.isdigit()):
        return None
    num = int(val)
    if num < 1 or num > MAX_MARKER_NUMBER:
        return None
    return num


def _parse_marker_line(line: str) -> tuple[str, str] | None:
    """Parse a single stripped line if it is a well-formed ``<!-- rag:... -->`` marker."""
    if not (line.startswith(_MARKER_PREFIX) and line.endswith(_MARKER_SUFFIX)):
        return None
    inner = line[len(_MARKER_PREFIX) : -len(_MARKER_SUFFIX)]
    if "=" not in inner:
        return None
    key, val = inner.split("=", 1)
    if key not in _VALID_MARKER_KEYS:
        return None
    if "-->" in val or "\r" in val or "\n" in val:
        return None
    if key in ("page", "slide") and _parse_bounded_positive_int(val) is None:
        return None
    return key, val


def _is_marker_or_prefix(line: str) -> bool:
    """Return True if ``line`` is a marker or starts with the canonical marker prefix."""
    if line.lstrip().startswith(_MARKER_PREFIX):
        return True
    return _parse_marker_line(line.strip(" \t")) is not None


def strip_markers(text: str) -> str:
    """Remove canonical ``<!-- rag:... -->`` marker lines from text."""
    lines = text.split("\n")
    kept: list[str] = []
    saw_marker = False
    for line in lines:
        if _parse_marker_line(line.strip(" \t")) is not None:
            saw_marker = True
            continue
        kept.append(line)
    if not saw_marker:
        return text
    return "\n".join(_trim_empty_lines(kept))


def split_units(canonical: str) -> tuple[list[Unit], str | None]:
    """Split canonical Markdown on locator markers and extract any converter header.

    Returns ``(units, converter_label)`` with all marker lines stripped from unit bodies.
    """
    units: list[Unit] = []
    converter: str | None = None
    cur_page: int | None = None
    cur_slide: int | None = None
    cur_sheet: str | None = None
    buf: list[str] = []

    def _flush() -> None:
        body = "\n".join(_trim_empty_lines(buf))
        buf.clear()
        if body:
            units.append(Unit(page=cur_page, slide=cur_slide, sheet=cur_sheet, body=body))

    for line in canonical.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        parsed = _parse_marker_line(line.strip(" \t"))
        if parsed is None:
            buf.append(line)
            continue
        key, val = parsed
        if key == "converter":
            if converter is None:
                converter = val
            continue
        _flush()
        cur_page = int(val) if key == "page" else None
        cur_slide = int(val) if key == "slide" else None
        cur_sheet = val if key == "sheet" else None

    _flush()
    return units, converter


def fix_ligatures(text: str) -> str:
    """Apply NFKC normalization strictly to Unicode ligatures U+FB00..U+FB06."""
    return text.translate(_LIGATURE_TABLE)


def digit_runs(text: str) -> list[str]:
    """Return contiguous runs of Unicode ``Nd`` characters mapped to ASCII digits."""
    runs: list[str] = []
    buf: list[str] = []
    for ch in text:
        if unicodedata.category(ch) == "Nd":
            buf.append(str(unicodedata.digit(ch)))
        elif buf:
            runs.append("".join(buf))
            buf.clear()
    if buf:
        runs.append("".join(buf))
    return runs


def _collapse_line_ws(line: str) -> str:
    """Collapse runs of ASCII space/tab to a single space and strip trailing spaces/tabs."""
    if " " not in line and "\t" not in line:
        return line
    normalized = line.replace("\t", " ")
    has_leading = normalized.startswith(" ")
    collapsed = " ".join(part for part in normalized.split(" ") if part)
    if not collapsed:
        return ""
    return (" " + collapsed) if has_leading else collapsed


def _collapse_blank_lines(lines: Sequence[str]) -> list[str]:
    """Trim outer empty lines and collapse 3+ consecutive newlines to 2."""
    collapsed: list[str] = []
    empty_run = 0
    for line in lines:
        if line == "":
            empty_run += 1
            if empty_run <= 1:
                collapsed.append("")
        else:
            empty_run = 0
            collapsed.append(line)
    return _trim_empty_lines(collapsed)


def _normalize_pdf_page(raw_page: str) -> list[str]:
    """Clean a single PDF page: markers, soft hyphens, ligatures, dehyphenation, whitespace."""
    cleaned = neutralize_markers(raw_page.replace("\r\n", "\n").replace("\r", "\n"))
    cleaned = fix_ligatures(cleaned.replace("\u00ad", ""))
    ws_lines = [_collapse_line_ws(line) for line in cleaned.split("\n")]

    merged: list[str] = []
    for line in ws_lines:
        if (
            merged
            and len(merged[-1]) >= 2
            and merged[-1][-1] == "-"
            and merged[-1][-2].islower()
            and line
            and line[0].islower()
        ):
            merged[-1] = merged[-1][:-1] + line
        else:
            merged.append(line)

    safe_lines = [line for line in merged if not _is_marker_or_prefix(line)]
    return _collapse_blank_lines(safe_lines)


def _header_footer_signature(line: str) -> str:
    """Map every Unicode Nd digit in a trimmed line to ``#`` for header/footer matching."""
    return "".join("#" if unicodedata.category(ch) == "Nd" else ch for ch in line.strip(" \t"))


def postprocess_pdf_pages(pages: Sequence[str]) -> PostprocessResult:
    """Post-process per-page PDF text, drop repeated headers/footers, and insert page markers."""
    page_lines = [_normalize_pdf_page(page) for page in pages]
    dropped_lines: list[str] = []

    if len(pages) >= 3:
        threshold = max(3, math.ceil(0.6 * len(pages)))
        page_candidates: list[set[int]] = []
        sig_counts: Counter[str] = Counter()

        for lines in page_lines:
            non_empty = [idx for idx, line in enumerate(lines) if line != ""]
            candidates = set(non_empty[:2]) | set(non_empty[-2:])
            page_candidates.append(candidates)
            sigs_on_page = {_header_footer_signature(lines[idx]) for idx in candidates}
            for sig in sigs_on_page:
                sig_counts[sig] += 1

        repeated_sigs = {sig for sig, count in sig_counts.items() if count >= threshold}
        if repeated_sigs:
            filtered_pages: list[list[str]] = []
            for lines, candidates in zip(page_lines, page_candidates, strict=True):
                kept: list[str] = []
                for idx, line in enumerate(lines):
                    if idx in candidates and _header_footer_signature(line) in repeated_sigs:
                        dropped_lines.append(line)
                    else:
                        kept.append(line)
                filtered_pages.append(_collapse_blank_lines(kept))
            page_lines = filtered_pages

    blocks: list[str] = []
    for page_num, lines in enumerate(page_lines, start=1):
        if lines:
            body = "\n".join(lines)
            blocks.append(f"<!-- rag:page={page_num} -->\n{body}")

    return PostprocessResult(text="\n\n".join(blocks), dropped_lines=dropped_lines)


def _fence_transition(line: str, fence_char: str | None, fence_len: int) -> tuple[str | None, int]:
    """Update fenced-code state (````` or ``~~~``) for one line."""
    stripped = line.lstrip(" \t")
    if not stripped.startswith(("```", "~~~")):
        return fence_char, fence_len
    ch = stripped[0]
    run_len = len(stripped) - len(stripped.lstrip(ch))
    rest = stripped[run_len:]
    if fence_char is None:
        if ch == "`" and "`" in rest:
            return fence_char, fence_len
        return ch, run_len
    if ch == fence_char and run_len >= fence_len and not rest.strip(" \t"):
        return None, 0
    return fence_char, fence_len


def _parse_pptx_slide_number(line: str, *, expected: int | None = None) -> int | None:
    """Return slide number if ``line`` is an exact sequential ``<!-- Slide number: N -->``."""
    stripped = line.strip(" \t")
    if not (stripped.startswith(_PPTX_SLIDE_PREFIX) and stripped.endswith(_MARKER_SUFFIX)):
        return None
    num_str = stripped[len(_PPTX_SLIDE_PREFIX) : -len(_MARKER_SUFFIX)]
    num = _parse_bounded_positive_int(num_str)
    if num is None:
        return None
    if expected is not None and num != expected:
        return None
    return num


def _split_pipe_row(line: str) -> list[str]:
    """Split a Markdown pipe-table row into trimmed cells, respecting ``\\|`` escapes."""
    stripped = line.strip(" \t")
    if "\\" not in stripped:
        parts = stripped.split("|")
    else:
        parts = []
        cur: list[str] = []
        i = 0
        n = len(stripped)
        while i < n:
            ch = stripped[i]
            if ch == "\\" and i + 1 < n and stripped[i + 1] == "\\":
                cur.append("\\\\")
                i += 2
            elif ch == "\\" and i + 1 < n and stripped[i + 1] == "|":
                cur.append("|")
                i += 2
            elif ch == "|":
                parts.append("".join(cur))
                cur.clear()
                i += 1
            else:
                cur.append(ch)
                i += 1
        parts.append("".join(cur))

    if len(parts) >= 2 and parts[0].strip(" \t") == "" and parts[-1].strip(" \t") == "":
        parts = parts[1:-1]
    elif parts and parts[0].strip(" \t") == "" and stripped.startswith("|"):
        parts = parts[1:]
    elif parts and parts[-1].strip(" \t") == "" and stripped.endswith("|"):
        parts = parts[:-1]
    return [part.strip(" \t") for part in parts]


def _is_separator_row(cells: Sequence[str]) -> bool:
    """Return True if every cell is a Markdown table separator (``---``, ``:---``, etc.)."""
    if not cells:
        return False
    for cell in cells:
        token = cell.strip(" \t")
        if not token:
            return False
        dashes = token.removeprefix(":").removesuffix(":")
        if not dashes or set(dashes) != {"-"}:
            return False
    return True


def _is_unnamed_header(cell: str) -> bool:
    """Return True if ``cell`` matches ``Unnamed: <ASCII digits>`` after whitespace collapse."""
    normalized = _collapse_line_ws(cell).strip(" \t")
    if not normalized.startswith("Unnamed: "):
        return False
    suffix = normalized[len("Unnamed: ") :]
    return bool(suffix) and suffix.isascii() and suffix.isdigit()


def _is_blank_data_cell(cell: str) -> bool:
    """Return True if a table cell is empty, NaN, or an ``Unnamed: N`` placeholder."""
    val = _collapse_line_ws(cell).strip(" \t")
    return val in ("", "NaN", "nan") or _is_unnamed_header(val)


def _prepare_table_headers(
    header_cells: Sequence[str],
    first_row_cells: Sequence[str] | None,
    *,
    kind: str,
) -> tuple[list[str], bool, str | None]:
    """Normalize pipe-table headers, promoting xlsx row 1 when headers are mostly blank."""
    if kind not in ("xlsx", "xls"):
        clean = [
            _collapse_line_ws(col).strip(" \t") or f"col_{idx + 1}"
            for idx, col in enumerate(header_cells)
        ]
        return clean, False, None

    normalized = [_collapse_line_ws(col).strip(" \t") for col in header_cells]
    blank_flags = [not col or _is_unnamed_header(col) for col in normalized]
    blank_count = sum(1 for is_blank in blank_flags if is_blank)

    if (
        first_row_cells is not None
        and blank_count * 2 >= len(header_cells)
        and len(first_row_cells) == len(header_cells)
        and not any(_is_blank_data_cell(cell) for cell in first_row_cells)
    ):
        title = None if blank_flags[0] else normalized[0]
        promoted = [
            _collapse_line_ws(col).strip(" \t") or f"col_{idx + 1}"
            for idx, col in enumerate(first_row_cells)
        ]
        return promoted, True, title

    clean = [
        f"col_{idx + 1}" if is_blank else col
        for idx, (col, is_blank) in enumerate(zip(normalized, blank_flags, strict=True))
    ]
    return clean, False, None


def _row_line_needs_escape(line: str, *, markers: bool) -> bool:
    if markers and line.startswith(("<!-- rag:", "<!-- Slide number:")):
        return True
    for ch in "`~":
        if line.startswith(ch * 3):
            return True
    hashes = len(line) - len(line.lstrip("#"))
    return hashes > 0 and line[hashes : hashes + 1] in (" ", "\t")


def _format_table_row(
    headers: Sequence[str], cells: Sequence[str], *, escape_markers: bool = True
) -> str:
    """Format one pipe-table row as ``header: value; ...``, dropping empty and NaN cells."""
    pairs: list[str] = []
    for idx, raw_val in enumerate(cells):
        val = _collapse_line_ws(raw_val).strip(" \t")
        if val in ("", "NaN", "nan"):
            continue
        header = headers[idx] if idx < len(headers) else f"col_{idx + 1}"
        pairs.append(f"{header}: {val}")
    line = "; ".join(pairs)
    if _row_line_needs_escape(line, markers=escape_markers):
        return f"\\{line}"
    return line


def table_lines(pairs: Sequence[tuple[str, str]], limit: int = TABLE_LINE_MAX) -> list[str]:
    """Render a row as ``header: value`` lines (<= limit chars); split rows repeat the label."""
    texts = [f"{head}: {val}" for head, val in pairs]
    joined = "; ".join(texts)
    if len(joined) <= limit:
        return [joined] if texts else []
    short = [i for i, text in enumerate(texts) if len(text) <= 200]
    if not short:
        return [joined]
    first = next((i for i in short if any(ch.isalpha() for ch in pairs[i][1])), short[0])
    label = texts.pop(first)
    room = limit - len(label)
    lines: list[str] = []
    cur: list[str] = []
    size = 0
    for text in texts:
        for start in range(0, len(text), room - 2):
            piece = text[start : start + room - 2]
            if cur and size + len(piece) + 2 > room:
                lines.append("; ".join([label, *cur]))
                cur, size = [], 0
            cur.append(piece)
            size += len(piece) + 2
    if cur:
        lines.append("; ".join([label, *cur]))
    return lines


def _format_table_lines(headers: Sequence[str], cells: Sequence[str]) -> list[str]:
    """Format one pipe-table row as ``header: value`` lines, each starting with the row label."""
    pairs: list[tuple[str, str]] = []
    for idx, raw_val in enumerate(cells):
        val = _collapse_line_ws(raw_val).strip(" \t")
        if val in ("", "NaN", "nan"):
            continue
        pairs.append((headers[idx] if idx < len(headers) else f"col_{idx + 1}", val))
    return table_lines(pairs)


def format_table_row(headers: Sequence[str], cells: Sequence[str]) -> str:
    """Format one table row as ``header: value; ...``, dropping empty and NaN cells."""
    return _format_table_row(headers, cells)


def postprocess_markdown(md: str, *, kind: str) -> PostprocessResult:
    """Normalize converter Markdown: markers, ligatures, soft hyphens, tables, and whitespace."""
    normalized = md.replace("\r\n", "\n").replace("\r", "\n")
    normalized = fix_ligatures(normalized.replace("\u00ad", ""))
    lines = normalized.split("\n")

    # Store (marker_kind, in_fence, payload) so markers are neutralized last and
    # generated markers are inserted only after neutralization.
    entries: list[tuple[str | None, bool, str]] = []
    fence_char: str | None = None
    fence_len = 0
    last_slide = 0
    i = 0
    total = len(lines)

    while i < total:
        line = lines[i]
        next_char, next_len = _fence_transition(line, fence_char, fence_len)
        if fence_char is not None or next_char is not None:
            fence_char, fence_len = next_char, next_len
            entries.append((None, True, line))
            i += 1
            continue

        if kind == "pptx":
            slide_num = _parse_pptx_slide_number(line, expected=last_slide + 1)
            if slide_num is not None:
                entries.append(("slide", False, str(slide_num)))
                last_slide = slide_num
                i += 1
                continue

        if kind in ("xlsx", "xls") and line.startswith("## "):
            cleaned_sheet = clean_sheet_name(line[3:])
            if cleaned_sheet:
                entries.append(("sheet", False, cleaned_sheet))
                i += 1
                continue

        if "|" in line and i + 1 < total and "|" in lines[i + 1]:
            header_cells = _split_pipe_row(line)
            sep_cells = _split_pipe_row(lines[i + 1])
            if (
                header_cells
                and len(sep_cells) == len(header_cells)
                and _is_separator_row(sep_cells)
            ):
                i += 2
                raw_rows: list[list[str]] = []
                row_overflow = False
                while i < total:
                    row_line = lines[i]
                    r_char, _ = _fence_transition(row_line, None, 0)
                    if (
                        row_line.strip(" \t") == ""
                        or "|" not in row_line
                        or r_char is not None
                        or (
                            kind == "pptx"
                            and _parse_pptx_slide_number(row_line, expected=last_slide + 1)
                            is not None
                        )
                        or (kind in ("xlsx", "xls") and row_line.startswith("## "))
                    ):
                        break
                    row_cells = _split_pipe_row(row_line)
                    if not _is_separator_row(row_cells):
                        if len(raw_rows) <= MAX_CSV_ROWS:
                            raw_rows.append(row_cells)
                        else:
                            row_overflow = True
                    i += 1
                clean_headers, consumed_first, pending_title = _prepare_table_headers(
                    header_cells,
                    raw_rows[0] if raw_rows else None,
                    kind=kind,
                )
                row_start = 1 if consumed_first else 0
                for row_cells in raw_rows[row_start : row_start + MAX_CSV_ROWS]:
                    # Forged marker lines are dropped later by neutralize_markers, not escaped.
                    if kind in ("xlsx", "xls"):
                        formatted_lines = _format_table_lines(clean_headers, row_cells)
                    else:
                        formatted_lines = [
                            _format_table_row(clean_headers, row_cells, escape_markers=False)
                        ]
                    for formatted in formatted_lines:
                        if formatted:
                            if pending_title is not None:
                                formatted = f"{pending_title} | {formatted}"
                                pending_title = None
                            entries.append((None, False, formatted))
                if kind in ("xlsx", "xls") and (
                    row_overflow or len(raw_rows) > row_start + MAX_CSV_ROWS
                ):
                    entries.append((None, False, TABLE_TRUNCATED_NOTE))
                continue

        entries.append((None, False, _collapse_line_ws(line)))
        i += 1

    out_lines: list[str] = []
    empty_run = 0
    for marker_kind, in_fence, payload in entries:
        if marker_kind is None:
            if _is_marker_or_prefix(payload):
                continue
            out_line = payload
        else:
            out_line = f"<!-- rag:{marker_kind}={payload} -->"

        if in_fence:
            empty_run = 0
            out_lines.append(out_line)
        elif out_line == "":
            empty_run += 1
            if empty_run <= 1:
                out_lines.append("")
        else:
            empty_run = 0
            out_lines.append(out_line)

    if fence_char is None:
        out_lines = _trim_empty_lines(out_lines)

    return PostprocessResult(text="\n".join(out_lines), dropped_lines=[])
