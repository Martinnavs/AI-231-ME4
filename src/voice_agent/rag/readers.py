"""Direct pure-Python readers for XLSX, PPTX, HTML, and EPUB.

All readers use lazy imports so environments without optional packages report
``status="unsupported"`` without import-time failure. They emit raw Markdown
structured for ``voice_agent.rag.canonical.postprocess_markdown`` via
``convert.convert_with_direct``.
"""

from __future__ import annotations

import codecs
import datetime as dt
import importlib.metadata
import io
import math
import os
import posixpath
import re
import shutil
import subprocess
import threading
import urllib.parse
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voice_agent.rag import canonical, zipguard
from voice_agent.rag.convert import ConvertResult

# Spreadsheet caps: prevent memory/CPU blow-up on adversarial or huge workbooks.
MAX_XLSX_SHEETS = 64
MAX_XLSX_ROWS = 20_000
MAX_XLSX_COLS = 200
MAX_XLSX_CELL_CHARS = 4_000

# Presentation caps: bound slide count, paragraph line length, table rows, and group nesting.
MAX_PPTX_SLIDES = 1_000
MAX_PPTX_RUN_CHARS = 5_000
MAX_PPTX_TABLE_ROWS = 500
MAX_PPTX_GROUP_DEPTH = 20

# HTML and EPUB caps: bound input bytes, DOM nesting, visited nodes, table size, and spine items.
MAX_HTML_BYTES = 10 * 1024 * 1024
MAX_HTML_DEPTH = 100
MAX_HTML_NODES = 200_000
MAX_HTML_TABLE_ROWS = 20_000
MAX_HTML_TABLE_COLS = 200
MAX_HTML_TABLE_CELLS = 1_000_000
MAX_HTML_DOC_TABLE_CELLS = 2_000_000
MAX_HTML_HEADER_COLSPAN_COPY = 64
MAX_EPUB_SPINE_ITEMS = 2_000
MAX_EPUB_ITEM_BYTES = 10 * 1024 * 1024
MAX_EPUB_TOTAL_BYTES = 64 * 1024 * 1024
MAX_CATDOC_OUTPUT_BYTES = 5 * 1024 * 1024

_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_BU_NONE_TAG = f"{{{_A_NS}}}buNone"
_BU_CHAR_TAG = f"{{{_A_NS}}}buChar"
_BU_AUTONUM_TAG = f"{{{_A_NS}}}buAutoNum"


def _package_version(dist_name: str, module: Any | None = None) -> str:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        ver = getattr(module, "__version__", None) if module is not None else None
        return ver if isinstance(ver, str) and ver else "unknown"


def converter_label(kind: str) -> str:
    """Return the direct converter label for ``kind``."""
    if kind == "xlsx":
        return f"direct-openpyxl-{_package_version('openpyxl')}"
    if kind == "pptx":
        return f"direct-python-pptx-{_package_version('python-pptx')}"
    if kind in ("html", "epub"):
        return f"direct-lxml-{_package_version('lxml')}"
    if kind == "rtf":
        return f"direct-striprtf-{_package_version('striprtf')}"
    if kind == "xls":
        return f"direct-xlrd-{_package_version('xlrd')}"
    return "direct"


_UNICODE_LINE_SEPS_TABLE = str.maketrans(
    {
        "\u000b": " ",
        "\u000c": " ",
        "\u001c": " ",
        "\u001d": " ",
        "\u001e": " ",
        "\u0085": " ",
        "\u2028": " ",
        "\u2029": " ",
    }
)
_MD_HEADING_PREFIX_RE = re.compile(r"^#{1,6}[ \t]")


def _strip_unicode_line_seps(text: str) -> str:
    """Replace Unicode line/paragraph/record separators with single ASCII spaces."""
    return text.translate(_UNICODE_LINE_SEPS_TABLE)


def _neutralize_html_comments(text: str) -> str:
    """Neutralize HTML comment openers anywhere in text so markers cannot be forged."""
    return text.replace("<!--", "<\\!--")


def _escape_md_line(line: str) -> str:
    """Escape line-start headings and fence runs, and neutralize ``<!--`` anywhere."""
    if not line:
        return line
    line = _neutralize_html_comments(line)
    stripped = line.lstrip(" \t")
    if not stripped:
        return line
    leading = line[: len(line) - len(stripped)]
    if _MD_HEADING_PREFIX_RE.match(stripped) is not None or stripped.startswith(("```", "~~~")):
        return f"{leading}\\{stripped}"
    return line


def _escape_md_lines(text: str) -> str:
    if "\n" not in text:
        return _escape_md_line(text)
    return "\n".join(_escape_md_line(line) for line in text.split("\n"))


@dataclass
class _BudgetCounter:
    limit: int
    used: int = 0
    exceeded: bool = False

    def add(self, chars: int) -> bool:
        self.used += chars
        if self.used > self.limit:
            self.exceeded = True
            return False
        return True


def _render_xlsx_cell(value: Any) -> tuple[str, bool]:
    """Render one openpyxl cell value to Markdown-table-safe text and a truncation flag."""
    if value is None:
        return "", False
    if isinstance(value, bool):
        return ("TRUE" if value else "FALSE"), False
    if isinstance(value, int):
        return str(value), False
    if isinstance(value, float):
        if not math.isfinite(value):
            return "", False
        if value.is_integer() and abs(value) < 1e16:
            return str(int(value)), False
        return repr(value), False
    if isinstance(value, dt.datetime):
        if value.hour == 0 and value.minute == 0 and value.second == 0 and value.microsecond == 0:
            return value.date().isoformat(), False
        return value.isoformat(sep=" "), False
    if isinstance(value, dt.date):
        return value.isoformat(), False
    if isinstance(value, dt.time):
        return value.isoformat(), False

    text = (
        _strip_unicode_line_seps(str(value))
        .replace("\r\n", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )
    truncated = False
    if len(text) > MAX_XLSX_CELL_CHARS:
        text = text[:MAX_XLSX_CELL_CHARS]
        truncated = True
    return _neutralize_html_comments(text).replace("|", "\\|"), truncated


def _is_numeric_cell(raw_val: Any, rendered: str) -> bool:
    if raw_val is None or isinstance(raw_val, bool):
        return False
    if isinstance(raw_val, int):
        return True
    if isinstance(raw_val, float):
        return math.isfinite(raw_val)
    if isinstance(raw_val, str) and rendered:
        candidate = rendered.replace(",", "").strip()
        if not candidate:
            return False
        try:
            num = float(candidate)
        except ValueError:
            return False
        return math.isfinite(num)
    return False


def _is_header_like_row(raw_row: Sequence[Any], rendered_row: Sequence[str]) -> bool:
    has_non_empty = False
    for raw_val, text in zip(raw_row, rendered_row, strict=True):
        if not text:
            continue
        has_non_empty = True
        if not isinstance(raw_val, str) or _is_numeric_cell(raw_val, text):
            return False
    return has_non_empty


STACK_MAX_HEADER_ROWS = 15


def _table_headers_mode() -> str:
    return os.environ.get("VOICE_AGENT_RAG_TABLE_HEADERS", "legacy")


def stack_table_headers(
    rendered_rows: Sequence[Sequence[str]],
    raw_rows: Sequence[Sequence[Any]],
    start: int,
) -> tuple[list[str], int] | None:
    import re as _re

    header_rows: list[list[str]] = []
    data_start: int | None = None
    rows = rendered_rows
    limit = min(len(rows), start + STACK_MAX_HEADER_ROWS)
    for i in range(start, limit):
        rr = rows[i]
        raw_r = raw_rows[i] if i < len(raw_rows) else ()
        cells = [(j, text) for j, text in enumerate(rr) if text]
        if not cells:
            header_rows.append(list(rr))
            continue
        nums = [
            text
            for j, text in cells[1:]
            if _is_numeric_cell(raw_r[j] if j < len(raw_r) else None, text)
        ]
        if len(cells) >= 2 and len(nums) >= max(1, (len(cells) - 1) * 0.5):
            year_cells = [text for _, text in cells if _re.fullmatch(r"(19|20)\d{2}", text)]
            if len(year_cells) >= 2 and all(_re.fullmatch(r"(19|20)\d{2}", text) for text in nums):
                header_rows.append(list(rr))
                data_start = i + 1
                continue
            data_start = i
            break
        header_rows.append(list(rr))
    if data_start is None or not header_rows:
        return None
    ncols = max(len(r) for r in rendered_rows)
    for hr in header_rows:
        if len(hr) < ncols:
            hr.extend([""] * (ncols - len(hr)))
    for hr in header_rows[:-1]:
        for j in range(1, len(hr)):
            if not hr[j]:
                for k in range(j - 1, -1, -1):
                    if hr[k]:
                        hr[j] = hr[k]
                        break
    headers: list[str] = []
    for j in range(ncols):
        parts: list[str] = []
        for hr in header_rows:
            v = hr[j].strip() if j < len(hr) else ""
            if v and (not parts or v != parts[-1]):
                parts.append(v)
        headers.append(" / ".join(parts) if parts else f"col_{j + 1}")
    return headers, data_start


def _row_has_number(raw_row: Sequence[Any], rendered_row: Sequence[str]) -> bool:
    return any(
        bool(text) and _is_numeric_cell(raw_val, text)
        for raw_val, text in zip(raw_row, rendered_row, strict=True)
    )


def _cover_lines(wb: Any) -> list[str]:
    lines: list[str] = []
    for ws in wb.worksheets[:MAX_XLSX_SHEETS]:
        sheet_lines: list[str] = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i >= 36:
                break
            cells = [str(c) for c in row if c is not None and str(c).strip()]
            if cells:
                line = " | ".join(cells)[:500]
                sheet_lines.append(line)
        title = str(ws.title).lower().strip()
        cover_titles = frozenset(
            {"cover", "contents", "notes", "index", "readme", "table of contents"}
        )
        if title in cover_titles or len(sheet_lines) <= 35:
            lines.extend(sheet_lines[:35])
    return lines


def _sheet_captions(title: str, cover_lines: list[str]) -> list[str]:
    low = title.lower()
    tok = re.sub(r"[_\-]+", " ", low).strip()
    caps: list[str] = []
    for line in cover_lines:
        ln = line.lower()
        if low in ln or (len(tok) > 4 and tok in ln):
            caps.append(line)
            if len(caps) >= 3:
                break
    return caps


def _forward_fill(cells: Sequence[str]) -> list[str]:
    filled: list[str] = []
    current = ""
    for cell in cells:
        if cell:
            current = cell
        filled.append(current)
    return filled


def _join_two_level_headers(upper: Sequence[str], lower: Sequence[str]) -> list[str]:
    filled_upper = _forward_fill(upper)
    joined: list[str] = []
    for up, lo in zip(filled_upper, lower, strict=True):
        if not up:
            joined.append(lo)
        elif not lo or up == lo:
            joined.append(up)
        else:
            joined.append(f"{up} / {lo}")
    return joined


def _format_pipe_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _render_xlsx_sheet(
    ws: Any,
    *,
    budget: _BudgetCounter | None = None,
) -> tuple[list[str], bool]:
    """Render one worksheet into Markdown lines (without the sheet heading) and truncation flag."""
    truncated = False
    raw_rows: list[list[Any]] = []
    rendered_rows: list[list[str]] = []
    sheet_cell_chars = 0

    for row_idx, row_values in enumerate(ws.iter_rows(values_only=True)):
        if row_idx >= MAX_XLSX_ROWS:
            truncated = True
            break
        if len(row_values) > MAX_XLSX_COLS:
            truncated = True
            row_values = row_values[:MAX_XLSX_COLS]

        rendered_row: list[str] = []
        raw_slice: list[Any] = []
        for cell_val in row_values:
            text, cell_trunc = _render_xlsx_cell(cell_val)
            if cell_trunc:
                truncated = True
            rendered_row.append(text)
            raw_slice.append(cell_val)

        if any(rendered_row):
            sheet_cell_chars += sum(len(text) for text in rendered_row)
            if budget is not None and budget.used + sheet_cell_chars > budget.limit:
                budget.exceeded = True
                return [], truncated
            rendered_rows.append(rendered_row)
            raw_rows.append(raw_slice)

    if not rendered_rows:
        return [], truncated

    max_cols = max(len(r) for r in rendered_rows)
    for row_r, row_raw in zip(rendered_rows, raw_rows, strict=True):
        if len(row_r) < max_cols:
            pad = max_cols - len(row_r)
            row_r.extend([""] * pad)
            row_raw.extend([None] * pad)

    active_cols = [c for c in range(max_cols) if any(r[c] != "" for r in rendered_rows)]
    if not active_cols:
        return [], truncated

    rendered_rows = [[r[c] for c in active_cols] for r in rendered_rows]
    raw_rows = [[r[c] for c in active_cols] for r in raw_rows]
    ncols = len(active_cols)

    lines: list[str] = []

    def _append_line(line: str) -> bool:
        lines.append(line)
        return budget is None or budget.add(len(line) + 1)

    idx = 0
    if ncols >= 2:
        while idx < len(rendered_rows):
            non_empty = [cell for cell in rendered_rows[idx] if cell]
            if len(non_empty) == 1:
                if not _append_line(f"**{non_empty[0]}**"):
                    return [], truncated
                idx += 1
            else:
                break

    if idx >= len(rendered_rows):
        return lines, truncated

    if _table_headers_mode() == "stack":
        result = stack_table_headers(rendered_rows, raw_rows, idx)
        if result is not None:
            headers, data_start = result
            if len(headers) < ncols:
                headers = headers + [""] * (ncols - len(headers))
            headers = headers[:ncols]
            headers = [h.replace("|", "/") for h in headers]
            if lines:
                _append_line("")
            _append_line(_format_pipe_row(headers))
            _append_line(_format_pipe_row(["---"] * ncols))
            for row in rendered_rows[data_start:]:
                if not _append_line(_format_pipe_row(row)):
                    return [], truncated
            return lines, truncated

    header_idx: int | None = None
    for probe in range(idx, len(rendered_rows)):
        non_empty_count = sum(1 for cell in rendered_rows[probe] if cell)
        if non_empty_count * 2 >= ncols:
            header_idx = probe
            break
    if header_idx is None:
        header_idx = idx

    while idx < header_idx:
        plain_cells = [cell for cell in rendered_rows[idx] if cell]
        if plain_cells and not _append_line(_escape_md_line(" | ".join(plain_cells))):
            return [], truncated
        idx += 1

    headers = list(rendered_rows[header_idx])
    data_start = header_idx + 1
    if (
        header_idx + 2 < len(rendered_rows)
        and _is_header_like_row(raw_rows[header_idx + 1], rendered_rows[header_idx + 1])
        and _row_has_number(raw_rows[header_idx + 2], rendered_rows[header_idx + 2])
    ):
        headers = _join_two_level_headers(headers, rendered_rows[header_idx + 1])
        data_start = header_idx + 2

    if lines and not _append_line(""):
        return [], truncated
    if not _append_line(_format_pipe_row(headers)):
        return [], truncated
    if not _append_line(_format_pipe_row(["---"] * ncols)):
        return [], truncated
    for row in rendered_rows[data_start:]:
        if not _append_line(_format_pipe_row(row)):
            return [], truncated

    return lines, truncated


def read_xlsx(raw_bytes: bytes) -> ConvertResult:
    """Read an ``.xlsx`` workbook with ``openpyxl`` directly into raw Markdown."""
    try:
        import openpyxl  # type: ignore[import-untyped]
    except ImportError:
        return ConvertResult(
            text="",
            status="unsupported",
            error="openpyxl not installed",
        )

    label = f"direct-openpyxl-{_package_version('openpyxl', openpyxl)}"
    mode = _table_headers_mode()
    if mode == "stack":
        label += "+stack1"
    truncated = False
    sections: list[str] = []
    budget = _BudgetCounter(canonical.MAX_CANONICAL_CHARS * 2)
    wb = None

    try:
        wb = openpyxl.load_workbook(io.BytesIO(raw_bytes), read_only=True, data_only=True)
        cover_lines: list[str] = [] if mode != "stack" else _cover_lines(wb)
        for sheet_idx, ws in enumerate(wb.worksheets):
            if sheet_idx >= MAX_XLSX_SHEETS:
                truncated = True
                break
            sheet_title = _neutralize_html_comments(
                _strip_unicode_line_seps(str(ws.title))
                .replace("\r", " ")
                .replace("\n", " ")
                .strip()
            )
            sheet_header_cost = len(sheet_title) + 6
            if budget.used + sheet_header_cost > budget.limit:
                budget.exceeded = True
                break
            sheet_lines, sheet_trunc = _render_xlsx_sheet(ws, budget=budget)
            if budget.exceeded:
                break
            if sheet_trunc:
                truncated = True
            if not sheet_lines:
                continue
            if not budget.add(sheet_header_cost):
                break
            caption_lines: list[str] = []
            if mode == "stack":
                caps = _sheet_captions(sheet_title, cover_lines)
                if caps:
                    caption_lines = [
                        _escape_md_line("Context: " + " ; ".join(caps)),
                        "",
                    ]
            block = "\n".join([f"## {sheet_title}", "", *caption_lines, *sheet_lines])
            sections.append(block)
    except Exception as exc:
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"xlsx parse error: {type(exc).__name__}",
            converter=label,
        )
    finally:
        if wb is not None:
            wb.close()

    if budget.exceeded:
        return ConvertResult(
            text="",
            status="too_large",
            error=f"xlsx output exceeds {canonical.MAX_CANONICAL_CHARS} characters",
            converter=label,
        )

    raw_md = "\n\n".join(sections)
    if len(raw_md) > budget.limit:
        return ConvertResult(
            text="",
            status="too_large",
            error=f"xlsx output exceeds {canonical.MAX_CANONICAL_CHARS} characters",
            converter=label,
        )
    return ConvertResult(
        text=raw_md,
        status="ok",
        error=None,
        converter=label,
        raw_text=raw_md,
        fallback="truncated" if truncated else None,
    )


def _clean_pptx_text(text: str) -> tuple[str, bool]:
    cleaned = (
        _strip_unicode_line_seps(text).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    )
    cleaned = cleaned.strip()
    truncated = False
    if len(cleaned) > MAX_PPTX_RUN_CHARS:
        cleaned = cleaned[:MAX_PPTX_RUN_CHARS]
        truncated = True
    return _neutralize_html_comments(cleaned), truncated


def _clean_pptx_cell(text: str) -> tuple[str, bool]:
    cleaned, truncated = _clean_pptx_text(text)
    return cleaned.replace("|", "\\|"), truncated


def _is_bullet_paragraph(para: Any) -> bool:
    p_el = getattr(para, "_p", None)
    p_pr = getattr(p_el, "pPr", None) if p_el is not None else None
    if p_pr is not None:
        if p_pr.find(_BU_NONE_TAG) is not None:
            return False
        if p_pr.find(_BU_CHAR_TAG) is not None or p_pr.find(_BU_AUTONUM_TAG) is not None:
            return True
    level = getattr(para, "level", 0) or 0
    return level > 0


def _shape_sort_key(item: tuple[int, Any]) -> tuple[int, int, int]:
    idx, shape = item
    top = getattr(shape, "top", None)
    left = getattr(shape, "left", None)
    return (
        top if isinstance(top, int) else 0,
        left if isinstance(left, int) else 0,
        idx,
    )


def _extract_text_frame_lines(
    text_frame: Any,
    *,
    budget: _BudgetCounter | None = None,
) -> tuple[list[str], bool]:
    lines: list[str] = []
    truncated = False
    for para in text_frame.paragraphs:
        line, line_trunc = _clean_pptx_text(para.text)
        if line_trunc:
            truncated = True
        if not line:
            continue
        escaped = _escape_md_line(line)
        emitted = f"- {escaped}" if _is_bullet_paragraph(para) else escaped
        if budget is not None and not budget.add(len(emitted) + 1):
            return lines, truncated
        lines.append(emitted)
    return lines, truncated


def _extract_pptx_table_lines(
    table: Any,
    *,
    budget: _BudgetCounter | None = None,
) -> tuple[list[str], bool]:
    rows = list(table.rows)
    if not rows:
        return [], False

    truncated = False
    rendered_rows: list[list[str]] = []
    approx_chars = 0
    for row_idx, row in enumerate(rows):
        if row_idx >= MAX_PPTX_TABLE_ROWS:
            truncated = True
            break
        row_cells: list[str] = []
        for cell in row.cells:
            if cell.is_spanned:
                row_cells.append("")
            else:
                val, cell_trunc = _clean_pptx_cell(cell.text)
                if cell_trunc:
                    truncated = True
                row_cells.append(val)
        approx_chars += sum(len(c) + 3 for c in row_cells) + 2
        if budget is not None and budget.used + approx_chars > budget.limit:
            budget.exceeded = True
            return [], truncated
        rendered_rows.append(row_cells)

    if not rendered_rows:
        return [], truncated

    ncols = max(len(r) for r in rendered_rows)
    if ncols == 0:
        return [], truncated

    for row in rendered_rows:
        if len(row) < ncols:
            row.extend([""] * (ncols - len(row)))

    lines: list[str] = []
    for raw_row in (rendered_rows[0], ["---"] * ncols, *rendered_rows[1:]):
        line = _format_pipe_row(raw_row)
        if budget is not None and not budget.add(len(line) + 1):
            return [], truncated
        lines.append(line)
    return lines, truncated


def _collect_shape_blocks(
    shapes: Sequence[Any],
    *,
    skip_element: Any,
    skip_types: frozenset[Any],
    group_type: Any,
    depth: int,
    budget: _BudgetCounter | None = None,
) -> tuple[list[str], bool]:
    blocks: list[str] = []
    truncated = False
    ordered = [shape for _, shape in sorted(enumerate(shapes), key=_shape_sort_key)]

    for shape in ordered:
        if budget is not None and budget.exceeded:
            break
        if skip_element is not None and getattr(shape, "_element", None) is skip_element:
            continue
        shape_type = getattr(shape, "shape_type", None)
        if shape_type in skip_types:
            continue
        if shape_type == group_type:
            if depth >= MAX_PPTX_GROUP_DEPTH:
                truncated = True
                continue
            sub_blocks, sub_trunc = _collect_shape_blocks(
                shape.shapes,
                skip_element=None,
                skip_types=skip_types,
                group_type=group_type,
                depth=depth + 1,
                budget=budget,
            )
            if sub_trunc:
                truncated = True
            blocks.extend(sub_blocks)
            continue
        if getattr(shape, "has_table", False):
            tbl_lines, tbl_trunc = _extract_pptx_table_lines(shape.table, budget=budget)
            if tbl_trunc:
                truncated = True
            if tbl_lines:
                blocks.append("\n".join(tbl_lines))
            continue
        if getattr(shape, "has_text_frame", False):
            tf_lines, tf_trunc = _extract_text_frame_lines(shape.text_frame, budget=budget)
            if tf_trunc:
                truncated = True
            if tf_lines:
                blocks.append("\n".join(tf_lines))

    return blocks, truncated


def read_pptx(raw_bytes: bytes) -> ConvertResult:
    """Read a ``.pptx`` presentation with ``python-pptx`` directly into raw Markdown."""
    try:
        import pptx  # type: ignore[import-untyped]
        from pptx.enum.shapes import MSO_SHAPE_TYPE  # type: ignore[import-untyped]
    except ImportError:
        return ConvertResult(
            text="",
            status="unsupported",
            error="python-pptx not installed",
        )

    label = f"direct-python-pptx-{_package_version('python-pptx', pptx)}"
    skip_types = frozenset(
        t
        for t in (
            getattr(MSO_SHAPE_TYPE, "PICTURE", None),
            getattr(MSO_SHAPE_TYPE, "MEDIA", None),
            getattr(MSO_SHAPE_TYPE, "EMBEDDED_OLE_OBJECT", None),
            getattr(MSO_SHAPE_TYPE, "LINKED_OLE_OBJECT", None),
        )
        if t is not None
    )
    group_type = MSO_SHAPE_TYPE.GROUP

    truncated = False
    slide_blocks: list[str] = []
    budget = _BudgetCounter(canonical.MAX_CANONICAL_CHARS * 2)

    try:
        prs = pptx.Presentation(io.BytesIO(raw_bytes))
        for slide_num, slide in enumerate(prs.slides, start=1):
            if slide_num > MAX_PPTX_SLIDES:
                truncated = True
                break

            slide_marker = f"<!-- Slide number: {slide_num} -->"
            if not budget.add(len(slide_marker) + 2):
                break
            parts: list[str] = [slide_marker]
            title_shape = slide.shapes.title
            title_el = getattr(title_shape, "_element", None) if title_shape is not None else None
            if title_shape is not None:
                title_text, title_trunc = _clean_pptx_text(title_shape.text)
                if title_trunc:
                    truncated = True
                if title_text:
                    title_line = f"# {_escape_md_line(title_text)}"
                    if not budget.add(len(title_line) + 2):
                        break
                    parts.append(title_line)

            shape_blocks, shapes_trunc = _collect_shape_blocks(
                slide.shapes,
                skip_element=title_el,
                skip_types=skip_types,
                group_type=group_type,
                depth=0,
                budget=budget,
            )
            if shapes_trunc:
                truncated = True
            if budget.exceeded:
                break
            parts.extend(shape_blocks)

            if slide.has_notes_slide:
                notes_tf = slide.notes_slide.notes_text_frame
                if notes_tf is not None:
                    if not budget.add(len("### Notes:") + 2):
                        break
                    note_lines, notes_trunc = _extract_text_frame_lines(notes_tf, budget=budget)
                    if notes_trunc:
                        truncated = True
                    if budget.exceeded:
                        break
                    if note_lines:
                        parts.append("\n".join(["### Notes:", *note_lines]))

            slide_blocks.append("\n\n".join(parts))
    except Exception as exc:
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"pptx parse error: {type(exc).__name__}",
            converter=label,
        )

    if budget.exceeded:
        return ConvertResult(
            text="",
            status="too_large",
            error=f"pptx output exceeds {canonical.MAX_CANONICAL_CHARS} characters",
            converter=label,
        )

    raw_md = "\n\n".join(slide_blocks)
    if len(raw_md) > budget.limit:
        return ConvertResult(
            text="",
            status="too_large",
            error=f"pptx output exceeds {canonical.MAX_CANONICAL_CHARS} characters",
            converter=label,
        )
    return ConvertResult(
        text=raw_md,
        status="ok",
        error=None,
        converter=label,
        raw_text=raw_md,
        fallback="truncated" if truncated else None,
    )


_HTML_DROP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
        "iframe",
        "form",
        "button",
        "select",
        "input",
        "nav",
        "header",
        "footer",
        "aside",
    }
)
_HTML_DROP_TOKENS = frozenset(
    {
        "nav",
        "navbar",
        "menu",
        "sidebar",
        "breadcrumb",
        "footer",
        "cookie",
        "banner",
        "skip",
        "social",
        "share",
        "pagination",
    }
)
_HTML_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "figure",
        "figcaption",
        "details",
        "summary",
        "address",
        "fieldset",
        "legend",
        "center",
    }
)
_HTML_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_HTML_VOID_SKIP_TAGS = frozenset(
    {
        "area",
        "base",
        "col",
        "colgroup",
        "embed",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_META_TAG_RE = re.compile(r"<meta\b([^>]{1,512})>", re.IGNORECASE)
_HTML_ATTR_RE = re.compile(
    r"""([A-Za-z_:][A-Za-z0-9_.:-]{0,63})\s*=\s*(?:"([^"]{0,512})"|'([^']{0,512})'|([^\s"'=<>`]{1,256}))"""
)
_CONTENT_CHARSET_RE = re.compile(r"\bcharset\s*=\s*([A-Za-z0-9._-]+)", re.IGNORECASE)
_XML_ENCODING_RE = re.compile(
    r"""<\?xml\b[^>]{0,256}\bencoding\s*=\s*["']([A-Za-z0-9._-]+)["']""",
    re.IGNORECASE,
)
_BR_SENTINEL = "\x00"


_ALLOWED_ENCODINGS = frozenset(
    {
        "utf-8",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "ascii",
        "gb2312",
        "gbk",
        "gb18030",
        "big5",
        "shift_jis",
        "euc_jp",
        "euc_kr",
        "koi8-r",
        "koi8_r",
        *(f"iso8859-{i}" for i in range(1, 17)),
        *(f"cp{i}" for i in range(1250, 1259)),
    }
)


@dataclass
class _WalkCounter:
    visited: int = 0
    truncated: bool = False
    table_cells: int = 0
    output_chars: int = 0
    output_budget: int | None = None
    too_large: bool = False

    def visit(self) -> bool:
        self.visited += 1
        if self.visited > MAX_HTML_NODES:
            self.truncated = True
            return False
        return True

    def add_output(self, chars: int) -> bool:
        self.output_chars += chars
        if self.output_budget is not None and self.output_chars > self.output_budget:
            self.too_large = True
            return False
        return True


def _is_valid_encoding(name: str) -> bool:
    try:
        return codecs.lookup(name).name in _ALLOWED_ENCODINGS
    except LookupError:
        return False


def _normalize_allowed_encoding(name: str) -> str:
    try:
        canonical_name = codecs.lookup(name).name
    except LookupError:
        return "utf-8"
    return canonical_name if canonical_name in _ALLOWED_ENCODINGS else "utf-8"


def _extract_declared_charset(raw_bytes: bytes) -> str | None:
    head = raw_bytes[:8192].decode("latin-1")
    for match in _META_TAG_RE.finditer(head):
        attrs: dict[str, str] = {}
        for attr_match in _HTML_ATTR_RE.finditer(match.group(1)):
            key = attr_match.group(1).lower()
            val = attr_match.group(2) or attr_match.group(3) or attr_match.group(4) or ""
            attrs[key] = val.strip()
        if "charset" in attrs:
            return _normalize_allowed_encoding(attrs["charset"])
        if attrs.get("http-equiv", "").lower() == "content-type" and "content" in attrs:
            charset_match = _CONTENT_CHARSET_RE.search(attrs["content"])
            if charset_match:
                return _normalize_allowed_encoding(charset_match.group(1))

    xml_match = _XML_ENCODING_RE.search(head[:512])
    if xml_match:
        return _normalize_allowed_encoding(xml_match.group(1))
    return None


def _decode_html_bytes(raw_bytes: bytes) -> str:
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        text = raw_bytes[3:].decode("utf-8", errors="replace")
    elif raw_bytes.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        text = raw_bytes.decode("utf-32", errors="replace")
    elif raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw_bytes.decode("utf-16", errors="replace")
    else:
        declared = _extract_declared_charset(raw_bytes)
        if declared is not None:
            text = raw_bytes.decode(declared, errors="replace")
        else:
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                decoded: str | None = None
                try:
                    import charset_normalizer  # type: ignore[import-untyped]

                    best = charset_normalizer.from_bytes(raw_bytes).best()
                    if best is not None and best.encoding and _is_valid_encoding(best.encoding):
                        decoded = raw_bytes.decode(best.encoding, errors="replace")
                except Exception:
                    decoded = None
                if decoded is not None:
                    text = decoded
                else:
                    text = raw_bytes.decode("utf-8", errors="replace")

    stripped = text.lstrip()
    if stripped.startswith("<?xml"):
        end_idx = stripped.find("?>", 0, 512)
        if end_idx != -1:
            return stripped[end_idx + 2 :]
    return text


def _local_tag(elem: Any) -> str:
    tag = getattr(elem, "tag", None)
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        tag = tag.rsplit("}", 1)[1]
    return tag.lower()


_MODIFIER_PREFIX_TOKENS = frozenset({"has", "no", "with", "without"})


def _is_boilerplate(elem: Any, tag: str) -> bool:
    if not tag or tag in _HTML_DROP_TAGS:
        return True
    role = (elem.get("role") or "").strip().lower() if hasattr(elem, "get") else ""
    if tag in ("html", "body", "main") or role == "main":
        return False
    for attr_name in ("id", "class"):
        val = elem.get(attr_name)
        if not isinstance(val, str) or not val:
            continue
        for raw_word in val.lower().split():
            tokens = _TOKEN_RE.findall(raw_word)
            for idx, tok in enumerate(tokens):
                if tok in _HTML_DROP_TOKENS:
                    if idx > 0 and tokens[idx - 1] in _MODIFIER_PREFIX_TOKENS:
                        continue
                    return True
    return False


_HTML_HARD_DROP_TAGS = frozenset(
    {"script", "style", "noscript", "template", "svg", "canvas", "iframe", "head"}
)
_HTML_ALWAYS_DROP_TAGS = frozenset({"select", "input", "textarea"})
_HTML_ROLLBACK_FORM_CONTROL_TAGS = _HTML_ALWAYS_DROP_TAGS | {"option", "button"}
_HTML_CHROME_TAGS = frozenset({"nav", "aside", "footer", "header"})
_HTML_SOFT_DROP_TAGS = frozenset({"form", "button", "nav", "header", "footer", "aside"})
_HTML_KEEP_EXPANDABLE_TOKENS = frozenset({"faq", "accordion", "collaps"})
_HTML_OK_CANDIDATE_TOKENS = frozenset({"and", "article", "body", "column", "main", "shadow"})
_HTML_LAYOUT_TABLE_ROLES = frozenset({"presentation", "none"})
_HTML_LINK_BLOCK_TAGS = _HTML_BLOCK_TAGS | frozenset({"ul", "ol", "dl"})
HTML_FORM_WRAPPER_RATIO = 0.5
HTML_ARTICLE_KEEP_RATIO = 0.2
HTML_LINK_DENSITY_RATIO = 0.8
HTML_LINK_DENSITY_LONG_CHARS = 1000
HTML_ROLLBACK_MIN_CHARS = 250
HTML_ROLLBACK_RAW_DIVISOR = 7
HTML_LAYOUT_CELL_MIN_CHARS = 200
HTML_MAX_MERGED_HEADER_ROWS = 4


def _html_token_match(elem: Any, wanted: frozenset[str], *, prefix: bool = False) -> bool:
    for attr_name in ("id", "class"):
        val = elem.get(attr_name)
        if not isinstance(val, str) or not val:
            continue
        for raw_word in val.lower().split():
            tokens = _TOKEN_RE.findall(raw_word)
            for idx, tok in enumerate(tokens):
                hit = tok.startswith(tuple(wanted)) if prefix else tok in wanted
                if hit and not (idx > 0 and tokens[idx - 1] in _MODIFIER_PREFIX_TOKENS):
                    return True
    return False


def _html_piece_len(text: str | None) -> int:
    return len(" ".join(text.split())) if text else 0


def _html_metrics(
    elems: list[Any], pos: dict[Any, int]
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Per-element subtree text chars, link chars, non-empty link count, parent index."""
    n = len(elems)
    tags = [_local_tag(e) for e in elems]
    tl = [_html_piece_len(e.text) if tags[i] else 0 for i, e in enumerate(elems)]
    lk = [0] * n
    lc = [0] * n
    parent = [-1] * n
    for i in range(n - 1, -1, -1):
        elem = elems[i]
        tag = tags[i]
        if tag == "a":
            lk[i] = tl[i]
            lc[i] = 1 if tl[i] else 0
        par = elem.getparent()
        p = pos.get(par, -1) if par is not None else -1
        parent[i] = p
        if p < 0:
            continue
        tl[p] += _html_piece_len(elem.tail)
        if tag in _HTML_HARD_DROP_TAGS or not tag:
            continue
        tl[p] += tl[i]
        lk[p] += lk[i]
        lc[p] += lc[i]
    return tl, lk, lc, parent


def _html_drop_keep_tail(elems: Sequence[Any]) -> None:
    """Remove nodes (in document order), moving each tail to the nearest surviving anchor."""
    removed: dict[Any, tuple[Any, bool]] = {}
    pending: dict[Any, tuple[bool, list[str]]] = {}
    to_remove: list[tuple[Any, Any]] = []
    for elem in elems:
        parent = elem.getparent()
        if parent is None:
            continue
        prev = elem.getprevious()
        if prev is not None and prev in removed:
            anchor, is_text = removed[prev]
        elif prev is not None:
            anchor, is_text = prev, False
        else:
            anchor, is_text = parent, True
        removed[elem] = (anchor, is_text)
        if elem.tail:
            pending.setdefault(anchor, (is_text, []))[1].append(elem.tail)
        to_remove.append((parent, elem))
    for anchor, (is_text, tails) in pending.items():
        joined = "".join(tails)
        if is_text:
            anchor.text = (anchor.text or "") + joined
        else:
            anchor.tail = (anchor.tail or "") + joined
    for parent, elem in to_remove:
        parent.remove(elem)


def _html_find_body(elems: list[Any]) -> int:
    for i, elem in enumerate(elems):
        if _local_tag(elem) == "body":
            return i
    return 0


def _html_layout_table(table: Any, pos: dict[Any, int], tl: list[int]) -> bool:
    role = (table.get("role") or "").strip().lower()
    rows: list[Any] = []
    for child in table:
        ctag = _local_tag(child)
        if ctag == "tr":
            rows.append(child)
        elif ctag == "thead":
            return False
        elif ctag in ("tbody", "tfoot"):
            rows.extend(g for g in child if _local_tag(g) == "tr")
    if role in _HTML_LAYOUT_TABLE_ROLES:
        return True
    if not rows:
        return False
    biggest = 0
    single_col = True
    for row in rows:
        cells = [c for c in row if _local_tag(c) in ("td", "th")]
        if any(_local_tag(c) == "th" for c in cells):
            return False
        if len(cells) != 1:
            single_col = False
        for c in cells:
            biggest = max(biggest, tl[pos[c]] if c in pos else 0)
    return (len(rows) == 1 or single_col) and biggest >= HTML_LAYOUT_CELL_MIN_CHARS


def _prune_html_tree(doc: Any) -> int | None:
    """Drop boilerplate in place and return the raw body text length (None if skipped).

    Link-density rule: a block/table with link text > 80% of its text is dropped when it has
    more than one link or under 1000 chars; larger blocks are never dropped on density alone.
    """
    elems = list(doc.iter())
    if len(elems) > MAX_HTML_NODES:
        return None
    pos = {e: i for i, e in enumerate(elems)}
    tl, lk, lc, parent = _html_metrics(elems, pos)
    n = len(elems)
    body_i = _html_find_body(elems)
    body_len = tl[body_i]
    tags = [_local_tag(e) for e in elems]

    protect = [False] * n  # subtree holds a main/article/FAQ element: never drop
    has_main = [False] * n  # only a real main root protects nav/aside/footer/header ancestors
    faq_self = [False] * n
    for i in range(n - 1, -1, -1):
        e = elems[i]
        tag = tags[i]
        if not tag:
            continue
        role = (e.get("role") or "").strip().lower()
        faq_self[i] = _html_token_match(e, _HTML_KEEP_EXPANDABLE_TOKENS, prefix=True)
        if tag == "main" or role == "main":
            has_main[i] = True
        if has_main[i] or tag == "article" or faq_self[i]:
            protect[i] = True
        p = parent[i]
        if p < 0:
            continue
        if has_main[i]:
            has_main[p] = True
            protect[p] = True
        elif protect[i] and tags[p] not in _HTML_CHROME_TAGS:
            protect[p] = True

    gone = [False] * n
    kept = [False] * n
    retag: list[tuple[Any, str]] = []
    for i, e in enumerate(elems):
        tag = tags[i]
        if parent[i] >= 0 and faq_self[parent[i]]:
            faq_self[i] = True
        if not tag or i == body_i or tag in ("html", "body"):
            continue
        if parent[i] >= 0 and gone[parent[i]]:
            gone[i] = True
            continue
        if tag in _HTML_HARD_DROP_TAGS or tag in _HTML_ALWAYS_DROP_TAGS:
            gone[i] = True
            continue
        strong = _is_boilerplate(e, tag) and tag not in _HTML_SOFT_DROP_TAGS
        if tag in _HTML_SOFT_DROP_TAGS or strong:
            keep = protect[i] or faq_self[i]
            if not keep and tag == "form" and tl[i] > HTML_FORM_WRAPPER_RATIO * body_len:
                keep = True
            if (
                not keep
                and tag in _HTML_SOFT_DROP_TAGS
                and _html_token_match(e, _HTML_OK_CANDIDATE_TOKENS)
                and not _html_token_match(e, _HTML_DROP_TOKENS)
            ):
                keep = True
            if not keep:
                gone[i] = True
                continue
            kept[i] = True
            if tag in _HTML_SOFT_DROP_TAGS:
                retag.append((e, "div"))
        elif protect[i] or faq_self[i]:
            kept[i] = True

    layout_tables: list[Any] = []
    for i, e in enumerate(elems):
        if gone[i] or not tags[i]:
            continue
        tag = tags[i]
        if tag == "table" and _html_layout_table(e, pos, tl):
            layout_tables.append(e)
            tag = "div"
        if kept[i] or protect[i] or i == body_i or tl[i] <= 0:
            continue
        if tag == "table":
            dense = lk[i] > HTML_LINK_DENSITY_RATIO * tl[i] and tl[i] < HTML_LINK_DENSITY_LONG_CHARS
        elif tag in _HTML_LINK_BLOCK_TAGS:
            dense = lk[i] > HTML_LINK_DENSITY_RATIO * tl[i] and (
                lc[i] > 1 or tl[i] < HTML_LINK_DENSITY_LONG_CHARS
            )
        else:
            continue
        if dense:
            gone[i] = True

    # Propagate link-density drops to descendants, then remove only topmost dropped nodes.
    for i in range(n):
        if parent[i] >= 0 and gone[parent[i]]:
            gone[i] = True
    top_gone: list[Any] = []
    for i in range(n):
        if gone[i] and not (parent[i] >= 0 and gone[parent[i]]):
            top_gone.append(elems[i])
    _html_drop_keep_tail(top_gone)

    for elem, new_tag in retag:
        elem.tag = new_tag
    for table in layout_tables:
        table.tag = "div"
        for child in list(table):
            ctag = _local_tag(child)
            if ctag in ("tbody", "thead", "tfoot"):
                child.tag = "div"
                for row in child:
                    _html_unwrap_row(row)
            elif ctag == "tr":
                _html_unwrap_row(child)
    for i, e in enumerate(elems):
        if not gone[i] and tags[i] and _is_boilerplate(e, _local_tag(e)):
            for attr_name in ("id", "class"):
                if attr_name in e.attrib:
                    del e.attrib[attr_name]
    return body_len


def _html_unwrap_row(row: Any) -> None:
    if _local_tag(row) != "tr":
        return
    row.tag = "div"
    for cell in row:
        if _local_tag(cell) in ("td", "th"):
            cell.tag = "div"


def _html_body_roots(doc: Any) -> list[Any]:
    for elem in doc.iter():
        if _local_tag(elem) == "body":
            return [elem]
    return [doc]


def _html_plain_text(root: Any) -> str:
    pieces = (" ".join(_strip_unicode_line_seps(s).split()) for s in root.itertext())
    return _escape_md_lines("\n".join(p for p in pieces if p))


def _html_hard_only(doc: Any) -> None:
    for elem in list(doc.iter()):
        tag = _local_tag(elem)
        if not tag:
            continue
        if tag in _HTML_ROLLBACK_FORM_CONTROL_TAGS and elem.getparent() is not None:
            elem.drop_tree()
            continue
        if tag in _HTML_SOFT_DROP_TAGS:
            elem.tag = "div"
        for attr_name in ("id", "class"):
            if attr_name in elem.attrib:
                del elem.attrib[attr_name]


def _select_html_roots(doc: Any) -> list[Any]:
    elems = list(doc.iter())
    if len(elems) > MAX_HTML_NODES:
        bodies = [e for e in elems[:1000] if _local_tag(e) == "body"]
        return bodies or [doc]
    mains: list[Any] = []
    articles: list[Any] = []
    body: Any = None
    in_chrome: dict[Any, bool] = {}
    for e in elems:
        tag = _local_tag(e)
        par = e.getparent()
        in_chrome[e] = par is not None and (
            in_chrome.get(par, False) or _local_tag(par) in _HTML_CHROME_TAGS
        )
        role = (e.get("role") or "").strip().lower()
        if tag == "main" or role == "main":
            mains.append(e)
        elif tag == "article" and not in_chrome[e]:
            articles.append(e)
        elif tag == "body" and body is None:
            body = e
    chosen = mains
    if not chosen and articles:
        pos = {e: i for i, e in enumerate(elems)}
        tl, _, _, _ = _html_metrics(elems, pos)
        longest = max(tl[pos[a]] for a in articles)
        chosen = [a for a in articles if tl[pos[a]] >= HTML_ARTICLE_KEEP_RATIO * longest]
    if not chosen:
        return [body if body is not None else doc]
    picked: set[Any] = set()
    out: list[Any] = []
    for e in chosen:
        if any(anc in picked for anc in e.iterancestors()):
            continue
        picked.add(e)
        out.append(e)
    return out


def _collapse_inline(parts: Sequence[str], *, single_line: bool = False) -> str:
    if not parts:
        return ""
    joined = _strip_unicode_line_seps("".join(parts))
    if not joined:
        return ""
    if single_line:
        return _neutralize_html_comments(" ".join(joined.replace(_BR_SENTINEL, " ").split()))
    raw_lines = joined.split(_BR_SENTINEL)
    cleaned_lines = [" ".join(segment.split()) for segment in raw_lines]
    start = 0
    end = len(cleaned_lines)
    while start < end and not cleaned_lines[start]:
        start += 1
    while end > start and not cleaned_lines[end - 1]:
        end -= 1
    if start >= end:
        return ""
    out: list[str] = []
    empty_run = 0
    for line in cleaned_lines[start:end]:
        if line == "":
            empty_run += 1
            if empty_run <= 1:
                out.append("")
        else:
            empty_run = 0
            out.append(line)
    return _neutralize_html_comments("\n".join(out))


def _longest_backtick_run(text: str) -> int:
    if "`" not in text:
        return 0
    longest = 0
    current = 0
    for ch in text:
        if ch == "`":
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


def _extract_pre_text(pre_elem: Any, start_depth: int, counter: _WalkCounter) -> str:
    parts: list[str] = []
    pre_chars = 0
    stack: list[tuple[str, Any, int]] = [("enter", pre_elem, start_depth)]

    while stack:
        kind, payload, depth = stack.pop()
        if kind == "text":
            piece = _strip_unicode_line_seps(str(payload)).replace(_BR_SENTINEL, "")
            parts.append(piece)
            pre_chars += len(piece)
            if (
                counter.output_budget is not None
                and counter.output_chars + pre_chars > counter.output_budget
            ):
                counter.too_large = True
                return ""
            continue
        elem = payload
        if not counter.visit():
            break
        tag = _local_tag(elem)
        if elem is not pre_elem and _is_boilerplate(elem, tag):
            continue
        if tag == "br":
            parts.append("\n")
            pre_chars += 1
            continue
        if depth >= MAX_HTML_DEPTH:
            counter.truncated = True
            continue
        children = list(elem)
        for child in reversed(children):
            if child.tail:
                stack.append(("text", child.tail, depth + 1))
            stack.append(("enter", child, depth + 1))
        if elem.text:
            stack.append(("text", elem.text, depth))

    text = "".join(parts).replace("\r\n", "\n").replace("\r", "\n")
    return _neutralize_html_comments(text.strip("\n"))


def _extract_cell_text(cell_elem: Any, start_depth: int, counter: _WalkCounter) -> str:
    parts: list[str] = []
    stack: list[tuple[str, Any, int]] = [("enter", cell_elem, start_depth)]

    while stack:
        kind, payload, depth = stack.pop()
        if kind == "text":
            parts.append(_strip_unicode_line_seps(str(payload)).replace(_BR_SENTINEL, " "))
            continue
        elem = payload
        if not counter.visit():
            break
        tag = _local_tag(elem)
        if elem is not cell_elem and _is_boilerplate(elem, tag):
            continue
        if tag in _HTML_VOID_SKIP_TAGS:
            continue
        if tag == "br":
            parts.append(" ")
            continue
        if tag == "img":
            alt = (
                _strip_unicode_line_seps(elem.get("alt") or "").strip()
                if hasattr(elem, "get")
                else ""
            )
            if alt:
                parts.append(f" {alt} ")
            continue
        if depth >= MAX_HTML_DEPTH:
            counter.truncated = True
            continue
        children = list(elem)
        for child in reversed(children):
            if child.tail:
                stack.append(("text", child.tail, depth + 1))
            stack.append(("enter", child, depth + 1))
        if elem.text:
            stack.append(("text", elem.text, depth))

    collapsed = " ".join(_strip_unicode_line_seps("".join(parts)).split())
    return _neutralize_html_comments(collapsed).replace("|", "\\|")


def _parse_span_attr(raw: str | None, *, default: int = 1, maximum: int = 200) -> int:
    if not raw:
        return default
    cleaned = raw.strip()
    if not (cleaned.isascii() and cleaned.isdigit()):
        return default
    val = int(cleaned)
    if val < 1:
        return default
    return min(val, maximum)


def _render_html_table(table_elem: Any, start_depth: int, counter: _WalkCounter) -> list[str]:
    if start_depth >= MAX_HTML_DEPTH:
        counter.truncated = True
        return []
    if counter.table_cells >= MAX_HTML_DOC_TABLE_CELLS:
        counter.truncated = True
        return []

    caption_text = ""
    thead_trs: list[tuple[Any, int]] = []
    body_trs: list[tuple[Any, int]] = []
    tfoot_trs: list[tuple[Any, int]] = []

    for child in table_elem:
        if not counter.visit():
            break
        ctag = _local_tag(child)
        if _is_boilerplate(child, ctag):
            continue
        if ctag == "caption" and not caption_text:
            caption_text = _extract_cell_text(child, start_depth + 1, counter).replace("\\|", "|")
        elif ctag == "tr":
            body_trs.append((child, start_depth + 1))
        elif ctag in ("thead", "tbody", "tfoot"):
            if start_depth + 1 >= MAX_HTML_DEPTH:
                counter.truncated = True
                continue
            target = thead_trs if ctag == "thead" else (tfoot_trs if ctag == "tfoot" else body_trs)
            for grand in child:
                if not counter.visit():
                    break
                gtag = _local_tag(grand)
                if gtag == "tr" and not _is_boilerplate(grand, gtag):
                    target.append((grand, start_depth + 2))

    ordered_trs = thead_trs + body_trs + tfoot_trs
    if not ordered_trs:
        return []
    if len(ordered_trs) > MAX_HTML_TABLE_ROWS:
        ordered_trs = ordered_trs[:MAX_HTML_TABLE_ROWS]
        counter.truncated = True

    thead_ids = {id(tr) for tr, _ in thead_trs}
    parsed_rows: list[list[tuple[str, int, int]]] = []
    parsed_hdr: list[bool] = []
    for tr_elem, tr_depth in ordered_trs:
        if tr_depth >= MAX_HTML_DEPTH:
            counter.truncated = True
            continue
        row_cells: list[tuple[str, int, int]] = []
        all_th = True
        for cell in tr_elem:
            if not counter.visit():
                break
            cell_tag = _local_tag(cell)
            if cell_tag not in ("th", "td") or _is_boilerplate(cell, cell_tag):
                continue
            all_th = all_th and cell_tag == "th"
            cell_text = _extract_cell_text(cell, tr_depth + 1, counter)
            colspan = _parse_span_attr(cell.get("colspan"), maximum=MAX_HTML_TABLE_COLS)
            rowspan = _parse_span_attr(cell.get("rowspan"), maximum=len(ordered_trs))
            row_cells.append((cell_text, colspan, rowspan))
        if row_cells:
            parsed_rows.append(row_cells)
            parsed_hdr.append(all_th or id(tr_elem) in thead_ids)

    if not parsed_rows:
        return [f"**{caption_text}**"] if caption_text else []

    num_rows = len(parsed_rows)
    leading_hdr = 0
    for flag in parsed_hdr:
        if not flag:
            break
        leading_hdr += 1
    leading_hdr = min(leading_hdr, HTML_MAX_MERGED_HEADER_ROWS)
    header_row_count = len(thead_trs) if thead_trs else max(1, leading_hdr)
    hdr_in_grid = 0
    hdr_open = True
    grid: list[list[str]] = []
    occupied: dict[int, dict[int, str]] = {}
    pending_occupied = 0
    table_cells = 0
    hit_cell_cap = False

    def _can_add_cell() -> bool:
        return table_cells < MAX_HTML_TABLE_CELLS and counter.table_cells < MAX_HTML_DOC_TABLE_CELLS

    def _can_reserve_future() -> bool:
        return (
            table_cells + pending_occupied < MAX_HTML_TABLE_CELLS
            and counter.table_cells + pending_occupied < MAX_HTML_DOC_TABLE_CELLS
        )

    for r_idx, row_specs in enumerate(parsed_rows):
        row_occ = occupied.pop(r_idx, {})
        pending_occupied -= len(row_occ)
        row_out: list[str] = []
        c_idx = 0
        is_header_row = r_idx < header_row_count

        for cell_text, colspan, rowspan in row_specs:
            while c_idx in row_occ and c_idx < MAX_HTML_TABLE_COLS:
                if not _can_add_cell():
                    hit_cell_cap = True
                    break
                row_out.append(row_occ[c_idx])
                table_cells += 1
                counter.table_cells += 1
                c_idx += 1
            if hit_cell_cap:
                break
            if c_idx >= MAX_HTML_TABLE_COLS:
                counter.truncated = True
                break

            for dc in range(colspan):
                col = c_idx + dc
                if col >= MAX_HTML_TABLE_COLS:
                    counter.truncated = True
                    break
                val = (
                    cell_text
                    if (dc == 0 or (is_header_row and dc < MAX_HTML_HEADER_COLSPAN_COPY))
                    else ""
                )
                if col in row_occ and len(row_out) == col:
                    if not _can_add_cell():
                        hit_cell_cap = True
                        break
                    row_out.append(row_occ[col])
                    table_cells += 1
                    counter.table_cells += 1
                    continue
                while len(row_out) < col:
                    if not _can_add_cell():
                        hit_cell_cap = True
                        break
                    row_out.append(row_occ.get(len(row_out), ""))
                    table_cells += 1
                    counter.table_cells += 1
                if hit_cell_cap:
                    break
                if not _can_add_cell():
                    hit_cell_cap = True
                    break
                row_out.append(val)
                table_cells += 1
                counter.table_cells += 1
                if rowspan > 1:
                    for dr in range(1, rowspan):
                        target_r = r_idx + dr
                        if target_r >= num_rows:
                            break
                        target_occ = occupied.setdefault(target_r, {})
                        if col not in target_occ:
                            if not _can_reserve_future():
                                hit_cell_cap = True
                                break
                            target_occ[col] = val
                            pending_occupied += 1
                    if hit_cell_cap:
                        break
            if hit_cell_cap:
                break
            c_idx += colspan

        if not hit_cell_cap and row_occ:
            trailing_cols = [col for col in row_occ if col >= len(row_out)]
            if trailing_cols:
                max_trailing = min(max(trailing_cols), MAX_HTML_TABLE_COLS - 1)
                for col in range(len(row_out), max_trailing + 1):
                    if not _can_add_cell():
                        hit_cell_cap = True
                        break
                    row_out.append(row_occ.get(col, ""))
                    table_cells += 1
                    counter.table_cells += 1

        if any(row_out):
            grid.append(row_out)
            if hdr_open and parsed_hdr[r_idx]:
                hdr_in_grid += 1
            else:
                hdr_open = False
        elif not (hdr_open and parsed_hdr[r_idx]):
            hdr_open = False
        if hit_cell_cap:
            counter.truncated = True
            break

    if not grid:
        return [f"**{caption_text}**"] if caption_text else []

    ncols = max(len(r) for r in grid)
    if ncols == 0:
        return [f"**{caption_text}**"] if caption_text else []
    max_rows_for_table = max(1, MAX_HTML_TABLE_CELLS // ncols)
    if len(grid) > max_rows_for_table:
        grid = grid[:max_rows_for_table]
        counter.truncated = True
    for row in grid:
        if len(row) < ncols:
            row.extend([""] * (ncols - len(row)))

    active_cols = [c for c in range(ncols) if any(r[c] != "" for r in grid)]
    if not active_cols:
        return [f"**{caption_text}**"] if caption_text else []
    grid = [[r[c] for c in active_cols] for r in grid]
    ncols = len(active_cols)

    hdr_rows = min(hdr_in_grid, HTML_MAX_MERGED_HEADER_ROWS)
    if hdr_rows >= 2 and len(grid) > hdr_rows:
        merged: list[str] = []
        for c in range(ncols):
            parts: list[str] = []
            for r in range(hdr_rows):
                if grid[r][c] and grid[r][c] not in parts:
                    parts.append(grid[r][c])
            merged.append(" / ".join(parts))
        grid = [merged, *grid[hdr_rows:]]

    out_blocks: list[str] = []
    if caption_text:
        out_blocks.append(f"**{caption_text}**")
    if len(grid) == 1:
        plain = _escape_md_line(" | ".join(cell for cell in grid[0] if cell))
        if plain:
            out_blocks.append(plain)
        return out_blocks

    table_lines = [_format_pipe_row(grid[0]), _format_pipe_row(["---"] * ncols)]
    table_chars = sum(len(line) + 1 for line in table_lines)
    for row in grid[1:]:
        line = _format_pipe_row(row)
        table_chars += len(line) + 1
        if (
            counter.output_budget is not None
            and counter.output_chars + table_chars > counter.output_budget
        ):
            counter.too_large = True
            return []
        table_lines.append(line)
    out_blocks.append("\n".join(table_lines))
    return out_blocks


def _walk_html_to_markdown(roots: Sequence[Any], counter: _WalkCounter) -> str:
    blocks: list[tuple[str, bool]] = []  # (text, is_list_item)
    inline_parts: list[str] = []
    quote_depth = 0
    list_stack: list[tuple[str, int]] = []
    li_prefix_stack: list[str | None] = []
    pending_dt: str | None = None
    pending_dd_prefix: str | None = None
    heading_depth = 0

    def _emit_block(text: str, *, is_list_item: bool = False, raw_block: bool = False) -> None:
        nonlocal pending_dd_prefix
        if not text:
            return
        if not raw_block:
            text = _escape_md_lines(text)
        for idx in range(len(li_prefix_stack) - 1, -1, -1):
            prefix = li_prefix_stack[idx]
            if prefix is not None:
                text = prefix + text
                li_prefix_stack[idx] = None
                is_list_item = True
                break
        if pending_dd_prefix is not None:
            text = pending_dd_prefix + text
            pending_dd_prefix = None
        if quote_depth > 0:
            q_prefix = "> " * quote_depth
            text = "\n".join(
                f"{q_prefix}{line}" if line else q_prefix.rstrip() for line in text.split("\n")
            )
        if not counter.add_output(len(text) + 2):
            return
        blocks.append((text, is_list_item))

    def _flush_inline(*, single_line: bool = False) -> str:
        text = _collapse_inline(inline_parts, single_line=single_line)
        inline_parts.clear()
        if not single_line and text:
            _emit_block(text)
        return text

    stack: list[tuple[str, Any, int]] = [("enter", r, 0) for r in reversed(roots)]
    inline_chars = 0
    while stack:
        if counter.too_large:
            break
        action, payload, depth = stack.pop()
        if action == "text":
            piece = _strip_unicode_line_seps(str(payload)).replace(_BR_SENTINEL, "")
            inline_parts.append(piece)
            inline_chars += len(piece.strip())
            if (
                counter.output_budget is not None
                and counter.output_chars + inline_chars > counter.output_budget
            ):
                counter.too_large = True
                break
            continue

        if action == "exit":
            tag = str(payload)
            if heading_depth > 0 and tag in _HTML_HEADING_TAGS:
                level = int(tag[1])
                heading_text = _flush_inline(single_line=True)
                inline_chars = 0
                heading_depth = 0
                if heading_text:
                    _emit_block(f"{'#' * level} {_escape_md_line(heading_text)}", raw_block=True)
            elif heading_depth > 0:
                continue
            elif tag == "dt":
                dt_text = _flush_inline(single_line=True)
                inline_chars = 0
                if dt_text:
                    pending_dt = dt_text
            elif tag == "dd":
                _flush_inline()
                inline_chars = 0
                pending_dd_prefix = None
            elif tag == "dl":
                _flush_inline()
                inline_chars = 0
                if pending_dt is not None:
                    _emit_block(pending_dt)
                    pending_dt = None
            elif tag == "li":
                _flush_inline()
                inline_chars = 0
                if li_prefix_stack:
                    li_prefix_stack.pop()
            elif tag in ("ul", "ol"):
                _flush_inline()
                inline_chars = 0
                if list_stack:
                    list_stack.pop()
            elif tag == "blockquote":
                _flush_inline()
                inline_chars = 0
                quote_depth = max(0, quote_depth - 1)
            elif tag in _HTML_BLOCK_TAGS:
                _flush_inline()
                inline_chars = 0
            continue

        elem = payload
        if not counter.visit():
            break
        tag = _local_tag(elem)
        if _is_boilerplate(elem, tag) or tag in _HTML_VOID_SKIP_TAGS:
            continue

        if tag == "br":
            inline_parts.append(_BR_SENTINEL if heading_depth == 0 else " ")
            continue
        if tag == "hr":
            if heading_depth == 0:
                _flush_inline()
                inline_chars = 0
            continue
        if tag == "img":
            alt = _strip_unicode_line_seps(elem.get("alt") or "").replace(_BR_SENTINEL, "").strip()
            if alt:
                inline_parts.append(f" {alt} ")
            continue

        if heading_depth == 0:
            if tag == "pre":
                _flush_inline()
                inline_chars = 0
                code_text = _extract_pre_text(elem, depth, counter)
                if counter.too_large:
                    break
                if code_text.strip():
                    fence = "`" * max(3, _longest_backtick_run(code_text) + 1)
                    _emit_block(f"{fence}\n{code_text}\n{fence}", raw_block=True)
                continue
            if tag == "table":
                _flush_inline()
                inline_chars = 0
                for tbl_block in _render_html_table(elem, depth, counter):
                    if counter.too_large:
                        break
                    _emit_block(tbl_block, raw_block=True)
                continue

            if tag in _HTML_HEADING_TAGS:
                _flush_inline()
                inline_chars = 0
                heading_depth = int(tag[1])
            elif tag == "blockquote":
                _flush_inline()
                inline_chars = 0
                quote_depth += 1
            elif tag in ("ul", "ol"):
                _flush_inline()
                inline_chars = 0
                list_stack.append((tag, 0))
            elif tag == "li":
                _flush_inline()
                inline_chars = 0
                indent = "  " * max(0, len(list_stack) - 1)
                if list_stack:
                    list_kind, count = list_stack[-1]
                    count += 1
                    list_stack[-1] = (list_kind, count)
                    prefix = f"{indent}{count}. " if list_kind == "ol" else f"{indent}- "
                else:
                    prefix = "- "
                li_prefix_stack.append(prefix)
            elif tag == "dt":
                _flush_inline()
                inline_chars = 0
                if pending_dt is not None:
                    _emit_block(pending_dt)
                    pending_dt = None
            elif tag == "dd":
                _flush_inline()
                inline_chars = 0
                if pending_dt is not None:
                    pending_dd_prefix = f"{_escape_md_line(pending_dt)}: "
                    pending_dt = None
            elif tag == "dl" or tag in _HTML_BLOCK_TAGS:
                _flush_inline()
                inline_chars = 0

        if depth >= MAX_HTML_DEPTH:
            counter.truncated = True
            if heading_depth > 0 and tag in _HTML_HEADING_TAGS:
                heading_depth = 0
            elif tag == "blockquote":
                quote_depth = max(0, quote_depth - 1)
            elif tag in ("ul", "ol") and list_stack:
                list_stack.pop()
            elif tag == "li" and li_prefix_stack:
                li_prefix_stack.pop()
            continue

        stack.append(("exit", tag, depth))
        children = list(elem)
        for child in reversed(children):
            if child.tail:
                stack.append(("text", child.tail, depth + 1))
            stack.append(("enter", child, depth + 1))
        if elem.text:
            stack.append(("text", elem.text, depth))

    _flush_inline()
    if pending_dt is not None:
        _emit_block(pending_dt)

    if not blocks:
        return ""

    out_parts: list[str] = []
    prev_is_li = False
    for text, is_li in blocks:
        if out_parts:
            sep = "\n" if (prev_is_li and is_li) else "\n\n"
            out_parts.append(sep)
        out_parts.append(text)
        prev_is_li = is_li
    return "".join(out_parts)


def _read_html_with_lxml(
    raw_bytes: bytes,
    *,
    lxml_mod: Any,
    lxml_html: Any,
    lxml_etree: Any,
    kind: str = "html",
    output_budget: int | None = None,
) -> ConvertResult:
    label = f"direct-lxml-{_package_version('lxml', lxml_mod)}"
    if len(raw_bytes) > MAX_HTML_BYTES:
        return ConvertResult(
            text="",
            status="too_large",
            error=f"{kind} input exceeds {MAX_HTML_BYTES} bytes",
            converter=label,
        )

    decoded = _decode_html_bytes(raw_bytes)
    if not decoded.strip():
        return ConvertResult(
            text="",
            status="ok",
            error=None,
            converter=label,
            raw_text="",
        )

    budget = canonical.MAX_CANONICAL_CHARS * 2 if output_budget is None else output_budget
    counter = _WalkCounter(output_budget=budget)
    try:
        parser = lxml_html.HTMLParser(
            recover=True,
            no_network=True,
            remove_comments=True,
            remove_pis=True,
            huge_tree=False,
        )
        doc = lxml_html.fromstring(decoded, parser=parser)
        if doc is None:
            return ConvertResult(text="", status="ok", error=None, converter=label, raw_text="")
        raw_body_chars = _prune_html_tree(doc)
        raw_md = _walk_html_to_markdown(_select_html_roots(doc), counter)
        if (
            raw_body_chars is not None
            and raw_body_chars >= HTML_ROLLBACK_MIN_CHARS
            and len(raw_md)
            < max(HTML_ROLLBACK_MIN_CHARS, raw_body_chars // HTML_ROLLBACK_RAW_DIVISOR)
            and not counter.too_large
        ):
            # Over-pruned: redo with hard removals only; fallback stays unset (docindex rejects it).
            rb_doc = lxml_html.fromstring(decoded, parser=parser)
            _html_hard_only(rb_doc)
            rb_counter = _WalkCounter(output_budget=budget)
            rb_md = _walk_html_to_markdown(_html_body_roots(rb_doc), rb_counter)
            if len(rb_md) < HTML_ROLLBACK_MIN_CHARS and not rb_counter.too_large:
                lxml_etree.strip_elements(
                    rb_doc,
                    *_HTML_HARD_DROP_TAGS,
                    *_HTML_ROLLBACK_FORM_CONTROL_TAGS,
                    with_tail=False,
                )
                plain = _html_plain_text(_html_body_roots(rb_doc)[0])
                if len(plain) > len(rb_md):
                    rb_md = plain
            if len(rb_md) > len(raw_md) and not rb_counter.too_large:
                raw_md = rb_md
                counter.truncated = counter.truncated or rb_counter.truncated
    except Exception as exc:
        if isinstance(exc, lxml_etree.ParserError) and "empty" in str(exc).lower():
            return ConvertResult(text="", status="ok", error=None, converter=label, raw_text="")
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"{kind} parse error: {type(exc).__name__}",
            converter=label,
        )

    if counter.too_large or len(raw_md) > budget:
        return ConvertResult(
            text="",
            status="too_large",
            error=f"{kind} output exceeds {canonical.MAX_CANONICAL_CHARS} characters",
            converter=label,
        )

    return ConvertResult(
        text=raw_md,
        status="ok",
        error=None,
        converter=label,
        raw_text=raw_md,
        fallback="truncated" if counter.truncated else None,
    )


def read_html(raw_bytes: bytes) -> ConvertResult:
    """Read HTML bytes directly with ``lxml.html`` into canonical-ready Markdown."""
    try:
        import lxml  # type: ignore[import-untyped]
        import lxml.etree as lxml_etree  # type: ignore[import-untyped]
        import lxml.html as lxml_html  # type: ignore[import-untyped]
    except ImportError:
        return ConvertResult(
            text="",
            status="unsupported",
            error="lxml not installed",
        )

    return _read_html_with_lxml(
        raw_bytes,
        lxml_mod=lxml,
        lxml_html=lxml_html,
        lxml_etree=lxml_etree,
        kind="html",
    )


def _find_zip_member(
    infos_by_name: dict[str, zipfile.ZipInfo],
    path: str,
) -> zipfile.ZipInfo | None:
    if path in infos_by_name:
        return infos_by_name[path]
    norm = posixpath.normpath(path.lstrip("/"))
    return infos_by_name.get(norm)


def _xml_local_tag(elem: Any) -> str:
    tag = getattr(elem, "tag", "")
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        return tag.rsplit("}", 1)[1].lower()
    return tag.lower()


def read_epub(raw_bytes: bytes) -> ConvertResult:
    """Read an ``.epub`` archive directly with zipguard + ``lxml.html``."""
    try:
        import lxml  # type: ignore[import-untyped]
        import lxml.etree as lxml_etree  # type: ignore[import-untyped]
        import lxml.html as lxml_html  # type: ignore[import-untyped]
    except ImportError:
        return ConvertResult(
            text="",
            status="unsupported",
            error="lxml not installed",
        )

    label = f"direct-lxml-{_package_version('lxml', lxml)}"
    preflight_err = zipguard.preflight_zip(
        raw_bytes,
        label="epub",
        scan_xml_members=True,
    )
    if preflight_err is not None:
        return ConvertResult(
            text="",
            status="unreadable",
            error=preflight_err,
            converter=label,
        )

    truncated = False
    chapters: list[str] = []
    total_chars = 0

    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            infos_by_name = {info.filename: info for info in zf.infolist()}

            container_info = _find_zip_member(infos_by_name, "META-INF/container.xml")
            if container_info is None:
                return ConvertResult(
                    text="",
                    status="unreadable",
                    error="epub missing META-INF/container.xml",
                    converter=label,
                )
            if container_info.file_size > MAX_EPUB_ITEM_BYTES:
                return ConvertResult(
                    text="",
                    status="too_large",
                    error=f"epub item exceeds {MAX_EPUB_ITEM_BYTES} bytes",
                    converter=label,
                )
            container_bytes = zipguard.read_member_bounded(
                zf,
                container_info,
                MAX_EPUB_ITEM_BYTES,
                keep=True,
                label="epub",
            )
            if zipguard.has_forbidden_xml_decl(container_bytes, allow_plain_doctype=False):
                return ConvertResult(
                    text="",
                    status="unreadable",
                    error="epub XML contains forbidden DOCTYPE or ENTITY declaration",
                    converter=label,
                )
            container_root, container_err = zipguard.parse_xml_bounded(
                container_bytes,
                zipguard.MAX_DOCX_XML_DEPTH,
                label="epub",
            )
            if container_err is not None or container_root is None:
                return ConvertResult(
                    text="",
                    status="unreadable",
                    error=container_err or "epub XML parse error",
                    converter=label,
                )

            opf_raw_path: str | None = None
            for elem in container_root.iter():
                if _xml_local_tag(elem) == "rootfile":
                    full_path = elem.get("full-path")
                    if full_path:
                        media_type = (elem.get("media-type") or "").strip().lower()
                        if media_type == "application/oebps-package+xml" or opf_raw_path is None:
                            opf_raw_path = full_path
                            if media_type == "application/oebps-package+xml":
                                break
            if not opf_raw_path:
                return ConvertResult(
                    text="",
                    status="unreadable",
                    error="epub container.xml missing rootfile",
                    converter=label,
                )

            opf_path = posixpath.normpath(urllib.parse.unquote(opf_raw_path).lstrip("/"))
            if opf_path in (".", "") or opf_path.startswith(".."):
                return ConvertResult(
                    text="",
                    status="unreadable",
                    error="epub invalid rootfile path",
                    converter=label,
                )
            opf_info = _find_zip_member(infos_by_name, opf_path)
            if opf_info is None:
                return ConvertResult(
                    text="",
                    status="unreadable",
                    error="epub missing OPF",
                    converter=label,
                )
            if opf_info.file_size > MAX_EPUB_ITEM_BYTES:
                return ConvertResult(
                    text="",
                    status="too_large",
                    error=f"epub item exceeds {MAX_EPUB_ITEM_BYTES} bytes",
                    converter=label,
                )
            opf_bytes = zipguard.read_member_bounded(
                zf,
                opf_info,
                MAX_EPUB_ITEM_BYTES,
                keep=True,
                label="epub",
            )
            if zipguard.has_forbidden_xml_decl(opf_bytes, allow_plain_doctype=False):
                return ConvertResult(
                    text="",
                    status="unreadable",
                    error="epub XML contains forbidden DOCTYPE or ENTITY declaration",
                    converter=label,
                )
            opf_root, opf_err = zipguard.parse_xml_bounded(
                opf_bytes,
                zipguard.MAX_DOCX_XML_DEPTH,
                label="epub",
            )
            if opf_err is not None or opf_root is None:
                return ConvertResult(
                    text="",
                    status="unreadable",
                    error=opf_err or "epub XML parse error",
                    converter=label,
                )

            opf_dir = posixpath.dirname(opf_path)
            manifest: dict[str, tuple[str, str]] = {}
            spine_elem: Any = None
            for child in opf_root:
                ctag = _xml_local_tag(child)
                if ctag == "manifest":
                    for item in child:
                        if _xml_local_tag(item) != "item":
                            continue
                        item_id = item.get("id")
                        href = item.get("href")
                        media_type = (item.get("media-type") or "").split(";", 1)[0].strip().lower()
                        if not item_id or not href:
                            continue
                        href_clean = urllib.parse.unquote(href.split("#", 1)[0])
                        joined = posixpath.join(opf_dir, href_clean) if opf_dir else href_clean
                        resolved = posixpath.normpath(joined.lstrip("/"))
                        if resolved.startswith("..") or resolved in ("", "."):
                            continue
                        manifest[item_id] = (resolved, media_type)
                elif ctag == "spine":
                    spine_elem = child

            if spine_elem is None:
                return ConvertResult(
                    text="",
                    status="ok",
                    error=None,
                    converter=label,
                    raw_text="",
                )

            spine_count = 0
            seen_idrefs: set[str] = set()
            total_chapter_bytes = 0
            budget_limit = canonical.MAX_CANONICAL_CHARS * 2
            for itemref in spine_elem:
                if _xml_local_tag(itemref) != "itemref":
                    continue
                if spine_count >= MAX_EPUB_SPINE_ITEMS:
                    truncated = True
                    break
                spine_count += 1
                idref = itemref.get("idref")
                if not idref or idref in seen_idrefs or idref not in manifest:
                    continue
                seen_idrefs.add(idref)
                item_path, media_type = manifest[idref]
                if media_type != "application/xhtml+xml":
                    continue
                item_info = _find_zip_member(infos_by_name, item_path)
                if item_info is None:
                    continue
                if item_info.file_size > MAX_EPUB_ITEM_BYTES:
                    return ConvertResult(
                        text="",
                        status="too_large",
                        error=f"epub item exceeds {MAX_EPUB_ITEM_BYTES} bytes",
                        converter=label,
                    )
                if total_chapter_bytes + max(0, item_info.file_size) > MAX_EPUB_TOTAL_BYTES:
                    return ConvertResult(
                        text="",
                        status="too_large",
                        error=f"epub cumulative chapter bytes exceed {MAX_EPUB_TOTAL_BYTES} bytes",
                        converter=label,
                    )
                try:
                    item_bytes = zipguard.read_member_bounded(
                        zf,
                        item_info,
                        MAX_EPUB_ITEM_BYTES,
                        keep=True,
                        label="epub",
                    )
                except ValueError:
                    return ConvertResult(
                        text="",
                        status="too_large",
                        error=f"epub item exceeds {MAX_EPUB_ITEM_BYTES} bytes",
                        converter=label,
                    )

                total_chapter_bytes += len(item_bytes)
                if total_chapter_bytes > MAX_EPUB_TOTAL_BYTES:
                    return ConvertResult(
                        text="",
                        status="too_large",
                        error=f"epub cumulative chapter bytes exceed {MAX_EPUB_TOTAL_BYTES} bytes",
                        converter=label,
                    )

                sep_cost = 2 if chapters else 0
                remaining_budget = max(0, budget_limit - total_chars - sep_cost)
                chapter_res = _read_html_with_lxml(
                    item_bytes,
                    lxml_mod=lxml,
                    lxml_html=lxml_html,
                    lxml_etree=lxml_etree,
                    kind="epub",
                    output_budget=remaining_budget,
                )
                if chapter_res.status != "ok":
                    return chapter_res
                if chapter_res.fallback == "truncated":
                    truncated = True
                if chapter_res.text.strip():
                    total_chars += len(chapter_res.text) + sep_cost
                    if total_chars > budget_limit:
                        return ConvertResult(
                            text="",
                            status="too_large",
                            error=f"epub output exceeds {canonical.MAX_CANONICAL_CHARS} characters",
                            converter=label,
                        )
                    chapters.append(chapter_res.text)
    except Exception as exc:
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"epub parse error: {type(exc).__name__}",
            converter=label,
        )

    raw_md = "\n\n".join(chapters)
    return ConvertResult(
        text=raw_md,
        status="ok",
        error=None,
        converter=label,
        raw_text=raw_md,
        fallback="truncated" if truncated else None,
    )


def read_rtf(raw_bytes: bytes) -> ConvertResult:
    """Read RTF into plain text via striprtf."""
    try:
        from striprtf import striprtf  # type: ignore[import-untyped]
    except ImportError:
        return ConvertResult(
            text="",
            status="unsupported",
            error="rtf (striprtf not installed)",
        )
    try:
        text = striprtf.rtf_to_text(raw_bytes.decode("latin-1", errors="replace"))  # type: ignore[no-untyped-call]
    except Exception as exc:
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"rtf parse error: {type(exc).__name__}",
        )
    text = text.strip()
    if not text:
        return ConvertResult(text="", status="empty", error="rtf contains no text")
    return ConvertResult(
        text=text, status="ok", error=None, converter=converter_label("rtf"), raw_text=text
    )


def read_xls(raw_bytes: bytes) -> ConvertResult:
    """Read .xls (Excel 97-2003) into pipe-table Markdown via xlrd."""
    try:
        import xlrd  # type: ignore[import-untyped]
    except ImportError:
        return ConvertResult(
            text="",
            status="unsupported",
            error="xls (xlrd not installed)",
        )
    try:
        wb = xlrd.open_workbook(file_contents=raw_bytes, on_demand=True)
    except Exception as exc:
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"xls parse error: {type(exc).__name__}",
        )
    if _table_headers_mode() == "stack":
        return _read_xls_stacked(wb, xlrd)
    truncated = False
    sections: list[str] = []
    datemode = wb.datemode
    for sheet_idx in range(min(wb.nsheets, MAX_XLSX_SHEETS + 1)):
        if sheet_idx >= MAX_XLSX_SHEETS:
            truncated = True
            break
        sheet = wb.sheet_by_index(sheet_idx)
        lines: list[str] = []
        for row_idx in range(min(sheet.nrows, canonical.MAX_CSV_ROWS)):
            last = -1
            for ci in range(sheet.ncols):
                value = sheet.cell(row_idx, ci).value
                if value is not None and value != "":
                    last = ci
            if last < 0:
                continue
            cells = [
                _render_xls_cell(sheet.cell(row_idx, ci), xlrd, datemode) for ci in range(last + 1)
            ]
            lines.append("| " + " | ".join(cells) + " |")
        if sheet.nrows >= canonical.MAX_CSV_ROWS:
            lines.append(canonical.TABLE_TRUNCATED_NOTE)
            truncated = True
        if lines:
            sections.append(f"## {sheet.name}\n\n" + "\n".join(lines))
    if not sections:
        return ConvertResult(text="", status="empty", error="xls contains no text")
    return ConvertResult(
        text="\n\n".join(sections),
        status="ok",
        error=None,
        converter=converter_label("xls"),
        raw_text="\n\n".join(sections),
        fallback="truncated" if truncated else None,
    )


def _render_xls_cell(cell: Any, xlrd: Any, datemode: int) -> str:
    """Render one xlrd cell to pipe-table-safe text.

    Text follows the xlsx reader convention: newlines collapse to single
    spaces (cells stay on one pipe-table row) and literal pipes are
    escaped as ``\\|``.
    """
    if cell.ctype == xlrd.XL_CELL_TEXT:
        return (
            str(cell.value)
            .replace("\r\n", " ")
            .replace("\r", " ")
            .replace("\n", " ")
            .replace("|", "\\|")
        )
    if cell.ctype == xlrd.XL_CELL_NUMBER:
        v = cell.value
        if v == int(v):
            return str(int(v))
        return repr(v)
    if cell.ctype == xlrd.XL_CELL_BOOLEAN:
        return "TRUE" if cell.value else "FALSE"
    if cell.ctype == xlrd.XL_CELL_DATE:
        return str(xlrd.xldate_as_datetime(cell.value, datemode).isoformat())
    return ""


def _read_xls_stacked(wb: Any, xlrd: Any) -> ConvertResult:
    class _XlsAdapter:
        def __init__(self, sheet: Any, datemode: int) -> None:
            self._sheet = sheet
            self._datemode = datemode

        def iter_rows(self, values_only: bool = True) -> Any:
            for row_idx in range(min(self._sheet.nrows, canonical.MAX_CSV_ROWS)):
                row: list[Any] = []
                for ci in range(self._sheet.ncols):
                    cell = self._sheet.cell(row_idx, ci)
                    if cell.ctype == xlrd.XL_CELL_NUMBER:
                        v = cell.value
                        row.append(int(v) if v == int(v) else float(v))
                    elif cell.ctype == xlrd.XL_CELL_DATE:
                        row.append(xlrd.xldate_as_datetime(cell.value, self._datemode))
                    elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
                        row.append(bool(cell.value))
                    elif cell.value is not None and cell.value != "":
                        row.append(str(cell.value))
                    else:
                        row.append(None)
                yield tuple(row)

    budget = _BudgetCounter(canonical.MAX_CANONICAL_CHARS * 2)
    sections: list[str] = []
    truncated = False
    for sheet_idx in range(min(wb.nsheets, MAX_XLSX_SHEETS + 1)):
        if sheet_idx >= MAX_XLSX_SHEETS:
            truncated = True
            break
        sheet = wb.sheet_by_index(sheet_idx)
        adapter = _XlsAdapter(sheet, wb.datemode)
        sheet_lines, sheet_trunc = _render_xlsx_sheet(adapter, budget=budget)
        if budget.exceeded:
            break
        if sheet_trunc:
            truncated = True
        if sheet_lines:
            sections.append(f"## {sheet.name}\n\n" + "\n".join(sheet_lines))
    if budget.exceeded:
        return ConvertResult(text="", status="too_large", error="xls output exceeds limit")
    if not sections:
        return ConvertResult(text="", status="empty", error="xls contains no text")
    return ConvertResult(
        text="\n\n".join(sections),
        status="ok",
        error=None,
        converter=converter_label("xls") + "+stack1",
        raw_text="\n\n".join(sections),
        fallback="truncated" if truncated else None,
    )


def read_doc(path: Path) -> ConvertResult:
    """Read .doc via catdoc subprocess."""
    return _catdoc_read(path, "catdoc", "doc")


def read_ppt(path: Path) -> ConvertResult:
    """Read .ppt via catppt subprocess."""
    return _catdoc_read(path, "catppt", "ppt")


def _catdoc_read(path: Path, binary: str, kind: str) -> ConvertResult:
    """Run catdoc or catppt on a file path and return the text."""
    bin_path = shutil.which(binary)
    if bin_path is None:
        return ConvertResult(
            text="",
            status="unsupported",
            error=f"{kind} ({binary} not installed)",
        )
    timed_out_event = threading.Event()

    def _kill_and_flag() -> None:
        timed_out_event.set()
        proc.kill()

    timer = threading.Timer(20, _kill_and_flag)
    timer.start()
    proc = subprocess.Popen(
        [bin_path, str(path.resolve())],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    data = b""
    try:
        assert proc.stdout is not None
        data = proc.stdout.read(MAX_CATDOC_OUTPUT_BYTES + 1)
        if len(data) > MAX_CATDOC_OUTPUT_BYTES:
            proc.kill()
    finally:
        timer.cancel()
        if proc.stdout is not None:
            proc.stdout.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if timed_out_event.is_set():
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"{kind}: {binary} timed out",
        )
    if len(data) > MAX_CATDOC_OUTPUT_BYTES:
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"{kind}: {binary} output too large",
        )
    if proc.returncode not in (None, 0):
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"{kind}: {binary} failed",
        )
    text = data.decode("utf-8", errors="replace").strip()
    if not text:
        return ConvertResult(
            text="",
            status="empty",
            error=f"{kind} contains no text",
        )
    return ConvertResult(text=text, status="ok", error=None, raw_text=text)
