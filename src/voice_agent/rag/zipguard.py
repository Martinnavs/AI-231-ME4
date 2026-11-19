"""Zip and XML preflight guards for archive-backed document formats."""

from __future__ import annotations

import html.entities
import io
import re
import warnings
import xml.etree.ElementTree as ET
import zipfile
import zlib

MAX_ZIP_MEMBER_BYTES = 20 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 64 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 100.0
MAX_ZIP_MEMBERS = 1000
MAX_DOCX_XML_DEPTH = 100
HIGH_RATIO_MEDIA_MAX_BYTES = 4 * 1024 * 1024
HIGH_RATIO_MEDIA_TOTAL_BYTES = 8 * 1024 * 1024

XML_MEMBER_SUFFIXES = (".xml", ".rels", ".opf", ".ncx", ".xhtml", ".html", ".htm")
_PLAIN_DOCTYPE_SUFFIXES = (".html", ".htm", ".xhtml", ".ncx", ".svg")
_HTML_ENTITY_SUFFIXES = (".html", ".htm", ".xhtml")
_XML_BUILTIN_ENTITIES = frozenset({"lt", "gt", "amp", "apos", "quot"})
_BOMS = (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff", b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")
_SNIFF_CHUNK_BYTES = 512

_XML_FORBIDDEN_RE = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
_XML_FORBIDDEN_BYTES_RE = re.compile(rb"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
_ENTITY_DECL_RE = re.compile(r"<!\s*ENTITY\b", re.IGNORECASE)
_DOCTYPE_DECL_RE = re.compile(r"<!\s*DOCTYPE\b", re.IGNORECASE)
_DOCTYPE_SCAN_RE = re.compile(r'[\[<>"\']')
_NAMED_ENTITY_RE = re.compile(r"&([A-Za-z][A-Za-z0-9]{1,31});")
_NAMED_ENTITY_BYTES_RE = re.compile(rb"&([A-Za-z][A-Za-z0-9]{1,31});")


def _has_forbidden_doctype(text: str) -> bool:
    """Return True if any DOCTYPE in ``text`` contains an internal subset (``[``) or is unclosed."""
    pos = 0
    while True:
        match = _DOCTYPE_DECL_RE.search(text, pos)
        if match is None:
            return False
        quote: str | None = None
        closed = False
        for tok in _DOCTYPE_SCAN_RE.finditer(text, match.end()):
            ch = tok.group(0)
            if ch == "[":
                return True
            if quote is None:
                if ch in ('"', "'"):
                    quote = ch
                elif ch == "<":
                    return True
                elif ch == ">":
                    closed = True
                    pos = tok.end()
                    break
            elif ch == quote:
                quote = None
        if not closed:
            return True


def has_forbidden_xml_decl(xml_bytes: bytes, *, allow_plain_doctype: bool = False) -> bool:
    """Return True if raw XML bytes contain a forbidden DOCTYPE or ENTITY declaration."""
    if not allow_plain_doctype:
        if _XML_FORBIDDEN_BYTES_RE.search(xml_bytes):
            return True
        if b"\x00" in xml_bytes and _XML_FORBIDDEN_BYTES_RE.search(xml_bytes.replace(b"\x00", b"")):
            return True

    candidates: list[str] = []
    if xml_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates.append(xml_bytes.decode("utf-16", errors="replace"))
    else:
        candidates.append(xml_bytes.decode("utf-8", errors="replace"))
    if b"\x00" in xml_bytes:
        candidates.append(xml_bytes.replace(b"\x00", b"").decode("utf-8", errors="replace"))

    for text in candidates:
        if _ENTITY_DECL_RE.search(text):
            return True
        if allow_plain_doctype:
            if _has_forbidden_doctype(text):
                return True
        elif _DOCTYPE_DECL_RE.search(text):
            return True
    return False


def read_member_bounded(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    max_bytes: int,
    *,
    keep: bool,
    label: str = "zip",
) -> bytes:
    """Read a zip member up to ``max_bytes`` uncompressed bytes, raising ValueError if exceeded."""
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
                    f"{label} member {info.filename!r} exceeds {max_bytes} uncompressed bytes"
                )
            if keep:
                chunks.append(chunk)
    return b"".join(chunks) if keep else b""


def parse_xml_bounded(
    xml_bytes: bytes,
    max_depth: int,
    *,
    label: str = "zip",
) -> tuple[ET.Element | None, str | None]:
    """Parse XML with iterparse, aborting as soon as element depth exceeds ``max_depth``."""
    root: ET.Element | None = None
    depth = 0
    for event, elem in ET.iterparse(io.BytesIO(xml_bytes), events=("start", "end")):
        if event == "start":
            depth += 1
            if root is None:
                root = elem
            if depth > max_depth:
                return None, f"{label} XML is nested deeper than {max_depth} levels"
        else:
            depth -= 1
    if root is None:
        return None, f"{label} XML parse error: empty document"
    return root, None


def format_zip_error(exc: Exception, *, label: str = "zip") -> str:
    """Map a zipfile/zlib/OS exception to a consistent unreadable error message."""
    if isinstance(exc, (UserWarning, zipfile.BadZipFile)):
        if "overlap" in str(exc).lower():
            return f"{label} has overlapping zip entries"
        return f"invalid {label} archive: {exc}"
    if isinstance(exc, ValueError):
        return str(exc)
    return f"invalid {label} archive: {exc}"


def _is_xml_member(filename: str) -> bool:
    return filename.lower().endswith(XML_MEMBER_SUFFIXES)


def _allows_plain_doctype(filename: str) -> bool:
    return filename.lower().endswith(_PLAIN_DOCTYPE_SUFFIXES)


def _replace_named_entity_bytes(match: re.Match[bytes]) -> bytes:
    name = match.group(1).decode("ascii")
    if name in _XML_BUILTIN_ENTITIES:
        return match.group(0)
    codepoint = html.entities.name2codepoint.get(name)
    if codepoint is None:
        return match.group(0)
    return f"&#{codepoint};".encode("ascii")


def _replace_named_entity_str(match: re.Match[str]) -> str:
    name = match.group(1)
    if name in _XML_BUILTIN_ENTITIES:
        return match.group(0)
    codepoint = html.entities.name2codepoint.get(name)
    if codepoint is None:
        return match.group(0)
    return f"&#{codepoint};"


def _normalize_html_entities_for_xml(xml_bytes: bytes) -> bytes:
    """Replace HTML named entities (e.g. ``&nbsp;``) with numeric XML references for iterparse."""
    if xml_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = xml_bytes.decode("utf-16", errors="replace")
        replaced = _NAMED_ENTITY_RE.sub(_replace_named_entity_str, text)
        return replaced.encode("utf-16")
    return _NAMED_ENTITY_BYTES_RE.sub(_replace_named_entity_bytes, xml_bytes)


def _member_starts_with_xml(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    max_bytes: int,
) -> bool:
    """Return True if a member's first non-BOM/NUL/whitespace bytes start with ``<``."""
    checked = 0
    saw_bom = False
    with zf.open(info, "r") as member:
        while checked < max_bytes:
            to_read = min(_SNIFF_CHUNK_BYTES, max_bytes - checked)
            chunk = member.read(to_read)
            if not chunk:
                return False
            checked += len(chunk)
            if not saw_bom:
                saw_bom = True
                for bom in _BOMS:
                    if chunk.startswith(bom):
                        chunk = chunk[len(bom) :]
                        break
            cleaned = chunk.replace(b"\x00", b"").lstrip(b" \t\r\n")
            if cleaned:
                return cleaned.startswith(b"<")
    return False


def preflight_zip(
    raw_bytes: bytes,
    *,
    label: str = "zip",
    scan_xml_members: bool = True,
    max_members: int = MAX_ZIP_MEMBERS,
    max_member_bytes: int = MAX_ZIP_MEMBER_BYTES,
    max_total_bytes: int = MAX_ZIP_TOTAL_BYTES,
    max_compression_ratio: float = MAX_ZIP_COMPRESSION_RATIO,
    max_xml_depth: int = MAX_DOCX_XML_DEPTH,
) -> str | None:
    """Validate zip archive limits and optionally scan XML/HTML members for DOCTYPE/depth."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                infos = zf.infolist()
                if len(infos) > max_members:
                    return f"{label} has too many zip members ({len(infos)} > {max_members})"

                ratio_checked_suffixes = XML_MEMBER_SUFFIXES + (
                    ".txt",
                    ".csv",
                    ".json",
                    ".svg",
                )
                seen_names: set[str] = set()
                total_declared_bytes = 0
                non_xml_declared_bytes = 0
                high_ratio_media_bytes = 0
                xml_infos: list[zipfile.ZipInfo] = []
                other_infos: list[zipfile.ZipInfo] = []
                for info in infos:
                    if info.filename in seen_names:
                        return f"duplicate zip member name: {info.filename!r}"
                    seen_names.add(info.filename)
                    is_high_ratio = info.file_size > 0 and (
                        info.compress_size == 0
                        or (info.file_size / info.compress_size) > max_compression_ratio
                    )
                    if is_high_ratio:
                        if info.filename.lower().endswith(ratio_checked_suffixes):
                            return f"suspicious compression ratio in {info.filename!r}"
                        if info.file_size > HIGH_RATIO_MEDIA_MAX_BYTES:
                            return f"suspicious compression ratio in {info.filename!r}"
                        high_ratio_media_bytes += info.file_size
                        if high_ratio_media_bytes > HIGH_RATIO_MEDIA_TOTAL_BYTES:
                            return (
                                "high-ratio media members exceed "
                                f"{HIGH_RATIO_MEDIA_TOTAL_BYTES} bytes"
                            )
                    if info.file_size > max_member_bytes:
                        return (
                            f"{label} member {info.filename!r} exceeds {max_member_bytes} "
                            "uncompressed bytes"
                        )
                    member_declared = max(0, info.file_size)
                    total_declared_bytes += member_declared
                    if total_declared_bytes > max_total_bytes:
                        return f"{label} total uncompressed size exceeds {max_total_bytes} bytes"
                    if not info.is_dir():
                        if _is_xml_member(info.filename):
                            xml_infos.append(info)
                        else:
                            other_infos.append(info)
                            non_xml_declared_bytes += member_declared

                by_offset = sorted(infos, key=lambda item: item.header_offset)
                for prev, curr in zip(by_offset, by_offset[1:], strict=False):
                    min_next_offset = prev.header_offset + 30 + max(0, prev.compress_size)
                    if curr.header_offset < min_next_offset:
                        return f"{label} has overlapping zip entries"

                if scan_xml_members:
                    for info in other_infos:
                        sniff_limit = min(max_member_bytes, max_total_bytes)
                        if _member_starts_with_xml(zf, info, sniff_limit):
                            xml_infos.append(info)
                            non_xml_declared_bytes -= max(0, info.file_size)

                    total_actual_bytes = non_xml_declared_bytes
                    for info in xml_infos:
                        remaining = max(0, max_total_bytes - total_actual_bytes)
                        bounded_by_total = remaining < max_member_bytes
                        limit = remaining if bounded_by_total else max_member_bytes
                        try:
                            member_bytes = read_member_bounded(
                                zf, info, limit, keep=True, label=label
                            )
                        except ValueError:
                            if bounded_by_total:
                                return (
                                    f"{label} total uncompressed size exceeds "
                                    f"{max_total_bytes} bytes"
                                )
                            raise
                        total_actual_bytes += len(member_bytes)
                        allow_plain = _allows_plain_doctype(info.filename)
                        if has_forbidden_xml_decl(member_bytes, allow_plain_doctype=allow_plain):
                            return f"{label} XML contains forbidden DOCTYPE or ENTITY declaration"
                        parse_bytes = (
                            _normalize_html_entities_for_xml(member_bytes)
                            if info.filename.lower().endswith(_HTML_ENTITY_SUFFIXES)
                            else member_bytes
                        )
                        try:
                            root, depth_error = parse_xml_bounded(
                                parse_bytes, max_xml_depth, label=label
                            )
                        except ET.ParseError as exc:
                            return f"{label} XML parse error: {exc}"
                        if depth_error is not None or root is None:
                            return depth_error or f"{label} XML parse error"
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
        return format_zip_error(exc, label=label)
    return None
