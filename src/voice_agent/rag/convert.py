"""MarkItDown adapter with zip/XML preflight guards and canonical post-processing."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from voice_agent.rag import canonical, zipguard

SUPPORTED_MARKITDOWN_KINDS = frozenset({"pptx", "xlsx", "html", "epub", "pdf", "docx"})
SUPPORTED_DIRECT_KINDS = frozenset({"xlsx", "pptx", "html", "epub"})
ZIP_PREFLIGHT_KINDS = frozenset({"pptx", "xlsx", "epub"})
_ALLOWED_ENGINES = frozenset({"markitdown", "direct"})
_ALLOWED_CHILD_ENV_VARS = ("PATH", "HOME", "LANG", "LC_ALL")
_ALLOWED_STATUSES = frozenset({"ok", "unsupported", "unreadable", "empty", "too_large"})
_RESULT_FIELDS = frozenset({"text", "status", "error", "converter", "raw_text", "fallback"})
_MAX_CHILD_STDOUT_BYTES = 32 * 1024 * 1024
_DEFAULT_MAX_ADDRESS_SPACE_MB = 4096
_MAX_CHILD_ERROR_CHARS = 200
_MAX_CHILD_META_CHARS = 120
_SAFE_CHILD_META_RE = re.compile(r"[A-Za-z0-9._+/=-]*")

_STREAM_METADATA: dict[str, tuple[str, str]] = {
    "pptx": (
        ".pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    "xlsx": (
        ".xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    "html": (".html", "text/html"),
    "epub": (".epub", "application/epub+zip"),
    "pdf": (".pdf", "application/pdf"),
    "docx": (
        ".docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
}

_CONVERTER_CLASSES: dict[str, str] = {
    "pptx": "PptxConverter",
    "xlsx": "XlsxConverter",
    "html": "HtmlConverter",
    "epub": "EpubConverter",
    "pdf": "PdfConverter",
    "docx": "DocxConverter",
}


@dataclass(frozen=True)
class ConvertResult:
    """Outcome of converting raw document bytes into canonical Markdown."""

    text: str
    status: str
    error: str | None = None
    converter: str = ""
    # Diagnostic raw converter output for in-process digit tests; omitted from IPC and equality.
    raw_text: str = field(default="", compare=False)
    fallback: str | None = None


def _normalize_kind(kind: str) -> str:
    norm = kind.strip().lower().lstrip(".")
    if norm == "htm":
        return "html"
    return norm


def _import_markitdown_converter(norm_kind: str) -> tuple[str, Any, Any] | ConvertResult:
    """Import MarkItDown converter class and StreamInfo without instantiating MarkItDown()."""
    try:
        import markitdown
        from markitdown import StreamInfo
        from markitdown import converters as md_converters
    except ImportError:
        return ConvertResult(
            text="",
            status="unsupported",
            error="markitdown not installed",
        )

    converter_cls = getattr(md_converters, _CONVERTER_CLASSES[norm_kind], None)
    if converter_cls is None:
        return ConvertResult(
            text="",
            status="unsupported",
            error="markitdown not installed",
        )

    version = getattr(markitdown, "__version__", None) or "unknown"
    ext, mimetype = _STREAM_METADATA[norm_kind]
    return f"markitdown-{version}", converter_cls, StreamInfo(extension=ext, mimetype=mimetype)


def _run_converter(raw_bytes: bytes, converter_cls: Any, stream_info: Any) -> str:
    """Invoke a MarkItDown DocumentConverter on ``raw_bytes`` with warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        converter = converter_cls()
        with io.BytesIO(raw_bytes) as stream:
            conv_result = converter.convert(stream, stream_info)
    return getattr(conv_result, "markdown", None) or ""


def _finalize_markdown(
    raw_text: str,
    *,
    kind: str,
    converter: str,
    fallback: str | None = None,
) -> ConvertResult:
    """Post-process Markdown text and enforce non-empty and character-cap checks."""
    if not raw_text.strip():
        return ConvertResult(
            text="",
            status="empty",
            error=f"{kind} contains no text",
            converter=converter,
            fallback=fallback,
        )

    processed = canonical.postprocess_markdown(raw_text, kind=kind)
    if not processed.text.strip():
        return ConvertResult(
            text="",
            status="empty",
            error=f"{kind} contains no text",
            converter=converter,
            fallback=fallback,
        )

    if len(processed.text) > canonical.MAX_CANONICAL_CHARS:
        return ConvertResult(
            text="",
            status="too_large",
            error=f"{kind} output exceeds {canonical.MAX_CANONICAL_CHARS} characters",
            converter=converter,
            fallback=fallback,
        )

    return ConvertResult(
        text=processed.text,
        status="ok",
        error=None,
        converter=converter,
        raw_text=raw_text,
        fallback=fallback,
    )


def _finalize_pdf_pages(
    pages: Sequence[str],
    *,
    converter: str,
    raw_text: str,
    fallback: str | None = None,
) -> ConvertResult:
    """Post-process per-page PDF text and enforce non-empty and character-cap checks."""
    processed = canonical.postprocess_pdf_pages(pages)
    if not processed.text.strip():
        return ConvertResult(
            text="",
            status="empty",
            error="pdf contains no text",
            converter=converter,
            fallback=fallback,
        )

    if len(processed.text) > canonical.MAX_CANONICAL_CHARS:
        return ConvertResult(
            text="",
            status="too_large",
            error=f"pdf output exceeds {canonical.MAX_CANONICAL_CHARS} characters",
            converter=converter,
            fallback=fallback,
        )

    return ConvertResult(
        text=processed.text,
        status="ok",
        error=None,
        converter=converter,
        raw_text=raw_text,
        fallback=fallback,
    )


def _pdf_metadata(raw_bytes: bytes, pypdf_pages: Sequence[tuple[int, str]]) -> tuple[int, str]:
    """Return ``(total_page_count, pypdf_version)`` for a validated PDF."""
    max_page = max((page_num for page_num, _ in pypdf_pages), default=0)
    try:
        import pypdf
    except ImportError:
        return max_page, "unknown"
    ver = getattr(pypdf, "__version__", None) or "unknown"
    try:
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
        return len(reader.pages), ver
    except Exception:
        return max_page, ver


def _split_pdf_pages(raw_text: str, page_count: int) -> list[str] | None:
    """Split PdfConverter output on ``\\f`` when it matches ``page_count`` pages."""
    if page_count <= 0 or "\f" not in raw_text:
        return None
    parts = raw_text.split("\f")
    if len(parts) == page_count + 1 and not parts[-1].strip():
        return parts[:-1]
    if len(parts) == page_count:
        return parts
    return None


def convert_pdf(raw_bytes: bytes, mode: str = "markitdown") -> ConvertResult:
    """Convert PDF bytes using ``markitdown`` (with pypdf fallback) or ``pypdf`` only.

    Runs ``docindex.load_pdf_pages`` first so encrypted, corrupt, over-limit
    (>500 pages), and no-text PDFs return the exact legacy ``(status, error)``.
    """
    if mode not in ("markitdown", "pypdf"):
        raise ValueError(f"unsupported pdf conversion mode: {mode!r}")

    from voice_agent.rag import docindex

    pypdf_pages, status, err = docindex.load_pdf_pages(raw_bytes)
    if status != "ok":
        return ConvertResult(text="", status=status, error=err)

    page_count, pypdf_ver = _pdf_metadata(raw_bytes, pypdf_pages)
    slot_count = max(page_count, max((p for p, _ in pypdf_pages), default=0))
    pypdf_page_list = [""] * slot_count
    for page_num, page_text in pypdf_pages:
        pypdf_page_list[page_num - 1] = page_text
    pypdf_raw_text = "\n\n".join(page for page in pypdf_page_list if page)
    pypdf_converter = f"pypdf-{pypdf_ver}"

    if mode == "pypdf":
        return _finalize_pdf_pages(
            pypdf_page_list,
            converter=pypdf_converter,
            raw_text=pypdf_raw_text,
        )

    imported = _import_markitdown_converter("pdf")
    if isinstance(imported, ConvertResult):
        return imported
    converter_label, converter_cls, stream_info = imported

    try:
        raw_text = _run_converter(raw_bytes, converter_cls, stream_info)
    except Exception:
        return _finalize_pdf_pages(
            pypdf_page_list,
            converter=pypdf_converter,
            raw_text=pypdf_raw_text,
            fallback="legacy",
        )

    md_pages = _split_pdf_pages(raw_text, page_count)
    if md_pages is None:
        return _finalize_pdf_pages(
            pypdf_page_list,
            converter=pypdf_converter,
            raw_text=pypdf_raw_text,
            fallback="pypdf-pages",
        )

    return _finalize_pdf_pages(
        md_pages,
        converter=converter_label,
        raw_text=raw_text,
    )


def _convert_docx(raw_bytes: bytes) -> ConvertResult:
    """Convert DOCX bytes after legacy guards and full XML-member zip preflight."""
    from voice_agent.rag import docindex

    legacy_text, status, err = docindex.load_docx(raw_bytes)
    if status != "ok":
        return ConvertResult(text="", status=status, error=err)

    preflight_err = zipguard.preflight_zip(
        raw_bytes,
        label="docx",
        scan_xml_members=True,
    )
    if preflight_err is not None:
        return ConvertResult(text="", status="unreadable", error=preflight_err)

    imported = _import_markitdown_converter("docx")
    if isinstance(imported, ConvertResult):
        return imported
    converter_label, converter_cls, stream_info = imported

    try:
        raw_text = _run_converter(raw_bytes, converter_cls, stream_info)
    except Exception:
        return _finalize_markdown(
            legacy_text,
            kind="docx",
            converter="legacy",
            fallback="legacy",
        )

    return _finalize_markdown(
        raw_text,
        kind="docx",
        converter=converter_label,
    )


def convert_with_markitdown(raw_bytes: bytes, kind: str) -> ConvertResult:
    """Convert supported document bytes using MarkItDown converters directly.

    Runs format guards and ``zipguard.preflight_zip(scan_xml_members=True)``
    before any converter for archive-backed formats. Never instantiates
    ``MarkItDown()`` (which loads magika) and never passes an ``llm_client``.
    """
    norm_kind = _normalize_kind(kind)
    if norm_kind not in SUPPORTED_MARKITDOWN_KINDS:
        return ConvertResult(
            text="",
            status="unsupported",
            error=f"unsupported format: {norm_kind or kind}",
        )

    if norm_kind == "pdf":
        return convert_pdf(raw_bytes, mode="markitdown")
    if norm_kind == "docx":
        return _convert_docx(raw_bytes)

    if norm_kind in ZIP_PREFLIGHT_KINDS:
        preflight_err = zipguard.preflight_zip(
            raw_bytes,
            label=norm_kind,
            scan_xml_members=True,
        )
        if preflight_err is not None:
            return ConvertResult(
                text="",
                status="unreadable",
                error=preflight_err,
            )

    imported = _import_markitdown_converter(norm_kind)
    if isinstance(imported, ConvertResult):
        return imported
    converter_label, converter_cls, stream_info = imported

    try:
        raw_text = _run_converter(raw_bytes, converter_cls, stream_info)
    except Exception as exc:
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"{norm_kind} parse error: {type(exc).__name__}",
            converter=converter_label,
        )

    return _finalize_markdown(
        raw_text,
        kind=norm_kind,
        converter=converter_label,
    )


def convert_with_direct(raw_bytes: bytes, kind: str) -> ConvertResult:
    """Convert XLSX, PPTX, HTML, or EPUB bytes using pure-Python direct readers."""
    norm_kind = _normalize_kind(kind)
    if norm_kind not in SUPPORTED_DIRECT_KINDS:
        return ConvertResult(
            text="",
            status="unsupported",
            error=f"unsupported format: {norm_kind or kind}",
        )

    if norm_kind in ZIP_PREFLIGHT_KINDS:
        preflight_err = zipguard.preflight_zip(
            raw_bytes,
            label=norm_kind,
            scan_xml_members=True,
        )
        if preflight_err is not None:
            return ConvertResult(
                text="",
                status="unreadable",
                error=preflight_err,
            )

    from voice_agent.rag import readers

    try:
        if norm_kind == "xlsx":
            reader_res = readers.read_xlsx(raw_bytes)
        elif norm_kind == "pptx":
            reader_res = readers.read_pptx(raw_bytes)
        elif norm_kind == "html":
            reader_res = readers.read_html(raw_bytes)
        else:
            reader_res = readers.read_epub(raw_bytes)
    except Exception as exc:
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"{norm_kind} parse error: {type(exc).__name__}",
            converter=readers.converter_label(norm_kind),
        )

    if reader_res.status != "ok":
        return reader_res

    return _finalize_markdown(
        reader_res.text,
        kind=norm_kind,
        converter=reader_res.converter,
        fallback=reader_res.fallback,
    )


def _child_env(engine: str = "markitdown") -> dict[str, str]:
    """Return a minimal environment containing only allow-listed variables."""
    del engine
    return {key: os.environ[key] for key in _ALLOWED_CHILD_ENV_VARS if key in os.environ}


def _disable_core_dumps() -> None:
    """Set RLIMIT_CORE=0 in the worker process when supported by the platform."""
    try:
        import resource
    except ImportError:
        return
    rlimit_core = getattr(resource, "RLIMIT_CORE", None)
    if rlimit_core is None:
        return
    with contextlib.suppress(OSError, ValueError):
        resource.setrlimit(rlimit_core, (0, 0))


def _apply_address_space_limit(max_address_space_mb: int) -> None:
    """Apply RLIMIT_AS in the worker process when supported by the platform."""
    if max_address_space_mb <= 0:
        return
    try:
        import resource
    except ImportError:
        return
    rlimit_as = getattr(resource, "RLIMIT_AS", None)
    if rlimit_as is None:
        return
    limit_bytes = max_address_space_mb * 1024 * 1024
    try:
        _soft, hard = resource.getrlimit(rlimit_as)
        target = limit_bytes if hard == resource.RLIM_INFINITY else min(limit_bytes, hard)
        resource.setrlimit(rlimit_as, (target, target))
    except (OSError, ValueError):
        return


def _close_pipes(proc: subprocess.Popen[bytes]) -> None:
    """Close any open subprocess pipes without raising OSError."""
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe is not None:
            with contextlib.suppress(OSError):
                pipe.close()


def _kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    """Kill the isolated child's own process group without raising."""
    pid = proc.pid
    if pid <= 0:
        return
    killpg = getattr(os, "killpg", None)
    sigkill = getattr(signal, "SIGKILL", None)
    if killpg is not None and sigkill is not None:
        try:
            killpg(pid, sigkill)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()


def _communicate_bounded(
    proc: subprocess.Popen[bytes],
    raw_bytes: bytes,
    *,
    timeout_s: float,
    max_stdout_bytes: int,
) -> tuple[bytearray, bool]:
    """Stream stdin and read stdout in chunks, aborting as soon as stdout exceeds cap."""
    stdin_pipe = proc.stdin
    stdout_pipe = proc.stdout
    assert stdin_pipe is not None and stdout_pipe is not None

    deadline = time.monotonic() + timeout_s
    stdout_buf = bytearray()
    stdin_view = memoryview(raw_bytes)
    stdin_offset = 0

    with selectors.DefaultSelector() as sel:
        if len(stdin_view) == 0:
            with contextlib.suppress(OSError):
                stdin_pipe.close()
            stdin_open = False
        else:
            os.set_blocking(stdin_pipe.fileno(), False)
            sel.register(stdin_pipe, selectors.EVENT_WRITE, data="stdin")
            stdin_open = True

        os.set_blocking(stdout_pipe.fileno(), False)
        sel.register(stdout_pipe, selectors.EVENT_READ, data="stdout")
        stdout_open = True

        while stdin_open or stdout_open:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, timeout_s)
            events = sel.select(timeout=remaining)
            if not events:
                raise subprocess.TimeoutExpired(proc.args, timeout_s)
            for key, _mask in events:
                if key.data == "stdin":
                    try:
                        written = os.write(key.fd, stdin_view[stdin_offset : stdin_offset + 65536])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        sel.unregister(key.fileobj)
                        with contextlib.suppress(OSError):
                            stdin_pipe.close()
                        stdin_open = False
                        continue
                    stdin_offset += written
                    if stdin_offset >= len(stdin_view):
                        sel.unregister(key.fileobj)
                        with contextlib.suppress(OSError):
                            stdin_pipe.close()
                        stdin_open = False
                elif key.data == "stdout":
                    try:
                        chunk = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        chunk = b""
                    if not chunk:
                        sel.unregister(key.fileobj)
                        with contextlib.suppress(OSError):
                            stdout_pipe.close()
                        stdout_open = False
                        continue
                    if len(stdout_buf) + len(chunk) > max_stdout_bytes:
                        return stdout_buf, True
                    stdout_buf.extend(chunk)

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(proc.args, timeout_s)
    proc.wait(timeout=remaining)
    return stdout_buf, False


def _sanitize_child_field(value: str, max_len: int) -> str:
    """Keep only printable characters up to ``max_len``."""
    chars: list[str] = []
    for ch in value:
        if ch.isprintable():
            chars.append(ch)
            if len(chars) >= max_len:
                break
    return "".join(chars)


def _parse_child_result(
    stdout_bytes: bytes | bytearray,
    *,
    engine: str = "markitdown",
) -> ConvertResult:
    """Validate and deserialize one ConvertResult JSON object from child stdout."""
    if len(stdout_bytes) > _MAX_CHILD_STDOUT_BYTES:
        return ConvertResult(
            text="",
            status="unreadable",
            error="isolated converter output exceeded 32 MB",
            converter=engine,
        )
    try:
        stdout_text = stdout_bytes.decode("utf-8")
        payload = json.loads(stdout_text)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        return ConvertResult(
            text="",
            status="unreadable",
            error="isolated converter returned invalid JSON",
            converter=engine,
        )

    if not isinstance(payload, dict) or set(payload.keys()) != _RESULT_FIELDS:
        return ConvertResult(
            text="",
            status="unreadable",
            error="isolated converter returned malformed result",
            converter=engine,
        )

    text = payload["text"]
    status = payload["status"]
    error = payload["error"]
    converter = payload["converter"]
    raw_text = payload["raw_text"]
    fallback = payload["fallback"]

    if (
        not isinstance(text, str)
        or not isinstance(status, str)
        or status not in _ALLOWED_STATUSES
        or (error is not None and not isinstance(error, str))
        or not isinstance(converter, str)
        or not isinstance(raw_text, str)
        or (fallback is not None and not isinstance(fallback, str))
    ):
        return ConvertResult(
            text="",
            status="unreadable",
            error="isolated converter returned invalid field types",
            converter=engine,
        )

    clean_error = (
        _sanitize_child_field(error, _MAX_CHILD_ERROR_CHARS) if error is not None else None
    )
    clean_converter = _sanitize_child_field(converter, _MAX_CHILD_META_CHARS)
    clean_fallback = (
        _sanitize_child_field(fallback, _MAX_CHILD_META_CHARS) if fallback is not None else None
    )
    if not _SAFE_CHILD_META_RE.fullmatch(clean_converter) or (
        clean_fallback is not None and not _SAFE_CHILD_META_RE.fullmatch(clean_fallback)
    ):
        return ConvertResult(
            text="",
            status="unreadable",
            error="isolated converter returned invalid metadata",
            converter=engine,
        )

    if status == "ok" and len(text) > canonical.MAX_CANONICAL_CHARS:
        return ConvertResult(
            text="",
            status="unreadable",
            error="isolated converter output exceeded max characters",
            converter=engine,
        )

    return ConvertResult(
        text=text,
        status=status,
        error=clean_error,
        converter=clean_converter,
        raw_text=raw_text,
        fallback=clean_fallback,
    )


def convert_isolated(
    raw_bytes: bytes,
    kind: str,
    *,
    engine: str = "markitdown",
    timeout_s: float = 30.0,
    max_address_space_mb: int = _DEFAULT_MAX_ADDRESS_SPACE_MB,
) -> ConvertResult:
    """Run the selected converter engine in an isolated Python subprocess.

    Spawns ``python -I -m voice_agent.rag.convert`` with a bare environment,
    address-space cap, discarded stderr, and a wall-clock timeout. Any child
    timeout, crash, non-zero exit, oversize output, or malformed payload
    returns an ``unreadable`` ``ConvertResult`` without raising.
    """
    if engine not in _ALLOWED_ENGINES:
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"unsupported conversion engine: {engine!r}",
            converter=str(engine),
        )
    if timeout_s <= 0:
        return ConvertResult(
            text="",
            status="unreadable",
            error="invalid conversion timeout",
            converter=engine,
        )
    if max_address_space_mb <= 0:
        return ConvertResult(
            text="",
            status="unreadable",
            error="invalid address space limit",
            converter=engine,
        )

    cmd = [
        sys.executable,
        "-I",
        "-m",
        "voice_agent.rag.convert",
        kind,
        "--max-as-mb",
        str(max_address_space_mb),
    ]
    if engine != "markitdown":
        cmd.extend(["--engine", engine])
    try:
        proc: subprocess.Popen[bytes] = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_child_env(engine),
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"isolated converter spawn failed: {type(exc).__name__}",
            converter=engine,
        )

    try:
        stdout_bytes, exceeded_cap = _communicate_bounded(
            proc,
            raw_bytes,
            timeout_s=timeout_s,
            max_stdout_bytes=_MAX_CHILD_STDOUT_BYTES,
        )
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        _close_pipes(proc)
        proc.wait()
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"conversion timed out after {timeout_s:g}s",
            converter=engine,
        )
    except Exception as exc:
        _kill_process_group(proc)
        _close_pipes(proc)
        proc.wait()
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"isolated convert failed: {type(exc).__name__}",
            converter=engine,
        )
    except BaseException:
        _kill_process_group(proc)
        _close_pipes(proc)
        proc.wait()
        raise

    if exceeded_cap or len(stdout_bytes) > _MAX_CHILD_STDOUT_BYTES:
        _kill_process_group(proc)
        _close_pipes(proc)
        proc.wait()
        return ConvertResult(
            text="",
            status="unreadable",
            error="isolated converter output exceeded 32 MB",
            converter=engine,
        )
    if proc.returncode != 0:
        return ConvertResult(
            text="",
            status="unreadable",
            error=f"isolated converter exited with code {proc.returncode}",
            converter=engine,
        )
    return _parse_child_result(stdout_bytes, engine=engine)


def _worker_main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the isolated conversion child process."""
    _disable_core_dumps()
    args = sys.argv[1:] if argv is None else argv
    if not args:
        return 2
    kind = args[0]
    max_address_space_mb = _DEFAULT_MAX_ADDRESS_SPACE_MB
    engine = "markitdown"
    rest = args[1:]
    idx = 0
    while idx < len(rest):
        flag = rest[idx]
        if idx + 1 >= len(rest):
            return 2
        val = rest[idx + 1]
        if flag == "--max-as-mb":
            try:
                max_address_space_mb = int(val)
            except ValueError:
                return 2
            if max_address_space_mb <= 0:
                return 2
        elif flag == "--engine":
            if val not in _ALLOWED_ENGINES:
                return 2
            engine = val
        else:
            return 2
        idx += 2

    _apply_address_space_limit(max_address_space_mb)
    raw_bytes = sys.stdin.buffer.read()
    if engine == "direct":
        result = convert_with_direct(raw_bytes, kind)
    else:
        result = convert_with_markitdown(raw_bytes, kind)
    payload = asdict(result)
    payload["raw_text"] = ""
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main())
