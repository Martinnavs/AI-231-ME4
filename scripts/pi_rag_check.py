"""Diagnostic and benchmark check for SQLite FTS5 RAG retrieval on Raspberry Pi.

    python scripts/pi_rag_check.py --folder DIR [--queries FILE] [--k 4]
        [--in-place] [--compare-bm25]

Measures Python/SQLite environment, FTS5 probe & compile options, lean retrieval import
time/RSS, cold DocIndex refresh, FTS5 build, reopen-to-first-query, query p50/p95, peak RSS,
DB size, and optional rank-bm25 comparison without writing into DIR unless --in-place.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import platform
import resource
import sqlite3
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voice_agent.rag.docindex import Chunk, Retriever
    from voice_agent.rag.fts5 import Fts5TokenizerMode

REPO_ROOT = Path(__file__).resolve().parent.parent
HEAVY_MODULES: tuple[str, ...] = (
    "numpy",
    "fastmcp",
    "markitdown",
    "rank_bm25",
)
DEFAULT_QUERIES: tuple[str, ...] = (
    "daily meal allowance expense claims",
    "factory reset pinhole twelve seconds",
    "approved renovation budget March 2026",
    "soil moisture probe calibration maintenance menu",
    "fault code ERR-408 thermistor response",
    "fasting hours prior to blood draw",
    "drip irrigation lines garden schedule",
    "Pump House morning shift Tuesday",
    "IT desk floor room number",
    "facilities extension equipment return pickup",
)


def _ensure_src_on_path() -> None:
    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _read_proc_status_kb(field: str) -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(f"{field}:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1])
    except OSError:
        return None
    return None


def _current_rss_kb() -> int:
    vm_rss = _read_proc_status_kb("VmRSS")
    if vm_rss is not None:
        return vm_rss
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _peak_rss_kb() -> int:
    vm_hwm = _read_proc_status_kb("VmHWM")
    if vm_hwm is not None:
        return vm_hwm
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _format_mib(kb: int) -> str:
    return f"{kb / 1024.0:.2f} MiB ({kb} kB)"


def _sqlite_info() -> tuple[str, list[str]]:
    conn = sqlite3.connect(":memory:")
    try:
        rows = conn.execute("PRAGMA compile_options").fetchall()
        options = [str(row[0]) for row in rows if row]
    finally:
        conn.close()
    return sqlite3.sqlite_version, options


def _parse_query_line(line: str, line_no: int) -> str:
    if line.startswith("{"):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON query on line {line_no}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"expected JSON object on line {line_no}")
        for key in ("q", "query", "question"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        raise ValueError(f"missing 'q'/'query'/'question' field on line {line_no}")
    return line


def _load_queries(path: Path | None) -> tuple[list[str], str]:
    if path is None:
        return list(DEFAULT_QUERIES), "builtin"
    if not path.is_file():
        raise FileNotFoundError(f"queries file not found: {path}")
    queries: list[str] = []
    text = path.read_text(encoding="utf-8")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        queries.append(_parse_query_line(stripped, line_no))
    if not queries:
        raise ValueError(f"queries file contains no queries: {path}")
    return queries, str(path)


def _percentile_ms(samples: Sequence[float], pct: float) -> float:
    if not samples:
        return 0.0
    if pct == 50.0:
        return float(statistics.median(samples))
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, int((pct / 100.0) * (len(ordered) - 1))))
    return float(ordered[idx])


def _loaded_heavy_modules() -> list[str]:
    return [
        mod
        for mod in HEAVY_MODULES
        if any(name == mod or name.startswith(f"{mod}.") for name in sys.modules)
    ]


class _TimedBuildRetriever:
    """Thin proxy recording ``Retriever.build`` duration without altering behaviour."""

    def __init__(self, inner: Retriever, build_seconds: list[float]) -> None:
        self._inner = inner
        self._build_seconds = build_seconds

    def build(self, chunks: Sequence[Chunk]) -> None:
        t0 = time.perf_counter()
        self._inner.build(chunks)
        self._build_seconds.append(time.perf_counter() - t0)

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        return self._inner.rank(query, k)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _benchmark_queries(
    rank_fn: Callable[[str], Sequence[object]],
    queries: Sequence[str],
) -> tuple[float, float, int]:
    latencies_ms: list[float] = []
    nonempty = 0
    for q in queries:
        t0 = time.perf_counter()
        hits = rank_fn(q)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if hits:
            nonempty += 1
    return _percentile_ms(latencies_ms, 50.0), _percentile_ms(latencies_ms, 95.0), nonempty


def _close_quietly(retriever: object) -> None:
    close_fn = getattr(retriever, "close", None)
    if callable(close_fn):
        with contextlib.suppress(Exception):
            close_fn()


def _run_bm25_comparison(
    chunks: Sequence[Chunk],
    queries: Sequence[str],
    bm25_cls: type[Retriever],
) -> dict[str, Any]:
    if importlib.util.find_spec("rank_bm25") is None:
        return {"status": "skipped (rank-bm25 not installed)"}

    t_build = time.perf_counter()
    bm25 = bm25_cls()
    bm25.build(chunks)
    build_s = time.perf_counter() - t_build

    rank_limit = max(len(chunks), 1)
    t_first = time.perf_counter()
    bm25.rank(queries[0], rank_limit)
    first_query_ms = (time.perf_counter() - t_first) * 1000.0

    p50_ms, p95_ms, nonempty = _benchmark_queries(
        lambda q: bm25.rank(q, rank_limit),
        queries,
    )
    rss_kb = _current_rss_kb()
    return {
        "status": "ok",
        "build_s": build_s,
        "first_query_ms": first_query_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "nonempty": nonempty,
        "rss_kb": rss_kb,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Documents folder to index and query.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Optional query file (one query per line or JSONL with 'q'/'query').",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Top-k results per query (default: 4).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Store FTS5 cache DB under DIR/.rag_cache instead of a temporary directory.",
    )
    parser.add_argument(
        "--compare-bm25",
        action="store_true",
        help="Also benchmark rank-bm25 on the same chunks when installed.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("pretok", "porter"),
        default="pretok",
        help="FTS5 tokenizer mode (default: pretok).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _format_compile_options(options: Sequence[str]) -> str:
    has_fts5 = any(opt == "ENABLE_FTS5" or opt.startswith("ENABLE_FTS5=") for opt in options)
    fts_opts = [opt for opt in options if "FTS" in opt]
    flag = "1" if has_fts5 else "0"
    if fts_opts:
        return f"ENABLE_FTS5={flag} ({', '.join(fts_opts)})"
    return f"ENABLE_FTS5={flag}"


def _run_check(args: argparse.Namespace) -> list[str]:
    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"folder does not exist or is not a directory: {folder}")
    if args.k <= 0:
        raise ValueError(f"--k must be >= 1, got {args.k}")

    queries, queries_source = _load_queries(args.queries)
    sqlite_ver, compile_opts = _sqlite_info()

    _ensure_src_on_path()
    t_imp0 = time.perf_counter()
    import voice_agent.rag.backends as backends_mod
    import voice_agent.rag.docindex as docindex_mod
    import voice_agent.rag.fts5 as fts5_mod

    import_ms = (time.perf_counter() - t_imp0) * 1000.0
    rss_after_import_kb = _current_rss_kb()
    heavy_loaded = _loaded_heavy_modules()

    from voice_agent.rag.fts5_store import fts5_available as store_fts5_available

    tok_mode: Fts5TokenizerMode = args.tokenizer
    fts_tok_str = fts5_mod.PRETOK_TOKENIZER if tok_mode == "pretok" else fts5_mod.PORTER_TOKENIZER
    probe_fn: Callable[[str], tuple[bool, str]] = getattr(
        fts5_mod,
        "fts5_available",
        store_fts5_available,
    )
    probe_ok, probe_reason = probe_fn(fts_tok_str)

    with contextlib.ExitStack() as stack:
        if args.in_place:
            cache_dir = folder.resolve() / ".rag_cache"
            db_loc_kind = "in-place"
        else:
            tmp_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix="pi_rag_check_"))
            cache_dir = Path(tmp_dir).resolve() / ".rag_cache"
            db_loc_kind = "temp"

        db_path = cache_dir / "fts5" / "index.sqlite3"
        build_seconds: list[float] = []
        base_factory = backends_mod.make_retriever_factory(
            mode="fts5",
            cache_dir=cache_dir,
            fts5_tokenizer=tok_mode,
        )
        initial_retriever = base_factory()

        def _timed_factory() -> Retriever:
            return _TimedBuildRetriever(initial_retriever, build_seconds)

        idx = docindex_mod.DocIndex(
            folder,
            retriever_factory=_timed_factory,
            canonical_cache_dir=None,
        )
        t_ref0 = time.perf_counter()
        idx.refresh()
        refresh_s = time.perf_counter() - t_ref0

        chunks = idx.chunks
        if chunks:
            _ = idx.search(queries[0], k=args.k)

        status_fn = getattr(initial_retriever, "status", None)
        retriever_status = str(status_fn()) if callable(status_fn) else "unknown"
        _close_quietly(initial_retriever)

        db_size_bytes = db_path.stat().st_size if db_path.is_file() else 0
        fallback_label = (
            "rank-bm25" if (not probe_ok or retriever_status.startswith("bm25")) else "none"
        )

        rank_limit = max(len(chunks), 1)
        t_reopen0 = time.perf_counter()
        reopen_factory = backends_mod.make_retriever_factory(
            mode="fts5",
            cache_dir=cache_dir,
            fts5_tokenizer=tok_mode,
        )
        reopened = reopen_factory()
        reopened.build(chunks)
        _ = reopened.rank(queries[0], rank_limit)[: args.k]
        reopen_ms = (time.perf_counter() - t_reopen0) * 1000.0

        sync_stats = getattr(reopened, "last_sync_stats", None)
        p50_ms, p95_ms, nonempty = _benchmark_queries(
            lambda q: reopened.rank(q, rank_limit)[: args.k],
            queries,
        )
        _close_quietly(reopened)
        peak_rss_kb = _peak_rss_kb()

        bm25_cmp = (
            _run_bm25_comparison(chunks, queries, docindex_mod.Bm25Retriever)
            if args.compare_bm25
            else None
        )

    records = idx.records()
    ok_files = sum(1 for r in records if r.status == "ok")
    failed_files = len(records) - ok_files
    fts5_build_s = sum(build_seconds)

    lines = [
        "pi_rag_check report",
        f"python: {platform.python_version()} ({platform.python_implementation()})",
        f"platform: {platform.platform()} ({platform.machine()})",
        f"sqlite_version: {sqlite_ver}",
        f"sqlite_compile_options: {_format_compile_options(compile_opts)}",
        f"import_time_ms: {import_ms:.2f}",
        f"rss_after_import: {_format_mib(rss_after_import_kb)}",
        f"heavy_modules_loaded: {', '.join(heavy_loaded) if heavy_loaded else 'none'}",
        f"tokenizer: {tok_mode}",
        f"fts5_probe: {'ok' if probe_ok else f'failed ({probe_reason})'}",
        f"retriever_status: {retriever_status}",
        f"fallback: {fallback_label}",
        f"folder: {folder}",
        (
            f"docindex_refresh: files={len(records)} (ok={ok_files}, failed={failed_files}), "
            f"chunks={len(chunks)}, seconds={refresh_s:.4f}"
        ),
        f"fts5_build_s: {fts5_build_s:.4f}",
    ]
    if retriever_status == "fts5:file":
        lines.append(f"db_location: {db_loc_kind} ({db_path})")
    else:
        lines.append(f"db_location: none ({retriever_status})")
    lines.append(f"db_size_bytes: {db_size_bytes} ({db_size_bytes / (1024.0 * 1024.0):.3f} MiB)")

    if sync_stats is not None:
        lines.append(
            f"reopen_to_first_query_ms: {reopen_ms:.2f} "
            f"(inserted={sync_stats.inserted}, deleted={sync_stats.deleted})"
        )
    else:
        lines.append(f"reopen_to_first_query_ms: {reopen_ms:.2f}")

    lines.extend(
        [
            f"queries: count={len(queries)}, k={args.k}, source={queries_source}",
            f"query_p50_ms: {p50_ms:.2f}",
            f"query_p95_ms: {p95_ms:.2f}",
            f"nonempty_queries: {nonempty}/{len(queries)}",
            f"peak_rss: {_format_mib(peak_rss_kb)}",
        ]
    )

    if bm25_cmp is not None:
        lines.append(f"bm25_compare: {bm25_cmp['status']}")
        if bm25_cmp["status"] == "ok":
            lines.extend(
                [
                    f"bm25_build_s: {bm25_cmp['build_s']:.4f}",
                    f"bm25_first_query_ms: {bm25_cmp['first_query_ms']:.2f}",
                    f"bm25_p50_ms: {bm25_cmp['p50_ms']:.2f}",
                    f"bm25_p95_ms: {bm25_cmp['p95_ms']:.2f}",
                    f"bm25_nonempty_queries: {bm25_cmp['nonempty']}/{len(queries)}",
                    f"bm25_rss: {_format_mib(int(bm25_cmp['rss_kb']))}",
                ]
            )

    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Pi RAG readiness and latency check and print the report."""
    args = _parse_args(argv)
    try:
        lines = _run_check(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"pi_rag_check error: {exc}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
