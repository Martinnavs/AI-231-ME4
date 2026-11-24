"""Standalone ingest writer: parse a source folder once, write sqlite DB + FTS5 sibling."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voice_agent.rag.canonical_cache import CanonicalCache
from voice_agent.rag.docindex import (
    SUPPORTED_EXTENSIONS,
    Chunk,
    LoadedFile,
    _resolve_converters,
    _walk_folder,
    load_file,
    looks_like_text,
    plain_fallback_enabled,
)
from voice_agent.rag.fts5_store import DEFAULT_TOKENIZER

INGEST_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class IngestReport:
    """Summary returned by :func:`ingest`."""

    files_total: int
    ok: int
    failed: int
    chunks: int
    reused: int
    seconds: float


@dataclass(frozen=True)
class _MetaInfo:
    """Metadata from existing ingest DB."""

    complete: str
    schema_version: str
    tokenizer: str
    table_headers: str = "legacy"
    plain_fallback: str = "0"


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS files ("
        "rel_key TEXT PRIMARY KEY, status TEXT, error TEXT, kind TEXT, size INTEGER,"
        " mtime_ns INTEGER, ctime_ns INTEGER, sha256 TEXT, canonical_key TEXT,"
        " note TEXT, dates_json TEXT, n_chunks INTEGER)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunk_data ("
        "chunk_id TEXT PRIMARY KEY, rel_key TEXT, ord INTEGER, file_sha256 TEXT,"
        " chunk_sha256 TEXT, position INTEGER, page INTEGER, section TEXT,"
        " file_date TEXT, source TEXT, header TEXT, text TEXT, fmt TEXT,"
        " slide INTEGER, sheet TEXT, canonical_sha256 TEXT, converter TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS images ("
        "id INTEGER PRIMARY KEY, rel_key TEXT, locator TEXT, sha256 TEXT,"
        " w INTEGER, h INTEGER, status TEXT)"
    )
    conn.commit()


def _remove_db(out: Path) -> None:
    """Remove the db file, its WAL sidecars, and the FTS5 sibling."""
    for base in (out.name, out.name + ".fts"):
        for suffix in ("", "-wal", "-shm", "-journal"):
            p = out.parent / (base + suffix)
            p.unlink(missing_ok=True)


def _init_meta(conn: sqlite3.Connection, src: Path, *, complete: bool = False) -> None:
    now_iso = datetime.now(UTC).isoformat(timespec="seconds")
    rows = [
        ("schema_version", INGEST_SCHEMA_VERSION),
        ("created_at", now_iso),
        ("src", src.name),
        ("tokenizer", DEFAULT_TOKENIZER),
        ("chunker_version", "1"),
        ("complete", "0" if not complete else "1"),
        ("table_headers", os.environ.get("VOICE_AGENT_RAG_TABLE_HEADERS", "legacy")),
        ("plain_fallback", os.environ.get("VOICE_AGENT_RAG_PLAIN_FALLBACK", "0")),
    ]
    conn.execute("DELETE FROM meta")
    conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", rows)
    conn.commit()


def ingest(
    src: Path,
    out: Path,
    *,
    converters: Mapping[str, str] | None = None,
    workers: int = 0,
    worker_mem_mb: int = 0,
) -> IngestReport:
    """Ingest ``src`` folder into sqlite DB at ``out`` and FTS5 sibling ``out.fts``.

    Returns an :class:`IngestReport` with counts and timing.
    """
    converters_resolved = _resolve_converters(converters)
    now_iso = datetime.now(UTC).isoformat(timespec="seconds")
    cache = CanonicalCache(src / ".rag_cache" / "canonical", src)

    existing_meta: _MetaInfo | None = None
    existing_files: dict[str, Any] | None = None
    rebuild = True
    if out.exists():
        existing_meta, existing_files = _load_manifest(out)
        if existing_meta is not None:
            old_complete = existing_meta.complete
            if old_complete == "1":
                old_sv = existing_meta.schema_version
                old_tok = existing_meta.tokenizer
                if old_sv == INGEST_SCHEMA_VERSION and old_tok == DEFAULT_TOKENIZER:
                    cur_th = os.environ.get("VOICE_AGENT_RAG_TABLE_HEADERS", "legacy")
                    cur_pf = os.environ.get("VOICE_AGENT_RAG_PLAIN_FALLBACK", "0")
                    if (
                        existing_meta.table_headers == cur_th
                        and existing_meta.plain_fallback == cur_pf
                    ):
                        rebuild = False
            if rebuild:
                _remove_db(out)
        else:
            _remove_db(out)

    start = time.monotonic()
    conn = sqlite3.connect(str(out))
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        _create_schema(conn)

        if rebuild or existing_meta is None:
            _init_meta(conn, src, complete=False)
            existing_files = None
        else:
            # Crash safety: mark incomplete before touching rows
            conn.execute("UPDATE meta SET value = '0' WHERE key = 'complete'")
            conn.commit()

        files, _ = _walk_folder(src)
        pf_on = plain_fallback_enabled()
        supported = [
            f
            for f in files
            if f.suffix.lower() in SUPPORTED_EXTENSIONS or (pf_on and looks_like_text(f))
        ]
        seen_keys: set[str] = set()

        all_chunks: list[Chunk] = []
        global_ord = 0
        total_ok = 0
        total_failed = 0
        total_reused = 0

        # Separate unchanged files from files needing parsing
        parse_list: list[tuple[Path, str]] = []
        for file_path in supported:
            rel_key = file_path.relative_to(src).as_posix()
            seen_keys.add(rel_key)

            old = existing_files.get(rel_key) if existing_files else None
            unchanged = False
            if old is not None and old["status"] == "ok":
                try:
                    st = file_path.stat()
                    unchanged = (
                        int(old["size"]) == st.st_size
                        and int(old["mtime_ns"]) == st.st_mtime_ns
                        and int(old["ctime_ns"]) == st.st_ctime_ns
                    )
                except OSError:
                    pass

            if unchanged:
                rows = conn.execute(
                    "SELECT chunk_id, file_sha256, chunk_sha256, position, page, section,"
                    " file_date, source, header, text, fmt, slide, sheet,"
                    " canonical_sha256, converter FROM chunk_data WHERE rel_key = ?"
                    " ORDER BY ord",
                    (rel_key,),
                ).fetchall()
                chunks = [
                    Chunk(
                        id=r[0],
                        file_sha256=r[1],
                        chunk_sha256=r[2],
                        position=r[3],
                        page=r[4],
                        section=r[5],
                        file_date=r[6],
                        source=r[7],
                        header=r[8],
                        text=r[9],
                        fmt=r[10],
                        slide=r[11],
                        sheet=r[12],
                        canonical_sha256=r[13],
                        converter=r[14],
                    )
                    for r in rows
                ]
                conn.execute("DELETE FROM chunk_data WHERE rel_key = ?", (rel_key,))
                for chunk in chunks:
                    conn.execute(
                        "INSERT INTO chunk_data VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            chunk.id,
                            rel_key,
                            global_ord,
                            chunk.file_sha256,
                            chunk.chunk_sha256,
                            chunk.position,
                            chunk.page,
                            chunk.section,
                            chunk.file_date,
                            chunk.source,
                            chunk.header,
                            chunk.text,
                            chunk.fmt,
                            chunk.slide,
                            chunk.sheet,
                            chunk.canonical_sha256,
                            chunk.converter,
                        ),
                    )
                    global_ord += 1
                all_chunks.extend(chunks)
                total_reused += len(chunks)
                total_ok += 1
                conn.commit()
            else:
                parse_list.append((file_path, rel_key))

        # Parse files needing work
        from voice_agent.rag.ingest_workers import supervisor_parse

        parsed_map: dict[str, LoadedFile] = {}
        if parse_list:
            if workers > 0:
                results = supervisor_parse(
                    parse_list,
                    converters_resolved,
                    str(cache.directory) if cache.directory else "",
                    src,
                    workers=workers,
                    worker_mem_mb=worker_mem_mb,
                )
                parsed_map = {rk: lf for rk, lf in results}
            else:
                for file_path, rel_key in parse_list:
                    lf = load_file(
                        file_path,
                        rel_key,
                        converters=converters_resolved,
                        canonical_cache=cache,
                        prev=None,
                        now_iso=now_iso,
                    )
                    parsed_map[rel_key] = lf

        # Write parsed results to DB in sorted rel_key order
        parse_rels = [(rk, parsed_map[rk]) for (_fp, rk) in parse_list]
        parse_rels.sort(key=lambda item: item[0])
        for rel_key, lf in parse_rels:
            if lf.status != "ok":
                total_failed += 1
                conn.execute(
                    "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        rel_key,
                        lf.status,
                        lf.error,
                        lf.kind,
                        lf.size,
                        lf.mtime_ns,
                        lf.ctime_ns,
                        lf.sha256,
                        lf.canonical_key,
                        lf.note,
                        json.dumps(lf.dates),
                        0,
                    ),
                )
                conn.commit()
                continue

            for chunk in lf.chunks:
                all_chunks.append(chunk)
                conn.execute(
                    "INSERT OR REPLACE INTO chunk_data VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        chunk.id,
                        rel_key,
                        global_ord,
                        chunk.file_sha256,
                        chunk.chunk_sha256,
                        chunk.position,
                        chunk.page,
                        chunk.section,
                        chunk.file_date,
                        chunk.source,
                        chunk.header,
                        chunk.text,
                        chunk.fmt,
                        chunk.slide,
                        chunk.sheet,
                        chunk.canonical_sha256,
                        chunk.converter,
                    ),
                )
                global_ord += 1
            total_ok += 1
            conn.execute(
                "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rel_key,
                    lf.status,
                    None,
                    lf.kind,
                    lf.size,
                    lf.mtime_ns,
                    lf.ctime_ns,
                    lf.sha256,
                    lf.canonical_key,
                    lf.note,
                    json.dumps(lf.dates),
                    len(lf.chunks),
                ),
            )
            conn.commit()

        if existing_files:
            for old_key in existing_files:
                if old_key not in seen_keys:
                    conn.execute("DELETE FROM chunk_data WHERE rel_key = ?", (old_key,))
                    conn.execute("DELETE FROM files WHERE rel_key = ?", (old_key,))
        conn.commit()

        # Renumber ord 0..N-1 (gaps appear when a file was removed)
        conn.execute(
            "UPDATE chunk_data SET ord = x.r FROM ("
            "SELECT chunk_id AS cid, row_number() OVER (ORDER BY ord) - 1 AS r"
            " FROM chunk_data) AS x WHERE chunk_data.chunk_id = x.cid"
        )
        conn.commit()

        # Build FTS5 sibling (fix item 1)
        from voice_agent.rag.fts5 import Fts5Retriever

        fts_path = out.with_name(out.name + ".fts")
        retriever = Fts5Retriever(db_path=fts_path, root=out.parent)
        retriever.build(all_chunks)
        retriever.close()

        _update_meta(conn, src, complete=True)
    finally:
        conn.close()

    elapsed = time.monotonic() - start
    return IngestReport(
        files_total=len(supported),
        ok=total_ok,
        failed=total_failed,
        chunks=len(all_chunks),
        reused=total_reused,
        seconds=round(elapsed, 3),
    )


def _load_manifest(out: Path) -> tuple[_MetaInfo | None, dict[str, Any] | None]:
    """Return ``(meta, files)`` where files is keyed by rel_key only."""
    try:
        conn = sqlite3.connect(str(out))
        try:
            tables = {
                str(r[0])
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "meta" not in tables or "files" not in tables:
                return None, None

            meta_rows = conn.execute("SELECT key, value FROM meta").fetchall()
            meta_dict = {str(r[0]): str(r[1]) for r in meta_rows}

            rows = conn.execute(
                "SELECT rel_key, status, error, kind, size, mtime_ns, ctime_ns,"
                " sha256, canonical_key, note, dates_json, n_chunks FROM files"
            ).fetchall()
            files: dict[str, Any] = {}
            for row in rows:
                files[str(row[0])] = {
                    "rel_key": str(row[0]),
                    "status": str(row[1]),
                    "error": row[2],
                    "kind": str(row[3]),
                    "size": str(row[4]),
                    "mtime_ns": str(row[5]),
                    "ctime_ns": str(row[6]),
                    "sha256": str(row[7]),
                    "canonical_key": row[8],
                    "note": row[9],
                    "dates_json": row[10],
                    "n_chunks": str(row[11]),
                }

            meta = _MetaInfo(
                complete=meta_dict.get("complete", ""),
                schema_version=meta_dict.get("schema_version", ""),
                tokenizer=meta_dict.get("tokenizer", ""),
                table_headers=meta_dict.get("table_headers", "legacy"),
                plain_fallback=meta_dict.get("plain_fallback", "0"),
            )
            return meta, files
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return None, None


def _update_meta(conn: sqlite3.Connection, src: Path, *, complete: bool) -> None:
    complete_val = "1" if complete else "0"
    conn.execute("UPDATE meta SET value = ? WHERE key = 'complete'", (complete_val,))
    conn.commit()


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point: ``python -m voice_agent.rag.ingest SRC --out DB``."""
    parser = argparse.ArgumentParser(
        prog="voice_agent.rag.ingest",
        description="Ingest a source folder into a sqlite DB (DB) + FTS5 sibling (DB.fts)",
    )
    parser.add_argument("src", help="Source folder to index")
    parser.add_argument(
        "--out",
        default="index.db",
        help="Output sqlite DB path; FTS5 index goes to <path>.fts (default: index.db)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(3, os.cpu_count() or 1),
        help="Worker count (0=in-process, default=min(3, cpu_count))",
    )
    parser.add_argument(
        "--worker-mem-mb",
        type=int,
        default=0,
        help="Per-worker address-space limit in MiB (0=unlimited)",
    )
    parser.add_argument(
        "--file-timeout-s",
        type=int,
        default=300,
        help="Per-file wall-clock timeout in seconds (default: 300)",
    )
    args = parser.parse_args(argv)

    src = Path(args.src)
    if not src.is_dir():
        print(f"error: source directory does not exist: {src}", file=sys.stderr)
        return 2

    report = ingest(
        src,
        Path(args.out),
        workers=args.workers,
        worker_mem_mb=args.worker_mem_mb,
    )
    print(
        f"ingest: {report.files_total} files, {report.ok} ok, {report.failed} failed,"
        f" {report.chunks} chunks, {report.reused} reused in {report.seconds:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
