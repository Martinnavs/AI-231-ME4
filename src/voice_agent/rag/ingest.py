"""Standalone ingest writer: parse a source folder once, write sqlite DB + FTS5 sibling."""

from __future__ import annotations

import argparse
import collections.abc
import hashlib
import json
import os
import re
import resource
import shutil
import sqlite3
import sys
import time
import zipfile
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

INGEST_SCHEMA_VERSION = "3"


@dataclass(frozen=True)
class IngestReport:
    """Summary returned by :func:`ingest`."""

    files_total: int
    ok: int
    failed: int
    chunks: int
    reused: int
    seconds: float
    skipped: int = 0
    peak_rss_mb: float = 0.0


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
        " slide INTEGER, sheet TEXT, canonical_sha256 TEXT, converter TEXT,"
        " ocr_confidence REAL DEFAULT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS images ("
        "id INTEGER PRIMARY KEY, rel_key TEXT, locator TEXT, sha256 TEXT,"
        " w INTEGER, h INTEGER, status TEXT, kind TEXT, error TEXT, note TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS file_cards (rel_key TEXT PRIMARY KEY, card_text TEXT NOT NULL)"
    )
    conn.commit()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add missing columns for schema v3."""
    try:
        conn.execute("ALTER TABLE chunk_data ADD COLUMN ocr_confidence REAL DEFAULT NULL")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE images ADD COLUMN error TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE images ADD COLUMN note TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass


def _run_ocr_phase(
    conn: sqlite3.Connection,
    src: Path,
    fts_path: Path,
    retriever: Any,
    *,
    ocr_deadline_s: float,
    raw_root: Path,
) -> tuple[int, int, int]:
    """OCR phase-2: process pending images, write OCR chunks, sync FTS5.

    Returns (read_count, skipped_count, unread_count).
    """
    from voice_agent.rag import ocr as ocr_mod  # noqa: PLC0415
    from voice_agent.rag.images import ImageRef  # noqa: PLC0415

    pending_rows = conn.execute(
        "SELECT id, rel_key, locator, sha256, w, h, kind FROM images"
        " WHERE status = 'pending' ORDER BY rowid"
    ).fetchall()

    if pending_rows and not ocr_mod.engine_available():
        conn.execute(
            "UPDATE images SET status = 'skipped', error = 'no ocr engine' WHERE status = 'pending'"
        )
        conn.commit()
        _ocr_file_notes(conn)
        print(f"ocr: no engine (rapidocr models or tesseract); {len(pending_rows)} images not read")
        return 0, len(pending_rows), 0

    start = time.monotonic()
    session = ocr_mod.OcrSession()
    seen_hashes: dict[str, int] = {}
    ocr_chunks: list[Chunk] = []

    for row in pending_rows:
        img_id, rel_key, locator, sha256, w, h, kind = row

        # Time governor
        if ocr_deadline_s > 0 and time.monotonic() - start > ocr_deadline_s:
            conn.execute(
                "UPDATE images SET status = 'skipped', error = 'time governor' WHERE id = ?",
                (img_id,),
            )
            session.skipped_count += 1
            continue

        # Dedupe
        if sha256 in seen_hashes:
            first = conn.execute(
                "SELECT status FROM images WHERE id = ?", (seen_hashes[sha256],)
            ).fetchone()
            same = first[0] if first and first[0] in ("read", "unread") else "unread"
            conn.execute(
                "UPDATE images SET status = ?, note = 'duplicate' WHERE id = ?", (same, img_id)
            )
            continue
        seen_hashes[sha256] = img_id

        ref = ImageRef(
            rel_key=rel_key, locator=locator, kind=kind, w=w, h=h, nbytes=0, sha256=sha256
        )

        try:
            raw = ocr_mod.extract_image_bytes(ref, raw_root)
        except OSError:
            conn.execute(
                "UPDATE images SET status = 'skipped', error = 'extract failed' WHERE id = ?",
                (img_id,),
            )
            session.skipped_count += 1
            continue

        result_holder: list[tuple[str, float, str] | None] = [None]

        def _ocr_target(
            _ref: Any = ref,
            _raw: bytes = raw,
            _session: Any = session,
            _kind: str = kind,
            _holder: list[tuple[str, float, str] | None] = result_holder,
        ) -> None:
            max_long = 1280 if _kind == "pdf_scan_page" else 960
            text, conf, note = ocr_mod.ocr_image(_ref, _raw, _session, max_long=max_long)
            _holder[0] = (text, conf, note)

        import threading  # noqa: PLC0415

        t = threading.Thread(target=_ocr_target, daemon=True)
        t.start()
        t.join(timeout=30)
        if t.is_alive():
            conn.execute(
                "UPDATE images SET status = 'skipped', error = 'ocr timeout' WHERE id = ?",
                (img_id,),
            )
            session.skipped_count += 1
            continue

        text, conf, note = result_holder[0]  # type: ignore[misc]

        if note.startswith("skipped"):  # too small, solid or unreadable: the engine never ran
            conn.execute(
                "UPDATE images SET status = 'skipped', error = ? WHERE id = ?", (note, img_id)
            )
            session.skipped_count += 1
            continue

        if text.strip():
            page = None
            m = re.search(r"(?:page|slide)\s+(\d+)", locator)
            if m:
                page = int(m.group(1))

            file_sha_row = conn.execute(
                "SELECT sha256 FROM files WHERE rel_key = ?", (rel_key,)
            ).fetchone()
            file_sha_val = file_sha_row[0] if file_sha_row else ""
            file_date_row = conn.execute(
                "SELECT dates_json FROM files WHERE rel_key = ?", (rel_key,)
            ).fetchone()
            file_date_val = ""
            if file_date_row and file_date_row[0]:
                try:
                    dates = json.loads(file_date_row[0])
                    file_date_val = dates.get("first", "") if dates else ""
                except Exception:
                    pass

            chunk_text_list = ocr_mod.chunk_ocr_text(text)
            for i, chunk_text in enumerate(chunk_text_list):
                cid = f"{rel_key}#ocr-{sha256[:8]}-{i}"
                c_sha = hashlib.sha256(chunk_text.encode()).hexdigest()

                ocr_chunks.append(
                    Chunk(
                        id=cid,
                        file_sha256=file_sha_val,
                        chunk_sha256=c_sha,
                        position=0,
                        page=page,
                        section=f"{locator} [OCR]",
                        file_date=file_date_val,
                        source=rel_key,
                        header="",
                        text=chunk_text,
                        fmt="ocr",
                        slide=page,
                        sheet=None,
                        canonical_sha256="",
                        converter="ocr",
                        ocr_confidence=conf,
                    )
                )

                conn.execute(
                    "INSERT OR REPLACE INTO chunk_data "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        cid,
                        rel_key,
                        0,
                        file_sha_val,
                        c_sha,
                        0,
                        page,
                        f"{locator} [OCR]",
                        file_date_val,
                        rel_key,
                        "",
                        chunk_text,
                        "ocr",
                        page,
                        None,
                        "",
                        "ocr",
                        conf,
                    ),
                )

            conn.execute(
                "UPDATE images SET status = 'read', note = ? WHERE id = ?",
                (text[:80], img_id),
            )
        else:
            conn.execute(
                "UPDATE images SET status = 'unread', note = ? WHERE id = ?",
                (note or "ocr returned empty", img_id),
            )
            session.unread_count += 1

        conn.commit()

    conn.execute(
        "UPDATE chunk_data SET ord = x.r FROM ("
        "SELECT chunk_id AS cid, row_number() OVER (ORDER BY ord) - 1 AS r"
        " FROM chunk_data) AS x WHERE chunk_data.chunk_id = x.cid"
    )
    conn.commit()

    if ocr_chunks:
        retriever.build_streaming(iter(ocr_chunks), prune_missing=False)

    _ocr_file_notes(conn)

    print(
        f"ocr: {session.read_count} read, {session.skipped_count} skipped,"
        f" {session.unread_count} unread"
    )
    return session.read_count, session.skipped_count, session.unread_count


def _remove_db(out: Path) -> None:
    """Remove the db file, its WAL sidecars, and the FTS5 sibling."""
    for base in (out.name, out.name + ".fts"):
        for suffix in ("", "-wal", "-shm", "-journal"):
            p = out.parent / (base + suffix)
            p.unlink(missing_ok=True)


def _init_meta(
    conn: sqlite3.Connection,
    src: Path,
    *,
    complete: bool = False,
    fts_optimized: int = 1,
) -> None:
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
        ("fts_optimized", str(fts_optimized)),
        ("has_file_cards", "1"),
    ]
    conn.execute("DELETE FROM meta")
    conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", rows)
    conn.commit()


def _chunk_generator(
    conn: sqlite3.Connection, page_size: int = 5000
) -> collections.abc.Iterator[Chunk]:
    """Yield Chunk objects from chunk_data in pages, ordered by ord."""
    cursor = conn.execute(
        "SELECT chunk_id, file_sha256, chunk_sha256, position, page, section,"
        " file_date, source, header, text, fmt, slide, sheet,"
        " canonical_sha256, converter, ocr_confidence FROM chunk_data ORDER BY ord"
    )
    while True:
        rows = cursor.fetchmany(page_size)
        if not rows:
            break
        for r in rows:
            ocr_val = None
            if len(r) > 15 and r[15] is not None:
                try:  # noqa: SIM105
                    ocr_val = float(r[15])
                except (ValueError, TypeError):
                    pass
            yield Chunk(
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
                converter=r[14] or "",
                ocr_confidence=ocr_val,
            )


def _auto_workers() -> int:
    """Compute workers from available RAM: min(3, max(1, (MemAvailable_MB - 800) // 600))."""
    try:
        memavail_kb = 0
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    memavail_kb = int(line.split()[1])
                    break
        memavail_mb = memavail_kb / 1024
        return min(3, max(1, int((memavail_mb - 800) // 600)))
    except (OSError, ValueError):
        return 1


def _check_disk_governor(
    path: Path, *, total_input_bytes: int, force: bool, min_free_mb: int
) -> None:
    """Refuse with exit 3 if free space is below threshold."""
    st = os.statvfs(str(path))
    free_mb = st.f_bavail * st.f_frsize / (1024 * 1024)
    threshold = max(1024, 3 * total_input_bytes / (1024 * 1024))
    if free_mb < threshold and not force:
        print(
            f"error: disk governor: free {free_mb:.0f} MB < required {threshold:.0f} MB"
            f" (use --force to override)",
            file=sys.stderr,
        )
        sys.exit(3)


def _mark_remaining_skipped(
    conn: sqlite3.Connection,
    remaining: list[tuple[Path, str]],
    reason: str,
) -> int:
    """Mark files as skipped in the files table. Returns count marked."""
    count = 0
    for fp, rk in remaining:
        try:
            st = fp.stat()
            conn.execute(
                "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rk,
                    "skipped",
                    reason,
                    "unknown",
                    st.st_size,
                    st.st_mtime_ns,
                    st.st_ctime_ns,
                    "",
                    "",
                    None,
                    "[]",
                    0,
                ),
            )
            count += 1
        except OSError:
            pass
    conn.commit()
    return count


def _image_refs(rel_key: str, lf: Any, chunks: list[Any], fpath: Path | None) -> list[Any]:
    """The images of one parsed file that wait for OCR (read while the file still exists)."""
    if fpath is None or lf.status != "ok":
        return []
    try:
        from voice_agent.rag.images import (  # noqa: PLC0415
            image_file_ref,
            list_pdf_images,
            list_pptx_images,
        )

        if lf.kind == "pptx":
            return list(list_pptx_images(fpath.read_bytes(), rel_key))
        if lf.kind == "pdf":
            text_chars: dict[int, int] = {}
            for c in chunks:
                if c.page is not None:
                    text_chars[c.page] = text_chars.get(c.page, 0) + len(c.text)
            return list(
                list_pdf_images(
                    fpath, rel_key, text_chars_by_page=text_chars, file_sha256=lf.sha256
                )
            )
        if lf.kind == "image":
            ref = image_file_ref(fpath, rel_key)
            return [ref] if ref is not None else []
    except Exception:
        pass
    return []


def _ocr_file_notes(conn: sqlite3.Connection) -> None:
    """Replace each file's 'N images waiting for OCR' note with 'N images read, M not read'."""
    for (rel_key,) in conn.execute("SELECT DISTINCT rel_key FROM images").fetchall():
        read_c, other_c = conn.execute(
            "SELECT COALESCE(SUM(status = 'read'), 0), COALESCE(SUM(status != 'read'), 0)"
            " FROM images WHERE rel_key = ?",
            (rel_key,),
        ).fetchone()
        row = conn.execute("SELECT note FROM files WHERE rel_key = ?", (rel_key,)).fetchone()
        old = re.sub(r";?\s*\d+ images (?:waiting for OCR|read, \d+ not read)", "", row[0] or "")
        new = f"{read_c} images read, {other_c} not read"
        conn.execute(
            "UPDATE files SET note = ? WHERE rel_key = ?",
            (f"{old.strip('; ')}; {new}" if old.strip("; ") else new, rel_key),
        )
    conn.commit()


def _write_image_rows(conn: sqlite3.Connection, rel_key: str, refs: list[Any]) -> int:
    """Replace the file's `images` rows (pending OCR); returns how many were written."""
    conn.execute("DELETE FROM images WHERE rel_key = ?", (rel_key,))
    for ir in refs:
        conn.execute(
            "INSERT INTO images (rel_key, locator, sha256, w, h, status, kind)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ir.rel_key, ir.locator, ir.sha256, ir.w, ir.h, "pending", ir.kind),
        )
    conn.commit()
    return len(refs)


def ingest(
    src: Path,
    out: Path,
    *,
    converters: Mapping[str, str] | None = None,
    workers: int | str = "auto",
    worker_mem_mb: int = 0,
    fts_build: str = "stream",
    no_optimize: bool = True,
    deadline_min: int = 0,
    max_file_chunks: int = 50_000,
    min_free_mb: int = 600,
    force: bool = False,
    ocr_run: bool = False,
    ocr_deadline_min: int = 15,
) -> IngestReport:
    """Ingest ``src`` folder into sqlite DB at ``out`` and FTS5 sibling ``out.fts``.

    Returns an :class:`IngestReport` with counts and timing.
    """
    converters_resolved = _resolve_converters(converters)
    now_iso = datetime.now(UTC).isoformat(timespec="seconds")
    cache = CanonicalCache(src / ".rag_cache" / "canonical", src)

    # Resolve workers
    if workers == "auto":
        resolved_workers: int = _auto_workers()
    else:
        resolved_workers = int(workers)
    if resolved_workers <= 0:
        resolved_workers = 0

    # Handle zip input
    if src.suffix.lower() == ".zip":
        from voice_agent.rag import zipguard, zipingest  # noqa: PLC0415

        # The upload-sized defaults (64 MB total, 1000 members) would refuse a real corpus zip;
        # the disk governor bounds the extraction, the ratio and member guards stay on.
        preflight_err = zipguard.preflight_zip(
            src,
            label="zip",
            scan_xml_members=False,  # each document is guarded again by its own reader
            max_members=200_000,
            max_member_bytes=512 * 1024 * 1024,
            max_total_bytes=64 * 1024 * 1024 * 1024,
        )
        if preflight_err is not None:
            print(f"error: zip preflight failed: {preflight_err}", file=sys.stderr)
            sys.exit(1)

        work_dir = out.parent / (out.name + ".work")
        work_dir.mkdir(parents=True, exist_ok=True)

        zf = zipfile.ZipFile(src)
        accepted, refused, junk = zipingest.validate_members(zf.infolist(), work_dir)

        # Record refused members
        zip_conn = sqlite3.connect(str(out))
        try:
            zip_conn.execute("PRAGMA journal_mode = WAL")
            _create_schema(zip_conn)
            _migrate_schema(zip_conn)
            _init_meta(zip_conn, src, complete=False, fts_optimized=0 if no_optimize else 1)
            for reason in refused:
                member_name = reason.split(":")[0].strip()
                zip_conn.execute(
                    "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (member_name, "refused", reason, "unknown", 0, 0, 0, "", "", None, "[]", 0),
                )
            zip_conn.commit()
        finally:
            zip_conn.close()

        # Extract, parse, commit per-file with staging limit
        all_parsed: list[tuple[str, LoadedFile]] = []
        zip_images: dict[str, list[Any]] = {}
        staged_bytes = 0
        staging_files: list[Path] = []

        for info, target in accepted:
            # Wait if staging limit exceeded
            if staged_bytes >= 512 * 1024 * 1024 and staging_files:
                # Clean up oldest staged files
                oldest = staging_files.pop(0)
                if oldest.exists():
                    staged_bytes -= oldest.stat().st_size
                    oldest.unlink()

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src_member, target.open("wb") as dst:
                shutil.copyfileobj(src_member, dst)
            staged_bytes += target.stat().st_size
            staging_files.append(target)

            rel_key = info.filename.replace("\\", "/")
            lf = load_file(
                target,
                rel_key,
                converters=converters_resolved,
                canonical_cache=cache,
                prev=None,
                now_iso=now_iso,
            )
            all_parsed.append((rel_key, lf))
            zip_images[rel_key] = _image_refs(rel_key, lf, lf.chunks, target)

            # Clean up extracted file after parsing
            target.unlink(missing_ok=True)
            staged_bytes -= target.stat().st_size if target.exists() else 0
            if target in staging_files:
                staging_files.remove(target)

        zf.close()
        shutil.rmtree(work_dir, ignore_errors=True)

        # Write parsed results
        conn = sqlite3.connect(str(out))
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            total_ok = 0
            total_failed = 0
            total_chunks = 0
            zip_chunks: list[Chunk] = []  # kept only for the legacy FTS build

            for rel_key, lf in sorted(all_parsed, key=lambda x: x[0]):
                if lf.status != "ok":
                    total_failed += 1
                else:
                    if fts_build != "stream":
                        zip_chunks.extend(lf.chunks)
                    total_ok += 1
                    total_chunks += len(lf.chunks)

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
                        len(lf.chunks) if lf.status == "ok" else 0,
                    ),
                )

                if zip_images.get(rel_key):
                    n_img = _write_image_rows(conn, rel_key, zip_images[rel_key])
                    conn.execute(
                        "UPDATE files SET note = ? WHERE rel_key = ?",
                        (
                            f"{lf.note}; {n_img} images waiting for OCR"
                            if lf.note
                            else f"{n_img} images waiting for OCR",
                            rel_key,
                        ),
                    )

                if lf.status == "ok":
                    global_ord_offset = total_chunks - len(lf.chunks)
                    for chunk in lf.chunks:
                        conn.execute(
                            "INSERT OR REPLACE INTO chunk_data "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                chunk.id,
                                rel_key,
                                global_ord_offset,
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
                                None,
                            ),
                        )
                        global_ord_offset += 1

                    # Build and insert file card
                    from voice_agent.rag.file_cards import (  # noqa: PLC0415
                        build_card,
                        card_to_json,
                    )

                    try:
                        card = build_card(rel_key, lf.chunks, kind=lf.kind)
                        conn.execute(
                            "INSERT OR REPLACE INTO file_cards VALUES (?, ?)",
                            (rel_key, card_to_json(card)),
                        )
                    except Exception:
                        pass

                conn.commit()

            # Renumber ord
            conn.execute(
                "UPDATE chunk_data SET ord = x.r FROM ("
                "SELECT chunk_id AS cid, row_number() OVER (ORDER BY ord) - 1 AS r"
                " FROM chunk_data) AS x WHERE chunk_data.chunk_id = x.cid"
            )
            conn.commit()

            # FTS build
            from voice_agent.rag.fts5 import Fts5Retriever  # noqa: PLC0415

            fts_path = out.with_name(out.name + ".fts")
            retriever = Fts5Retriever(db_path=fts_path, root=out.parent)
            if fts_build == "stream":
                retriever.build_streaming(_chunk_generator(conn))
            else:
                retriever.build(zip_chunks)

            if ocr_run:
                _run_ocr_phase(
                    conn,
                    src,
                    fts_path,
                    retriever,
                    ocr_deadline_s=ocr_deadline_min * 60,
                    raw_root=src,
                )

            retriever.close()

            fts_opt_val = 0 if no_optimize else 1
            _update_meta(conn, src, complete=True, fts_optimized=fts_opt_val)
        finally:
            conn.close()

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        return IngestReport(
            files_total=total_ok + total_failed,
            ok=total_ok,
            failed=total_failed,
            chunks=total_chunks,
            reused=0,
            seconds=0.0,
            skipped=len(refused),
            peak_rss_mb=round(peak_rss, 1),
        )

    # Folder source path
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

    # Disk governor preflight
    total_input_bytes = sum(f.stat().st_size for f in src.iterdir() if f.is_file())
    _check_disk_governor(
        out.parent,
        total_input_bytes=total_input_bytes,
        force=force,
        min_free_mb=min_free_mb,
    )

    start = time.monotonic()
    deadline = start + deadline_min * 60 if deadline_min > 0 else 0.0
    conn = sqlite3.connect(str(out))
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        _create_schema(conn)
        _migrate_schema(conn)

        if rebuild or existing_meta is None:
            _init_meta(conn, src, complete=False, fts_optimized=0 if no_optimize else 1)
            existing_files = None
        else:
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

        # Collect rel_keys
        file_list: list[tuple[Path, str]] = []
        for file_path in supported:
            rel_key = file_path.relative_to(src).as_posix()
            seen_keys.add(rel_key)
            file_list.append((file_path, rel_key))

        # Separate unchanged files from files needing parsing
        parse_list: list[tuple[Path, str]] = []
        global_ord = 0
        total_ok = 0
        total_failed = 0
        total_reused = 0
        all_chunks: list[Chunk] = []  # kept only for the legacy FTS build
        keep_chunks = fts_build != "stream"
        n_chunks = 0

        for file_path, rel_key in file_list:
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
                    " canonical_sha256, converter, ocr_confidence FROM chunk_data WHERE rel_key = ?"
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
                        converter=r[14] or "",
                        ocr_confidence=float(r[15]) if r[15] is not None else None,
                    )
                    for r in rows
                ]
                conn.execute("DELETE FROM chunk_data WHERE rel_key = ?", (rel_key,))
                for chunk in chunks:
                    conn.execute(
                        "INSERT INTO chunk_data VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                            None,
                        ),
                    )
                    global_ord += 1
                n_chunks += len(chunks)
                if keep_chunks:
                    all_chunks.extend(chunks)
                total_reused += len(chunks)
                total_ok += 1

                # Build and insert file card for unchanged file
                from voice_agent.rag.file_cards import (  # noqa: PLC0415
                    build_card,
                    card_to_json,
                )

                assert old is not None  # unchanged is only True when old exists
                file_kind: str = old["kind"]
                try:
                    card = build_card(rel_key, chunks, kind=file_kind)
                    conn.execute(
                        "INSERT OR REPLACE INTO file_cards VALUES (?, ?)",
                        (rel_key, card_to_json(card)),
                    )
                except Exception:
                    pass

                conn.commit()
            else:
                parse_list.append((file_path, rel_key))

        # Governor state
        governor_stopped = False
        total_skipped = 0

        # Governor-aware dispatch
        if parse_list:
            from voice_agent.rag.ingest_workers import supervisor_parse  # noqa: PLC0415

            def _governor_check(_in_flight: int) -> None:
                nonlocal governor_stopped
                if governor_stopped:
                    return
                # Time governor
                if deadline > 0 and time.monotonic() >= deadline:
                    governor_stopped = True
                    return
                # Disk governor
                st = os.statvfs(str(out.parent))
                free_mb = st.f_bavail * st.f_frsize / (1024 * 1024)
                if free_mb < min_free_mb:
                    governor_stopped = True

            parsed_map: dict[str, LoadedFile] = {}
            if resolved_workers > 0:
                results = supervisor_parse(
                    parse_list,
                    converters_resolved,
                    str(cache.directory) if cache.directory else "",
                    src,
                    workers=resolved_workers,
                    worker_mem_mb=worker_mem_mb,
                    window_check=_governor_check,
                )
                parsed_map = {rk: lf for rk, lf in results}
            else:
                for file_path, rel_key in parse_list:
                    _governor_check(0)
                    if governor_stopped:
                        break
                    lf = load_file(
                        file_path,
                        rel_key,
                        converters=converters_resolved,
                        canonical_cache=cache,
                        prev=None,
                        now_iso=now_iso,
                    )
                    parsed_map[rel_key] = lf

            # Mark remaining unparsed files as skipped
            parsed_rels = set()
            for _fp, rk in parse_list:
                if rk in parsed_map or not governor_stopped:
                    parsed_rels.add(rk)

            remaining = [(fp, rk) for fp, rk in parse_list if rk not in parsed_rels]
            if remaining:
                timed = deadline > 0 and time.monotonic() >= deadline
                reason = "time governor" if timed else "disk governor"
                total_skipped = _mark_remaining_skipped(conn, remaining, reason)

            # Write parsed results to DB in sorted rel_key order
            rel_key_to_path: dict[str, Path] = {rk: fp for fp, rk in file_list}
            parse_rels = [(rk, parsed_map[rk]) for rk in parsed_rels if rk in parsed_map]
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

                # Bounded per-file chunks
                chunks = lf.chunks[:max_file_chunks]
                note = lf.note
                if len(lf.chunks) > max_file_chunks:
                    n_orig = len(lf.chunks)
                    note = (note or "") + f" [truncated: {n_orig} -> {max_file_chunks} chunks]"

                for chunk in chunks:
                    n_chunks += 1
                    if keep_chunks:
                        all_chunks.append(chunk)
                    conn.execute(
                        "INSERT OR REPLACE INTO chunk_data "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                            None,
                        ),
                    )
                    global_ord += 1
                total_ok += 1

                # Build and insert file card
                from voice_agent.rag.file_cards import (  # noqa: PLC0415
                    build_card,
                    card_to_json,
                )

                try:
                    card = build_card(rel_key, chunks, kind=lf.kind)
                    conn.execute(
                        "INSERT OR REPLACE INTO file_cards VALUES (?, ?)",
                        (rel_key, card_to_json(card)),
                    )
                except Exception:
                    pass

                refs = _image_refs(rel_key, lf, chunks, rel_key_to_path.get(rel_key))
                n_image_rows = _write_image_rows(conn, rel_key, refs)

                # Build file note with image count
                if n_image_rows > 0:
                    img_note = f"{n_image_rows} images waiting for OCR"
                    note = f"{note}; {img_note}" if note else img_note

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
                        note,
                        json.dumps(lf.dates),
                        len(chunks),
                    ),
                )
                conn.commit()

        if existing_files:
            for old_key in existing_files:
                if old_key not in seen_keys:
                    conn.execute("DELETE FROM chunk_data WHERE rel_key = ?", (old_key,))
                    conn.execute("DELETE FROM images WHERE rel_key = ?", (old_key,))
                    conn.execute("DELETE FROM files WHERE rel_key = ?", (old_key,))
        conn.commit()

        # Renumber ord 0..N-1 (gaps appear when a file was removed)
        conn.execute(
            "UPDATE chunk_data SET ord = x.r FROM ("
            "SELECT chunk_id AS cid, row_number() OVER (ORDER BY ord) - 1 AS r"
            " FROM chunk_data) AS x WHERE chunk_data.chunk_id = x.cid"
        )
        conn.commit()

        # Build FTS5
        from voice_agent.rag.fts5 import Fts5Retriever  # noqa: PLC0415

        fts_path = out.with_name(out.name + ".fts")
        retriever = Fts5Retriever(db_path=fts_path, root=out.parent)
        if fts_build == "stream":
            retriever.build_streaming(_chunk_generator(conn))
        else:
            retriever.build(all_chunks)

        # OCR phase-2
        if ocr_run:
            _run_ocr_phase(
                conn,
                src,
                fts_path,
                retriever,
                ocr_deadline_s=ocr_deadline_min * 60,
                raw_root=src,
            )

        retriever.close()

        fts_opt_val = 0 if no_optimize else 1
        _update_meta(conn, src, complete=True, fts_optimized=fts_opt_val)
    finally:
        conn.close()

    elapsed = time.monotonic() - start
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return IngestReport(
        files_total=len(file_list),
        ok=total_ok,
        failed=total_failed,
        chunks=n_chunks,
        reused=total_reused,
        seconds=round(elapsed, 3),
        skipped=total_skipped,
        peak_rss_mb=round(peak_rss, 1),
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


def _update_meta(
    conn: sqlite3.Connection, src: Path, *, complete: bool, fts_optimized: int = 1
) -> None:
    complete_val = "1" if complete else "0"
    conn.execute("UPDATE meta SET value = ? WHERE key = 'complete'", (complete_val,))
    conn.execute("UPDATE meta SET value = ? WHERE key = 'fts_optimized'", (str(fts_optimized),))
    conn.commit()


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point: ``python -m voice_agent.rag.ingest SRC --out DB``."""
    parser = argparse.ArgumentParser(
        prog="voice_agent.rag.ingest",
        description="Ingest a source folder into a sqlite DB (DB) + FTS5 sibling (DB.fts)",
    )
    parser.add_argument("src", help="Source folder or .zip to index")
    parser.add_argument(
        "--out",
        default="index.db",
        help="Output sqlite DB path; FTS5 index goes to <path>.fts (default: index.db)",
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help=(
            'Worker count: "auto" = min(3, max(1, (MemAvail_MB-800)//600)),'
            " 0=in-process (default: auto)"
        ),
    )
    parser.add_argument(
        "--worker-mem-mb",
        type=int,
        default=0,
        help=(
            "Per-worker address-space limit in MiB (0=unlimited). numpy and the PDF libraries "
            "map far more address space than they use: a small value fails files."
        ),
    )
    parser.add_argument(
        "--fts-build",
        choices=["stream", "legacy"],
        default="stream",
        help="FTS build mode (default: stream)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        default=True,
        help="Skip FTS5 optimize (default: on)",
    )
    parser.add_argument(
        "--deadline-min",
        type=int,
        default=0,
        help="Stop dispatching after M minutes (0=none, default: 0)",
    )
    parser.add_argument(
        "--max-file-chunks",
        type=int,
        default=50_000,
        help="Per-file chunk cap (default: 50000)",
    )
    parser.add_argument(
        "--min-free-mb",
        type=int,
        default=600,
        help="Stop dispatching when free disk < N MB (default: 600)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Override disk governor preflight",
    )
    parser.add_argument(
        "--file-timeout-s",
        type=int,
        default=300,
        help="Per-file wall-clock timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        default=False,
        help="Run OCR on pending images after ingest (default: off)",
    )
    parser.add_argument(
        "--ocr-deadline-min",
        type=int,
        default=15,
        help="OCR time budget in minutes (default: 15)",
    )
    args = parser.parse_args(argv)

    # Resolve workers
    workers_val = "auto" if args.workers == "auto" else int(args.workers)

    src = Path(args.src)
    if not src.exists():
        print(f"error: source does not exist: {src}", file=sys.stderr)
        return 2

    report = ingest(
        src,
        Path(args.out),
        workers=workers_val,
        worker_mem_mb=args.worker_mem_mb,
        fts_build=args.fts_build,
        no_optimize=args.no_optimize,
        deadline_min=args.deadline_min,
        max_file_chunks=args.max_file_chunks,
        min_free_mb=args.min_free_mb,
        force=args.force,
        ocr_run=args.ocr,
        ocr_deadline_min=args.ocr_deadline_min,
    )
    print(
        f"ingest: {report.files_total} files, {report.ok} ok, {report.failed} failed,"
        f" {report.skipped} skipped (disk/time), {report.chunks} chunks,"
        f" {report.peak_rss_mb:.0f} MB RSS, {report.seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
