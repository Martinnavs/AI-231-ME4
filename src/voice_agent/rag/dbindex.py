"""Read-only reader over an ingest DB (ticket 107).

Opens a pre-built sqlite DB + FTS5 sibling from ``voice_agent.rag.ingest`` and
exposes the same search/records/failures interface as ``DocIndex``.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doc_chat.corpus import tokenize
from voice_agent.rag.docindex import (
    HEADER_WEIGHT,
    Chunk,
    FileRecord,
    RefreshReport,
    select_results,
)
from voice_agent.rag.fts5 import (
    _build_match_query,
)
from voice_agent.rag.fts5_store import (
    DEFAULT_COLUMNS,
    DEFAULT_TOKENIZER,
    Fts5Store,
)
from voice_agent.rag.ingest import INGEST_SCHEMA_VERSION

# Column layout written by ingest: (chunk_id, rel_key, ord, file_sha256,
# chunk_sha256, position, page, section, file_date, source, header, text,
# fmt, slide, sheet, canonical_sha256, converter)
_CHUNK_COLS = (
    "chunk_id, rel_key, ord, file_sha256, chunk_sha256, position, page,"
    " section, file_date, source, header, text, fmt, slide, sheet,"
    " canonical_sha256, converter"
)
_FILE_COLS = (
    "rel_key, status, error, kind, size, mtime_ns, ctime_ns,"
    " sha256, canonical_key, note, dates_json, n_chunks"
)


class DbIndexError(OSError):
    """Raised when the ingest DB cannot be opened."""


def _make_chunk(row: tuple[Any, ...]) -> Chunk:
    return Chunk(
        id=str(row[0]),
        file_sha256=str(row[3]),
        chunk_sha256=str(row[4]),
        position=int(row[5]),
        page=int(row[6]) if row[6] is not None else None,
        section=str(row[7]) if row[7] else "",
        file_date=str(row[8]) if row[8] else "",
        source=str(row[9]),
        header=str(row[10]),
        text=str(row[11]),
        fmt=str(row[12]),
        slide=int(row[13]) if row[13] is not None else None,
        sheet=str(row[14]) if row[14] else None,
        canonical_sha256=str(row[15]) if row[15] else "",
        converter=str(row[16]) if row[16] else "",
    )


@dataclass(frozen=True)
class _FakeRetriever:
    """Minimal duck-type for ``index.retriever`` so ``docs_status`` works."""

    def status(self) -> str:
        return "fts5-db"


class DbIndex:
    """Read-only document index backed by an ingest DB artifact."""

    def __init__(self, db: Path) -> None:
        self.folder = db.parent
        db = Path(db)
        fts_db = db.with_name(db.name + ".fts")

        if not db.exists():
            raise DbIndexError(f"ingest DB not found: {db}")
        if not fts_db.exists():
            raise DbIndexError(f"FTS5 sibling not found: {fts_db}")

        # Validate meta
        self._meta = self._read_meta(db)
        if self._meta.get("complete") != "1":
            raise DbIndexError(f"ingest DB is incomplete (complete={self._meta.get('complete')!r})")
        sv = self._meta.get("schema_version", "")
        if sv != INGEST_SCHEMA_VERSION:
            raise DbIndexError(f"schema mismatch: expected {INGEST_SCHEMA_VERSION!r}, got {sv!r}")

        # Open main DB read-only
        uri = db.as_uri() + "?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA query_only = ON")

        # Open FTS5 store — opened in rw mode but we never write
        tok = self._meta.get("tokenizer", DEFAULT_TOKENIZER)
        cols_str = self._meta.get("columns", "")
        if cols_str:
            self._cols = tuple(cols_str.split(","))
        else:
            self._cols = DEFAULT_COLUMNS
        self._fts = Fts5Store(fts_db, db.parent, tokenizer=tok, columns=self._cols)
        self._weights = (1.0, HEADER_WEIGHT)

        self._lock = threading.Lock()
        self._chunks_cache: tuple[Chunk, ...] | None = None

    @property
    def retriever(self) -> _FakeRetriever:
        return _FakeRetriever()

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        with self._lock:
            if self._chunks_cache is None:
                rows = self._conn.execute(
                    f"SELECT {_CHUNK_COLS} FROM chunk_data ORDER BY ord"
                ).fetchall()
                self._chunks_cache = tuple(_make_chunk(r) for r in rows)
            return self._chunks_cache

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
        if k <= 0 or not query.strip():
            return []
        match = _build_match_query(query)
        if not match:
            return []
        q_tokens = set(tokenize(query))

        with self._lock:
            conn = self._conn
            limits = (64, 512, None)
            for idx, limit in enumerate(limits):
                hits = self._fts.query(match, limit=limit, weights=self._weights)
                if not hits:
                    return []
                hit_ids = [cid for cid, _ in hits]
                chunks_by_id = self._load_chunks(conn, hit_ids)
                ranked: Sequence[tuple[Chunk, float]] = [
                    (chunks_by_id[cid], score) for cid, score in hits if cid in chunks_by_id
                ]
                if not ranked:
                    return []
                results, filled = select_results(
                    ranked,
                    k,
                    q_tokens,
                    per_file_cap=per_file_cap,
                    adaptive_cap=adaptive_cap,
                    whole_files=whole_files,
                    whole_max_chunks=whole_max_chunks,
                )
                if filled or idx == len(limits) - 1:
                    return results
        return []

    def _load_chunks(self, conn: sqlite3.Connection, hit_ids: list[str]) -> dict[str, Chunk]:
        chunks: dict[str, Chunk] = {}
        batch_size = 500
        for i in range(0, len(hit_ids), batch_size):
            batch = hit_ids[i : i + batch_size]
            placeholders = ",".join("?" for _ in batch)
            sql = f"SELECT {_CHUNK_COLS} FROM chunk_data WHERE chunk_id IN ({placeholders})"
            rows = conn.execute(sql, batch).fetchall()
            for row in rows:
                chunks[str(row[0])] = _make_chunk(row)
        return chunks

    def records(self) -> list[FileRecord]:
        with self._lock:
            rows = self._conn.execute(f"SELECT {_FILE_COLS} FROM files ORDER BY rel_key").fetchall()
        recs: list[FileRecord] = []
        for row in rows:
            mtime_ns = int(row[5]) if row[5] else 0
            recs.append(
                FileRecord(
                    path=str(self.folder / row[0]),
                    name=str(row[0]),
                    mtime=mtime_ns / 1e9,
                    ingested_at="",
                    sha256=str(row[7]) if row[7] else "",
                    size=int(row[4]) if row[4] else 0,
                    kind=str(row[3]) if row[3] else "unknown",
                    dates_in_text=[],
                    status=str(row[1]),
                    error=str(row[2]) if row[2] else None,
                )
            )
        return recs

    def failures(self) -> list[tuple[str, str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT rel_key, status, error FROM files WHERE status != 'ok' ORDER BY rel_key"
            ).fetchall()
        return [(str(r[0]), str(r[1]), str(r[2]) if r[2] else "") for r in rows]

    def refresh(self) -> RefreshReport:
        return RefreshReport()

    def parsing(self) -> str | None:
        return None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
            self._fts.close()

    @staticmethod
    def _read_meta(db: Path) -> dict[str, str]:
        uri = db.as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
            return {str(r[0]): str(r[1]) for r in rows}
        finally:
            conn.close()
