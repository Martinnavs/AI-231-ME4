"""Read-only reader over an ingest DB (ticket 107).

Opens a pre-built sqlite DB + FTS5 sibling from ``voice_agent.rag.ingest`` and
exposes the same search/records/failures interface as ``DocIndex``.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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

GREP_STOP: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    ]
)
GREP_STOP = GREP_STOP | frozenset(
    ["what", "which", "according", "documents", "document", "file", "many", "much"]
)
ORDINAL_RE = re.compile(r"\b(first|earliest|last|latest|final|oldest|newest|most recent)\b", re.I)
TEXTLIKE = {".csv", ".tsv", ".log", ".txt", ".md", ".ini", ".toml", ".json"}
MAX_GREP_BYTES = 64 * 1024 * 1024  # 64 MiB


def grep_terms(question: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    terms: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        if t in GREP_STOP or len(t) <= 1:
            continue
        key = t[:5]
        if key not in seen:
            seen.add(key)
            terms.append(key)
    return sorted(terms)


def grep_lines(text: str, question: str, top: int) -> list[str]:
    lines = text.split("\n")
    terms = grep_terms(question)
    if not terms:
        return []
    scored: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        low = line.lower()
        s = sum(1 for t in terms if t in low)
        if s > 0:
            scored.append((i, s))
    if not scored:
        return []
    best = sorted(scored, key=lambda x: (-x[1], x[0]))[:top]
    keep: set[int] = {0} | {idx for idx, _ in best}
    if ORDINAL_RE.search(question):
        mx = max(s for _, s in scored)
        good = sorted(
            (idx for idx, s in scored if s >= max(1, mx - 1)),
        )
        keep |= set(good[:3]) | set(good[-3:])
    return [f"L{i + 1}: {lines[i]}" for i in sorted(keep)]


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

    def __init__(self, db: Path, *, raw_root: Path | None = None) -> None:
        self.folder = db.parent
        self._raw_root = raw_root
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

    def file_context(
        self,
        question: str,
        *,
        n_files: int = 3,
        whole_max_chars: int = 15_000,
        per_big: int = 4,
        grep_top: int = 20,
        rank_k: int = 300,
    ) -> list[dict[str, Any]]:
        ranked = self.search(question, rank_k, per_file_cap=rank_k)
        # Collect unique source order (first appearance)
        seen: set[str] = set()
        order: list[str] = []
        for p in ranked:
            src = p["source"]
            if src not in seen:
                seen.add(src)
                order.append(src)
        order = order[:n_files]

        # Gather per-file chunk data
        file_chunks: dict[str, list[Chunk]] = {}
        file_date: dict[str, str] = {}
        for f in order:
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT {_CHUNK_COLS} FROM chunk_data WHERE rel_key = ? ORDER BY ord",
                    (f,),
                ).fetchall()
            chunks = [_make_chunk(r) for r in rows]
            file_chunks[f] = chunks
            file_date[f] = chunks[0].file_date if chunks else ""

        passages: list[dict[str, Any]] = []
        for f in order:
            chunks = file_chunks[f]
            full = "\n".join(c.text for c in chunks)
            if len(full) <= whole_max_chars:
                passages.append(
                    {
                        "source": f,
                        "id": f"{f}#whole",
                        "section": "",
                        "page": None,
                        "file_date": file_date[f],
                        "text": full,
                        "score": 0.0,
                        "coverage": 0.0,
                        "part": "whole",
                    }
                )
            else:
                # grep passage
                if grep_top > 0:
                    raw = self._get_source_text(f, full)
                    glines = grep_lines(raw, question, grep_top)
                    if glines:
                        passages.append(
                            {
                                "source": f,
                                "id": f"{f}#grep",
                                "section": "",
                                "page": None,
                                "file_date": file_date[f],
                                "text": "\n".join(glines),
                                "score": 0.0,
                                "coverage": 0.0,
                                "part": "grep",
                            }
                        )
                # chunk passages
                file_ranked = [p for p in ranked if p["source"] == f]
                for p in file_ranked[:per_big]:
                    passages.append(
                        {
                            "source": p["source"],
                            "id": p["id"],
                            "section": p["section"],
                            "page": p["page"],
                            "file_date": p["file_date"],
                            "text": p["text"],
                            "score": p["score"],
                            "coverage": p["coverage"],
                            "part": "chunk",
                        }
                    )
        return passages

    def _get_source_text(self, f: str, full: str) -> str:
        if (
            self._raw_root is not None
            and (self._raw_root / f).is_file()
            and Path(f).suffix.lower() in TEXTLIKE
        ):
            raw_path = self._raw_root / f
            data = raw_path.read_bytes()
            if len(data) > MAX_GREP_BYTES:
                data = data[:MAX_GREP_BYTES]
            return data.decode("utf-8", errors="replace")
        return full

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

    def file_card(self, rel_key: str) -> dict[str, Any] | None:
        """Return the FileCard dict for a single file, or None."""
        row = self._conn.execute(
            "SELECT card_text FROM file_cards WHERE rel_key = ?", (rel_key,)
        ).fetchone()
        if row is None:
            return None
        return cast("dict[str, Any]", json.loads(row[0]))

    def catalog(self, max_chars: int = 2000) -> str:
        """Build a catalogue string for all files, capped at max_chars."""
        try:
            rows = self._conn.execute(
                "SELECT rel_key, card_text FROM file_cards ORDER BY rel_key"
            ).fetchall()
        except sqlite3.OperationalError:
            return ""
        lines: list[str] = []
        for rel_key, card_text in rows:
            card = json.loads(card_text)
            sections = card.get("sections", [])
            entities = card.get("top_entities", [])
            date_range = ""
            if card.get("date_min") and card.get("date_max"):
                date_range = f"{card['date_min']} to {card['date_max']}"
            elif card.get("date_min"):
                date_range = card["date_min"]
            line = (
                f"{rel_key} | {card.get('title', '')} | {card.get('kind', '')}"
                f" | {', '.join(sections[:3])}"
                f" | {', '.join(entities[:5])}"
                f" | {date_range}"
                f" | {card.get('row_count', 0)} rows"
            )
            full = "\n".join(lines + [line]) if lines else line
            if len(full.encode("utf-8")) > max_chars:
                break
            lines.append(line)
        return "\n".join(lines)

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
