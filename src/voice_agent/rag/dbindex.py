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
_IDENTIFY_CUE = re.compile(r"\bwhich\s+(?:other\s+)?files?\s+(?:are|is)\b", re.IGNORECASE)
_DOC_CUE_TERMS = frozenset({"file", "files", "document", "documents", "note", "notes"})


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
# fmt, slide, sheet, canonical_sha256, converter, ocr_confidence)
_CHUNK_COLS = (
    "chunk_id, rel_key, ord, file_sha256, chunk_sha256, position, page,"
    " section, file_date, source, header, text, fmt, slide, sheet,"
    " canonical_sha256, converter, ocr_confidence"
)
_FILE_COLS = (
    "rel_key, status, error, kind, size, mtime_ns, ctime_ns,"
    " sha256, canonical_key, note, dates_json, n_chunks"
)


class DbIndexError(OSError):
    """Raised when the ingest DB cannot be opened."""


def _make_chunk(row: tuple[Any, ...]) -> Chunk:
    ocr_val = None
    if len(row) > 17 and row[17] is not None:
        try:  # noqa: SIM105
            ocr_val = float(row[17])
        except (ValueError, TypeError):
            pass
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
        ocr_confidence=ocr_val,
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

        # Create missing indexes on already-built DBs (writable open only)
        if db.is_file() and db.parent.is_dir():
            try:
                rw = sqlite3.connect(str(db))
                try:
                    rw.execute(
                        "CREATE INDEX IF NOT EXISTS idx_chunk_rel_ord ON chunk_data(rel_key, ord)"
                    )
                    rw.execute("CREATE INDEX IF NOT EXISTS idx_images_status ON images(status)")
                    rw.commit()
                finally:
                    rw.close()
            except sqlite3.Error:
                pass

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

    def _file_context_legacy(
        self,
        question: str,
        *,
        n_files: int = 3,
        whole_max_chars: int = 15_000,
        per_big: int = 4,
        grep_top: int = 20,
        rank_k: int = 300,
    ) -> list[dict[str, Any]]:
        """Original implementation kept for parity testing (remove after verification)."""
        ranked = self.search(question, rank_k, per_file_cap=rank_k)
        seen: set[str] = set()
        order: list[str] = []
        for p in ranked:
            src = p["source"]
            if src not in seen:
                seen.add(src)
                order.append(src)
        order = order[:n_files]

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
        match = _build_match_query(question)
        if not match:
            return []

        hits = self._fts.query(match, limit=rank_k, weights=self._weights)
        if not hits:
            return []

        order: list[str] = []
        hits_by_file: dict[str, list[tuple[str, float]]] = {}
        for cid, score in hits:
            rk = cid.rsplit("#", 1)[0]
            if rk not in hits_by_file:
                if len(order) >= n_files:
                    continue
                order.append(rk)
                hits_by_file[rk] = []
            hits_by_file[rk].append((cid, score))

        passages: list[dict[str, Any]] = []
        for f in order:
            row = self._conn.execute(
                "SELECT SUM(length(text)), file_date FROM chunk_data WHERE rel_key = ?", (f,)
            ).fetchone()
            char_count = row[0] or 0
            file_date_val = row[1] or "" if row[1] else ""

            if char_count <= whole_max_chars:
                rows = self._conn.execute(
                    f"SELECT {_CHUNK_COLS} FROM chunk_data WHERE rel_key = ? ORDER BY ord", (f,)
                ).fetchall()
                full = "\n".join(str(r[11]) for r in rows)
                passages.append(
                    {
                        "source": f,
                        "id": f"{f}#whole",
                        "section": "",
                        "page": None,
                        "file_date": file_date_val,
                        "text": full,
                        "score": 0.0,
                        "coverage": 0.0,
                        "part": "whole",
                    }
                )
            else:
                if grep_top > 0:
                    text_rows = self._conn.execute(
                        "SELECT text FROM chunk_data WHERE rel_key = ? ORDER BY ord", (f,)
                    ).fetchall()
                    full = "\n".join(str(r[0]) for r in text_rows)
                    raw = self._get_source_text(f, full)
                    glines = grep_lines(raw, question, grep_top)
                    if glines:
                        passages.append(
                            {
                                "source": f,
                                "id": f"{f}#grep",
                                "section": "",
                                "page": None,
                                "file_date": file_date_val,
                                "text": "\n".join(glines),
                                "score": 0.0,
                                "coverage": 0.0,
                                "part": "grep",
                            }
                        )
                hit_ids = [cid for cid, _ in hits_by_file[f][:per_big]]
                if hit_ids:
                    chunks = self._load_chunks(self._conn, hit_ids)
                    for cid, _ in hits_by_file[f][:per_big]:
                        c = chunks.get(cid)
                        if c:
                            passages.append(
                                {
                                    "source": c.source,
                                    "id": c.id,
                                    "section": c.section,
                                    "page": c.page,
                                    "file_date": c.file_date,
                                    "text": c.text,
                                    "score": 0.0,
                                    "coverage": 0.0,
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

    def overview_context(self, max_chars: int = 6000) -> list[dict[str, Any]]:
        """One compact line per file card, all files, grouped by folder, fitting max_chars."""
        try:
            rows = self._conn.execute(
                "SELECT rel_key, card_text FROM file_cards ORDER BY rel_key"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        cards = [(str(rk), json.loads(ct)) for rk, ct in rows]
        if len(cards) > 120:
            return self._overview_rollup(cards, max_chars)
        lines: dict[str, str] = {}
        for depth in (6, 5, 4, 3, 2, 1, 0):
            lines = {rk: self._overview_line(rk, c, depth) for rk, c in cards}
            if len(self._overview_render(cards, lines)) <= max_chars:
                break
        if len(self._overview_render(cards, lines)) > max_chars:
            return []
        return [{"folder": self._folder(rk), "rel_key": rk, "line": lines[rk]} for rk, _ in cards]

    @staticmethod
    def _folder(rel_key: str) -> str:
        return rel_key.split("/", 1)[0] if "/" in rel_key else "(root)"

    @staticmethod
    def _overview_line(rk: str, card: dict[str, Any], depth: int) -> str:
        parts: list[str] = [rk]
        if depth >= 1:
            parts += [str(card.get("title") or ""), str(card.get("kind") or "")]
        if depth >= 2:
            parts.append(", ".join(str(s) for s in (card.get("sections") or [])[:3]))
        if depth >= 3:
            parts.append(", ".join(str(e) for e in (card.get("top_entities") or [])[:5]))
        if depth >= 4:
            dmin = card.get("date_min")
            dmax = card.get("date_max")
            if dmin and dmax:
                parts.append(f"{dmin} to {dmax}")
            elif dmin:
                parts.append(str(dmin))
        if depth >= 5:
            parts.append(f"{card.get('row_count', 0)} rows")
        return " | ".join(p for p in parts if p)

    @staticmethod
    def _overview_render(cards: list[tuple[str, dict[str, Any]]], lines: dict[str, str]) -> str:
        folders: dict[str, list[str]] = {}
        for rk, _ in cards:
            folders.setdefault(DbIndex._folder(rk), []).append(lines[rk])
        out = [f"{len(cards)} files in {len(folders)} folders:"]
        for f in sorted(folders):
            out.append(f"{f} ({len(folders[f])} files):")
            out.extend(folders[f])
        return "\n".join(out)

    @staticmethod
    def _overview_rollup(
        cards: list[tuple[str, dict[str, Any]]], max_chars: int
    ) -> list[dict[str, Any]]:
        folders: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        titles: dict[str, str] = {}
        for rk, card in cards:
            folders.setdefault(DbIndex._folder(rk), []).append((rk, card))
            titles[rk] = (str(card.get("title") or "") or Path(rk).stem)[:40]
        for n_titles in (3, 2, 1):
            out = [f"{len(cards)} files in {len(folders)} folders:"]
            for f in sorted(folders):
                files = folders[f]
                shown = "; ".join(titles[rk] for rk, _ in files[:n_titles])
                out.append(f"{f} ({len(files)} files): {shown}")
            if len("\n".join(out)) <= max_chars:
                return [
                    {
                        "folder": f,
                        "n_files": len(folders[f]),
                        "titles": [titles[rk] for rk, _ in folders[f][:n_titles]],
                    }
                    for f in sorted(folders)
                ]
        return []

    @staticmethod
    def _topic_terms(query: str) -> list[str]:
        return [t for t in dict.fromkeys(tokenize(query)) if t not in _DOC_CUE_TERMS]

    def enumerate_files(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """Every file matching `query` (topic terms, not document-cue words), one best line each."""
        match = _build_match_query(query)
        if not match or limit <= 0:
            return []
        terms = self._topic_terms(query)
        identity = bool(_IDENTIFY_CUE.search(query)) and bool(terms)
        best: dict[str, str] = {}
        order: list[str] = []
        for cid, _ in self._fts.query(match, limit=None, weights=self._weights):
            rk = cid.rsplit("#", 1)[0]
            if rk not in best:
                best[rk] = cid
                if len(order) < limit:
                    order.append(rk)
        titles: dict[str, str] = {}
        if identity:
            titles = {
                str(rk): str(json.loads(ct).get("title") or "")
                for rk, ct in self._conn.execute(
                    "SELECT rel_key, card_text FROM file_cards"
                ).fetchall()
            }
        rows: list[dict[str, Any]] = []
        for rk in order:
            if identity and not (
                (set(tokenize(rk)) | set(tokenize(titles.get(rk, "")))) & set(terms)
            ):
                continue
            text = " ".join(
                str(r[0])
                for r in self._conn.execute(
                    "SELECT text FROM chunk_data WHERE rel_key = ? ORDER BY ord", (rk,)
                ).fetchall()
            )
            if not identity and not (set(tokenize(text)) & set(terms)):
                continue
            row = self._conn.execute(
                "SELECT text FROM chunk_data WHERE chunk_id = ?", (best[rk],)
            ).fetchone()
            chunk_text = str(row[0]) if row else ""
            line = next((ln for ln in chunk_text.splitlines() if ln.strip()), "")[:200]
            rows.append({"rel_key": rk, "line": line})
            if len(rows) >= limit:
                break
        return rows

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


def _normalize_ocr(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return s


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def ocr_quote_ok(quote: str, passage_text: str) -> bool:
    """OCR-safe quote check: normalization + exact numeric subset + substring."""
    qn = _normalize_ocr(quote)
    pn = _normalize_ocr(passage_text)
    if qn not in pn:
        return False
    qnums = _NUM_RE.findall(qn)
    pnums = _NUM_RE.findall(pn)
    if not qnums:
        return True
    idx = 0
    for qn_val in qnums:
        found = False
        for i in range(idx, len(pnums)):
            if pnums[i] == qn_val:
                idx = i + 1
                found = True
                break
        if not found:
            return False
    return True
