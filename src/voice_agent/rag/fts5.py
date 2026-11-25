"""SQLite FTS5 lexical retriever implementing the ``Retriever`` protocol with BM25 fallback."""

from __future__ import annotations

import collections.abc
import contextlib
import hashlib
import math
import re
import sqlite3
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from doc_chat.corpus import tokenize
from voice_agent.rag.docindex import HEADER_WEIGHT, Chunk, Retriever, _tokenize_header
from voice_agent.rag.fts5_store import (
    DEFAULT_BUSY_TIMEOUT_S,
    DEFAULT_COLUMNS,
    DEFAULT_LOCK_TIMEOUT_S,
    DEFAULT_RETRY_DELAY_S,
    DEFAULT_TOKENIZER,
    Fts5Store,
    Fts5StoreError,
    SyncStats,
    fts5_available,
    is_sqlite_transient,
)

Fts5TokenizerMode = Literal["pretok", "porter"]

MAX_QUERY_TERMS = 64
PRETOK_COLUMNS: tuple[str, ...] = DEFAULT_COLUMNS
PRETOK_TOKENIZER = DEFAULT_TOKENIZER
PORTER_COLUMNS: tuple[str, ...] = ("header", "text")
PORTER_TOKENIZER = "porter unicode61 remove_diacritics 2"
_ABS_PATH_RE = re.compile(r"/(?:[^\s/'\":]+/)+[^\s/'\":]*")


def _clean_reason(reason: str) -> str:
    single = " ".join(reason.splitlines()).strip()
    return single or "unknown error"


def _safe_disk_reason(exc: BaseException, db_path: Path, root: Path) -> str:
    """Return ``ExcType: short reason`` without filesystem paths or document text."""
    exc_type = type(exc).__name__
    if isinstance(exc, OSError) and not isinstance(exc, Fts5StoreError) and exc.strerror:
        raw = exc.strerror
    else:
        raw = str(exc)

    candidates: list[str] = []
    for p in (db_path, db_path.parent, root):
        candidates.append(str(p))
        with contextlib.suppress(OSError):
            candidates.append(str(p.resolve()))
    for needle in sorted(set(candidates), key=len, reverse=True):
        if needle:
            raw = raw.replace(needle, "<path>")

    raw = _ABS_PATH_RE.sub("<path>", raw)
    cleaned = _clean_reason(raw)
    if len(cleaned) > 120:
        cleaned = cleaned[:117] + "..."
    return f"{exc_type}: {cleaned}"


def _chunk_key(header: str, text: str) -> str:
    """Return the sha256 hex digest of ``header + '\\n' + text`` (matches ``embed_key``)."""
    payload = f"{header}\n{text}".encode()
    return hashlib.sha256(payload).hexdigest()


def _build_match_query(query: str) -> str:
    """Build a safe OR-joined FTS5 MATCH string of at most 64 unique quoted terms."""
    terms = list(dict.fromkeys(tokenize(query)))[:MAX_QUERY_TERMS]
    if not terms:
        return ""
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def _pretok_row(chunk: Chunk) -> tuple[str, str, str, str]:
    header_tokens = _tokenize_header(chunk.header)
    body_tokens = tokenize(chunk.text)
    return (
        chunk.id,
        _chunk_key(chunk.header, chunk.text),
        " ".join(body_tokens),
        " ".join(header_tokens),
    )


def _porter_row(chunk: Chunk) -> tuple[str, str, str, str]:
    return (
        chunk.id,
        _chunk_key(chunk.header, chunk.text),
        chunk.header,
        chunk.text,
    )


class Fts5Retriever:
    """SQLite FTS5 lexical retriever with incremental sync and lazy BM25 fallback."""

    def __init__(
        self,
        db_path: Path | None,
        root: Path,
        *,
        tokenizer: Fts5TokenizerMode = "pretok",
        header_weight: float = HEADER_WEIGHT,
        _busy_timeout: float = DEFAULT_BUSY_TIMEOUT_S,
        _lock_timeout: float = DEFAULT_LOCK_TIMEOUT_S,
        _retry_delay: float = DEFAULT_RETRY_DELAY_S,
    ) -> None:
        if tokenizer not in ("pretok", "porter"):
            raise ValueError(
                f"unknown FTS5 tokenizer mode {tokenizer!r}; expected 'pretok' or 'porter'"
            )
        weight = float(header_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"header_weight must be a finite non-negative float, got {header_weight!r}"
            )

        self._db_path: Path | None = None if db_path is None else Path(db_path)
        self._root = Path(root)
        self._tokenizer_mode: Fts5TokenizerMode = tokenizer
        self._header_weight = weight
        self._busy_timeout = max(0.0, float(_busy_timeout))
        self._lock_timeout = max(0.0, float(_lock_timeout))
        self._retry_delay = max(0.0, float(_retry_delay))

        if tokenizer == "pretok":
            self._columns = PRETOK_COLUMNS
            self._fts_tokenizer = PRETOK_TOKENIZER
            self._weights: tuple[float, float] = (1.0, weight)
        else:
            self._columns = PORTER_COLUMNS
            self._fts_tokenizer = PORTER_TOKENIZER
            self._weights = (weight, 1.0)

        self._lock = threading.Lock()
        self._store: Fts5Store | None = None
        self._is_memory_store = self._db_path is None
        self._bm25: Retriever | None = None
        self._chunks: list[Chunk] = []
        self._id_to_index: dict[str, int] = {}
        self._last_sync_stats: SyncStats | None = None
        self._broken_reason: str | None = None
        self._logged_unavailable = False
        self._logged_disk_unavailable = False
        self._sync_pending = False
        self._closed = False

        ok, reason = fts5_available(self._fts_tokenizer)
        if not ok:
            self._fallback_to_bm25_locked(reason, rebuild=False)
            return

        store, open_err = self._open_store()
        if store is not None:
            self._store = store
            self._is_memory_store = store.is_memory
        else:
            self._fallback_to_bm25_locked(open_err, rebuild=False)

    @property
    def last_sync_stats(self) -> SyncStats | None:
        """Counts from the most recent ``Fts5Store.sync`` call, if any."""
        with self._lock:
            return self._last_sync_stats

    def _store_timeout_kwargs(self) -> dict[str, float]:
        kwargs: dict[str, float] = {}
        if self._busy_timeout != DEFAULT_BUSY_TIMEOUT_S:
            kwargs["_busy_timeout"] = self._busy_timeout
        if self._lock_timeout != DEFAULT_LOCK_TIMEOUT_S:
            kwargs["_lock_timeout"] = self._lock_timeout
        if self._retry_delay != DEFAULT_RETRY_DELAY_S:
            kwargs["_retry_delay"] = self._retry_delay
        return kwargs

    def _open_store(self) -> tuple[Fts5Store | None, str]:
        if self._db_path is not None:
            try:
                store = Fts5Store(
                    self._db_path,
                    self._root,
                    tokenizer=self._fts_tokenizer,
                    columns=self._columns,
                    **self._store_timeout_kwargs(),
                )
                return store, ""
            except (Fts5StoreError, OSError, sqlite3.Error) as disk_exc:
                safe_reason = _safe_disk_reason(disk_exc, self._db_path, self._root)
                try:
                    mem_store = Fts5Store(
                        None,
                        self._root,
                        tokenizer=self._fts_tokenizer,
                        columns=self._columns,
                    )
                except (Fts5StoreError, OSError, sqlite3.Error) as mem_exc:
                    mem_err = f"{type(mem_exc).__name__}: {mem_exc}"
                    return None, f"{safe_reason}; memory fallback failed: {mem_err}"
                if not self._logged_disk_unavailable:
                    self._logged_disk_unavailable = True
                    print(
                        f"rag: fts5 disk store unavailable ({safe_reason}); using memory store",
                        file=sys.stderr,
                        flush=True,
                    )
                return mem_store, ""

        try:
            mem_store = Fts5Store(
                None,
                self._root,
                tokenizer=self._fts_tokenizer,
                columns=self._columns,
            )
            return mem_store, ""
        except (Fts5StoreError, OSError, sqlite3.Error) as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _fallback_to_bm25_locked(self, reason: str, *, rebuild: bool) -> Retriever:
        cleaned = _clean_reason(reason)
        if self._broken_reason is None:
            self._broken_reason = cleaned
        if self._store is not None:
            with contextlib.suppress(sqlite3.Error):
                self._store.close()
            self._store = None
        if not self._logged_unavailable:
            self._logged_unavailable = True
            print(
                f"rag: fts5 unavailable ({cleaned}); serving BM25",
                file=sys.stderr,
                flush=True,
            )
        if self._bm25 is None:
            from voice_agent.rag.docindex import Bm25Retriever

            bm25 = Bm25Retriever()
            if rebuild:
                bm25.build(self._chunks)
            self._bm25 = bm25
        elif rebuild:
            self._bm25.build(self._chunks)
        return self._bm25

    def status(self) -> str:
        """Return ``fts5:file``, ``fts5:memory``, or ``bm25 (fts5 unavailable: ...)``."""
        with self._lock:
            if self._bm25 is not None or self._broken_reason is not None:
                reason = self._broken_reason or "unknown error"
                return f"bm25 (fts5 unavailable: {reason})"
            if self._store is not None:
                return "fts5:memory" if self._store.is_memory else "fts5:file"
            return "fts5:memory" if self._is_memory_store else "fts5:file"

    def _ephemeral_bm25_rank_locked(self, query: str, k: int) -> list[tuple[int, float]]:
        from voice_agent.rag.docindex import Bm25Retriever

        ephemeral = Bm25Retriever()
        ephemeral.build(self._chunks)
        return ephemeral.rank(query, k)

    def _sync_store_locked(self, store: Fts5Store, chunk_list: list[Chunk]) -> bool:
        row_fn = _pretok_row if self._tokenizer_mode == "pretok" else _porter_row
        keyed = [(chunk.id, _chunk_key(chunk.header, chunk.text)) for chunk in chunk_list]
        chunks_by_id = {chunk.id: chunk for chunk in chunk_list}

        def _make_row(chunk_id: str) -> tuple[str, ...]:
            return row_fn(chunks_by_id[chunk_id])

        for attempt in range(2):
            try:
                self._last_sync_stats = store.sync_lazy(keyed, _make_row)
                self._sync_pending = False
                return True
            except sqlite3.Error as exc:
                if is_sqlite_transient(exc):
                    if attempt == 0:
                        if self._retry_delay > 0.0:
                            time.sleep(self._retry_delay)
                        continue
                    self._sync_pending = True
                    return False
                self._sync_pending = False
                self._fallback_to_bm25_locked(f"{type(exc).__name__}: {exc}", rebuild=True)
                return False
        return False

    def build(self, chunks: Sequence[Chunk]) -> None:
        """Incrementally sync ``chunks`` into FTS5 (or delegate to BM25 on fallback)."""
        chunk_list = list(chunks)
        with self._lock:
            if self._closed:
                raise sqlite3.ProgrammingError("Cannot build on a closed Fts5Retriever.")
            self._chunks = chunk_list
            self._id_to_index = {chunk.id: idx for idx, chunk in enumerate(chunk_list)}

            if self._bm25 is not None:
                self._bm25.build(chunk_list)
                return

            store = self._store
            if store is None:
                bm25 = self._fallback_to_bm25_locked(
                    self._broken_reason or "store unavailable", rebuild=False
                )
                bm25.build(chunk_list)
                return

            self._sync_store_locked(store, chunk_list)

    def build_streaming(
        self,
        chunks: collections.abc.Iterable[Chunk],
        *,
        batch: int = 5000,
        prune_missing: bool = True,
    ) -> int:
        """Stream ``chunks`` into a fresh FTS5 store in batches (never > ``batch`` in memory)."""
        if batch <= 0:
            raise ValueError("batch must be positive")

        with self._lock:
            if self._closed:
                raise sqlite3.ProgrammingError("Cannot build on a closed Fts5Retriever.")

            if self._bm25 is not None:
                cl: list[Chunk] = []
                for chunk in chunks:
                    cl.append(chunk)
                self._chunks = cl
                self._id_to_index = {c.id: i for i, c in enumerate(cl)}
                self._bm25.build(cl)
                return len(cl)

            store = self._store
            if store is None:
                bm25 = self._fallback_to_bm25_locked(
                    self._broken_reason or "store unavailable", rebuild=False
                )
                cl = list(chunks)
                self._chunks = cl
                self._id_to_index = {c.id: i for i, c in enumerate(cl)}
                bm25.build(cl)
                return len(cl)

        row_fn = _pretok_row if self._tokenizer_mode == "pretok" else _porter_row
        total = 0
        batch_buf: list[Chunk] = []

        seen: set[str] = set()

        def flush(buf: list[Chunk]) -> None:
            # sync WITHOUT pruning: a pruning sync would delete the earlier batches
            by_id = {c.id: c for c in buf}
            keyed = [(c.id, _chunk_key(c.header, c.text)) for c in buf]
            seen.update(by_id)
            for attempt in range(2):
                try:
                    store.sync_lazy(keyed, lambda cid: row_fn(by_id[cid]), prune=False)
                    return
                except sqlite3.Error as exc:
                    if is_sqlite_transient(exc) and attempt == 0:
                        if self._retry_delay > 0.0:
                            time.sleep(self._retry_delay)
                        continue
                    raise

        for chunk in chunks:
            batch_buf.append(chunk)
            if len(batch_buf) >= batch:
                flush(batch_buf)
                total += len(batch_buf)
                batch_buf = []
        if batch_buf:
            flush(batch_buf)
            total += len(batch_buf)

        if prune_missing:
            store.prune_except(seen)
        # a streamed build never holds the chunk list: the reader (DbIndex) loads rows on demand
        self._chunks = []
        self._id_to_index = {}
        return total

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(chunk_index, score)`` pairs ordered best-first."""
        if k <= 0:
            return []
        with self._lock:
            if self._closed or not self._chunks:
                return []
            if self._bm25 is not None:
                return self._bm25.rank(query, k)

            match = _build_match_query(query)
            if not match:
                return []

            store = self._store
            if store is None:
                bm25 = self._fallback_to_bm25_locked(
                    self._broken_reason or "store unavailable", rebuild=True
                )
                return bm25.rank(query, k)

            if self._sync_pending and not self._sync_store_locked(store, self._chunks):
                if self._bm25 is not None:
                    return self._bm25.rank(query, k)
                return self._ephemeral_bm25_rank_locked(query, k)

            hits: list[tuple[str, float]] = []
            for attempt in range(2):
                try:
                    hits = store.query(match, limit=None, weights=self._weights)
                    break
                except sqlite3.Error as exc:
                    if is_sqlite_transient(exc):
                        if attempt == 0:
                            if self._retry_delay > 0.0:
                                time.sleep(self._retry_delay)
                            continue
                        return self._ephemeral_bm25_rank_locked(query, k)
                    bm25 = self._fallback_to_bm25_locked(
                        f"{type(exc).__name__}: {exc}", rebuild=True
                    )
                    return bm25.rank(query, k)

            ranked: list[tuple[int, float]] = []
            for chunk_id, score in hits:
                idx = self._id_to_index.get(chunk_id)
                if idx is not None:
                    ranked.append((idx, float(score)))
                    if len(ranked) >= k:
                        break
            return ranked

    def close(self) -> None:
        """Close the underlying FTS5 store idempotently."""
        with self._lock:
            self._closed = True
            if self._store is not None:
                with contextlib.suppress(sqlite3.Error):
                    self._store.close()
                self._store = None

    def __enter__(self) -> Fts5Retriever:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
