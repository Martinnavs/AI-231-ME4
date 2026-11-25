"""Hardened SQLite FTS5 storage backend for RAG lexical retrieval."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import os
import re
import sqlite3
import stat
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from doc_chat.corpus import _STOPWORDS, tokenize
from voice_agent.rag.canonical_cache import validate_cache_dir

SCHEMA_VERSION = "2"
DEFAULT_COLUMNS: tuple[str, ...] = ("all_tokens", "header_tokens")
DEFAULT_TOKENIZER = "unicode61 remove_diacritics 0 tokenchars ''''"
DEFAULT_BUSY_TIMEOUT_S = 5.0
DEFAULT_LOCK_TIMEOUT_S = 2.0
DEFAULT_RETRY_DELAY_S = 0.05

_COLUMN_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_FINGERPRINT_PROBE = (
    "Boxes's churches wishes passes cats\u2019 running walked 2026 1,599.00 3.14 the and about"
    " café naïve Straße ﬁnance œuvre"
)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _O_CLOEXEC
_DB_SIDECAR_SUFFIXES: tuple[str, ...] = ("-journal", "-wal", "-shm")
_DEFENSIVE_CONFIG: int | None = getattr(sqlite3, "SQLITE_DBCONFIG_DEFENSIVE", None)
_SQLITE_BUSY: int = getattr(sqlite3, "SQLITE_BUSY", 5)
_SQLITE_LOCKED: int = getattr(sqlite3, "SQLITE_LOCKED", 6)
_SQLITE_FULL: int = getattr(sqlite3, "SQLITE_FULL", 13)

_SELECT_MASTER_SQL = "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
_SELECT_META_SQL = "SELECT key, value FROM meta"
_INSERT_META_SQL = "INSERT INTO meta (key, value) VALUES (?, ?)"
_SELECT_CHUNKS_SQL = "SELECT rowid, chunk_id, key FROM chunks"
_DELETE_FTS_SQL = "DELETE FROM fts WHERE rowid = ?"
_DELETE_CHUNKS_SQL = "DELETE FROM chunks WHERE rowid = ?"
_INSERT_CHUNKS_SQL = "INSERT INTO chunks (rowid, chunk_id, key) VALUES (?, ?, ?)"

_CREATE_META_SQL = "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
_CREATE_CHUNKS_SQL = (
    "CREATE TABLE chunks ("
    "rowid INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE, key TEXT NOT NULL)"
)

_MASTER_CACHE_LOCK = threading.Lock()
_MASTER_CACHE: dict[tuple[tuple[str, ...], str], tuple[tuple[str, str, str, str | None], ...]] = {}


class Fts5StoreError(OSError):
    """Raised when the FTS5 store refuses an unsafe path or cannot initialize."""


@dataclass(frozen=True, slots=True)
class SyncStats:
    """Counts of row mutations applied during a single ``Fts5Store.sync`` call."""

    inserted: int
    deleted: int


def tokenize_fingerprint() -> str:
    """Return a sha256 digest capturing ``doc_chat.corpus.tokenize`` and ``_STOPWORDS``."""
    probed = " ".join(tokenize(_FINGERPRINT_PROBE))
    stops = ",".join(sorted(_STOPWORDS))
    payload = (probed + "\n" + stops).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_columns(columns: Sequence[str]) -> tuple[str, ...]:
    cols = tuple(columns)
    if not cols:
        raise ValueError("columns must be non-empty")
    seen: set[str] = set()
    for col in cols:
        if not isinstance(col, str) or not _COLUMN_RE.fullmatch(col):
            raise ValueError("invalid FTS column name: " + repr(col))
        if col in seen:
            raise ValueError("duplicate FTS column name: " + repr(col))
        seen.add(col)
    return cols


def _validate_tokenizer(tokenizer: str) -> str:
    if not isinstance(tokenizer, str) or not tokenizer.strip():
        raise ValueError("tokenizer must be a non-empty string")
    if any(ch in tokenizer for ch in ("\x00", "\n", "\r", ";")):
        raise ValueError("tokenizer contains disallowed control characters")
    return tokenizer


def _build_create_fts_sql(columns: tuple[str, ...], tokenizer: str) -> str:
    escaped_tokenizer = tokenizer.replace("'", "''")
    col_list = ", ".join(columns)
    return (
        "CREATE VIRTUAL TABLE fts USING fts5("
        + col_list
        + ", tokenize='"
        + escaped_tokenizer
        + "')"
    )


def _build_insert_fts_sql(columns: tuple[str, ...]) -> str:
    col_list = ", ".join(("rowid", *columns))
    placeholders = ", ".join("?" for _ in range(1 + len(columns)))
    return "INSERT INTO fts (" + col_list + ") VALUES (" + placeholders + ")"


def _build_query_sql(num_columns: int, *, with_limit: bool) -> str:
    weight_placeholders = ", ".join("?" for _ in range(num_columns))
    bm25_expr = "bm25(fts, " + weight_placeholders + ")"
    sql = (
        "SELECT chunks.chunk_id, -"
        + bm25_expr
        + " AS score FROM fts JOIN chunks ON chunks.rowid = fts.rowid "
        "WHERE fts MATCH ? ORDER BY " + bm25_expr
    )
    if with_limit:
        return sql + " LIMIT ?"
    return sql


def _harden_connection(conn: sqlite3.Connection) -> None:
    """Apply defensive SQLite configuration and pragmas to ``conn``."""
    # Connection.setconfig / SQLITE_DBCONFIG_DEFENSIVE exist on Python 3.12; guard defensively.
    setconfig = getattr(conn, "setconfig", None)
    if _DEFENSIVE_CONFIG is not None and callable(setconfig):
        setconfig(_DEFENSIVE_CONFIG, True)
    conn.execute("PRAGMA trusted_schema = OFF")
    conn.execute("PRAGMA cell_size_check = ON")


def _init_schema(
    conn: sqlite3.Connection,
    columns: tuple[str, ...],
    tokenizer: str,
) -> None:
    create_fts_sql = _build_create_fts_sql(columns, tokenizer)
    conn.execute("BEGIN")
    try:
        conn.execute(_CREATE_META_SQL)
        conn.execute(_CREATE_CHUNKS_SQL)
        conn.execute(create_fts_sql)
        conn.execute("COMMIT")
    except BaseException:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("ROLLBACK")
        raise


def _write_meta(
    conn: sqlite3.Connection,
    columns: tuple[str, ...],
    tokenizer: str,
) -> None:
    meta_rows = (
        ("schema_version", SCHEMA_VERSION),
        ("tokenizer", tokenizer),
        ("columns", ",".join(columns)),
        ("tokenize_fingerprint", tokenize_fingerprint()),
    )
    conn.execute("BEGIN")
    try:
        conn.executemany(_INSERT_META_SQL, meta_rows)
        conn.execute("COMMIT")
    except BaseException:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("ROLLBACK")
        raise


def _expected_master_rows(
    columns: tuple[str, ...],
    tokenizer: str,
) -> tuple[tuple[str, str, str, str | None], ...]:
    cache_key = (columns, tokenizer)
    with _MASTER_CACHE_LOCK:
        cached = _MASTER_CACHE.get(cache_key)
        if cached is not None:
            return cached

    probe_conn = sqlite3.connect(":memory:", isolation_level=None)
    try:
        _harden_connection(probe_conn)
        _init_schema(probe_conn, columns, tokenizer)
        rows = tuple(
            (str(r[0]), str(r[1]), str(r[2]), None if r[3] is None else str(r[3]))
            for r in probe_conn.execute(_SELECT_MASTER_SQL).fetchall()
        )
    finally:
        probe_conn.close()

    with _MASTER_CACHE_LOCK:
        _MASTER_CACHE[cache_key] = rows
    return rows


def fts5_available(tokenizer: str = DEFAULT_TOKENIZER) -> tuple[bool, str]:
    """Probe in-memory FTS5 virtual table creation, insert, ``MATCH``, and ``bm25()``."""
    try:
        tok = _validate_tokenizer(tokenizer)
    except ValueError as exc:
        return False, str(exc)

    probe_ddl = _build_create_fts_sql(("body", "header"), tok)
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        _harden_connection(conn)
        conn.execute(probe_ddl)
        conn.execute(
            "INSERT INTO fts (rowid, body, header) VALUES (?, ?, ?)",
            (1, "alpha beta", "gamma"),
        )
        rows = conn.execute(
            "SELECT rowid, -bm25(fts, ?, ?) FROM fts WHERE fts MATCH ?",
            (1.0, 1.5, '"alpha"'),
        ).fetchall()
        if not rows or rows[0][0] != 1 or not isinstance(rows[0][1], (int, float)):
            return False, "fts5 probe query returned unexpected result"
        return True, ""
    except (sqlite3.Error, ValueError, RuntimeError) as exc:
        reason = str(exc).strip() or type(exc).__name__
        return False, type(exc).__name__ + ": " + reason
    finally:
        if conn is not None:
            with contextlib.suppress(sqlite3.Error):
                conn.close()


def _euid() -> int:
    geteuid = getattr(os, "geteuid", None)
    if callable(geteuid):
        return int(geteuid())
    return 0


def _open_validated_dir(dir_path: Path, root_path: Path, *, create: bool) -> int:
    validated = validate_cache_dir(dir_path, root_path)
    if validated is None:
        raise Fts5StoreError("refusing unsafe or symlinked cache directory")
    base_path, rel_parts = validated

    if base_path.is_symlink():
        raise Fts5StoreError("refusing symlinked cache root directory")

    try:
        fd = os.open(str(base_path), _DIR_FLAGS)
    except FileNotFoundError as exc:
        if not create:
            raise Fts5StoreError("cache root directory does not exist") from exc
        base_path.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(str(base_path), _DIR_FLAGS)

    try:
        for part in rel_parts:
            try:
                nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError as exc:
                if not create:
                    raise Fts5StoreError("cache directory does not exist") from exc
                with contextlib.suppress(FileExistsError):
                    os.mkdir(part, 0o700, dir_fd=fd)
                nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nxt
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise
    return fd


def _verify_existing_file(dfd: int, name: str) -> os.stat_result | None:
    """Return ``lstat`` result if ``name`` exists under ``dfd``, or ``None`` if absent.

    Refuses symlinks, non-regular files, and files not owned by the effective UID.
    """
    try:
        st = os.lstat(name, dir_fd=dfd)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(st.st_mode):
        raise Fts5StoreError("refusing symlinked database file: " + name)
    if not stat.S_ISREG(st.st_mode):
        raise Fts5StoreError("refusing non-regular database file: " + name)
    if st.st_uid != _euid():
        raise Fts5StoreError("refusing database file not owned by effective UID: " + name)
    return st


def _create_db_file(dfd: int, name: str) -> None:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _O_CLOEXEC
    fd = os.open(name, flags, 0o600, dir_fd=dfd)
    os.close(fd)


def _unlink_db_and_sidecars(dfd: int, db_name: str) -> None:
    """Safely unlink the DB file and its non-symlink sidecars inside ``dfd``."""
    names = (db_name, *(db_name + suffix for suffix in _DB_SIDECAR_SUFFIXES))
    for name in names:
        st = _verify_existing_file(dfd, name)
        if st is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(name, dir_fd=dfd)


def _same_inode(a: os.stat_result, b: os.stat_result) -> bool:
    return a.st_dev == b.st_dev and a.st_ino == b.st_ino


def is_sqlite_busy_or_locked(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` is a SQLite BUSY or LOCKED ``OperationalError``."""
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int) and (code & 0xFF) in (_SQLITE_BUSY, _SQLITE_LOCKED):
        return True
    name = getattr(exc, "sqlite_errorname", "")
    if isinstance(name, str) and (
        name.startswith("SQLITE_BUSY") or name.startswith("SQLITE_LOCKED")
    ):
        return True
    msg = str(exc).lower()
    return (
        "database is locked" in msg
        or "database table is locked" in msg
        or "database schema is locked" in msg
        or "sqlite_busy" in msg
        or "sqlite_locked" in msg
        or "busy" in msg
    )


def is_sqlite_transient(exc: BaseException) -> bool:
    """Return ``True`` for transient ``OperationalError`` (BUSY, LOCKED, or disk-full)."""
    if is_sqlite_busy_or_locked(exc):
        return True
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int) and (code & 0xFF) == _SQLITE_FULL:
        return True
    name = getattr(exc, "sqlite_errorname", "")
    if isinstance(name, str) and name.startswith("SQLITE_FULL"):
        return True
    msg = str(exc).lower()
    return "database or disk is full" in msg or "disk full" in msg or "disk is full" in msg


def _acquire_store_lock(dfd: int, lock_name: str, *, timeout: float) -> int:
    """Open ``lock_name`` inside ``dfd`` safely (0600, O_NOFOLLOW) and take ``flock(LOCK_EX)``."""
    _verify_existing_file(dfd, lock_name)
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | _O_CLOEXEC
    try:
        lock_fd = os.open(lock_name, flags, 0o600, dir_fd=dfd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise Fts5StoreError("refusing symlinked lock file: " + lock_name) from exc
        raise

    try:
        st = os.fstat(lock_fd)
        if not stat.S_ISREG(st.st_mode):
            raise Fts5StoreError("refusing non-regular lock file: " + lock_name)
        if st.st_uid != _euid():
            raise Fts5StoreError("refusing lock file not owned by effective UID: " + lock_name)

        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_fd
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise Fts5StoreError("timed out waiting for FTS5 store lock") from exc
                time.sleep(min(0.02, max(0.001, remaining)))
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(lock_fd)
        raise


def _release_store_lock(lock_fd: int) -> None:
    """Release and close an ``flock`` descriptor acquired by ``_acquire_store_lock``."""
    try:
        with contextlib.suppress(OSError):
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        with contextlib.suppress(OSError):
            os.close(lock_fd)


class Fts5Store:
    """Symlink-safe, validated SQLite FTS5 store with incremental chunk sync."""

    def __init__(
        self,
        path: Path | None,
        root: Path,
        *,
        tokenizer: str = DEFAULT_TOKENIZER,
        columns: Sequence[str] = DEFAULT_COLUMNS,
        _busy_timeout: float = DEFAULT_BUSY_TIMEOUT_S,
        _lock_timeout: float = DEFAULT_LOCK_TIMEOUT_S,
        _retry_delay: float = DEFAULT_RETRY_DELAY_S,
    ) -> None:
        self._columns = _validate_columns(columns)
        self._tokenizer = _validate_tokenizer(tokenizer)
        self._path = None if path is None else Path(path)
        self._root = Path(root)
        self._busy_timeout = max(0.0, float(_busy_timeout))
        self._lock_timeout = max(0.0, float(_lock_timeout))
        self._retry_delay = max(0.0, float(_retry_delay))
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

        self._insert_fts_sql = _build_insert_fts_sql(self._columns)
        self._query_sql = _build_query_sql(len(self._columns), with_limit=False)
        self._query_limit_sql = _build_query_sql(len(self._columns), with_limit=True)

        if self._path is None:
            self._open_memory()
        else:
            self._open_disk(self._path, self._root)

    @property
    def path(self) -> Path | None:
        """Database path, or ``None`` when running in memory."""
        return self._path

    @property
    def root(self) -> Path:
        """Trusted root directory constraining the database path."""
        return self._root

    @property
    def tokenizer(self) -> str:
        """Configured FTS5 tokenizer specification."""
        return self._tokenizer

    @property
    def columns(self) -> tuple[str, ...]:
        """Configured FTS5 column names."""
        return self._columns

    @property
    def is_memory(self) -> bool:
        """Whether this store operates purely in memory."""
        return self._path is None

    def _open_memory(self) -> None:
        conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        try:
            _harden_connection(conn)
            _init_schema(conn, self._columns, self._tokenizer)
            _write_meta(conn, self._columns, self._tokenizer)
        except BaseException:
            conn.close()
            raise
        self._conn = conn

    def _connect_disk_uri(self, dfd: int, db_path: Path, db_name: str) -> sqlite3.Connection:
        dir_stat = os.fstat(dfd)
        parent_stat = os.lstat(db_path.parent)
        if not _same_inode(dir_stat, parent_stat):
            raise Fts5StoreError("database parent directory changed during open")

        file_stat = _verify_existing_file(dfd, db_name)
        if file_stat is None:
            raise Fts5StoreError("database file disappeared before connect")

        abs_db = Path(os.path.abspath(db_path))
        uri = abs_db.as_uri() + "?mode=rw"
        conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=self._busy_timeout,
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            post_stat = os.lstat(db_path)
            if stat.S_ISLNK(post_stat.st_mode) or not _same_inode(file_stat, post_stat):
                raise Fts5StoreError("database file changed during connect")
            _harden_connection(conn)
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                conn.close()
            raise
        return conn

    def _validate_existing_db(self, conn: sqlite3.Connection) -> bool:
        expected_master = _expected_master_rows(self._columns, self._tokenizer)
        actual_master = tuple(
            (str(r[0]), str(r[1]), str(r[2]), None if r[3] is None else str(r[3]))
            for r in conn.execute(_SELECT_MASTER_SQL).fetchall()
        )
        if actual_master != expected_master:
            return False

        quick_rows = conn.execute("PRAGMA quick_check").fetchall()
        if quick_rows != [("ok",)]:
            return False

        meta = {str(k): str(v) for k, v in conn.execute(_SELECT_META_SQL).fetchall()}
        return (
            meta.get("schema_version") == SCHEMA_VERSION
            and meta.get("tokenizer") == self._tokenizer
            and meta.get("columns") == ",".join(self._columns)
            and meta.get("tokenize_fingerprint") == tokenize_fingerprint()
        )

    def _create_and_init_disk_db(
        self,
        dfd: int,
        db_path: Path,
        db_name: str,
        *,
        validate: bool,
    ) -> sqlite3.Connection:
        _create_db_file(dfd, db_name)
        conn = self._connect_disk_uri(dfd, db_path, db_name)
        try:
            _init_schema(conn, self._columns, self._tokenizer)
            _write_meta(conn, self._columns, self._tokenizer)
            if validate and not self._validate_existing_db(conn):
                raise Fts5StoreError("rebuilt database failed validation")
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                conn.close()
            raise
        return conn

    def _open_and_validate_existing(
        self,
        dfd: int,
        db_path: Path,
        db_name: str,
    ) -> tuple[sqlite3.Connection | None, bool]:
        for attempt in range(2):
            conn: sqlite3.Connection | None = None
            try:
                conn = self._connect_disk_uri(dfd, db_path, db_name)
                valid = self._validate_existing_db(conn)
                return conn, valid
            except sqlite3.DatabaseError as exc:
                if conn is not None:
                    with contextlib.suppress(sqlite3.Error):
                        conn.close()
                if is_sqlite_busy_or_locked(exc):
                    if attempt == 0:
                        if self._retry_delay > 0.0:
                            time.sleep(self._retry_delay)
                        continue
                    raise
                return None, False
        return None, False

    def _open_disk(self, db_path: Path, root_path: Path) -> None:
        db_name = db_path.name
        if not db_name or db_name in (".", ".."):
            raise Fts5StoreError("invalid database filename")
        if db_path.is_symlink():
            raise Fts5StoreError("refusing symlinked database file")

        dfd = _open_validated_dir(db_path.parent, root_path, create=True)
        lock_fd: int | None = None
        try:
            lock_fd = _acquire_store_lock(dfd, db_name + ".lock", timeout=self._lock_timeout)
            for suffix in _DB_SIDECAR_SUFFIXES:
                _verify_existing_file(dfd, db_name + suffix)

            existing = _verify_existing_file(dfd, db_name)
            if existing is None:
                self._conn = self._create_and_init_disk_db(dfd, db_path, db_name, validate=False)
                return

            existing_conn, valid = self._open_and_validate_existing(dfd, db_path, db_name)
            if valid and existing_conn is not None:
                self._conn = existing_conn
                return

            if existing_conn is not None:
                with contextlib.suppress(sqlite3.Error):
                    existing_conn.close()

            _unlink_db_and_sidecars(dfd, db_name)
            self._conn = self._create_and_init_disk_db(dfd, db_path, db_name, validate=True)
        finally:
            if lock_fd is not None:
                _release_store_lock(lock_fd)
            os.close(dfd)

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise sqlite3.ProgrammingError("Cannot operate on a closed Fts5Store.")
        return self._conn

    def _sync_with_keys(
        self,
        incoming_keys: dict[str, str],
        row_for_id: Callable[[str], tuple[str, ...]],
        *,
        prune: bool = True,
    ) -> SyncStats:
        with self._lock:
            conn = self._require_conn()
            existing_rows = conn.execute(_SELECT_CHUNKS_SQL).fetchall()
            existing_by_id: dict[str, tuple[int, str]] = {}
            max_rowid = 0
            for raw_rowid, raw_cid, raw_key in existing_rows:
                rid = int(raw_rowid)
                if rid > max_rowid:
                    max_rowid = rid
                existing_by_id[str(raw_cid)] = (rid, str(raw_key))

            to_delete: list[int] = []
            for cid, (rid, old_key) in existing_by_id.items():
                new_key = incoming_keys.get(cid)
                if (new_key is None and prune) or (new_key is not None and new_key != old_key):
                    to_delete.append(rid)

            to_insert: list[tuple[str, ...]] = []
            for cid, new_key in incoming_keys.items():
                existing_entry = existing_by_id.get(cid)
                if existing_entry is None or existing_entry[1] != new_key:
                    to_insert.append(row_for_id(cid))

            if not to_delete and not to_insert:
                return SyncStats(inserted=0, deleted=0)

            conn.execute("BEGIN")
            try:
                if to_delete:
                    delete_params = [(rid,) for rid in to_delete]
                    conn.executemany(_DELETE_FTS_SQL, delete_params)
                    conn.executemany(_DELETE_CHUNKS_SQL, delete_params)

                if to_insert:
                    chunk_params: list[tuple[int, str, str]] = []
                    fts_params: list[tuple[object, ...]] = []
                    next_rowid = max_rowid + 1
                    for offset, row in enumerate(to_insert):
                        rid = next_rowid + offset
                        chunk_params.append((rid, row[0], row[1]))
                        fts_params.append((rid, *row[2:]))
                    conn.executemany(_INSERT_CHUNKS_SQL, chunk_params)
                    conn.executemany(self._insert_fts_sql, fts_params)

                conn.execute("COMMIT")
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
                raise

            return SyncStats(inserted=len(to_insert), deleted=len(to_delete))

    def prune_except(self, keep: set[str]) -> int:
        """Delete every row whose chunk_id is not in ``keep``; returns how many."""
        with self._lock:
            conn = self._require_conn()
            gone = [
                (int(rid),)
                for rid, cid, _key in conn.execute(_SELECT_CHUNKS_SQL).fetchall()
                if str(cid) not in keep
            ]
            if gone:
                conn.execute("BEGIN")
                try:
                    conn.executemany(_DELETE_FTS_SQL, gone)
                    conn.executemany(_DELETE_CHUNKS_SQL, gone)
                    conn.execute("COMMIT")
                except BaseException:
                    with contextlib.suppress(sqlite3.Error):
                        conn.execute("ROLLBACK")
                    raise
            return len(gone)

    def append(self, rows: Sequence[tuple[str, ...]]) -> SyncStats:
        """Append ``rows`` of ``(chunk_id, key, *col_values)`` in one transaction."""
        with self._lock:
            conn = self._require_conn()
            max_rowid = 0
            top = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM chunks").fetchone()
            if top and top[0] is not None:
                max_rowid = int(top[0])

            if not rows:
                return SyncStats(inserted=0, deleted=0)

            conn.execute("BEGIN")
            try:
                chunk_params: list[tuple[int, str, str]] = []
                fts_params: list[tuple[object, ...]] = []
                for offset, row in enumerate(rows):
                    rid = max_rowid + 1 + offset
                    chunk_params.append((rid, row[0], row[1]))
                    fts_params.append((rid, *row[2:]))
                conn.executemany(_INSERT_CHUNKS_SQL, chunk_params)
                conn.executemany(self._insert_fts_sql, fts_params)
                conn.execute("COMMIT")
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
                raise

            return SyncStats(inserted=len(rows), deleted=0)

    def sync(self, rows: Sequence[tuple[str, ...]]) -> SyncStats:
        """Incrementally sync ``rows`` of ``(chunk_id, key, *col_values)`` in one transaction."""
        expected_len = 2 + len(self._columns)
        incoming_by_id: dict[str, tuple[str, ...]] = {}
        incoming_keys: dict[str, str] = {}
        for row in rows:
            if len(row) != expected_len:
                raise ValueError(
                    "expected row of length " + str(expected_len) + ", got " + str(len(row))
                )
            chunk_id = row[0]
            if chunk_id in incoming_by_id:
                raise ValueError("duplicate chunk_id in sync rows: " + repr(chunk_id))
            incoming_by_id[chunk_id] = row
            incoming_keys[chunk_id] = row[1]

        return self._sync_with_keys(incoming_keys, incoming_by_id.__getitem__)

    def sync_lazy(
        self,
        keyed: Sequence[tuple[str, str]],
        make_row: Callable[[str], tuple[str, ...]],
        *,
        prune: bool = True,
    ) -> SyncStats:
        """Incrementally sync keyed rows, calling ``make_row(chunk_id)`` only for changed IDs.

        ``prune=False`` leaves rows that are not in ``keyed`` alone (a batch of a longer stream).
        """
        incoming_keys: dict[str, str] = {}
        for item in keyed:
            if len(item) != 2:
                raise ValueError("expected (chunk_id, key) pair of length 2, got " + str(len(item)))
            chunk_id, key = item
            if chunk_id in incoming_keys:
                raise ValueError("duplicate chunk_id in sync rows: " + repr(chunk_id))
            incoming_keys[chunk_id] = key

        expected_len = 2 + len(self._columns)

        def _validated_make_row(cid: str) -> tuple[str, ...]:
            row = make_row(cid)
            if len(row) != expected_len:
                raise ValueError(
                    "expected row of length " + str(expected_len) + ", got " + str(len(row))
                )
            if row[0] != cid:
                raise ValueError(
                    "make_row returned mismatched chunk_id: expected "
                    + repr(cid)
                    + ", got "
                    + repr(row[0])
                )
            if row[1] != incoming_keys[cid]:
                raise ValueError("make_row returned mismatched key for " + repr(cid))
            return row

        return self._sync_with_keys(incoming_keys, _validated_make_row, prune=prune)

    def query(
        self,
        match: str,
        limit: int | None = None,
        *,
        weights: tuple[float, ...] | None = None,
    ) -> list[tuple[str, float]]:
        """Run an FTS5 ``MATCH`` query and return ``[(chunk_id, -bm25)]`` best-first."""
        if weights is None:
            resolved_weights: tuple[float, ...] = (1.0,) * len(self._columns)
        else:
            if len(weights) != len(self._columns):
                raise ValueError("weights length must match column count")
            resolved_weights = tuple(float(w) for w in weights)

        if not match or not match.strip():
            return []
        if limit is not None and limit <= 0:
            return []

        with self._lock:
            conn = self._require_conn()
            if limit is None:
                params: tuple[object, ...] = (
                    *resolved_weights,
                    match,
                    *resolved_weights,
                )
                rows = conn.execute(self._query_sql, params).fetchall()
            else:
                params = (
                    *resolved_weights,
                    match,
                    *resolved_weights,
                    int(limit),
                )
                rows = conn.execute(self._query_limit_sql, params).fetchall()

        results: list[tuple[str, float]] = []
        for raw_cid, raw_score in rows:
            score = float(raw_score)
            if score == 0.0:
                score = 0.0
            results.append((str(raw_cid), score))
        return results

    def close(self) -> None:
        """Close the underlying SQLite connection idempotently."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> Fts5Store:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
