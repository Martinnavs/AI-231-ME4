"""Offline unit tests for voice_agent.rag.fts5_store."""

from __future__ import annotations

import ast
import fcntl
import os
import re
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from voice_agent.rag import canonical_cache
from voice_agent.rag import fts5_store as fts5_store_mod
from voice_agent.rag.fts5_store import (
    DEFAULT_TOKENIZER,
    SCHEMA_VERSION,
    Fts5Store,
    Fts5StoreError,
    SyncStats,
    fts5_available,
    tokenize_fingerprint,
)

PORTER_TOKENIZER = "porter unicode61 remove_diacritics 2"


def _make_rows(count: int, prefix: str = "k") -> list[tuple[str, str, str, str]]:
    return [
        (f"chunk-{i}", f"{prefix}-{i}", f"alpha token{i} body", f"header{i}") for i in range(count)
    ]


def test_canonical_cache_exposes_validate_cache_dir_alias() -> None:
    assert canonical_cache.validate_cache_dir is canonical_cache._validate_directory


def test_probe_success_and_failure_reasons() -> None:
    ok_pretok, reason_pretok = fts5_available(DEFAULT_TOKENIZER)
    assert ok_pretok is True
    assert reason_pretok == ""

    ok_porter, reason_porter = fts5_available(PORTER_TOKENIZER)
    assert ok_porter is True
    assert reason_porter == ""

    ok_bad, reason_bad = fts5_available("no_such_fts5_tokenizer_xyz")
    assert ok_bad is False
    assert reason_bad != ""

    ok_empty, reason_empty = fts5_available("   ")
    assert ok_empty is False
    assert reason_empty != ""

    ok_semi, reason_semi = fts5_available("unicode61; DROP TABLE x")
    assert ok_semi is False
    assert reason_semi != ""


def test_in_memory_mode_sync_query_weights_and_close(tmp_path: Path) -> None:
    with Fts5Store(None, tmp_path, tokenizer=DEFAULT_TOKENIZER) as store:
        assert store.is_memory is True
        assert store.path is None
        assert list(tmp_path.iterdir()) == []

        stats = store.sync(
            [
                ("c1", "k1", "alpha alpha beta", "other"),
                ("c2", "k2", "delta epsilon", "alpha"),
                ("c3", "k3", "zeta eta", "theta"),
            ]
        )
        assert stats == SyncStats(inserted=3, deleted=0)

        hits = store.query('"alpha"', weights=(10.0, 1.0))
        assert [cid for cid, _ in hits] == ["c1", "c2"]
        assert all(score > 0.0 for _, score in hits)
        assert hits[0][1] >= hits[1][1]

        # Weighting header column higher flips ranking so header match comes first.
        weighted = store.query('"alpha"', weights=(1.0, 10.0))
        assert [cid for cid, _ in weighted] == ["c2", "c1"]

        limited = store.query('"alpha"', limit=1, weights=(1.0, 10.0))
        assert len(limited) == 1
        assert limited[0][0] == "c2"

        assert store.query("") == []
        assert store.query("   ") == []
        assert store.query('"alpha"', limit=0) == []

    # close() is idempotent; operations after close raise ProgrammingError.
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        store.query('"alpha"')


def test_pretok_apostrophe_and_porter_stemming(tmp_path: Path) -> None:
    with Fts5Store(
        None,
        tmp_path,
        tokenizer=DEFAULT_TOKENIZER,
        columns=("all_tokens", "header_tokens"),
    ) as pretok:
        pretok.sync([("c1", "k1", "don't stop", "title")])
        assert [cid for cid, _ in pretok.query('"don\'t"')] == ["c1"]

    with Fts5Store(
        None,
        tmp_path,
        tokenizer=PORTER_TOKENIZER,
        columns=("header", "text"),
    ) as porter:
        porter.sync([("c1", "k1", "header", "running quickly")])
        assert [cid for cid, _ in porter.query('"run"')] == ["c1"]


def test_hardening_defensive_and_pragmas_on_python312(tmp_path: Path) -> None:
    assert hasattr(sqlite3, "SQLITE_DBCONFIG_DEFENSIVE")
    assert hasattr(sqlite3.Connection, "setconfig")

    db_path = tmp_path / ".rag_cache" / "fts5" / "index.sqlite3"
    with Fts5Store(db_path, tmp_path, tokenizer=DEFAULT_TOKENIZER) as store:
        conn = store._conn
        assert conn is not None
        assert conn.getconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE) is True
        assert conn.execute("PRAGMA trusted_schema").fetchone() == (0,)
        assert conn.execute("PRAGMA cell_size_check").fetchone() == (1,)

        # Defensive mode blocks direct writes to FTS5 shadow tables.
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM fts_data")


def test_symlinked_db_and_directory_refused(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    # 1. Symlinked DB file (pointing inside or outside root).
    target_file = outside / "target.sqlite3"
    target_file.write_bytes(b"precious")
    cache_dir = root / ".rag_cache" / "fts5"
    cache_dir.mkdir(parents=True)
    sym_db = cache_dir / "index.sqlite3"
    sym_db.symlink_to(target_file)
    with pytest.raises(Fts5StoreError):
        Fts5Store(sym_db, root, tokenizer=DEFAULT_TOKENIZER)
    assert target_file.read_bytes() == b"precious"
    sym_db.unlink()

    # 2. Symlinked -journal sidecar.
    sym_journal = cache_dir / "index.sqlite3-journal"
    sym_journal.symlink_to(target_file)
    with pytest.raises(Fts5StoreError):
        Fts5Store(sym_db, root, tokenizer=DEFAULT_TOKENIZER)
    assert target_file.read_bytes() == b"precious"
    sym_journal.unlink()

    # 3. Direct symlinked directory.
    sym_dir = root / "sym_cache"
    sym_dir.symlink_to(outside)
    with pytest.raises(Fts5StoreError):
        Fts5Store(sym_dir / "index.sqlite3", root, tokenizer=DEFAULT_TOKENIZER)

    # 4. Intermediate symlinked directory inside root pointing inside root.
    real_inner = root / "real_inner"
    real_inner.mkdir()
    inter_sym = root / "inter_sym"
    inter_sym.symlink_to(real_inner)
    with pytest.raises(Fts5StoreError):
        Fts5Store(inter_sym / "fts5" / "index.sqlite3", root, tokenizer=DEFAULT_TOKENIZER)

    # 5. Path escaping root via ..
    with pytest.raises(Fts5StoreError):
        Fts5Store(root / ".." / "outside" / "index.sqlite3", root, tokenizer=DEFAULT_TOKENIZER)

    # 6. Symlinked root directory itself.
    sym_root = tmp_path / "sym_root"
    sym_root.symlink_to(root)
    with pytest.raises(Fts5StoreError):
        Fts5Store(sym_root / "index.sqlite3", sym_root, tokenizer=DEFAULT_TOKENIZER)

    # 7. Non-regular file at DB path (directory).
    dir_as_db = cache_dir / "dir_db.sqlite3"
    dir_as_db.mkdir()
    with pytest.raises(Fts5StoreError):
        Fts5Store(dir_as_db, root, tokenizer=DEFAULT_TOKENIZER)


def test_wrong_uid_db_file_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    db_path = root / ".rag_cache" / "fts5" / "index.sqlite3"
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"")

    real_lstat = os.lstat

    class _FakeStat:
        def __init__(self, wrapped: os.stat_result) -> None:
            self._wrapped = wrapped
            self.st_uid = wrapped.st_uid + 1

        def __getattr__(self, name: str) -> object:
            return getattr(self._wrapped, name)

    def _fake_lstat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> object:
        res = real_lstat(path, dir_fd=dir_fd) if dir_fd is not None else real_lstat(path)
        if os.fspath(path) in (str(db_path), db_path.name):
            return _FakeStat(res)
        return res

    monkeypatch.setattr(os, "lstat", _fake_lstat)
    with pytest.raises(Fts5StoreError):
        Fts5Store(db_path, root, tokenizer=DEFAULT_TOKENIZER)


def test_permissions_db_0600_dirs_0700_and_journal_0600(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir(mode=0o755)
    rag_cache = root / ".rag_cache"
    fts_dir = rag_cache / "fts5"
    db_path = fts_dir / "index.sqlite3"

    with Fts5Store(db_path, root, tokenizer=DEFAULT_TOKENIZER) as store:
        store.sync(_make_rows(5))

    assert stat.S_IMODE(root.stat().st_mode) == 0o755
    assert stat.S_IMODE(rag_cache.stat().st_mode) == 0o700
    assert stat.S_IMODE(fts_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600

    # Verify SQLite rollback journal (-journal) inherits 0600 from the DB file.
    journal_path = db_path.with_name(db_path.name + "-journal")
    raw_conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        raw_conn.execute("PRAGMA journal_mode = DELETE")
        raw_conn.execute("PRAGMA cache_size = 1")
        raw_conn.execute("BEGIN IMMEDIATE")
        for i in range(100):
            raw_conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                (f"probe-{i}", "x" * 200),
            )
        assert journal_path.exists()
        assert stat.S_IMODE(journal_path.stat().st_mode) == 0o600
        raw_conn.execute("ROLLBACK")
    finally:
        raw_conn.close()


@pytest.mark.parametrize(
    "kind",
    ["foreign", "corrupt", "empty", "truncated", "old_schema", "extra_trigger", "extra_view"],
)
def test_invalid_or_tampered_db_is_deleted_and_rebuilt(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    db_path = root / ".rag_cache" / "fts5" / "index.sqlite3"
    db_path.parent.mkdir(parents=True)

    if kind == "foreign":
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE other_app (id INTEGER PRIMARY KEY, secret TEXT)")
        conn.execute("INSERT INTO other_app (secret) VALUES ('stale')")
        conn.commit()
        conn.close()
    elif kind == "corrupt":
        db_path.write_bytes(b"not a valid sqlite database at all" * 16)
    elif kind == "empty":
        db_path.write_bytes(b"")
    else:
        with Fts5Store(db_path, root, tokenizer=DEFAULT_TOKENIZER) as seed:
            seed.sync(_make_rows(4))

        if kind == "truncated":
            raw = db_path.read_bytes()
            db_path.write_bytes(raw[: len(raw) // 2])
        elif kind == "old_schema":
            conn = sqlite3.connect(db_path)
            conn.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
            conn.commit()
            conn.close()
        elif kind == "extra_trigger":
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TRIGGER evil AFTER INSERT ON chunks BEGIN DELETE FROM meta; END")
            conn.commit()
            conn.close()
        elif kind == "extra_view":
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE VIEW sneaky AS SELECT chunk_id FROM chunks")
            conn.commit()
            conn.close()

    # Plant regular leftover sidecars alongside the invalid DB to confirm cleanup.
    journal = db_path.with_name(db_path.name + "-journal")
    wal = db_path.with_name(db_path.name + "-wal")
    shm = db_path.with_name(db_path.name + "-shm")
    journal.write_bytes(b"stale-journal")
    wal.write_bytes(b"stale-wal")
    shm.write_bytes(b"stale-shm")

    with Fts5Store(db_path, root, tokenizer=DEFAULT_TOKENIZER) as rebuilt:
        assert not journal.exists()
        assert not wal.exists()
        assert not shm.exists()
        assert stat.S_IMODE(db_path.stat().st_mode) == 0o600

        stats = rebuilt.sync(_make_rows(2, prefix="fresh"))
        assert stats == SyncStats(inserted=2, deleted=0)
        hits = rebuilt.query('"token0"')
        assert [cid for cid, _ in hits] == ["chunk-0"]

        check_conn = rebuilt._conn
        assert check_conn is not None
        meta = dict(check_conn.execute("SELECT key, value FROM meta").fetchall())
        assert meta["schema_version"] == SCHEMA_VERSION
        assert meta["tokenizer"] == DEFAULT_TOKENIZER
        assert meta["tokenize_fingerprint"] == tokenize_fingerprint()
        extra = check_conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('trigger', 'view')"
        ).fetchall()
        assert extra == []


def test_sync_incremental_counts_and_zero_write_resync(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    db_path = root / ".rag_cache" / "fts5" / "index.sqlite3"

    rows = _make_rows(10, prefix="v1")
    with Fts5Store(db_path, root, tokenizer=DEFAULT_TOKENIZER) as store:
        assert store.sync(rows) == SyncStats(inserted=10, deleted=0)
        assert store._conn is not None

        # Unchanged resync performs 0 writes (only the initial SELECT).
        trace_log: list[str] = []
        store._conn.set_trace_callback(trace_log.append)
        changes_before = store._conn.total_changes
        assert store.sync(rows) == SyncStats(inserted=0, deleted=0)
        assert store._conn.total_changes == changes_before
        assert all(stmt.startswith("SELECT ") for stmt in trace_log)

        # Editing 1 of 10 rows deletes and inserts only that 1 row.
        trace_log.clear()
        edited = list(rows)
        edited[3] = ("chunk-3", "v2-3", "alpha updatedterm body", "header3")
        changes_before_edit = store._conn.total_changes
        assert store.sync(edited) == SyncStats(inserted=1, deleted=1)
        # Only 1 delete + 1 insert on chunks/fts (far fewer changes than rewriting 10 rows).
        delta_changes = store._conn.total_changes - changes_before_edit
        assert 4 <= delta_changes <= 20
        assert sum(1 for s in trace_log if s.startswith("DELETE FROM chunks")) == 1
        assert sum(1 for s in trace_log if s.startswith("INSERT INTO chunks")) == 1
        assert sum(1 for s in trace_log if s.startswith("DELETE FROM fts")) == 1
        assert sum(1 for s in trace_log if s.startswith("INSERT INTO fts")) == 1
        store._conn.set_trace_callback(None)

        assert [cid for cid, _ in store.query('"updatedterm"')] == ["chunk-3"]
        assert store.query('"token3"') == []
        assert [cid for cid, _ in store.query('"token0"')] == ["chunk-0"]

    # Reopen from disk and resync unchanged rows: 0 writes, data preserved.
    with Fts5Store(db_path, root, tokenizer=DEFAULT_TOKENIZER) as reopened:
        assert reopened._conn is not None
        changes_reopen = reopened._conn.total_changes
        assert reopened.sync(edited) == SyncStats(inserted=0, deleted=0)
        assert reopened._conn.total_changes == changes_reopen
        assert [cid for cid, _ in reopened.query('"updatedterm"')] == ["chunk-3"]

        # Remove 2 rows and add 1 row in one sync.
        shrunk = edited[:-2] + [("chunk-new", "k-new", "alpha brandnew", "hnew")]
        assert reopened.sync(shrunk) == SyncStats(inserted=1, deleted=2)
        assert reopened.query('"token8"') == []
        assert reopened.query('"token9"') == []
        assert [cid for cid, _ in reopened.query('"brandnew"')] == ["chunk-new"]


def test_sql_uses_bound_parameters_only_no_fstring_or_percent_near_execute() -> None:
    store_path = Path("src/voice_agent/rag/fts5_store.py")
    source = store_path.read_text(encoding="utf-8")
    lines = source.splitlines()

    for idx, line in enumerate(lines):
        if "execute" in line:
            window = "\n".join(lines[max(0, idx - 3) : min(len(lines), idx + 4)])
            assert not re.search(r"""\bf["']""", window), f"f-string near execute at line {idx + 1}"
            assert "%" not in window, f"% operator near execute at line {idx + 1}"

    tree = ast.parse(source, filename=str(store_path))
    for node in ast.walk(tree):
        assert not isinstance(node, ast.JoinedStr), "f-string found in fts5_store.py"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "executescript"


def test_concurrent_queries_and_injection_payloads(tmp_path: Path) -> None:
    with Fts5Store(None, tmp_path, tokenizer=DEFAULT_TOKENIZER) as store:
        payload_rows = [
            ("c-inject", "k1", "alpha '; DROP TABLE chunks; --", "hdr"),
            ("c-plain", "k2", "beta normal content", "hdr"),
        ]
        assert store.sync(payload_rows) == SyncStats(inserted=2, deleted=0)

        def _run_query(term: str) -> list[tuple[str, float]]:
            return store.query(term)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_run_query, ['"alpha"', '"beta"', '"alpha"', '"beta"']))

        assert [cid for cid, _ in results[0]] == ["c-inject"]
        assert [cid for cid, _ in results[1]] == ["c-plain"]
        assert [cid for cid, _ in results[2]] == ["c-inject"]
        assert [cid for cid, _ in results[3]] == ["c-plain"]


def test_sync_lazy_calls_make_row_only_for_new_and_changed_rows(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    db_path = root / ".rag_cache" / "fts5" / "index.sqlite3"

    rows_by_id: dict[str, tuple[str, str, str, str]] = {
        row[0]: row for row in _make_rows(5, prefix="v1")
    }
    calls: list[str] = []

    def _make_row(cid: str) -> tuple[str, ...]:
        calls.append(cid)
        return rows_by_id[cid]

    with Fts5Store(db_path, root, tokenizer=DEFAULT_TOKENIZER) as store:
        keyed = [(cid, row[1]) for cid, row in rows_by_id.items()]
        assert store.sync_lazy(keyed, _make_row) == SyncStats(inserted=5, deleted=0)
        assert sorted(calls) == sorted(rows_by_id.keys())
        assert store._conn is not None

        # Unchanged resync: make_row called 0 times and 0 DB writes.
        calls.clear()
        changes_before = store._conn.total_changes
        assert store.sync_lazy(keyed, _make_row) == SyncStats(inserted=0, deleted=0)
        assert calls == []
        assert store._conn.total_changes == changes_before

        # Edit 1 of 5 rows: make_row called once for the changed chunk only.
        calls.clear()
        rows_by_id["chunk-2"] = ("chunk-2", "v2-2", "alpha lazyedited body", "header2")
        keyed_edited = [(cid, row[1]) for cid, row in rows_by_id.items()]
        assert store.sync_lazy(keyed_edited, _make_row) == SyncStats(inserted=1, deleted=1)
        assert calls == ["chunk-2"]
        assert [cid for cid, _ in store.query('"lazyedited"')] == ["chunk-2"]
        assert store.query('"token2"') == []

        # Delete 1 (chunk-4), edit 1 (chunk-0), and add 1 (chunk-5): 2 make_row calls.
        calls.clear()
        del rows_by_id["chunk-4"]
        rows_by_id["chunk-0"] = ("chunk-0", "v2-0", "alpha zeroedited body", "header0")
        rows_by_id["chunk-5"] = ("chunk-5", "v1-5", "alpha brandnewlazy body", "header5")
        keyed_mixed = [(cid, row[1]) for cid, row in rows_by_id.items()]
        assert store.sync_lazy(keyed_mixed, _make_row) == SyncStats(inserted=2, deleted=2)
        assert sorted(calls) == ["chunk-0", "chunk-5"]
        assert store.query('"token4"') == []
        assert [cid for cid, _ in store.query('"zeroedited"')] == ["chunk-0"]
        assert [cid for cid, _ in store.query('"brandnewlazy"')] == ["chunk-5"]

    # Reopen from disk and sync_lazy unchanged: 0 make_row calls, 0 writes.
    calls.clear()
    with Fts5Store(db_path, root, tokenizer=DEFAULT_TOKENIZER) as reopened:
        assert reopened._conn is not None
        changes_reopen = reopened._conn.total_changes
        assert reopened.sync_lazy(keyed_mixed, _make_row) == SyncStats(inserted=0, deleted=0)
        assert calls == []
        assert reopened._conn.total_changes == changes_reopen
        assert [cid for cid, _ in reopened.query('"brandnewlazy"')] == ["chunk-5"]


def test_sync_lazy_validation_rejects_duplicates_and_bad_rows(tmp_path: Path) -> None:
    with Fts5Store(None, tmp_path, tokenizer=DEFAULT_TOKENIZER) as store:
        stats = store.sync_lazy([("c1", "k1")], lambda cid: (cid, "k1", "alpha", "hdr"))
        assert stats == SyncStats(inserted=1, deleted=0)

        # Duplicate chunk_id is rejected before make_row is invoked.
        calls: list[str] = []
        with pytest.raises(ValueError, match="duplicate chunk_id"):
            store.sync_lazy(
                [("c1", "k1"), ("c1", "k2")],
                lambda cid: calls.append(cid) or (cid, "k2", "beta", "hdr"),
            )
        assert calls == []

        # Mismatched row length, chunk_id, or key raises ValueError without mutating store.
        with pytest.raises(ValueError, match="expected row of length"):
            store.sync_lazy([("c2", "k2")], lambda cid: (cid, "k2", "only_one_col"))
        with pytest.raises(ValueError, match="mismatched chunk_id"):
            store.sync_lazy([("c2", "k2")], lambda _cid: ("other", "k2", "beta", "hdr"))
        with pytest.raises(ValueError, match="mismatched key"):
            store.sync_lazy([("c2", "k2")], lambda cid: (cid, "wrong-key", "beta", "hdr"))

        # Existing c1 remains intact and c2 was never inserted.
        assert [cid for cid, _ in store.query('"alpha"')] == ["c1"]
        assert store.query('"beta"') == []


def test_lock_contention_begin_exclusive_does_not_unlink_and_preserves_inode(
    tmp_path: Path,
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    db_path = root / ".rag_cache" / "fts5" / "index.sqlite3"
    lock_path = db_path.with_name(db_path.name + ".lock")

    with Fts5Store(db_path, root, tokenizer=DEFAULT_TOKENIZER) as seed:
        seed.sync(_make_rows(2, prefix="keep"))

    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
    before_stat = db_path.stat()

    holder = sqlite3.connect(db_path, isolation_level=None)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        with pytest.raises(sqlite3.OperationalError):
            Fts5Store(
                db_path,
                root,
                tokenizer=DEFAULT_TOKENIZER,
                _busy_timeout=0.05,
                _retry_delay=0.02,
            )
    finally:
        holder.execute("ROLLBACK")
        holder.close()

    after_stat = db_path.stat()
    assert after_stat.st_ino == before_stat.st_ino
    assert after_stat.st_dev == before_stat.st_dev

    with Fts5Store(db_path, root, tokenizer=DEFAULT_TOKENIZER) as reopened:
        assert [cid for cid, _ in reopened.query('"token0"')] == ["chunk-0"]


def test_lock_contention_clears_before_retry_succeeds(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    db_path = root / ".rag_cache" / "fts5" / "index.sqlite3"

    with Fts5Store(db_path, root, tokenizer=DEFAULT_TOKENIZER) as seed:
        seed.sync(_make_rows(2, prefix="retry"))

    before_ino = db_path.stat().st_ino
    holder = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    holder.execute("BEGIN EXCLUSIVE")

    def _release_soon() -> None:
        time.sleep(0.08)
        holder.execute("ROLLBACK")
        holder.close()

    releaser = threading.Thread(target=_release_soon)
    releaser.start()
    try:
        with Fts5Store(
            db_path,
            root,
            tokenizer=DEFAULT_TOKENIZER,
            _busy_timeout=0.02,
            _retry_delay=0.15,
        ) as store:
            assert db_path.stat().st_ino == before_ino
            assert [cid for cid, _ in store.query('"token1"')] == ["chunk-1"]
    finally:
        releaser.join(timeout=5.0)


def test_flock_refuses_symlink_times_out_and_serializes_corrupt_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    cache_dir = root / ".rag_cache" / "fts5"
    cache_dir.mkdir(parents=True)
    db_path = cache_dir / "index.sqlite3"
    lock_path = cache_dir / "index.sqlite3.lock"

    # 1. Symlinked lock file is refused.
    outside = tmp_path / "outside.txt"
    outside.write_text("do-not-touch", encoding="utf-8")
    lock_path.symlink_to(outside)
    with pytest.raises(Fts5StoreError):
        Fts5Store(db_path, root, tokenizer=DEFAULT_TOKENIZER)
    assert outside.read_text(encoding="utf-8") == "do-not-touch"
    lock_path.unlink()

    # 2. Held flock across processes times out cleanly without deleting existing DB.
    with Fts5Store(db_path, root, tokenizer=DEFAULT_TOKENIZER) as seed:
        seed.sync(_make_rows(1, prefix="v1"))
    before_ino = db_path.stat().st_ino

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, sys, time; "
                "fd = open(sys.argv[1], 'r+'); "
                "fcntl.flock(fd, fcntl.LOCK_EX); "
                "print('locked', flush=True); "
                "time.sleep(0.5); "
                "fcntl.flock(fd, fcntl.LOCK_UN); "
                "fd.close()"
            ),
            str(lock_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "locked"
        with pytest.raises(Fts5StoreError, match="lock"):
            Fts5Store(db_path, root, tokenizer=DEFAULT_TOKENIZER, _lock_timeout=0.05)
        assert db_path.stat().st_ino == before_ino
    finally:
        proc.wait(timeout=5)
        assert proc.returncode == 0

    # 3. Corrupt DB with concurrent opens: flock serializes rebuild so it rebuilds once.
    db_path.write_bytes(b"corrupted header" * 16)
    init_calls = 0
    init_lock = threading.Lock()
    orig_init_schema = fts5_store_mod._init_schema

    def _counting_init(
        conn: sqlite3.Connection,
        columns: tuple[str, ...],
        tokenizer: str,
    ) -> None:
        nonlocal init_calls
        with init_lock:
            init_calls += 1
        time.sleep(0.05)
        orig_init_schema(conn, columns, tokenizer)

    monkeypatch.setattr(fts5_store_mod, "_init_schema", _counting_init)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(Fts5Store, db_path, root, tokenizer=DEFAULT_TOKENIZER, _lock_timeout=2.0)
        f2 = pool.submit(Fts5Store, db_path, root, tokenizer=DEFAULT_TOKENIZER, _lock_timeout=2.0)
        s1 = f1.result(timeout=5.0)
        s2 = f2.result(timeout=5.0)
    try:
        assert init_calls == 1
        assert s1._conn is not None
        assert s2._conn is not None
    finally:
        s1.close()
        s2.close()

    # Ensure fcntl module check was exercised cleanly.
    with open(lock_path, "r+b") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
