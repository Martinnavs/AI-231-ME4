"""Tests for ingest time/disk governors with --workers (ticket 237)."""

from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voice_agent.rag import ingest as ingest_mod
from voice_agent.rag.docindex import LoadedFile, _resolve_converters, _walk_folder, load_file
from voice_agent.rag.ingest import ingest, main
from voice_agent.rag.ingest_workers import supervisor_parse


def _make_five_files(src: Path) -> None:
    for i in range(5):
        (src / f"doc_{i}.txt").write_text(
            f"Document {i} body with searchable token item{i}.\n",
            encoding="utf-8",
        )


def _read_files_status(db: Path) -> dict[str, tuple[str, str | None, int]]:
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    rows = conn.execute("SELECT rel_key, status, error, n_chunks FROM files").fetchall()
    conn.close()
    return {r[0]: (r[1], r[2], int(r[3])) for r in rows}


def _read_chunk_sources(db: Path) -> set[str]:
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    rows = conn.execute("SELECT DISTINCT rel_key FROM chunk_data").fetchall()
    conn.close()
    return {r[0] for r in rows}


def _slow_in_flight_loader(
    path: Path,
    rel_key: str,
    *,
    converters: Any,
    canonical_cache: Any,
    prev: Any,
    now_iso: Any,
) -> LoadedFile:
    time.sleep(0.15)
    return load_file(
        path,
        rel_key,
        converters=converters,
        canonical_cache=canonical_cache,
        prev=prev,
        now_iso=now_iso,
    )


class TestGovernorWithWorkers:
    def test_time_governor_stops_dispatch_and_marks_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        _make_five_files(src)

        # Fake clock in ingest.py only (leaves ingest_workers.py real time untouched):
        # 1st call: start = 100.0 (deadline = 160.0 for deadline_min=1)
        # 2nd call: _governor_check after file 1 -> 110.0 (< 160.0, continue)
        # 3rd+ call: _governor_check after file 2 -> 160.0 (>= 160.0, stop)
        calls = [0]

        def fake_monotonic() -> float:
            calls[0] += 1
            if calls[0] == 1:
                return 100.0
            if calls[0] == 2:
                return 110.0
            return 160.0

        monkeypatch.setattr(ingest_mod, "time", SimpleNamespace(monotonic=fake_monotonic))

        db = tmp_path / "gov_workers.db"
        report = ingest(src, db, workers=2, deadline_min=1)

        assert report.files_total == 5
        assert report.ok == 2
        assert report.failed == 0
        assert report.skipped == 3

        rows = _read_files_status(db)
        assert len(rows) == 5
        ok_keys = {k for k, (st, _err, _n) in rows.items() if st == "ok"}
        skipped_rows = {k: (st, err, n) for k, (st, err, n) in rows.items() if st == "skipped"}
        assert len(ok_keys) == 2
        assert len(skipped_rows) == 3
        for st, err, n_chunks in skipped_rows.values():
            assert st == "skipped"
            assert err == "time governor"
            assert n_chunks == 0
        assert _read_chunk_sources(db) == ok_keys

    def test_disk_governor_stops_dispatch_and_marks_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        _make_five_files(src)

        calls = [0]

        class _FakeOs:
            def __getattr__(self, name: str) -> Any:
                return getattr(os, name)

            def statvfs(self, _path: str) -> SimpleNamespace:
                calls[0] += 1
                # Call 1 = preflight, Call 2 = after file 1 -> plenty of space
                # Call 3+ = after file 2 -> below min_free_mb (600)
                mb = 10_000 if calls[0] <= 2 else 100
                frsize = 4096
                bavail = (mb * 1024 * 1024) // frsize
                return SimpleNamespace(f_bavail=bavail, f_frsize=frsize)

        monkeypatch.setattr(ingest_mod, "os", _FakeOs())

        db = tmp_path / "disk_gov_workers.db"
        report = ingest(src, db, workers=2, min_free_mb=600)

        assert report.files_total == 5
        assert report.ok == 2
        assert report.failed == 0
        assert report.skipped == 3

        rows = _read_files_status(db)
        skipped_rows = {k: (st, err, n) for k, (st, err, n) in rows.items() if st == "skipped"}
        assert len(skipped_rows) == 3
        for st, err, n_chunks in skipped_rows.values():
            assert st == "skipped"
            assert err == "disk governor"
            assert n_chunks == 0

    def test_no_deadline_parses_all_five_unchanged(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        _make_five_files(src)

        db_w0 = tmp_path / "w0.db"
        db_w2 = tmp_path / "w2.db"
        rep0 = ingest(src, db_w0, workers=0, deadline_min=0)
        rep2 = ingest(src, db_w2, workers=2, deadline_min=0)

        assert rep2.files_total == 5
        assert rep2.ok == 5
        assert rep2.failed == 0
        assert rep2.skipped == 0
        assert rep2.chunks == rep0.chunks
        assert _read_files_status(db_w2) == _read_files_status(db_w0)
        assert _read_chunk_sources(db_w2) == _read_chunk_sources(db_w0)

    def test_supervisor_parse_window_check_drains_in_flight_only(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        _make_five_files(src)

        files, _ = _walk_folder(src)
        parse_list = [(f, f.relative_to(src).as_posix()) for f in files]

        checks: list[int] = []

        def stop_after_two(in_flight: int) -> bool:
            checks.append(in_flight)
            return len(checks) >= 2

        before_children = set(mp.active_children())
        results = supervisor_parse(
            parse_list,
            _resolve_converters(None),
            "",
            src,
            workers=2,
            loader=_slow_in_flight_loader,
            window_check=stop_after_two,
        )
        after_children = set(mp.active_children())

        assert len(checks) == 2
        assert len(results) == 2
        assert all(lf.status == "ok" for _rk, lf in results)
        assert after_children - before_children == set()

    def test_cli_fractional_deadline_min(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        _make_five_files(src)

        calls = [0]

        def fake_monotonic() -> float:
            calls[0] += 1
            if calls[0] == 1:
                return 0.0
            if calls[0] == 2:
                return 0.1
            return 1.0

        monkeypatch.setattr(ingest_mod, "time", SimpleNamespace(monotonic=fake_monotonic))

        db = tmp_path / "cli.db"
        rc = main([str(src), "--out", str(db), "--workers", "auto", "--deadline-min", "0.01"])
        assert rc == 0
        rows = _read_files_status(db)
        assert sum(1 for st, _e, _n in rows.values() if st == "ok") < 5
        assert sum(1 for st, _e, _n in rows.values() if st == "skipped") > 0
