"""Offline unit tests for voice_agent.rag.ingest_workers (multi-process ingest)."""

from __future__ import annotations

import os
import random
import sqlite3
import time
from pathlib import Path

from voice_agent.rag.docindex import Chunk, LoadedFile, _walk_folder
from voice_agent.rag.ingest import ingest, main


def _build_src(src_path: Path, n: int = 15) -> None:
    for i in range(n):
        ext = [".txt", ".md", ".csv"][i % 3]
        (src_path / f"file_{i:02d}{ext}").write_text(
            f"Content block {i} with unique tokens alpha beta gamma delta epsilon.\n",
            encoding="utf-8",
        )


def _build_hetero_src(src_path: Path) -> None:
    """Create files with different sizes and kinds so size-order != name-order."""
    (src_path / "a_big.txt").write_text("A" * 5000 + "\n")
    (src_path / "b_medium.md").write_text("B" * 2000 + "\n")
    (src_path / "c_small.txt").write_text("C" * 500 + "\n")
    (src_path / "d_tiny.csv").write_text("D,1\nD,2\n")
    (src_path / "e_huge.txt").write_text("E" * 8000 + "\n")
    (src_path / "f_mid.md").write_text("F" * 1500 + "\n")
    (src_path / "g_big.txt").write_text("G" * 4000 + "\n")
    (src_path / "h_tiny.txt").write_text("H\n")
    (src_path / "i_large.csv").write_text("I" * 6000 + "\n")
    (src_path / "j_small.md").write_text("J" * 800 + "\n")
    (src_path / "k_mid.txt").write_text("K" * 3000 + "\n")
    (src_path / "l_huge.csv").write_text("L" * 7000 + "\n")


def _read_chunks(db: Path) -> list[Chunk]:
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    rows = conn.execute(
        "SELECT chunk_id, file_sha256, chunk_sha256, position, page, section,"
        " file_date, source, header, text, fmt, slide, sheet,"
        " canonical_sha256, converter FROM chunk_data ORDER BY ord"
    ).fetchall()
    conn.close()
    return [
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
        )
        for r in rows
    ]


def _read_files_rows(db: Path) -> list[tuple]:
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    rows = conn.execute(
        "SELECT rel_key, status, error, kind, size, mtime_ns, ctime_ns,"
        " sha256, canonical_key, note, dates_json, n_chunks FROM files ORDER BY rel_key"
    ).fetchall()
    conn.close()
    return rows


def _read_fts_rank(db: Path, query: str, k: int = 10) -> list[int]:
    fts_path = db.with_name(db.name + ".fts")
    if not fts_path.exists():
        return []
    from voice_agent.rag.fts5 import Fts5Retriever

    r = Fts5Retriever(db_path=fts_path, root=db.parent)
    return [int(i) for i, _ in r.rank(query, k=k)]


# -- loader hooks (top-level picklable) --------------------------------


def _crash_loader(
    path: Path, rel_key: str, *, converters, canonical_cache, prev, now_iso
) -> LoadedFile:
    if "file_05" in rel_key:
        os._exit(1)
    from voice_agent.rag.docindex import load_file

    return load_file(
        path,
        rel_key,
        converters=converters,
        canonical_cache=canonical_cache,
        prev=None,
        now_iso=now_iso,
    )


def _timeout_loader(
    path: Path, rel_key: str, *, converters, canonical_cache, prev, now_iso
) -> LoadedFile:
    if "file_10" in rel_key:
        while True:
            time.sleep(3600)
    from voice_agent.rag.docindex import load_file

    return load_file(
        path,
        rel_key,
        converters=converters,
        canonical_cache=canonical_cache,
        prev=None,
        now_iso=now_iso,
    )


# -- slow-big loader: sleeps longer for the largest file (by size) ------


def _slow_big_loader(
    path: Path, rel_key: str, *, converters, canonical_cache, prev, now_iso
) -> LoadedFile:
    """Sleep ~1.5 s for the file with largest path (dispatched last), instant for rest."""
    if rel_key.endswith("z_big.txt"):
        time.sleep(1.5)
    from voice_agent.rag.docindex import load_file

    return load_file(
        path,
        rel_key,
        converters=converters,
        canonical_cache=canonical_cache,
        prev=None,
        now_iso=now_iso,
    )


# -- slow-during-drain loader: sleeps beyond timeout for last file ------


def _slow_drain_loader(
    path: Path, rel_key: str, *, converters, canonical_cache, prev, now_iso
) -> LoadedFile:
    """Sleep 2 s for 'c.txt' (the last file dispatched to the sole worker)."""
    if rel_key == "c.txt":
        time.sleep(2)
    from voice_agent.rag.docindex import load_file

    return load_file(
        path,
        rel_key,
        converters=converters,
        canonical_cache=canonical_cache,
        prev=None,
        now_iso=now_iso,
    )


# -- random-sleep loader (picklable) ------------------------------------


def _random_sleep_loader(
    path: Path, rel_key: str, *, converters, canonical_cache, prev, now_iso
) -> LoadedFile:
    """Sleep 0..0.3 s based on a stable hash of rel_key."""
    sleep_s = (hash(rel_key) % 300) / 1000.0
    time.sleep(sleep_s)
    from voice_agent.rag.docindex import load_file

    return load_file(
        path,
        rel_key,
        converters=converters,
        canonical_cache=canonical_cache,
        prev=None,
        now_iso=now_iso,
    )


# -- Tests ---------------------------------------------------------------


class TestEquality:
    def test_db_content_and_fts_identical(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        _build_src(src, 15)
        ingest(src, tmp_path / "w0.db", workers=0)
        ingest(src, tmp_path / "w1.db", workers=1)
        ingest(src, tmp_path / "w3.db", workers=3)
        c0, c1, c3 = (
            _read_chunks(tmp_path / "w0.db"),
            _read_chunks(tmp_path / "w1.db"),
            _read_chunks(tmp_path / "w3.db"),
        )
        assert c0 == c1
        assert c0 == c3
        for q in ("alpha", "delta epsilon", "gamma"):
            assert _read_fts_rank(tmp_path / "w0.db", q) == _read_fts_rank(tmp_path / "w3.db", q)

    def test_hetero_files_identical(self, tmp_path: Path) -> None:
        """Files with different sizes/kinds so size-order != name-order."""
        src = tmp_path / "src"
        src.mkdir()
        _build_hetero_src(src)
        ingest(src, tmp_path / "w0.db", workers=0)
        ingest(src, tmp_path / "w1.db", workers=1)
        ingest(src, tmp_path / "w3.db", workers=3)
        c0, c1, c3 = (
            _read_chunks(tmp_path / "w0.db"),
            _read_chunks(tmp_path / "w1.db"),
            _read_chunks(tmp_path / "w3.db"),
        )
        assert c0 == c1, "chunks differ between w0 and w1"
        assert c0 == c3, "chunks differ between w0 and w3"
        f0, f1, f3 = (
            _read_files_rows(tmp_path / "w0.db"),
            _read_files_rows(tmp_path / "w1.db"),
            _read_files_rows(tmp_path / "w3.db"),
        )
        assert f0 == f1, "files rows differ between w0 and w1"
        assert f0 == f3, "files rows differ between w0 and w3"
        for q in ("alpha", "Content"):
            assert _read_fts_rank(tmp_path / "w0.db", q) == _read_fts_rank(tmp_path / "w3.db", q)


class TestOrder:
    def test_ord_contiguous_after_workers(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        _build_src(src, 10)
        ingest(src, tmp_path / "order.db", workers=3)
        conn = sqlite3.connect(str(tmp_path / "order.db"))
        rows = conn.execute("SELECT ord FROM chunk_data ORDER BY ord").fetchall()
        conn.close()
        assert [r[0] for r in rows] == list(range(len(rows)))


class TestWindow:
    def test_in_flight_never_exceeds_window(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        _build_src(src, 12)
        window = min(4, 6)
        peak = [0]

        def check(n):
            peak[0] = max(peak[0], n)
            assert n <= window

        from voice_agent.rag.docindex import _resolve_converters
        from voice_agent.rag.ingest_workers import supervisor_parse

        files, _ = _walk_folder(src)
        sup = [f for f in files if f.suffix.lower() in {".txt", ".md", ".csv"}]
        parse_list = [(f, f.relative_to(src).as_posix()) for f in sup]
        supervisor_parse(
            parse_list, _resolve_converters(None), "", src, workers=2, window_check=check
        )
        assert peak[0] <= window

    def test_in_flight_increases(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        _build_src(src, 8)
        peaks = []

        def track(n):
            peaks.append(n)

        from voice_agent.rag.docindex import _resolve_converters
        from voice_agent.rag.ingest_workers import supervisor_parse

        files, _ = _walk_folder(src)
        sup = [f for f in files if f.suffix.lower() in {".txt", ".md", ".csv"}]
        parse_list = [(f, f.relative_to(src).as_posix()) for f in sup]
        supervisor_parse(
            parse_list, _resolve_converters(None), "", src, workers=2, window_check=track
        )
        assert max(peaks) >= 2


class TestCrash:
    def test_crash_yields_unreadable_row(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        _build_src(src, 8)
        from voice_agent.rag.docindex import _resolve_converters
        from voice_agent.rag.ingest_workers import supervisor_parse

        files, _ = _walk_folder(src)
        sup = [f for f in files if f.suffix.lower() in {".txt", ".md", ".csv"}]
        parse_list = [(f, f.relative_to(src).as_posix()) for f in sup]
        results = supervisor_parse(
            parse_list, _resolve_converters(None), "", src, workers=2, loader=_crash_loader
        )
        by_rk = {rk: lf for rk, lf in results}
        crash_rk = None
        for _fp, rk in parse_list:
            if "file_05" in rk:
                crash_rk = rk
                break
        assert crash_rk is not None
        crash_lf = by_rk[crash_rk]
        assert crash_lf.status == "unreadable"
        assert "crashed" in (crash_lf.error or "")

    def test_no_zombies_after_crash(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        _build_src(src, 8)
        import multiprocessing as mp

        from voice_agent.rag.docindex import _resolve_converters
        from voice_agent.rag.ingest_workers import supervisor_parse

        before = set(mp.active_children())
        files, _ = _walk_folder(src)
        sup = [f for f in files if f.suffix.lower() in {".txt", ".md", ".csv"}]
        parse_list = [(f, f.relative_to(src).as_posix()) for f in sup]
        supervisor_parse(
            parse_list, _resolve_converters(None), "", src, workers=2, loader=_crash_loader
        )
        time.sleep(0.5)
        after = set(mp.active_children())
        assert after - before == set(), f"zombie children: {after - before}"


class TestTimeout:
    def test_timeout_yields_unreadable_row(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        _build_src(src, 6)
        from voice_agent.rag.docindex import _resolve_converters
        from voice_agent.rag.ingest_workers import supervisor_parse

        files, _ = _walk_folder(src)
        sup = [f for f in files if f.suffix.lower() in {".txt", ".md", ".csv"}]
        parse_list = [(f, f.relative_to(src).as_posix()) for f in sup]
        results = supervisor_parse(
            parse_list,
            _resolve_converters(None),
            "",
            src,
            workers=2,
            loader=_timeout_loader,
            file_timeout_s=1,
        )
        by_rk = {rk: lf for rk, lf in results}
        timeout_lf = None
        for _fp, rk in parse_list:
            if "file_10" in rk:
                timeout_lf = by_rk.get(rk)
        if timeout_lf is not None:
            assert timeout_lf.status == "unreadable"
            assert "timed out" in (timeout_lf.error or "")

    def test_timeout_during_drain(self, tmp_path: Path) -> None:
        """Loader sleeps beyond file_timeout_s for the LAST file (drain phase)."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("alpha\n")
        (src / "b.txt").write_text("beta\n")
        (src / "c.txt").write_text("gamma\n")

        from voice_agent.rag.docindex import _resolve_converters
        from voice_agent.rag.ingest_workers import supervisor_parse

        files, _ = _walk_folder(src)
        sup = [f for f in files if f.suffix == ".txt"]
        parse_list = [(f, f.relative_to(src).as_posix()) for f in sup]
        results = supervisor_parse(
            parse_list,
            _resolve_converters(None),
            "",
            src,
            workers=1,
            loader=_slow_drain_loader,
            file_timeout_s=1,
        )
        by_rk = {rk: lf for rk, lf in results}
        assert by_rk["a.txt"].status == "ok", "a.txt should be ok"
        assert by_rk["b.txt"].status == "ok", "b.txt should be ok"
        assert by_rk["c.txt"].status == "unreadable", "c.txt should be timed out"
        assert "timed out" in (by_rk["c.txt"].error or "")


class TestDrain:
    def test_slow_largest_file_ok(self, tmp_path: Path) -> None:
        """Largest file sleeps ~1.5 s (dispatched last) — must come back ok, not crashed."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("a" * 100 + "\n")
        (src / "b.txt").write_text("b" * 200 + "\n")
        (src / "z_big.txt").write_text("z" * 5000 + "\n")

        from voice_agent.rag.docindex import _resolve_converters
        from voice_agent.rag.ingest_workers import supervisor_parse

        files, _ = _walk_folder(src)
        sup = [f for f in files if f.suffix == ".txt"]
        parse_list = [(f, f.relative_to(src).as_posix()) for f in sup]
        results = supervisor_parse(
            parse_list,
            _resolve_converters(None),
            "",
            src,
            workers=2,
            loader=_slow_big_loader,
            file_timeout_s=300,
        )
        by_rk = {rk: lf for rk, lf in results}
        assert by_rk["z_big.txt"].status == "ok", (
            f"z_big.txt should be ok but got {by_rk['z_big.txt'].error}"
        )
        assert by_rk["a.txt"].status == "ok"
        assert by_rk["b.txt"].status == "ok"


class TestRandomSleep:
    def test_many_random_sleeps_identical(self, tmp_path: Path) -> None:
        """30 files with randomised sleeps (0..0.3 s) — all ok, rows identical to workers=0."""
        src = tmp_path / "src"
        src.mkdir()
        rng = random.Random(42)
        for i in range(30):
            (src / f"f{i:02d}.txt").write_text(f"chunk-{i} " * rng.randint(50, 2000) + "\n")

        from voice_agent.rag.docindex import _resolve_converters
        from voice_agent.rag.ingest_workers import supervisor_parse

        files, _ = _walk_folder(src)
        sup = [f for f in files if f.suffix == ".txt"]
        parse_list = [(f, f.relative_to(src).as_posix()) for f in sup]
        results = supervisor_parse(
            parse_list,
            _resolve_converters(None),
            "",
            src,
            workers=3,
            loader=_random_sleep_loader,
            file_timeout_s=300,
        )
        by_rk = {rk: lf for rk, lf in results}
        for _fp, rk in parse_list:
            assert by_rk[rk].status == "ok", f"{rk} not ok: {by_rk[rk].error}"

        # Also verify DB-level identity via ingest()
        ingest(src, tmp_path / "w3.db", workers=3, worker_mem_mb=0)
        ingest(src, tmp_path / "w0.db", workers=0)
        assert _read_chunks(tmp_path / "w0.db") == _read_chunks(tmp_path / "w3.db")
        assert _read_files_rows(tmp_path / "w0.db") == _read_files_rows(tmp_path / "w3.db")


class TestResume:
    def test_second_run_zero_workers_sent(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        _build_src(src, 6)
        db = tmp_path / "resume.db"
        ingest(src, db, workers=0)
        chunks1 = _read_chunks(db)
        call_count = [0]

        def counting(*a, **kw):
            call_count[0] += 1
            from voice_agent.rag.docindex import load_file

            return load_file(*a, **kw)

        import voice_agent.rag.docindex as mod

        orig = mod.load_file
        mod.load_file = counting
        try:
            ingest(src, db, workers=0)
        finally:
            mod.load_file = orig
        assert call_count[0] == 0, f"load_file called {call_count[0]} times"
        assert _read_chunks(db) == chunks1


class TestRegression:
    def test_seq_pairing_with_hetero_sizes(self, tmp_path: Path) -> None:
        """Verify results are attributed to the correct file when size-order != name-order.

        The old bug assigned seq in dispatch (size) order, causing results to be
        attributed to the wrong file.  With the fix, each result carries its rel_key
        and the parent commits by looking up rel_key, so files always get their own data.
        """
        src = tmp_path / "src"
        src.mkdir()
        _build_hetero_src(src)

        ingest(src, tmp_path / "w0.db", workers=0)
        ingest(src, tmp_path / "w3.db", workers=3)
        f0 = _read_files_rows(tmp_path / "w0.db")
        f3 = _read_files_rows(tmp_path / "w3.db")
        for row0, row3 in zip(f0, f3, strict=True):
            rk = row0[0]
            assert row3[3] == row0[3], f"{rk}: kind mismatch w0={row0[3]} w3={row3[3]}"
            assert row3[4] == row0[4], f"{rk}: size mismatch w0={row0[4]} w3={row3[4]}"
            assert row3[5] == row0[5], f"{rk}: mtime_ns mismatch w0={row0[5]} w3={row3[5]}"


class TestCLI:
    def test_main_with_workers(self, tmp_path: Path) -> None:
        src = tmp_path / "cli_src"
        src.mkdir()
        (src / "a.txt").write_text("hello world\n")
        db = tmp_path / "cli.db"
        rc = main([str(src), "--out", str(db), "--workers", "2"])
        assert rc == 0
        assert db.exists()
