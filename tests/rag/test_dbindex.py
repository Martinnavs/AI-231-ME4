"""Offline unit tests for voice_agent.rag.dbindex (ticket 107)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from voice_agent.rag.dbindex import DbIndex, DbIndexError
from voice_agent.rag.docindex import Bm25Retriever, Chunk, DocIndex, select_results
from voice_agent.rag.ingest import ingest


@pytest.fixture(scope="module")
def _ingested(tmp_path_factory: pytest.TempPathFactory) -> dict:
    src = tmp_path_factory.mktemp("src")
    db = tmp_path_factory.mktemp("db") / "test.db"

    (src / "guide.md").write_text(
        "# Setup\n\nInstall the package on 2026-03-15 following standard steps.\n\n"
        "## Usage\n\nRun the agent with the --verbose flag for detailed output.\n\n"
        "## Advanced\n\nFor production deployments configure the retry policy.\n\n"
        "## Troubleshooting\n\nCommon issues with the deployment pipeline.\n",
        encoding="utf-8",
    )
    (src / "notes.txt").write_text(
        "Meeting notes from November 18, 2026. Action items assigned.\n",
        encoding="utf-8",
    )
    (src / "data.csv").write_text(
        "item,price,updated\nwidget_a,100,2026-01-01\nwidget_b,200,2026-02-15\n",
        encoding="utf-8",
    )
    (src / "extra.txt").write_text(
        "Budget review for the quarterly planning cycle with action items.\n",
        encoding="utf-8",
    )
    (src / "corrupt.pdf").write_bytes(b"\x00\x01\x02\x80\xff\xfe\x04\x05")

    ingest(src, db)
    return {"src": src, "db": db}


@pytest.fixture
def src(_ingested: dict) -> Path:
    return _ingested["src"]


@pytest.fixture
def db(_ingested: dict) -> Path:
    return _ingested["db"]


class TestSelectResults:
    def test_filled(self) -> None:
        """Normal fill without overflow."""
        ranked = [
            (
                Chunk(
                    id="a#0",
                    file_sha256="x",
                    chunk_sha256="y",
                    position=0,
                    page=None,
                    section="",
                    file_date="2026-01-01",
                    source="a.txt",
                    header="a.txt",
                    text="quick brown fox jumps",
                ),
                10.0,
            ),
            (
                Chunk(
                    id="b#0",
                    file_sha256="x",
                    chunk_sha256="y",
                    position=0,
                    page=None,
                    section="",
                    file_date="2026-01-01",
                    source="b.txt",
                    header="b.txt",
                    text="lazy dog sleeps",
                ),
                9.0,
            ),
        ]
        results, filled = select_results(ranked, k=2, q_tokens={"quick", "lazy"})
        assert filled is True
        assert len(results) == 2

    def test_not_filled_overflow(self) -> None:
        """Only 1 unique source, k=2 -> overflow fills the gap."""
        chunks = [
            (
                Chunk(
                    id=f"single#{i}",
                    file_sha256="x",
                    chunk_sha256="y",
                    position=i,
                    page=None,
                    section="",
                    file_date="",
                    source="single.txt",
                    header="single.txt",
                    text=f"word {i}",
                ),
                float(10 - i),
            )
            for i in range(5)
        ]
        results, filled = select_results(chunks, k=2, q_tokens={"word"})
        # All from same source; MAX_CHUNKS_PER_FILE=2, so first 2 selected normally
        assert filled is True

    def test_not_filled_overflow_needed(self) -> None:
        """Soft cap blocks more than 2 per source, overflow fills."""
        chunks = [
            (
                Chunk(
                    id=f"same#{i}",
                    file_sha256="x",
                    chunk_sha256="y",
                    position=i,
                    page=None,
                    section="",
                    file_date="",
                    source="only.txt",
                    header="only.txt",
                    text=f"result item number {i}",
                ),
                float(10.0 - i * 0.5),
            )
            for i in range(6)
        ]
        results, filled = select_results(chunks, k=5, q_tokens={"result"})
        # Same source: 2 via normal, 2 via overflow check, then extend from overflow
        assert filled is False
        assert len(results) == 5

    def test_empty_ranked(self) -> None:
        results, filled = select_results([], k=3, q_tokens={"anything"})
        assert results == []
        assert filled is True  # no ranked items -> vacuously True


class TestDbIndexSearchParity:
    @pytest.mark.parametrize(
        "query",
        [
            "install steps",  # multi-word
            "budget",  # single word
            "nonexistentxyz",  # no match
            "the",  # very common word
            "November",  # date
            "action items",  # multi-word, multiple sources
            "verbose",  # single source
            "planning",  # in extra.txt
        ],
    )
    @pytest.mark.parametrize("k", [1, 4, 8])
    def test_search_parity(self, src: Path, db: Path, query: str, k: int) -> None:
        from voice_agent.rag.backends import _make_lexical_factory

        factory = _make_lexical_factory("fts5", cache_dir=None, fts5_tokenizer="pretok")
        di = DocIndex(src, retriever_factory=factory)
        di.refresh()
        dbi = DbIndex(db)

        doc_results = di.search(query, k=k)
        db_results = dbi.search(query, k=k)

        assert doc_results == db_results, (
            f"query={query!r} k={k}: DocIndex returned {len(doc_results)}, "
            f"DbIndex returned {len(db_results)}"
        )
        dbi.close()

    def test_records_equal(self, src: Path, db: Path) -> None:
        di = DocIndex(src, retriever_factory=Bm25Retriever)
        di.refresh()
        dbi = DbIndex(db)

        doc_recs = di.records()
        db_recs = dbi.records()

        assert len(doc_recs) == len(db_recs)
        for dr, br in zip(doc_recs, db_recs, strict=True):
            assert dr.name == br.name
            assert dr.status == br.status
            assert dr.kind == br.kind

        dbi.close()

    def test_failures_equal(self, src: Path, db: Path) -> None:
        di = DocIndex(src, retriever_factory=Bm25Retriever)
        di.refresh()
        dbi = DbIndex(db)

        doc_fails = di.failures()
        db_fails = dbi.failures()

        # Same failure names
        doc_names = {name for name, _, _ in doc_fails}
        db_names = {name for name, _, _ in db_fails}
        assert doc_names == db_names

        dbi.close()

    def test_incomplete_db_raises(self, src: Path, tmp_path: Path) -> None:
        db = tmp_path / "incomplete.db"
        ingest(src, db)

        # Tamper with meta to mark incomplete
        uri = db.as_uri() + "?mode=rw"
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("UPDATE meta SET value = '0' WHERE key = 'complete'")
        conn.commit()
        conn.close()

        with pytest.raises(DbIndexError):
            DbIndex(db)

    def test_missing_fts_raises(self, src: Path, tmp_path: Path) -> None:
        db = tmp_path / "no_fts.db"
        ingest(src, db)
        # Remove .fts
        (db.with_name(db.name + ".fts")).unlink(missing_ok=True)

        with pytest.raises(DbIndexError):
            DbIndex(db)

    def test_never_writes(self, src: Path, db: Path) -> None:
        def _stats(path: Path) -> tuple:
            st = path.stat()
            return st.st_mtime_ns, st.st_size

        db_stat = _stats(db)
        fts_path = db.with_name(db.name + ".fts")
        fts_stat = _stats(fts_path)

        dbi = DbIndex(db)
        for q in ("install", "budget", "test"):
            dbi.search(q, k=4)
            dbi.records()
            dbi.failures()

        assert _stats(db) == db_stat
        assert _stats(fts_path) == fts_stat
        dbi.close()


class TestDbIndexChunks:
    def test_chunks_match(self, src: Path, db: Path) -> None:
        dbi = DbIndex(db)
        di = DocIndex(src, retriever_factory=Bm25Retriever)
        di.refresh()

        db_chunks = dbi.chunks
        di_chunks = di.chunks

        assert len(db_chunks) == len(di_chunks)
        for dc, ic in zip(db_chunks, di_chunks, strict=True):
            assert dc.id == ic.id
            assert dc.text == ic.text
            assert dc.source == ic.source
            assert dc.converter == ic.converter

        dbi.close()


class TestDbIndexErrorSchemas:
    def test_schema_mismatch(self, src: Path, tmp_path: Path) -> None:
        db = tmp_path / "bad_schema.db"
        ingest(src, db)

        uri = db.as_uri() + "?mode=rw"
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
        conn.commit()
        conn.close()

        with pytest.raises(DbIndexError, match="schema mismatch"):
            DbIndex(db)


class TestDbIndexClose:
    def test_close_and_reopen(self, src: Path, db: Path) -> None:
        dbi = DbIndex(db)
        dbi.search("test", k=1)
        dbi.close()

        # Re-open
        dbi2 = DbIndex(db)
        dbi2.search("test", k=1)
        dbi2.close()


class TestDbIndexIndexes:
    """Ticket 216: indexes on chunk_data and images."""

    def test_idx_chunk_rel_ord_exists(self, db: Path) -> None:
        """idx_chunk_rel_ord exists after ingest."""
        indexes = {
            str(r[0])
            for r in sqlite3.connect(str(db))
            .execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='chunk_data'")
            .fetchall()
        }
        assert "idx_chunk_rel_ord" in indexes

    def test_idx_images_status_exists(self, db: Path) -> None:
        """idx_images_status exists after ingest."""
        indexes = {
            str(r[0])
            for r in sqlite3.connect(str(db))
            .execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='images'")
            .fetchall()
        }
        assert "idx_images_status" in indexes


class TestOverview:
    """Ticket 221a: overview_context and enumerate_files on a DbIndex."""

    def test_overview_covers_all_ok_files(self, db: Path) -> None:
        dbi = DbIndex(db)
        try:
            entries = dbi.overview_context()
            ok_names = {rec.name for rec in dbi.records() if rec.status == "ok"}
            keys = [e["rel_key"] for e in entries]
            assert set(keys) == ok_names
            assert len(keys) == len(set(keys))
            assert len("\n".join(e["line"] for e in entries)) <= 6000
        finally:
            dbi.close()

    def test_overview_folders_and_shorten(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "a").mkdir(parents=True)
        (src / "b").mkdir(parents=True)
        (src / "a" / "one.md").write_text("# Alpha\n\nText one.\n", encoding="utf-8")
        (src / "a" / "two.md").write_text("# Beta\n\nText two.\n", encoding="utf-8")
        (src / "b" / "three.md").write_text("# Gamma\n\nText three.\n", encoding="utf-8")
        (tmp_path / "db").mkdir()
        db = tmp_path / "db" / "test.db"
        ingest(src, db)
        dbi = DbIndex(db)
        try:
            entries = dbi.overview_context()
            keys = [e["rel_key"] for e in entries]
            assert keys == ["a/one.md", "a/two.md", "b/three.md"]
            assert [e["folder"] for e in entries] == ["a", "a", "b"]
            title = dbi.file_card("a/one.md")["title"]
            assert title in entries[0]["line"]
            short = dbi.overview_context(max_chars=80)
            assert [e["rel_key"] for e in short] == keys
            cards = [(rk, dbi.file_card(rk)) for rk in keys]
            lines = {e["rel_key"]: e["line"] for e in short}
            assert len(DbIndex._overview_render(cards, lines)) <= 80
            assert dbi.overview_context(max_chars=30) == []
        finally:
            dbi.close()

    def test_overview_rollup_above_120_files(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        for i in range(121):
            folder = src / f"f{i % 5}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"file_{i:03d}.txt").write_text(f"Plain content {i}.\n", encoding="utf-8")
        (tmp_path / "db").mkdir()
        db = tmp_path / "db" / "test.db"
        ingest(src, db)
        dbi = DbIndex(db)
        try:
            entries = dbi.overview_context()
            assert len(entries) == 5
            assert sum(e["n_files"] for e in entries) == 121
            assert all(len(e["titles"]) <= 3 for e in entries)
        finally:
            dbi.close()

    def test_enumerate_distinct_best_line(self, db: Path) -> None:
        dbi = DbIndex(db)
        try:
            assert [r["rel_key"] for r in dbi.enumerate_files("widget")] == ["data.csv"]
            assert [r["rel_key"] for r in dbi.enumerate_files("budget quarterly")] == ["extra.txt"]
            assert dbi.enumerate_files("the and of") == []
            assert dbi.enumerate_files("zzzqqq") == []
        finally:
            dbi.close()

    def test_enumerate_identity_vs_mention(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "calendars").mkdir(parents=True)
        (src / "notes").mkdir(parents=True)
        (src / "calendars" / "nyu_calendar.html").write_text(
            "<html><body>the university calendar lists exam dates</body></html>",
            encoding="utf-8",
        )
        (src / "notes" / "semester_memo.txt").write_text(
            "The calendar was printed. The calendar was mailed. The calendar was posted. "
            "The calendar was distributed. The calendar is final.\n",
            encoding="utf-8",
        )
        (tmp_path / "db").mkdir()
        db = tmp_path / "db" / "test.db"
        ingest(src, db)
        dbi = DbIndex(db)
        try:
            identity = dbi.enumerate_files("Which files are calendars?")
            assert [r["rel_key"] for r in identity] == ["calendars/nyu_calendar.html"]
            mention = dbi.enumerate_files("Which files mention calendars?")
            keys = [r["rel_key"] for r in mention]
            assert len(keys) == len(set(keys)) == 2
            assert set(keys) == {"calendars/nyu_calendar.html", "notes/semester_memo.txt"}
            assert all(r["line"] for r in mention)
        finally:
            dbi.close()

    def test_enumerate_limit(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir(parents=True)
        for i in range(3):
            (src / f"z{i}.txt").write_text("zephyr winds today.\n", encoding="utf-8")
        (tmp_path / "db").mkdir()
        db = tmp_path / "db" / "test.db"
        ingest(src, db)
        dbi = DbIndex(db)
        try:
            assert len(dbi.enumerate_files("zephyr", limit=2)) == 2
        finally:
            dbi.close()
