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
