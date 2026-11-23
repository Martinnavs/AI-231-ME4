"""Offline unit tests for voice_agent.rag.ingest."""

from __future__ import annotations

import io
import json
import sqlite3
import time
from pathlib import Path

import pytest

from voice_agent.rag.docindex import Chunk
from voice_agent.rag.ingest import IngestReport, ingest, main


def _build_docx_bytes(text: str) -> bytes:
    """Build a minimal .docx with one paragraph."""
    try:
        import pptx  # noqa: PLC0415
    except ImportError:
        pytest.skip("python-pptx not installed")

    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    assert slide.shapes.title is not None
    slide.shapes.title.text = "Summary"
    slide.placeholders[1].text = text
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


@pytest.fixture
def fixtures(tmp_path: Path) -> dict[str, Path]:
    """Create a small mixed-format source folder."""
    paths: dict[str, Path] = {}

    md = tmp_path / "guide.md"
    md.write_text(
        "# Setup\n\n"
        "Install the package on 2026-03-15 following standard steps.\n\n"
        "## Usage\n\n"
        "Run the agent with the --verbose flag for detailed output.\n",
        encoding="utf-8",
    )
    paths["md"] = md

    txt = tmp_path / "notes.txt"
    txt.write_text(
        "Meeting notes from November 18, 2026. Action items assigned.\n",
        encoding="utf-8",
    )
    paths["txt"] = txt

    csv = tmp_path / "inventory.csv"
    csv.write_text(
        "item,price,updated\nwidget_a,100,2026-01-01\nwidget_b,200,2026-02-15\n",
        encoding="utf-8",
    )
    paths["csv"] = csv

    docx = tmp_path / "brief.docx"
    docx.write_bytes(_build_docx_bytes("Quarterly budget review summary."))
    paths["docx"] = docx

    unsupported = tmp_path / "raw.xyz"
    unsupported.write_text("binary-like content", encoding="utf-8")
    paths["unsupported"] = unsupported

    corrupt = tmp_path / "bad.pdf"
    corrupt.write_bytes(b"\x00\x01\x02\x80\xff\xfe\x04\x05")
    paths["corrupt"] = corrupt

    return paths


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


def _read_files(db: Path) -> dict[str, dict]:
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    rows = conn.execute("SELECT rel_key, status, error, kind, size, n_chunks FROM files").fetchall()
    conn.close()
    keys = ["rel_key", "status", "error", "kind", "size", "n_chunks"]
    return {r[0]: dict(zip(keys, r, strict=False)) for r in rows}


def _read_meta(db: Path) -> dict[str, str]:
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


class TestIngestBasic:
    def test_report_fields(self, fixtures: dict[str, Path], tmp_path: Path) -> None:
        db = tmp_path / "ingest.db"
        src = fixtures["md"].parent

        report = ingest(src, db)
        assert isinstance(report, IngestReport)
        assert report.files_total >= 4  # md, txt, csv, docx
        assert report.ok >= 3  # md, txt, csv at minimum
        assert report.failed >= 1  # corrupt pdf
        assert report.chunks >= 3
        assert report.reused == 0
        assert report.seconds >= 0

    def test_chunks_match_docindex(self, fixtures: dict[str, Path], tmp_path: Path) -> None:
        db = tmp_path / "ingest.db"
        src = fixtures["md"].parent

        ingest(src, db)

        from voice_agent.rag.docindex import DocIndex

        idx = DocIndex(src)
        idx.refresh()

        db_chunks = _read_chunks(db)
        doc_chunks = idx.chunks

        assert len(db_chunks) == len(doc_chunks)

        for dc, di in zip(db_chunks, doc_chunks, strict=True):
            assert dc == di

    def test_fts5_rank_matches(self, fixtures: dict[str, Path], tmp_path: Path) -> None:
        db = tmp_path / "ingest.db"
        src = fixtures["md"].parent

        ingest(src, db)

        fts_path = db.with_name(db.name + ".fts")
        assert fts_path.exists(), "FTS5 sibling should exist"

        from voice_agent.rag.fts5 import Fts5Retriever

        db_retriever = Fts5Retriever(db_path=fts_path, root=src)

        from voice_agent.rag.docindex import DocIndex

        idx = DocIndex(src)
        idx.refresh()
        idx_chunks = list(idx.chunks)

        idx_retriever = Fts5Retriever(db_path=None, root=src)
        idx_retriever.build(idx_chunks)
        idx_retriever.close()

        queries = [
            "install steps",
            "budget review",
            "package",
            "inventory",
            "November",
        ]
        for q in queries:
            r1 = db_retriever.rank(q, k=10)
            r2 = idx_retriever.rank(q, k=10)
            ids1 = [str(i) for i, _ in r1]
            ids2 = [str(i) for i, _ in r2]
            assert ids1 == ids2, f"FTS5 rank mismatch for query {q!r}: {ids1} vs {ids2}"

    def test_files_rows(self, fixtures: dict[str, Path], tmp_path: Path) -> None:
        db = tmp_path / "ingest.db"
        src = fixtures["md"].parent

        ingest(src, db)
        files = _read_files(db)

        md_row = files["guide.md"]
        assert md_row["status"] == "ok"
        assert int(md_row["n_chunks"]) >= 1

        pdf_row = files["bad.pdf"]
        assert pdf_row["status"] == "unreadable"
        assert int(pdf_row["n_chunks"]) == 0

    def test_meta_fields(self, fixtures: dict[str, Path], tmp_path: Path) -> None:
        db = tmp_path / "ingest.db"
        src = fixtures["md"].parent

        ingest(src, db)
        meta = _read_meta(db)

        assert meta["complete"] == "1"
        assert meta["schema_version"] == "1"
        assert meta["src"] == src.name
        assert meta["tokenizer"]

    def test_dates_json_is_json(self, fixtures: dict[str, Path], tmp_path: Path) -> None:
        """dates_json must be valid JSON, not a Python repr."""
        db = tmp_path / "ingest.db"
        src = fixtures["md"].parent

        ingest(src, db)
        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT rel_key, dates_json FROM files").fetchall()
        conn.close()

        for _, dates_json in rows:
            parsed = json.loads(dates_json)
            assert isinstance(parsed, list)

    def test_converter_roundtrip(self, fixtures: dict[str, Path], tmp_path: Path) -> None:
        """chunk_data.converter is stored and read back correctly."""
        db = tmp_path / "ingest.db"
        src = fixtures["md"].parent

        ingest(src, db)
        db_chunks = _read_chunks(db)

        for chunk in db_chunks:
            assert isinstance(chunk.converter, str)
            assert chunk.converter != ""


class TestIngestResume:
    def test_second_run_parses_nothing(self, fixtures: dict[str, Path], tmp_path: Path) -> None:
        db = tmp_path / "ingest.db"
        src = fixtures["md"].parent

        ingest(src, db)
        first_chunks = _read_chunks(db)

        load_count = 0

        def counting_load(*args, **kwargs):
            nonlocal load_count
            load_count += 1
            from voice_agent.rag.docindex import load_file  # noqa: F811

            return load_file(*args, **kwargs)

        import voice_agent.rag.docindex as docindex_mod  # noqa: F811

        monkey = pytest.MonkeyPatch()
        monkey.setattr(docindex_mod, "load_file", counting_load)

        try:
            ingest(src, db)
        finally:
            monkey.undo()

        assert load_count == 0, f"load_file called {load_count} times on unchanged tree"

        second_chunks = _read_chunks(db)
        assert second_chunks == first_chunks

    def test_reused_equals_fresh(self, fixtures: dict[str, Path], tmp_path: Path) -> None:
        """Reused chunks equal freshly parsed chunks (full dataclass equality)."""
        db = tmp_path / "ingest.db"
        src = fixtures["md"].parent

        ingest(src, db)
        first_chunks = _read_chunks(db)

        # Run again — all files should be reused
        ingest(src, db)
        second_chunks = _read_chunks(db)

        assert second_chunks == first_chunks

    def test_failed_file_reparsed(self, fixtures: dict[str, Path], tmp_path: Path) -> None:
        """A previously failed file must be re-parsed on the second run."""
        # Create a folder where a file initially fails
        src = tmp_path / "fail_src"
        src.mkdir()
        (src / "good.txt").write_text("hello world\n")
        (src / "bad.pdf").write_bytes(b"\x00\x01\x02\x80\xff\xfe\x04\x05")

        db = tmp_path / "fail.db"
        ingest(src, db)
        files1 = _read_files(db)
        assert files1["bad.pdf"]["status"] == "unreadable"

        # Fix the bad file
        (src / "bad.pdf").unlink()
        (src / "good2.txt").write_text("this is the replaced file\n")

        # Second run: bad.pdf is gone, good2.txt is new
        ingest(src, db)
        files2 = _read_files(db)
        assert "bad.pdf" not in files2
        assert "good2.txt" in files2
        assert files2["good2.txt"]["status"] == "ok"

    def test_touch_one_file(self, fixtures: dict[str, Path], tmp_path: Path) -> None:
        db = tmp_path / "ingest.db"
        src = fixtures["md"].parent

        ingest(src, db)
        original_chunks = _read_chunks(db)
        total = len(original_chunks)

        time.sleep(0.05)
        fixtures["txt"].touch()

        ingest(src, db)
        final_chunks = _read_chunks(db)

        assert len(final_chunks) >= total

    def test_delete_file(self, fixtures: dict[str, Path], tmp_path: Path) -> None:
        db = tmp_path / "ingest.db"
        src = fixtures["md"].parent

        ingest(src, db)
        assert "notes.txt" in _read_files(db)

        fixtures["txt"].unlink()
        ingest(src, db)

        files = _read_files(db)
        assert "notes.txt" not in files

        conn = sqlite3.connect(str(db))
        cnt = conn.execute("SELECT COUNT(*) FROM chunk_data WHERE source = 'notes.txt'").fetchone()[
            0
        ]
        conn.close()
        assert cnt == 0

    def test_ord_contiguous_after_incremental(
        self, fixtures: dict[str, Path], tmp_path: Path
    ) -> None:
        """After adding a file in the middle of sort order, ord must be 0..N-1."""
        src = tmp_path / "ord_src"
        src.mkdir()
        (src / "a.txt").write_text("first\n")
        (src / "c.txt").write_text("third\n")

        db = tmp_path / "ord.db"
        ingest(src, db)

        # Add b.txt (alphabetically between a and c)
        (src / "b.txt").write_text("middle\n")
        ingest(src, db)

        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT ord FROM chunk_data ORDER BY ord").fetchall()
        conn.close()

        ords = [r[0] for r in rows]
        expected = list(range(len(ords)))
        assert ords == expected, f"ord not contiguous: {ords}"

    def test_crash_safety_complete_zero(self, fixtures: dict[str, Path], tmp_path: Path) -> None:
        """Crash during resume leaves complete=0; third run completes correctly."""
        crash_dir = tmp_path / "crash_src"
        crash_dir.mkdir()

        for i in range(4):
            (crash_dir / f"file_{i}.txt").write_text(f"Content for file {i} with unique words.\n")

        db = tmp_path / "crash.db"
        # First run: complete
        ingest(crash_dir, db)
        meta1 = _read_meta(db)
        assert meta1["complete"] == "1"

        # Touch a file to trigger re-parse on second run
        time.sleep(0.05)
        (crash_dir / "file_0.txt").write_text("Modified content.\n")

        # Second run: simulate crash (monkeypatch _update_meta to raise on complete=True)
        from voice_agent.rag import ingest as ingest_mod  # noqa: F811

        original_update = ingest_mod._update_meta

        def failing_update(conn, src, *, complete):
            if complete:
                raise SystemExit(1)
            original_update(conn, src, complete=complete)

        monkey = pytest.MonkeyPatch()
        monkey.setattr(ingest_mod, "_update_meta", failing_update)

        try:
            with pytest.raises(SystemExit):
                ingest(crash_dir, db)
        finally:
            monkey.undo()

        meta2 = _read_meta(db)
        assert meta2["complete"] == "0", "crash should leave complete=0"

        # Third run completes
        report3 = ingest(crash_dir, db)
        meta3 = _read_meta(db)
        assert meta3["complete"] == "1"
        assert report3.failed == 0


class TestIngestCLI:
    def test_main_success(self, fixtures: dict[str, Path], tmp_path: Path) -> None:
        db = tmp_path / "cli.db"
        src = str(fixtures["md"].parent)

        rc = main([src, "--out", str(db)])
        assert rc == 0
        assert db.exists()
        # FTS5 sibling should exist
        fts_path = db.with_name(db.name + ".fts")
        assert fts_path.exists()

    def test_main_missing_src(self, tmp_path: Path) -> None:
        rc = main(["/nonexistent/path", "--out", str(tmp_path / "x.db")])
        assert rc == 2
