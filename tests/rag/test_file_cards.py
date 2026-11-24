"""Tests for file_cards module and ingest/dbindex integration (ticket 215)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from voice_agent.rag.dbindex import DbIndex
from voice_agent.rag.docindex import Chunk
from voice_agent.rag.file_cards import (
    MAX_CARD_BYTES,
    MAX_ENTITIES,
    MAX_SECTIONS,
    FileCard,
    build_card,
    card_to_json,
)
from voice_agent.rag.ingest import ingest


def _make_chunk(
    text: str,
    *,
    section: str = "",
    source: str = "test.md",
    page: int | None = None,
    ord: int = 0,
) -> Chunk:
    return Chunk(
        id=f"test#{ord}",
        file_sha256="abc",
        chunk_sha256="def",
        position=ord,
        page=page,
        section=section,
        file_date="",
        source=source,
        header="",
        text=text,
    )


class TestBuildCardMarkdown:
    def test_title_from_heading(self) -> None:
        chunks = [
            _make_chunk("# Introduction\nWelcome to the guide.", section="Introduction"),
            _make_chunk("## Setup\nInstall the package.", section="Setup"),
            _make_chunk("## Usage\nRun the agent.", section="Usage"),
            _make_chunk("## Advanced\nConfiguration details.", section="Advanced"),
            _make_chunk("## FAQ\nCommon questions.", section="FAQ"),
        ]
        card = build_card("guide.md", chunks, kind="md")
        assert card.title == "Introduction"
        assert card.chunk_count == 5

    def test_sections_extracted(self) -> None:
        chunks = [
            _make_chunk("Section A content.", section="SectionA"),
            _make_chunk("Section B content.", section="SectionB"),
            _make_chunk("Section A again.", section="SectionA"),
            _make_chunk("Section C content.", section="SectionC"),
        ]
        card = build_card("test.md", chunks, kind="md")
        assert card.sections == ["SectionA", "SectionB", "SectionC"]

    def test_entities_found(self) -> None:
        chunks = [
            _make_chunk("The company Microsoft announced a new product from Google Cloud."),
            _make_chunk("Apple and Amazon also responded. See AWS documentation."),
            _make_chunk("Contact Sarah Johnson for details about the API."),
            _make_chunk("Read the RFC document for more info."),
            _make_chunk("Check OpenSSL and PostgreSQL compatibility."),
        ]
        card = build_card("news.md", chunks, kind="md")
        assert len(card.top_entities) >= 1

    def test_dates_found(self) -> None:
        chunks = [
            _make_chunk("Published on 2026-03-15. Updated 2026-06-01."),
            _make_chunk("Event date: November 18, 2026. Deadline is 2027-01-15."),
        ]
        card = build_card("dates.md", chunks, kind="md")
        assert card.date_min is not None
        assert card.date_max is not None


class TestBuildCardCSV:
    def test_table_columns(self) -> None:
        chunks = [
            _make_chunk("Name,Date,Amount", source="data.csv"),
            _make_chunk("Widget,2026-01-01,100", source="data.csv"),
            _make_chunk("Gadget,2026-02-15,200", source="data.csv"),
        ]
        card = build_card("data.csv", chunks, kind="csv")
        assert card.table_columns == ["Name", "Date", "Amount"]

    def test_csv_row_count(self) -> None:
        chunks = [
            _make_chunk("col1,col2", source="data.csv"),
        ] + [_make_chunk(f"val{i},val{j}", source="data.csv") for i in range(10) for j in range(1)]
        card = build_card("data.csv", chunks, kind="csv")
        assert card.row_count == 10  # data chunks = total - header

    def test_csv_title_from_filename(self) -> None:
        chunks = [_make_chunk("a,b", source="report.csv")]
        card = build_card("quarterly_report.csv", chunks, kind="csv")
        assert card.title == "quarterly_report"


class TestCaps:
    def test_sections_capped(self) -> None:
        chunks = [_make_chunk(f"content {i}", section=f"Section{i:02d}") for i in range(50)]
        card = build_card("big.md", chunks, kind="md")
        assert len(card.sections) <= MAX_SECTIONS

    def test_entities_capped(self) -> None:
        entities_text = " ".join(
            f"See {name} documentation." for name in [f"Name{i}" for i in range(30)]
        )
        chunks = [_make_chunk(entities_text)]
        card = build_card("entities.md", chunks, kind="md")
        assert len(card.top_entities) <= MAX_ENTITIES


class TestCardToJson:
    def test_size_cap(self) -> None:
        card = FileCard(
            rel_key="test.md",
            title="Test",
            kind="md",
            sections=[f"SectionNameWithVeryLongText{i}" for i in range(50)],
            top_entities=["Entity1"],
            chunk_count=1,
        )
        text = card_to_json(card)
        assert len(text.encode("utf-8")) <= MAX_CARD_BYTES

    def test_normal_card(self) -> None:
        card = FileCard(
            rel_key="test.md",
            title="Test",
            kind="md",
            sections=["Intro", "Setup"],
            top_entities=["Alice"],
            date_min="2026-01-01",
            date_max="2026-06-01",
            row_count=5,
            chunk_count=3,
        )
        text = card_to_json(card)
        data = json.loads(text)
        assert data["rel_key"] == "test.md"
        assert data["title"] == "Test"
        assert data["sections"] == ["Intro", "Setup"]


class TestDbIndexFileCard:
    def test_file_card_returns_dict(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "test.md").write_text("# Hello\n\nWorld.\n")
        (src / "data.csv").write_text("a,b\n1,2\n")
        db = tmp_path / "test.db"
        ingest(src, db)

        idx = DbIndex(db)
        card = idx.file_card("test.md")
        assert card is not None
        assert "title" in card
        assert "kind" in card
        assert "sections" in card
        assert "chunk_count" in card
        # Missing file returns None
        assert idx.file_card("nonexistent.md") is None
        idx.close()


class TestCatalog:
    def test_catalog_lines_and_capping(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        for i in range(5):
            (src / f"file{i}.md").write_text(f"# File {i}\nContent {i}.\n")
        db = tmp_path / "test.db"
        ingest(src, db)

        idx = DbIndex(db)
        # Full catalog: 5 lines
        cat = idx.catalog(max_chars=2000)
        lines = [line for line in cat.split("\n") if line]
        assert len(lines) == 5
        # Zero budget: empty
        assert idx.catalog(max_chars=0) == ""
        # Tight budget: at most 1 line
        cat = idx.catalog(max_chars=50)
        lines = [line for line in cat.split("\n") if line]
        assert len(lines) <= 1
        idx.close()


class TestIngestIntegration:
    def test_three_file_types(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "guide.md").write_text("# Guide\n\nContent here.\n")
        (src / "data.csv").write_text("col1,col2\nval1,val2\nval3,val4\n")
        (src / "notes.txt").write_text("Some notes about the project.\n")
        db = tmp_path / "test.db"
        ingest(src, db)

        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT rel_key, card_text FROM file_cards").fetchall()
        conn.close()

        rel_keys = {r[0] for r in rows}
        assert rel_keys == {"guide.md", "data.csv", "notes.txt"}
        for _, card_text in rows:
            assert len(card_text) > 0
            data = json.loads(card_text)
            assert "title" in data

    def test_graceful_skip(self, tmp_path: Path) -> None:
        """If build_card raises, chunks exist but no card row."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "good.md").write_text("# Good\n\nContent.\n")
        (src / "also.txt").write_text("Another file.\n")
        db = tmp_path / "test.db"
        ingest(src, db)

        conn = sqlite3.connect(str(db))
        chunks = conn.execute(
            "SELECT COUNT(*) FROM chunk_data WHERE rel_key = 'good.md'"
        ).fetchone()[0]
        card_count = conn.execute("SELECT COUNT(*) FROM file_cards").fetchone()[0]
        conn.close()

        assert chunks >= 1
        assert card_count == 2  # both files got cards
