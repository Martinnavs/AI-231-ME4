"""CSV table rows keep their row label in every chunk (WP5)."""

from __future__ import annotations

from voice_agent.rag import canonical, docindex


def _wide_csv(rows: int = 3, cols: int = 80) -> str:
    header = "Title," + ",".join(f"Metric number {c} quarterly estimate" for c in range(cols))
    body = [f"2020 Q{r}," + ",".join(str(r * c) for c in range(cols)) for r in range(1, rows + 1)]
    return "\n".join([header, *body])


def test_narrow_csv_rows_are_unchanged() -> None:
    text = docindex.parse_csv_text("name,qty\nbolt,5\nnut,7\n")
    assert text == "name: bolt; qty: 5\nname: nut; qty: 7"


def test_wide_csv_row_is_split_and_every_chunk_has_its_label() -> None:
    text = docindex.parse_csv_text(_wide_csv())
    chunks = docindex.chunk_text(
        text, kind="csv", source="t.csv", file_sha256="x", file_date="2026-01-01"
    )
    assert len(chunks) > 3
    assert all("Title: 2020 Q" in chunk.text for chunk in chunks)
    assert all(len(chunk.text) <= docindex.CHUNK_SIZE for chunk in chunks)


def test_csv_over_the_row_cap_ends_with_the_truncation_note() -> None:
    text = docindex.parse_csv_text("a,b\n" + "x,1\n" * 5, max_rows=3)
    lines = text.split("\n")
    assert len(lines) == 4
    assert lines[-1] == canonical.TABLE_TRUNCATED_NOTE
    assert docindex.parse_csv_text("a,b\nx,1\n", max_rows=3) == "a: x; b: 1"
