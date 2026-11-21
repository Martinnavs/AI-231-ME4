"""xlsx table rows keep their row label in every chunk (WP5)."""

from __future__ import annotations

from voice_agent.rag import canonical, docindex


def _pipe_table(rows: int = 3, cols: int = 60) -> str:
    head = "| Region | " + " | ".join(f"Metric {c} quarterly estimate" for c in range(cols)) + " |"
    sep = "| " + " | ".join(["---"] * (cols + 1)) + " |"
    body = [
        f"| Region{r:02d} | " + " | ".join(str(r * c) for c in range(cols)) + " |"
        for r in range(1, rows + 1)
    ]
    return "\n".join(["## Probe", "", head, sep, *body])


def test_xlsx_wide_rows_split_with_their_label() -> None:
    text = canonical.postprocess_markdown(_pipe_table(), kind="xlsx").text
    chunks = docindex.chunk_canonical(
        text,
        kind="xlsx",
        source="t.xlsx",
        file_sha256="x",
        file_date="2026-01-01",
        canonical_sha256="y",
        converter="direct-openpyxl-3.1.5",
    )
    assert len(chunks) > 3
    assert all("Region0" in chunk.text for chunk in chunks)
    assert all(len(chunk.text) <= docindex.CHUNK_SIZE for chunk in chunks)


def test_narrow_rows_are_unchanged_for_xlsx_and_html() -> None:
    md = "| a | b |\n| --- | --- |\n| x | 1 |"
    assert canonical.postprocess_markdown(md, kind="xlsx").text == "a: x; b: 1"
    assert canonical.postprocess_markdown(md, kind="html").text == "a: x; b: 1"


def test_rows_over_the_cap_end_with_the_truncation_note_for_xlsx_only() -> None:
    rows = ["| id | val |", "| --- | --- |"]
    rows += [f"| {i} | item |" for i in range(canonical.MAX_CSV_ROWS + 5)]
    xlsx = canonical.postprocess_markdown("\n".join(rows), kind="xlsx").text
    html = canonical.postprocess_markdown("\n".join(rows), kind="html").text
    assert xlsx.endswith(canonical.TABLE_TRUNCATED_NOTE)
    assert canonical.TABLE_TRUNCATED_NOTE not in html
