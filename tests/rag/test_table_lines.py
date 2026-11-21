"""Tests for canonical.table_lines (row-label-preserving table row rendering)."""

from __future__ import annotations

from voice_agent.rag import canonical


def test_short_row_is_one_unchanged_line() -> None:
    assert canonical.table_lines([("a", "x"), ("b", "y")]) == ["a: x; b: y"]
    assert canonical.table_lines([]) == []


def test_long_row_splits_and_every_line_keeps_the_label() -> None:
    pairs = [("Region", "North")] + [(f"metric {i}", f"{i}.5") for i in range(60)]
    lines = canonical.table_lines(pairs, limit=200)
    assert len(lines) > 1
    assert all(line.startswith("Region: North; ") and len(line) <= 200 for line in lines)
    assert "metric 59: 59.5" in lines[-1]


def test_label_skips_numeric_code_columns() -> None:
    pairs = [("SUMLEV", "010"), ("NAME", "United States")]
    pairs += [(f"c{i}", "x" * 20) for i in range(30)]
    lines = canonical.table_lines(pairs, limit=200)
    assert all(line.startswith("NAME: United States; ") for line in lines)
    assert "SUMLEV: 010" in lines[0]


def test_long_value_is_sliced_not_lost() -> None:
    lines = canonical.table_lines([("id", "A1"), ("note", "w" * 500)], limit=200)
    assert all(len(line) <= 200 for line in lines)
    assert sum(line.count("w") for line in lines) == 500
