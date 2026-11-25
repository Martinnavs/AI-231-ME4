"""Tests for pipeline/src/voice_pipeline/temporal.py (ticket 225)."""

from __future__ import annotations

import pytest

from voice_pipeline.temporal import has_temporal_cue, source_year


@pytest.mark.parametrize(
    "q",
    [
        "What's the current Social Security wage base?",
        "Who is the latest author?",
        "What's the newest model in the folder?",
    ],
)
def test_has_temporal_cue_true(q: str) -> None:
    assert has_temporal_cue(q) is True


@pytest.mark.parametrize(
    "q",
    [
        "What was the wage base in 2025?",
        "When is the next meeting?",
        "The report was filed in 2024.",
    ],
)
def test_has_temporal_cue_false(q: str) -> None:
    assert has_temporal_cue(q) is False


@pytest.mark.parametrize(
    ("date_max", "file_path", "title", "want"),
    [
        ("2026-10-07", "census/nst_est2024_pop.xlsx", "nst_est2024_pop", "2026"),
        (None, "irs/p15_2026_guide.pdf", "P15", "2026"),
        ("", "pico/pico_datasheet.pdf", "Pico 2 W Datasheet, 2024 edition", "2024"),
    ],
)
def test_source_year_order(date_max: str | None, file_path: str, title: str, want: str) -> None:
    assert source_year(date_max, file_path, title) == want


def test_source_year_none() -> None:
    assert source_year(None, "raspberrypi/pico_datasheet.pdf", "Pico Datasheet") is None
