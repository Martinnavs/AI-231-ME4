"""Tests for voice_pipeline.spoken_form: the voice-side spoken-form filter (ticket 231)."""

from __future__ import annotations

import pytest

from voice_pipeline.spoken_form import spoken_form


@pytest.mark.parametrize(
    "text, want",
    [
        # (a) the 'As per <file>:' lead-in becomes the base name, and the money gets a unit word
        ("As per gs_salary_table_2026.pdf: $45,980", "gs salary table 2026: 45,980 dollars"),
        ("As per security_incidents.md: 3 incidents", "security incidents: 3 incidents"),
        ("as per a/b.pdf: it is fine", "b: it is fine"),
        # (b) any remaining file token -> base name without extension; path slashes -> spaces
        ("See a/b.pdf for the details.", "See b for the details."),
        ("The notes.md file is short.", "The notes file is short."),
        ("Files live in docs/reports.", "Files live in docs reports."),
        # (c) markdown: pipes, asterisks, backticks, leading bullets, headers
        ("| name | amount |", "name amount"),
        ("The **total** is `5`.", "The total is 5."),
        ("- gs_salary_table_2026.pdf", "gs salary table 2026"),
        ("# Summary", "Summary"),
        # (d) a parenthetical year: dropped when the year is spoken elsewhere, else 'from YYYY'
        ("The report (2026) covers tax.", "The report from 2026 covers tax."),
        ("The 2026 report (2026) covers tax.", "The 2026 report covers tax."),
        ("Updated (2026). Next (2026) comes.", "Updated. Next comes."),
        # (e) digits inside numbers are never changed; currency and percent get a unit word
        ("$45,980", "45,980 dollars"),
        ("$5", "5 dollars"),
        ("29.41%", "29.41 percent"),
        ("29.41 cents per kilowatthour", "29.41 cents per kilowatthour"),
        ("51×21 mm", "51×21 mm"),
    ],
)
def test_spoken_form_rules(text: str, want: str) -> None:
    assert spoken_form(text) == want


def test_plain_text_mode_strings_unchanged() -> None:
    """Strings as the agent server sends them in text mode are unchanged where the filter is
    not applied (it is voice-side only), and untouched by it where the text has no marks."""
    for text in (
        "You have 2 files in 2 folders: a and b.",
        "I have 2 files in 2 folders: a (1), b (1).",
        "The documents don't say. Want me to look it up on the web?",
        "It is about 3 p.m. on a Tuesday.",
        "As per my notes, the meeting moved.",
    ):
        assert spoken_form(text) == text


def test_marked_text_changes_only_by_the_closed_rules() -> None:
    assert spoken_form("The salary is $45,980 a year (2026).") == (
        "The salary is 45,980 dollars a year from 2026."
    )


@pytest.mark.parametrize(
    "text",
    [
        "As per gs_salary_table_2026.pdf: $45,980",
        "The report (2026) covers tax.",
        "| name | amount |",
        "- **gs_salary_table_2026.pdf**",
        "29.41% of the 51×21 mm tiles cost $5 each.",
        "Updated (2026). Next (2026) comes.",
    ],
)
def test_spoken_form_is_idempotent(text: str) -> None:
    once = spoken_form(text)
    assert spoken_form(once) == once
