"""Tests for voice_pipeline.spoken_form: the voice-side spoken-form filter (tickets 231/242)."""

from __future__ import annotations

import pytest

from voice_pipeline.spoken_form import spoken_form


@pytest.mark.parametrize(
    "text, want",
    [
        # (a) the 'As per <file>:' citation lead becomes the document's spoken title words,
        #     and the money gets a unit word
        (
            "As per gs_salary_table_2026.pdf: $45,980",
            "From your gs salary table 2026 document: 45,980 dollars",
        ),
        (
            "As per security_incidents.md: 3 incidents",
            "From your security incidents document: 3 incidents",
        ),
        ("as per a/b.pdf: it is fine", "From your b document: it is fine"),
        # a leading date/number run in the stem is not spoken ('2024 01 travel policy')
        (
            "As per 2024_01_travel_policy.pdf: The policy starts January.",
            "From your travel policy document: The policy starts January.",
        ),
        # (b) the unknown-file fallback lead, no filename at all
        ("As per your documents: 3 incidents", "From your documents: 3 incidents"),
        # (c) the temporal year tag stays in the sentence and the year rules apply
        (
            "As per gs_salary_table_2026.md (2026): 45,980",
            "From your gs salary table 2026 document: 45,980",
        ),
        (
            "As per gs_plan.md (2026): 45,980",
            "From your gs plan document from 2026: 45,980",
        ),
        # (d) any remaining file token -> base name without extension; slashes -> spaces
        ("See a/b.pdf for the details.", "See b for the details."),
        ("The notes.md file is short.", "The notes file is short."),
        ("Files live in docs/reports.", "Files live in docs reports."),
        # (e) markdown: pipes, asterisks, backticks, leading bullets, headers
        ("| name | amount |", "name amount"),
        ("The **total** is `5`.", "The total is 5."),
        ("- gs_salary_table_2026.pdf", "gs salary table 2026"),
        ("# Summary", "Summary"),
        # (f) a parenthetical year: dropped when the year is spoken elsewhere, else 'from YYYY'
        ("The report (2026) covers tax.", "The report from 2026 covers tax."),
        ("The 2026 report (2026) covers tax.", "The 2026 report covers tax."),
        ("Updated (2026). Next (2026) comes.", "Updated. Next comes."),
        # (g) digits inside numbers are never changed; currency and percent get a unit word
        ("$45,980", "45,980 dollars"),
        ("$5", "5 dollars"),
        ("29.41%", "29.41 percent"),
        ("29.41 cents per kilowatthour", "29.41 cents per kilowatthour"),
        ("51×21 mm", "51×21 mm"),
        # (h) a bare negative number is said 'minus N'; cents and ranges stay as they are
        ("The reading is -215.", "The reading is minus 215."),
        ("The balance is -72.5 cents.", "The balance is minus 72.5 cents."),
        ("Steps 5-10 apply.", "Steps 5-10 apply."),
        ("AC 60-22 applies.", "AC 60-22 applies."),
        # (i) the web lead names publishers, has no file extension, and is untouched
        (
            "As per Wikipedia and CNN, the capital is Lima.",
            "As per Wikipedia and CNN, the capital is Lima.",
        ),
    ],
)
def test_spoken_form_rules(text: str, want: str) -> None:
    assert spoken_form(text) == want


@pytest.mark.parametrize(
    ("text", "doc_title", "want"),
    [
        # the card title wins over the stem when it is at hand
        (
            "As per travel_expenses_guide.pdf: The rate is 72.5 cents a mile.",
            "The Zorblat Field Guide: Handling, Storage and Care",
            "From your document on The Zorblat Field Guide: Handling, Storage and Care: "
            "The rate is 72.5 cents a mile.",
        ),
        # a table description is not a title: the stem stands
        (
            "As per inventory_2031.xlsx: 40 units",
            "A table with columns for item and count",
            "From your inventory 2031 document: 40 units",
        ),
        # a title over 60 chars is not spoken as a title
        (
            "As per long_name.md: ok",
            "A" * 61,
            "From your long name document: ok",
        ),
    ],
)
def test_spoken_form_doc_title(text: str, doc_title: str, want: str) -> None:
    assert spoken_form(text, doc_title=doc_title) == want


def test_plain_text_mode_strings_unchanged() -> None:
    """Strings as the agent server sends them in text mode are unchanged where the filter is
    not applied (it is voice-side only), and untouched by it where the text has no marks."""
    for text in (
        "You have 2 files in 2 folders: a and b.",
        "I have 2 files in 2 folders: a (1), b (1).",
        "The documents don't say. Want me to look it up on the web?",
        "It is about 3 p.m. on a Tuesday.",
        "As per my notes, the meeting moved.",
        "I checked your documents on the launch plan, beta memo and 2 more.",
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
        "As per your documents: 3 incidents",
        "As per gs_plan.md (2026): 45,980",
        "The reading is -215.",
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
