"""Tests for voice_pipeline.general.answer_general (ticket 221a)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from voice_pipeline.general import (
    GENERAL_SYSTEM,
    _counts_ok,
    _names_ok,
    _short_list,
    answer_general,
)

ENTRIES: list[dict[str, Any]] = [
    {"folder": "a", "rel_key": "a/guide.md", "line": "a/guide.md | Setup | md | Setup"},
    {"folder": "b", "rel_key": "b/notes.txt", "line": "b/notes.txt | Notes | txt"},
]
PLAIN = "I have 2 files in 2 folders: a (1), b (1)."
GOOD_ANSWER = "You have 2 files in 2 folders: a and b."


class _Fakes:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.captured: list[tuple[str, str]] = []
        self.enumerate_calls: list[str] = []

    async def request(self, system: str, content: str) -> str:
        self.captured.append((system, content))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    async def catalog(self) -> list[dict[str, Any]]:
        return ENTRIES

    async def enumerate_(self, query: str) -> list[dict[str, Any]]:
        self.enumerate_calls.append(query)
        return [{"rel_key": "b/notes.txt", "line": "budgets are reviewed"}]


def _good_fakes() -> _Fakes:
    return _Fakes(json.dumps({"status": "answer", "answer": GOOD_ANSWER}))


@pytest.mark.asyncio
async def test_general_model_answer_accepted() -> None:
    fakes = _good_fakes()
    r = await answer_general(
        "What are all the files about?",
        catalog=fakes.catalog,
        enumerate_=fakes.enumerate_,
        request=fakes.request,
    )
    assert r.status == "accepted"
    assert r.spoken == GOOD_ANSWER
    assert r.fallback is None
    assert r.calls == 1


@pytest.mark.asyncio
async def test_general_unknown_file_falls_back() -> None:
    fakes = _Fakes(json.dumps({"status": "answer", "answer": "Your files include ghost.md."}))
    r = await answer_general(
        "What are all the files about?",
        catalog=fakes.catalog,
        enumerate_=fakes.enumerate_,
        request=fakes.request,
    )
    assert r.status == "accepted"
    assert r.fallback == "plain_list"
    assert r.spoken == PLAIN


@pytest.mark.asyncio
async def test_general_wrong_count_falls_back() -> None:
    fakes = _Fakes(
        json.dumps({"status": "answer", "answer": "You have 5 files in 2 folders: a and b."})
    )
    r = await answer_general(
        "What are all the files about?",
        catalog=fakes.catalog,
        enumerate_=fakes.enumerate_,
        request=fakes.request,
    )
    assert r.status == "accepted"
    assert r.fallback == "plain_list"
    assert r.spoken == PLAIN


@pytest.mark.asyncio
async def test_general_number_not_in_context_falls_back() -> None:
    fakes = _Fakes(json.dumps({"status": "answer", "answer": "These documents span 300 pages."}))
    r = await answer_general(
        "What are all the files about?",
        catalog=fakes.catalog,
        enumerate_=fakes.enumerate_,
        request=fakes.request,
    )
    assert r.status == "accepted"
    assert r.fallback == "plain_list"
    assert r.spoken == PLAIN


@pytest.mark.asyncio
async def test_general_model_error_falls_back() -> None:
    fakes = _Fakes(RuntimeError("boom"))
    r = await answer_general(
        "What are all the files about?",
        catalog=fakes.catalog,
        enumerate_=fakes.enumerate_,
        request=fakes.request,
    )
    assert r.status == "accepted"
    assert r.fallback == "plain_list"
    assert r.spoken == PLAIN


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "should_enumerate"),
    [
        ("Which files mention budgets?", True),
        ("What are all the files about?", False),
    ],
)
async def test_general_enumerate_cue(question: str, should_enumerate: bool) -> None:
    fakes = _good_fakes()
    await answer_general(
        question,
        catalog=fakes.catalog,
        enumerate_=fakes.enumerate_,
        request=fakes.request,
    )
    assert (len(fakes.enumerate_calls) > 0) is should_enumerate
    if should_enumerate:
        assert "b/notes.txt: budgets are reviewed" in fakes.captured[0][1]


# ---- Ticket 226: count check accepts row counts; fallbacks carry the reason ----


def test_counts_ok_accepts_row_count_and_folder_counts() -> None:
    assert _counts_ok("7 files are academic calendars", 48, 30, (7,))
    assert not _counts_ok("9 files are academic calendars", 48, 30, (7,))
    assert _counts_ok("You have 48 files in 30 folders.", 48, 30)
    assert _counts_ok("There are 30 folders.", 48, 30)
    assert _counts_ok("admin has 5 files", 48, 30, (0, 5, 12))
    assert not _counts_ok("admin has 6 files", 48, 30, (0, 5, 12))


def _calendar_entries() -> list[dict[str, Any]]:
    """30 folders, 48 files, one folder with exactly 7 (so '7' is in the catalogue)."""
    entries: list[dict[str, Any]] = [{"folder": "acad", "n_files": 7, "titles": ["cal"]}]
    entries += [{"folder": f"f{i}", "n_files": 2, "titles": []} for i in range(12)]
    entries += [{"folder": f"g{i}", "n_files": 1, "titles": []} for i in range(17)]
    return entries


class _CalendarFakes:
    def __init__(self, answer: str, rows: int = 7) -> None:
        self.answer = answer
        self.rows = rows

    async def catalog(self) -> list[dict[str, Any]]:
        return _calendar_entries()

    async def enumerate_(self, query: str) -> list[dict[str, Any]]:
        return [
            {"rel_key": f"acad/calendar_{i}.md", "line": "academic calendar"}
            for i in range(self.rows)
        ]

    async def request(self, system: str, content: str) -> str:
        return json.dumps({"status": "answer", "answer": self.answer})


@pytest.mark.asyncio
async def test_general_row_count_answer_accepted() -> None:
    fakes = _CalendarFakes("7 files are academic calendars.")
    r = await answer_general(
        "Which files are academic calendars?",
        catalog=fakes.catalog,
        enumerate_=fakes.enumerate_,
        request=fakes.request,
    )
    assert r.status == "accepted"
    assert r.fallback is None
    assert r.spoken == "7 files are academic calendars."
    assert r.rejected is None


@pytest.mark.asyncio
async def test_general_wrong_row_count_answer_falls_back_with_counts_reason() -> None:
    fakes = _CalendarFakes("9 files are academic calendars.")
    r = await answer_general(
        "Which files are academic calendars?",
        catalog=fakes.catalog,
        enumerate_=fakes.enumerate_,
        request=fakes.request,
    )
    assert r.status == "accepted"
    assert r.fallback == "plain_list"
    assert r.rejected is not None
    assert r.rejected["failed"] == "counts"
    assert r.rejected["answer"] == "9 files are academic calendars."


@pytest.mark.asyncio
async def test_fallback_cares_names_reason() -> None:
    fakes = _Fakes(json.dumps({"status": "answer", "answer": "Your files include ghost.md."}))
    r = await answer_general(
        "What are all the files about?",
        catalog=fakes.catalog,
        enumerate_=fakes.enumerate_,
        request=fakes.request,
    )
    assert r.fallback == "plain_list"
    assert r.rejected is not None
    assert r.rejected["failed"] == "names"
    assert r.rejected["answer"] == "Your files include ghost.md."


@pytest.mark.asyncio
async def test_fallback_cares_timeout_reason() -> None:
    async def slow_request(system: str, content: str) -> str:
        await asyncio.sleep(0.05)
        return json.dumps({"status": "answer", "answer": GOOD_ANSWER})

    async def catalog() -> list[dict[str, Any]]:
        return ENTRIES

    async def enumerate_(query: str) -> list[dict[str, Any]]:
        return []

    r = await answer_general(
        "What are all the files about?",
        catalog=catalog,
        enumerate_=enumerate_,
        request=slow_request,
        budget_s=0.01,
    )
    assert r.status == "accepted"
    assert r.fallback == "plain_list"
    assert r.spoken == PLAIN
    assert r.rejected is not None
    assert r.rejected == {"failed": "timeout", "answer": ""}


# ---- Ticket 227: folder claims trigger only on 'folder(s)', never on 'file(s)' ----


def test_names_ok_folder_claim_only_for_folders() -> None:
    entries: list[dict[str, Any]] = [
        {
            "folder": "butler",
            "rel_key": "butler/academic_calendar_2026_2027.html",
            "line": "butler/academic_calendar_2026_2027.html | Calendar | html | Calendar",
        }
    ]
    assert _names_ok(
        "The academic calendar files include butler/academic_calendar_2026_2027.html.", entries
    )
    assert not _names_ok("The nasa folder has data.", entries)
    assert not _names_ok("Your files include ghost.md.", entries)


@pytest.mark.asyncio
async def test_general_file_claim_not_treated_as_folder() -> None:
    entries: list[dict[str, Any]] = [
        {
            "folder": "butler",
            "rel_key": "butler/academic_calendar_2026_2027.html",
            "line": "butler/academic_calendar_2026_2027.html | Calendar | html | Calendar",
        }
    ]

    async def catalog() -> list[dict[str, Any]]:
        return entries

    async def enumerate_(query: str) -> list[dict[str, Any]]:
        return []

    async def request(system: str, content: str) -> str:
        return json.dumps(
            {
                "status": "answer",
                "answer": "The academic calendar files include "
                "butler/academic_calendar_2026_2027.html.",
            }
        )

    r = await answer_general(
        "Which files are academic calendars?",
        catalog=catalog,
        enumerate_=enumerate_,
        request=request,
    )
    assert r.status == "accepted"
    assert r.fallback is None
    assert r.rejected is None


@pytest.mark.asyncio
async def test_general_folder_claim_with_count_accepted() -> None:
    entries: list[dict[str, Any]] = [
        {"folder": "census", "rel_key": f"census/c{i}.md", "line": f"census/c{i}.md | c | md | c"}
        for i in range(3)
    ]

    async def catalog() -> list[dict[str, Any]]:
        return entries

    async def enumerate_(query: str) -> list[dict[str, Any]]:
        return []

    async def request(system: str, content: str) -> str:
        return json.dumps({"status": "answer", "answer": "The census folder has 3 files."})

    r = await answer_general(
        "What is in the census folder?",
        catalog=catalog,
        enumerate_=enumerate_,
        request=request,
    )
    assert r.status == "accepted"
    assert r.fallback is None
    assert r.rejected is None


@pytest.mark.asyncio
async def test_general_unknown_folder_falls_back_with_names_reason() -> None:
    fakes = _Fakes(json.dumps({"status": "answer", "answer": "The nasa folder has data."}))
    r = await answer_general(
        "What is in the nasa folder?",
        catalog=fakes.catalog,
        enumerate_=fakes.enumerate_,
        request=fakes.request,
    )
    assert r.fallback == "plain_list"
    assert r.rejected is not None
    assert r.rejected["failed"] == "names"
    assert r.rejected["answer"] == "The nasa folder has data."


# ---- Ticket 231: spoken length cap on general answers ----

_PAD = [
    "the",
    "whole",
    "set",
    "is",
    "stored",
    "in",
    "your",
    "documents",
    "and",
    "nothing",
    "else",
    "beyond",
    "this",
    "list",
    "at",
    "all",
]


def _answer_of_words(base: str, total: int) -> str:
    """`base` padded with neutral words (no numbers, no file/folder claims) to `total` words."""
    parts = base.split()
    need = total - len(parts)
    assert need >= 0
    pad = (_PAD * (need // len(_PAD) + 1))[:need]
    answer = " ".join(parts + pad)
    assert len(answer.split()) == total
    return answer


def _long_entries() -> list[dict[str, Any]]:
    """8 files in 4 folders; top 3 by count: b (3), then a (2) before c (2) on the tie."""
    entries: list[dict[str, Any]] = []
    for folder, n in (("a", 2), ("b", 3), ("c", 2), ("d", 1)):
        entries += [
            {"folder": folder, "rel_key": f"{folder}/f{i}.md", "line": f"{folder}/f{i}.md | f | md"}
            for i in range(n)
        ]
    return entries


SHORT = "I have 8 files in 4 folders, mainly b (3), a (2), c (2)."


def test_general_system_prompts_a_short_answer() -> None:
    assert "Keep the whole answer under 30 words." in GENERAL_SYSTEM


def test_short_list_top_three_by_file_count() -> None:
    assert _short_list(_long_entries()) == SHORT
    assert _short_list(_calendar_entries()) == (
        "I have 48 files in 30 folders, mainly acad (7), f0 (2), f1 (2)."
    )


class _LongFakes:
    def __init__(self, entries: list[dict[str, Any]], answer: str) -> None:
        self.entries = entries
        self.answer = answer

    async def catalog(self) -> list[dict[str, Any]]:
        return self.entries

    async def enumerate_(self, query: str) -> list[dict[str, Any]]:
        return []

    async def request(self, system: str, content: str) -> str:
        return json.dumps({"status": "answer", "answer": self.answer})


@pytest.mark.asyncio
async def test_general_over_45_words_replaced_by_short_summary() -> None:
    answer = _answer_of_words("You have 8 files in 4 folders: a, b, c and d.", 46)
    r = await answer_general(
        "What are all the files about?",
        catalog=_LongFakes(_long_entries(), answer).catalog,
        enumerate_=_LongFakes(_long_entries(), answer).enumerate_,
        request=_LongFakes(_long_entries(), answer).request,
    )
    assert r.status == "accepted"
    assert r.fallback == "long"
    assert r.spoken == SHORT
    assert r.answer == SHORT
    assert r.rejected is not None
    assert r.rejected["failed"] == "long"
    assert r.rejected["answer"] == answer[:120]
    assert r.calls == 1


@pytest.mark.asyncio
async def test_general_exactly_45_words_not_capped() -> None:
    answer = _answer_of_words("You have 8 files in 4 folders: a, b, c and d.", 45)
    fakes = _LongFakes(_long_entries(), answer)
    r = await answer_general(
        "What are all the files about?",
        catalog=fakes.catalog,
        enumerate_=fakes.enumerate_,
        request=fakes.request,
    )
    assert r.status == "accepted"
    assert r.fallback is None
    assert r.rejected is None
    assert r.spoken == answer


@pytest.mark.asyncio
async def test_general_cap_uses_folder_counts_shape_too() -> None:
    """The n_files-shaped catalogue (no 'line' key) caps to the same deterministic summary."""
    answer = _answer_of_words("You have 48 files in 30 folders.", 46)
    fakes = _LongFakes(_calendar_entries(), answer)
    r = await answer_general(
        "What are all the files about?",
        catalog=fakes.catalog,
        enumerate_=fakes.enumerate_,
        request=fakes.request,
    )
    assert r.fallback == "long"
    assert r.spoken == "I have 48 files in 30 folders, mainly acad (7), f0 (2), f1 (2)."


def test_cap_failure_still_reports_other_reasons_first() -> None:
    """A long answer that names a file not in the list fails 'names', not 'long'."""
    entries = _long_entries()
    fakes = _LongFakes(entries, _answer_of_words("Your files include ghost.md.", 46))
    r = asyncio.run(
        answer_general(
            "What are all the files about?",
            catalog=fakes.catalog,
            enumerate_=fakes.enumerate_,
            request=fakes.request,
        )
    )
    assert r.fallback == "plain_list"
    assert r.rejected is not None
    assert r.rejected["failed"] == "names"
