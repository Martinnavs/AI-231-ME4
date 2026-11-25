"""Tests for voice_pipeline.general.answer_general (ticket 221a)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from voice_pipeline.general import _counts_ok, answer_general

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
