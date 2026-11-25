"""Tests for voice_pipeline.general.answer_general (ticket 221a)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from voice_pipeline.general import answer_general

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
