"""Offline contract tests for the bounded research runner."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from voice_agent.research.runner import Limits, Report, Runner, Source

SKILL = Path(__file__).parents[2] / "skills" / "deep-research" / "SKILL.md"


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value)


@pytest.mark.asyncio
async def test_followup_searches_and_sources_are_deduplicated_and_capped() -> None:
    model_instructions: list[str] = []
    model_inputs: list[dict[str, Any]] = []
    search_calls: list[str] = []
    model_outputs = iter(
        [
            _json({"queries": [" Alpha  topic ", "alpha topic", "Beta", "Ignored"]}),
            _json(
                {
                    "claims": [{"text": "First supported claim", "sources": ["S1"]}],
                    "gaps": ["Need one more official source"],
                    "queries": ["Gamma", " gamma ", "Delta"],
                }
            ),
            _json({"claims": [{"text": "Revised claim", "sources": ["S2", "S3"]}], "gaps": []}),
        ]
    )

    async def model(system: str, content: str) -> str:
        model_instructions.append(system)
        model_inputs.append(json.loads(content))
        return next(model_outputs)

    async def search(query: str) -> list[Source]:
        search_calls.append(query)
        if " ".join(query.casefold().split()) == "alpha topic":
            return [
                Source("Alpha", "https://EXAMPLE.org/a#top", "alpha evidence"),
                Source("Duplicate", "https://example.org/a#other", "duplicate evidence"),
                Source("Beta", "https://example.org/b", "beta evidence"),
            ]
        if query == "Beta":
            return [Source("Beta duplicate", "https://example.org/b#fragment", "duplicate")]
        if query == "Gamma":
            return [
                Source("Gamma", "https://example.org/c", "gamma evidence"),
                Source("Overflow", "https://example.org/d", "must be capped"),
            ]
        return [Source("Unexpected", "https://example.org/e", "must not run")]

    runner = Runner(
        SKILL,
        model,
        search,
        limits=Limits(searches=3, sources=3, concurrency=1),
    )
    report = await runner.run(
        "Compare Alpha and Beta", reference_at=datetime(2026, 1, 2, tzinfo=UTC)
    )

    assert search_calls == ["Alpha  topic", "Beta", "Gamma"]
    assert report.searches == 3
    assert report.model_calls == 3
    assert [item["id"] for item in report.evidence] == ["S1", "S2", "S3"]
    assert [item["url"] for item in report.evidence] == [
        "https://example.org/a",
        "https://example.org/b",
        "https://example.org/c",
    ]
    assert report.claims == [{"text": "Revised claim", "sources": ["S2", "S3"]}]
    assert report.status == "done"
    assert model_inputs[-1]["remaining_searches"] == 0


@pytest.mark.asyncio
async def test_claim_with_unknown_citation_is_omitted() -> None:
    calls = 0

    async def model(_system: str, _content: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _json({"queries": ["one source"]})
        return _json(
            {
                "claims": [
                    {"text": "Unsupported claim", "sources": ["S999"]},
                    {
                        "text": "See https://untrusted.example/evidence",
                        "sources": ["S1"],
                    },
                    {"text": "Supported claim", "sources": ["S1"]},
                ],
                "gaps": [],
                "queries": [],
            }
        )

    async def search(_query: str) -> list[Source]:
        return [
            Source("Source", "https://example.org/source", "evidence"),
            Source("Oversized URL", "https://example.org/" + "x" * 2000, "ignored"),
        ]

    report = await Runner(SKILL, model, search).run("Question")

    assert report.claims == [{"text": "Supported claim", "sources": ["S1"]}]
    assert report.gaps == [
        "A draft claim lacked valid evidence references and was omitted.",
        "A claim contained a URL instead of an evidence ID and was omitted.",
    ]
    assert [source["url"] for source in report.evidence] == ["https://example.org/source"]
    assert report.status == "partial"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seconds": float("nan")},
        {"seconds": float("inf")},
        {"seconds": True},
        {"searches": 1.5},
        {"model_calls": False},
    ],
)
def test_limits_reject_non_finite_or_non_integer_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Limits(**kwargs)


def test_report_escapes_markdown_and_html_in_untrusted_text() -> None:
    report = Report(
        question="# forged heading [link](https://evil.example)",
        reference_at="2026-11-26T00:00:00+00:00",
        timezone="UTC",
        status="done",
        claims=[{"text": "[click](https://evil.example) <script>", "sources": ["S1"]}],
        gaps=["> quote a fake heading"],
        evidence=[
            {
                "id": "S1",
                "title": "**bold** <img src=x>",
                "url": "https://example.org/source",
                "kind": "search_snippet",
            }
        ],
    )

    markdown = report.markdown()

    assert r"\# forged heading \[link\]\(https://evil\.example\)" in markdown
    assert r"\[click\]\(https://evil\.example\) &lt;script&gt;" in markdown
    assert "&gt; quote a fake heading" in markdown
    assert r"\*\*bold\*\* &lt;img src=x&gt;" in markdown
    assert "<script>" not in markdown
    assert "<img src=x>" not in markdown


@pytest.mark.asyncio
async def test_deadline_cancels_inflight_search() -> None:
    search_started = asyncio.Event()
    search_cancelled = asyncio.Event()
    model_calls = 0

    async def model(_system: str, _content: str) -> str:
        nonlocal model_calls
        model_calls += 1
        return _json({"queries": ["slow query"]})

    async def search(_query: str) -> list[Source]:
        search_started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            search_cancelled.set()
            raise
        return []

    report = await Runner(SKILL, model, search, limits=Limits(seconds=0.05, searches=1)).run(
        "Question"
    )

    assert search_started.is_set()
    assert search_cancelled.is_set()
    assert model_calls == 1
    assert report.searches == 1
    assert report.model_calls == 1
    assert report.status == "timed_out"


@pytest.mark.asyncio
async def test_external_cancellation_propagates_and_starts_no_next_calls() -> None:
    search_started = asyncio.Event()
    search_cancelled = asyncio.Event()
    model_calls = 0
    search_calls = 0
    snapshots: list[str] = []

    async def model(_system: str, _content: str) -> str:
        nonlocal model_calls
        model_calls += 1
        return _json({"queries": ["wait here"]})

    async def search(_query: str) -> list[Source]:
        nonlocal search_calls
        search_calls += 1
        search_started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            search_cancelled.set()
            raise
        return []

    runner = Runner(SKILL, model, search)
    task = asyncio.create_task(
        runner.run("Question", snapshot=lambda report: snapshots.append(report.status))
    )
    await search_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert search_cancelled.is_set()
    assert search_calls == 1
    assert model_calls == 1
    assert "cancelled" in snapshots


@pytest.mark.asyncio
async def test_malformed_json_fails_without_searching() -> None:
    search_calls = 0

    async def model(_system: str, _content: str) -> str:
        return "not-json"

    async def search(_query: str) -> list[Source]:
        nonlocal search_calls
        search_calls += 1
        return []

    report = await Runner(SKILL, model, search).run("Question")

    assert report.status == "failed"
    assert report.model_calls == 1
    assert report.searches == 0
    assert search_calls == 0
    assert report.gaps == ["Research stopped: JSONDecodeError."]


@pytest.mark.asyncio
async def test_reference_time_is_normalized_and_sent_to_model() -> None:
    seen: dict[str, Any] = {}
    local_time = datetime(2026, 11, 26, 15, 30, tzinfo=timezone(timedelta(hours=8)))

    async def model(_system: str, content: str) -> str:
        seen.update(json.loads(content))
        if "evidence" not in seen or not seen["evidence"]:
            return _json({"queries": []})
        return _json({"claims": [], "gaps": ["No source yet."], "queries": []})

    async def search(_query: str) -> list[Source]:
        return [Source("Source", "https://example.org/source", "evidence")]

    report = await Runner(SKILL, model, search).run(
        "Question", timezone="Asia/Manila", reference_at=local_time
    )

    assert report.reference_at == "2026-11-26T07:30:00+00:00"
    assert seen["reference_at"] == report.reference_at
    assert seen["timezone"] == "Asia/Manila"
