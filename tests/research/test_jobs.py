from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from voice_agent.research.jobs import ResearchJobs
from voice_agent.research.runner import Limits, Runner, Source


class FakeModel:
    def __init__(self, *, block_first: bool = False) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.block_first = block_first
        self.calls = 0

    async def __call__(self, system: str, content: str) -> str:
        self.calls += 1
        if self.block_first and self.calls == 1:
            self.entered.set()
            await self.release.wait()
        if self.calls == 1:
            return json.dumps({"queries": ["official source query"]})
        return json.dumps({"claims": [{"text": "A supported fact.", "sources": ["S1"]}]})


async def fake_search(query: str) -> list[Source]:
    return [Source("Official source", "https://example.org/fact", "A supported fact.")]


def make_runner(tmp_path: Path, model: FakeModel) -> Runner:
    skill = tmp_path / "research.md"
    skill.write_text("Use only returned evidence and cite source ids.")
    return Runner(skill, model, fake_search, limits=Limits(seconds=5))


@pytest.mark.asyncio
async def test_completed_job_persists_status_report_and_evidence(tmp_path: Path) -> None:
    jobs = ResearchJobs(tmp_path / "state", make_runner(tmp_path, FakeModel()))
    job = jobs.start("Compare two official sources.")
    task = jobs.tasks[job.id]
    await task
    await asyncio.sleep(0)  # Let the task done callback remove its handle.

    assert jobs.status(job.id).status == "done"
    result = jobs.result(job.id)
    assert result is not None
    assert result["status"] == "done"
    assert result["claims"] == [{"text": "A supported fact.", "sources": ["S1"]}]
    folder = Path(job.folder)
    assert (folder / "report.md").is_file()
    assert "https://example.org/fact" in (folder / "report.md").read_text()
    assert not jobs.tasks


@pytest.mark.asyncio
async def test_cancel_after_worker_starts_persists_partial_artifacts(tmp_path: Path) -> None:
    model = FakeModel(block_first=True)
    jobs = ResearchJobs(tmp_path / "state", make_runner(tmp_path, model))
    job = jobs.start("Research this question.")
    await asyncio.wait_for(model.entered.wait(), timeout=1)

    cancelled = await jobs.cancel(job.id)

    assert cancelled.status == "cancelled"
    result = jobs.result(job.id)
    assert result is not None
    assert result["status"] == "cancelled"
    assert "cancelled" in (Path(job.folder) / "report.md").read_text()
    assert not jobs.tasks


@pytest.mark.asyncio
async def test_immediate_cancel_sets_terminal_status_without_starting_worker(
    tmp_path: Path,
) -> None:
    model = FakeModel(block_first=True)
    jobs = ResearchJobs(tmp_path / "state", make_runner(tmp_path, model))
    job = jobs.start("Research this question.")

    await jobs.cancel(job.id)

    assert jobs.status(job.id).status == "cancelled"
    assert model.calls == 0
    assert jobs.result(job.id) is None


@pytest.mark.asyncio
async def test_queue_limit_and_close_cancel_active_jobs(tmp_path: Path) -> None:
    model = FakeModel(block_first=True)
    jobs = ResearchJobs(tmp_path / "state", make_runner(tmp_path, model), max_active=1)
    first = jobs.start("First research question.")
    await asyncio.wait_for(model.entered.wait(), timeout=1)

    with pytest.raises(ValueError, match="queue is full"):
        jobs.start("Second research question.")

    await jobs.close()

    assert jobs.status(first.id).status == "cancelled"
    assert not jobs.tasks
    result = jobs.result(first.id)
    assert result is not None and result["status"] == "cancelled"


@pytest.mark.asyncio
async def test_start_rejects_blank_or_overlong_question(tmp_path: Path) -> None:
    jobs = ResearchJobs(
        tmp_path / "state",
        make_runner(tmp_path, FakeModel()),
    )
    with pytest.raises(ValueError, match="invalid research question"):
        jobs.start("  ")
    with pytest.raises(ValueError, match="invalid research question"):
        jobs.start("x" * (jobs.runner.limits.question_chars + 1))
