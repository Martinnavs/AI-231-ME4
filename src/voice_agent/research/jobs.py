"""Optional background lifecycle over the existing JobTable in a dedicated directory."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path

from voice_agent.agent.jobs import CANCELLED, DONE, ERROR, RUNNING, Job, JobTable
from voice_agent.research.runner import Report, Runner


class ResearchJobs:
    """One process owns this store. Interrupted runs are lost, not auto-resumed."""

    def __init__(
        self, state_dir: Path, runner: Runner, *, max_active: int = 1, max_records: int = 100
    ) -> None:
        if (
            type(max_active) is not int
            or max_active < 1
            or type(max_records) is not int
            or max_records < 1
        ):
            raise ValueError("job limits must be positive integers")
        self.table = JobTable(state_dir)
        self.runner, self.max_active = runner, max_active
        self.max_records = max_records
        self.tasks: dict[str, asyncio.Task[None]] = {}

    def start(self, question: str, *, timezone: str = "Asia/Manila") -> Job:
        if len(self.table.jobs) >= self.max_records:
            raise ValueError("research record limit reached; archive the dedicated store")
        if sum(not t.done() for t in self.tasks.values()) >= self.max_active:
            raise ValueError("research queue is full")
        if not question.strip() or len(question) > self.runner.limits.question_chars:
            raise ValueError("invalid research question")
        job = self.table.create(question, "research")
        job.folder = str(self.table.path.parent / job.id)
        try:
            Path(job.folder).mkdir()
        except OSError:
            self.table.finish(job.id, ERROR, "Could not create research artifacts.")
            raise
        self.table.save()
        task = asyncio.create_task(self._work(job.id, timezone))
        self.tasks[job.id] = task
        task.add_done_callback(lambda _: self.tasks.pop(job.id, None))
        return job

    def status(self, job_id: str) -> Job:
        return self.table.jobs[job_id]

    def result(self, job_id: str) -> dict[str, object] | None:
        job = self.status(job_id)
        path = Path(job.folder) / "evidence.json"
        if not path.exists():
            return None
        value: dict[str, object] = json.loads(path.read_text())
        return value

    async def cancel(self, job_id: str) -> Job:
        job = self.status(job_id)
        task = self.tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if job.status == RUNNING:
            self.table.finish(job_id, CANCELLED, "Research was cancelled.")
        return job

    async def close(self) -> None:
        for job_id in list(self.tasks):
            await self.cancel(job_id)

    async def _work(self, job_id: str, timezone: str) -> None:
        job = self.status(job_id)
        folder = Path(job.folder)

        def persist(report: Report) -> None:
            for name, content in (
                ("evidence.json", json.dumps(asdict(report), ensure_ascii=False, indent=2)),
                ("report.md", report.markdown()),
            ):
                target = folder / name
                temp = target.with_suffix(target.suffix + ".tmp")
                temp.write_text(content)
                temp.replace(target)

        try:
            report = await self.runner.run(job.task, timezone=timezone, snapshot=persist)
        except asyncio.CancelledError:
            self.table.finish(job_id, CANCELLED, "Research was cancelled.")
            raise
        except Exception as exc:
            self.table.finish(job_id, ERROR, f"Research failed: {type(exc).__name__}.")
        else:
            status = DONE if report.status == "done" else ERROR
            self.table.finish(job_id, status, f"Research {report.status}.\nFiles: report.md")
        self.table.notify(job)
