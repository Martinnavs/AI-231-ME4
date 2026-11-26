"""Offline research-skill smoke demo; synthetic sources, no model or web API calls."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from voice_agent.research.jobs import ResearchJobs
from voice_agent.research.runner import Runner, Source


async def demo(output: Path) -> None:
    replies = iter(
        [
            {"queries": ["sample service announced 2027 rate"]},
            {
                "claims": [
                    {
                        "text": "The sample service announced a 2027 rate of 8 units.",
                        "sources": ["S1"],
                    }
                ],
                "gaps": [],
                "queries": [],
            },
        ]
    )

    async def model(system: str, content: str) -> str:
        return json.dumps(next(replies))

    async def search(query: str) -> list[Source]:
        return [
            Source(
                "Synthetic announcement",
                "https://example.org/sample-announcement",
                "Synthetic fixture: the service announced 8 units for calendar 2027.",
                "2026-11-26",
            )
        ]

    skill = Path(__file__).resolve().parents[1] / "skills/deep-research/SKILL.md"
    jobs = ResearchJobs(output, Runner(skill, model, search))
    job = jobs.start("What has the sample service announced for next year?")
    task = jobs.tasks[job.id]
    await task
    print(json.dumps({"job_id": job.id, "status": job.status, "folder": job.folder}))
    print((Path(job.folder) / "report.md").read_text())
    await jobs.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="dedicated demo state directory")
    args = parser.parse_args()
    asyncio.run(demo(args.out))


if __name__ == "__main__":
    main()
