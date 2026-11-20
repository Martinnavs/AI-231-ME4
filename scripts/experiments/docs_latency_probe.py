"""Measure docs-server search latency over stdio while a big PDF is being indexed.

uv run --no-sync python scripts/experiments/docs_latency_probe.py --tmp <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

PDF = Path(
    "/mnt/jfs_hpc/home/anthony.martin.navarez/rag-testsets/ipo-2026/raw/irs/"
    "i1040gi_form1040_instructions.pdf"
)
SEED = {
    "alpha.md": "# Alpha\n\nzebra stripes are black and white.\n",
    "beta.md": "# Beta\n\nplatypus is a venomous mammal.\n",
    "gamma.md": "# Gamma\n\nkiwi is a flightless bird from New Zealand.\n",
}


async def run(tmp: Path, duration: int, copy_at: int, pdf_query: str) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    for name, body in SEED.items():
        (tmp / name).write_text(body)
    from mcp.client.stdio import get_default_environment

    env = {
        **get_default_environment(),
        **{k: v for k, v in os.environ.items() if k.startswith("VOICE_AGENT_")},
    }
    transport = StdioTransport(
        sys.executable,
        ["-m", "voice_agent.mcp_servers.docs", "--corpus", str(tmp), "--retriever", "fts5"],
        env=env,
    )
    lat: list[float] = []
    pdf_ready_at: float | None = None
    copied = False
    async with Client(transport) as client:
        start = time.monotonic()
        for sec in range(duration):
            target = start + sec
            await asyncio.sleep(max(0.0, target - time.monotonic()))
            t = time.monotonic() - start
            if not copied and t >= copy_at:
                shutil.copy(PDF, tmp / PDF.name)
                copied = True
                copy_t = t
            t0 = time.monotonic()
            await client.call_tool("search_docs", {"query": "zebra stripes"})
            ms = (time.monotonic() - t0) * 1000
            lat.append(ms)
            st = (await client.call_tool("docs_status", {})).structured_content or {}
            print(f"t={t:5.1f} ms={ms:7.1f} indexing={st.get('indexing')}", flush=True)
            if copied and pdf_ready_at is None:
                hits = await client.call_tool("search_docs", {"query": pdf_query})
                got = (hits.structured_content or {}).get("result", [])
                if any(h["source"] == PDF.name for h in got):
                    pdf_ready_at = time.monotonic() - start - copy_t
    lat.sort()
    p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
    print(f"calls={len(lat)} max_ms={lat[-1]:.1f} p95_ms={p95:.1f}")
    print(
        f"pdf_searchable_after_s={pdf_ready_at if pdf_ready_at is None else round(pdf_ready_at, 1)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", type=Path, required=True)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--copy-at", type=int, default=5)
    ap.add_argument("--pdf-query", default="standard deduction filing status")
    args = ap.parse_args()
    asyncio.run(run(args.tmp, args.duration, args.copy_at, args.pdf_query))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
