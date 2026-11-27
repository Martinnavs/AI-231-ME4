"""Paid check (about 34 grounded Gemini lookups plus their directness calls): what would the sufficiency
supplement say? Takes the web questions the verdict produced on the dev split (gold mode report), asks
the real `web_answer` tool in-process (grounded Gemini backend, the production directness check and
attribution), and writes question, web question, gold document answer, reply and timing for hand reading.
Key from $GEMINI_API_KEY or ~/.secrets/google-ai-studio/api-key.txt, never printed. Run from the repo root:
    PYTHONPATH=src .venv/bin/python scripts/supplement_web_check.py out/sufficiency_eval/gemini-dev-gold.json
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from fastmcp import Client

from voice_agent.mcp_servers import gemini_grounding as g
from voice_agent.mcp_servers import tavily_search as t

MAX_LOOKUPS = 40  # hard stop, well under the live-spend cap


async def main(report: Path, out: Path) -> None:
    key = g.load_key()
    if not key:
        sys.exit("no Gemini key ($GEMINI_API_KEY or ~/.secrets/google-ai-studio/api-key.txt)")
    rows = [
        r
        for r in json.loads(report.read_text())["rows"]
        if r["verdict"] in ("stale", "missing_half") and r["web_question"]
    ][:MAX_LOOKUPS]
    ledger = g.Ledger(None)
    results = []
    server = t.build_server(
        None, key=None, gemini_key=key, ping_every_s=None, grounding=True, ledger=ledger
    )
    async with Client(server) as client:
        for i, r in enumerate(rows, 1):
            started = time.perf_counter()
            try:
                reply = await client.call_tool("web_answer", {"query": r["web_question"]})
                text = reply.structured_content["result"]
            except Exception as exc:  # noqa: BLE001
                text = f"ERROR {type(exc).__name__}"
            results.append(
                {
                    "id": r["id"],
                    "kind": r["kind"],
                    "question": r["question"],
                    "web_question": r["web_question"],
                    "doc_answer": r.get("doc_spoken"),
                    "reply": text,
                    "hedge": g.HEDGE in text,
                    "seconds": round(time.perf_counter() - started, 2),
                }
            )
            print(f"[{i}/{len(rows)}] {r['id']} hedge={results[-1]['hedge']} {results[-1]['seconds']}s", file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}; hedges {sum(x['hedge'] for x in results)}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), Path("out/sufficiency_eval/supplement-web-check.json")))
