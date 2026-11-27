"""Live check of `ask_documents` (about $0.04): dev rows, honesty rows, latency vs size.

Run in the pipeline venv from a checkout where `read_key("google-ai-studio")` works (the keyring
patch is applied to a scratch worktree only, never committed). Needs --i-have-approval-to-spend.
Every call is counted against --max-usd and a ledger; the run stops before it would exceed it.
Prices: Gemini Flash-Lite, $0.25/M input, $1.50/M output (the figures in ticket 209).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from me4_ask_eval import RAG, _rows
from voice_pipeline.agent import MODEL, read_key, warm_http_options
from voice_pipeline.ask_documents import REPLY_SCHEMA, ask_documents
from voice_pipeline.fast_path import turn_now

from voice_agent.rag.dbindex import DbIndex

PRICE_IN, PRICE_OUT = 0.25e-6, 1.5e-6
HONESTY = ["d-14", "d-15", "d-16", "unans-12", "unans-7", "u-1", "u-2", "off-2"]


class Spent(Exception):
    pass


class Meter:
    def __init__(self, cap: float, ledger: Path) -> None:
        self.cap, self.ledger = cap, ledger
        self.usd = self.tin = self.tout = self.calls = 0
        self.log: list[dict[str, Any]] = []

    def add(self, tin: int, tout: int, secs: float, note: str) -> None:
        usd = tin * PRICE_IN + tout * PRICE_OUT
        self.usd += usd
        self.tin += tin
        self.tout += tout
        self.calls += 1
        self.log.append({"note": note, "in": tin, "out": tout, "s": round(secs, 2), "usd": usd})

    def check(self, est_in: int) -> None:
        if self.usd + est_in * PRICE_IN + 200 * PRICE_OUT > self.cap:
            raise Spent(f"cap {self.cap} would be exceeded after ${self.usd:.4f}")

    def commit(self) -> None:
        data = (
            json.loads(self.ledger.read_text()) if self.ledger.exists() else {"usd": 0, "runs": []}
        )
        data["usd"] = round(data["usd"] + self.usd, 6)
        data["runs"].append(
            {
                "id": "me4-live-209-210",
                "usd": round(self.usd, 6),
                "prompt": self.tin,
                "completion": self.tout,
                "requests": self.calls,
                "searches": 0,
            }
        )
        self.ledger.write_text(json.dumps(data, indent=1) + "\n")


def make_request(meter: Meter, tag: dict[str, str]) -> Any:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=read_key("google-ai-studio"), http_options=warm_http_options(None, False)
    )
    thinking = types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)

    async def req(system: str, content: str) -> str:
        meter.check(len(content) // 3)
        config = types.GenerateContentConfig(
            system_instruction=system,
            thinking_config=thinking,
            response_mime_type="application/json",
            response_json_schema=REPLY_SCHEMA,
        )
        t0 = time.perf_counter()
        reply = await client.aio.models.generate_content(
            model=MODEL, contents=content, config=config
        )
        usage = reply.usage_metadata
        meter.add(
            int(usage.prompt_token_count or 0),
            int(usage.candidates_token_count or 0),
            time.perf_counter() - t0,
            tag["id"],
        )
        return reply.text or ""

    return req


async def run_row(row: dict[str, Any], db: DbIndex, meter: Meter) -> dict[str, Any]:
    async def context(q: str) -> list[dict[str, Any]]:
        return db.file_context(q)

    async def web(q: str) -> str | None:  # no live web leg in this check
        return None

    tag = {"id": row["id"]}
    start = time.perf_counter()
    r = await ask_documents(
        row["q"],
        context=context,
        request=make_request(meter, tag),
        web=web,
        budget_s=20.0,
        allow_fallback=False,
        file_names=[r.name for r in db.records()],
        now=turn_now(),  # ticket 268: the turn's clock, in the demo's zone
    )
    want = row["accept"] + ([row["expected"]] if row["expected"] else [])
    spoken = r.spoken or ""
    return {
        "id": row["id"],
        "unanswerable": row["unanswerable"],
        "status": r.status,
        "spoken": spoken,
        "rejected": r.rejected or {},
        "web_question": r.web_question,
        "seconds": round(time.perf_counter() - start, 2),
        "correct_doc_half": (
            not row["unanswerable"]
            and r.status == "accepted"
            and (not want or any(w.lower() in spoken.lower() for w in want[:-1] or want))
        ),
        "false_accept": row["unanswerable"] and r.status == "accepted",
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db44", required=True)
    ap.add_argument("--raw44", required=True)
    ap.add_argument("--dbB", required=True)
    ap.add_argument("--rawB", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-have-approval-to-spend", action="store_true")
    args = ap.parse_args()
    if not args.i_have_approval_to_spend:
        raise SystemExit("refusing to spend: pass --i-have-approval-to-spend")
    meter = Meter(args.max_usd, Path(args.ledger))
    a_db = DbIndex(Path(args.db44), raw_root=Path(args.raw44))
    b_db = DbIndex(Path(args.dbB), raw_root=Path(args.rawB))
    results: list[dict[str, Any]] = []
    stopped = ""
    try:
        for row in _rows("dev", RAG / "heldout/dev.jsonl"):
            results.append(await run_row(row, a_db, meter))
        extra = {
            "golden": ("ipo-2026/golden.jsonl", a_db),
            "A": ("ipo-2026/me4-frozen/q.jsonl", a_db),
            "B": ("ipo-2026-b/q2.jsonl", b_db),
        }
        seen = {r["id"] for r in results}
        for name, (rel, db) in extra.items():
            for row in _rows(name, RAG / rel):
                if row["id"] in HONESTY and row["id"] not in seen:
                    seen.add(row["id"])
                    results.append(await run_row(row, db, meter))
    except Spent as exc:
        stopped = str(exc)
    finally:
        meter.commit()
        a_db.close()
        b_db.close()
    secs = [m["s"] for m in meter.log]
    summary = {
        "calls": meter.calls,
        "usd": round(meter.usd, 5),
        "stopped": stopped,
        "median_s": round(statistics.median(secs), 2) if secs else None,
        "max_s": max(secs) if secs else None,
        "tokens_in_median": int(statistics.median([m["in"] for m in meter.log])) if secs else None,
        "dev_accepted": [
            r["id"] for r in results if r["id"].startswith("d-") and r["status"] == "accepted"
        ],
        "false_accepts": [r["id"] for r in results if r["false_accept"]],
    }
    Path(args.out).write_text(
        json.dumps({"summary": summary, "rows": results, "calls": meter.log}, indent=1) + "\n"
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
