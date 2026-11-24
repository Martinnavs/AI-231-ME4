"""Tempted-oracle false accepts through the OLD path (db.search k=8 + doc_answer.answer).

Same oracle and rows as me4_ask_eval.py, so the two numbers are comparable (ticket 209).
Run in the pipeline venv.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from me4_ask_eval import RAG, TemptedOracle, _rows
from voice_pipeline.doc_answer import answer

from voice_agent.rag.dbindex import DbIndex


async def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--db44", "--raw44", "--dbB", "--rawB"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()
    dbs = {
        "A": DbIndex(Path(args.db44), raw_root=Path(args.raw44)),
        "B": DbIndex(Path(args.dbB), raw_root=Path(args.rawB)),
    }
    sets = [
        ("dev", "heldout/dev.jsonl", "A"),
        ("golden", "ipo-2026/golden.jsonl", "A"),
        ("A", "ipo-2026/me4-frozen/q.jsonl", "A"),
        ("B", "ipo-2026-b/q2.jsonl", "B"),
    ]
    out: dict[str, Any] = {}
    for name, rel, which in sets:
        db = dbs[which]
        fa: list[str] = []
        n = 0
        for row in _rows(name, RAG / rel):
            if not row["unanswerable"]:
                continue
            n += 1
            oracle = TemptedOracle(row)

            async def request(system: str, content: str, o: TemptedOracle = oracle) -> str:
                r = json.loads(await o(system, content))
                if r.get("status") != "answer":
                    return "NONE"
                return f"ANSWER: {r['answer']}\nQUOTE: {r['quotes'][0]}"

            got = await answer(request, db.search(row["q"], 8), row["q"])
            if got.status == "accepted":
                fa.append(row["id"])
        out[name] = {"unanswerable": n, "false_accepts_old_path": fa}
    nd = [x for x in (RAG / "heldout/nondoc25.txt").read_text().splitlines() if x.strip()]
    fa = []
    for i, q in enumerate(nd):
        row = {
            "set": "nondoc25",
            "id": f"nondoc-{i + 1:02d}",
            "q": q,
            "facts": [],
            "unanswerable": True,
            "web": None,
            "expected": "",
            "accept": [],
        }
        oracle = TemptedOracle(row)

        async def request2(system: str, content: str, o: TemptedOracle = oracle) -> str:
            r = json.loads(await o(system, content))
            if r.get("status") != "answer":
                return "NONE"
            return f"ANSWER: {r['answer']}\nQUOTE: {r['quotes'][0]}"

        got = await answer(request2, dbs["A"].search(q, 8), q)
        if got.status == "accepted":
            fa.append(row["id"])
    out["nondoc25"] = {"unanswerable": len(nd), "false_accepts_old_path": fa}
    for k, v in out.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
