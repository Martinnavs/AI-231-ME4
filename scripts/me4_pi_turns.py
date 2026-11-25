"""Ten timed text turns against a running agent server (run on the Pi after `up.sh text`).

    PYTHONPATH=src python3 scripts/me4_pi_turns.py --out ~/pi-check/turns2.json

Each turn records seconds to the first answer text, total seconds, the answer, and whether every
`expect` string occurs in it (case-insensitive) and no `forbid` string does. The expectations come
from the golden-v2 rows and the 44-file corpus; "ok" is a hint, read the answers yourself too.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from text_chat.client import AgentClient

# (question, expect any of, forbid, why)
TURNS: list[tuple[str, list[str], list[str], str]] = [
    (
        "How quickly do I need to notify GSA IT after spotting a suspected security incident?",
        ["1 hour", "one hour"],
        [],
        "single fact from a .md file",
    ),
    (
        "Where does my base schedule live?",
        ["hrlinks"],
        [],
        "the first Pi run missed this on the old path",
    ),
    (
        "What was the estimated United States population in 2024 in the census file?",
        ["340,110,988", "340110988"],
        [],
        "a number from a spreadsheet",
    ),
    ("What's the GS-7 step 3 salary?", ["45,980", "45980"], [], "a table row, wrong-row lure"),
    ("Which file did that come from?", ["gs_salary"], [], "follow-up on sources"),
    (
        "What is the weather today?",
        ["cebu", "°", "degrees", "rain", "cloud", "clear", "drizzle", "sunny"],
        ["as per"],
        "must NOT come from documents",
    ),
    (
        "What is the GSA TTS policy on bringing pets to the office?",
        ["documents don't say", "documents do not say"],
        [],
        "not in the files: must say so",
    ),
    (
        "What was the average home electricity price in Guam?",
        ["documents don't say", "documents do not say"],
        [],
        "decoy: Guam is not in the table",
    ),
    ("Set a timer for five minutes.", [], ["as per"], "a command must not hit the documents"),
    (
        "How many hours does a full-time TTS employee work in a two-week pay period?",
        ["80"],
        [],
        "single fact, second document question",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8781")
    ap.add_argument("--out", default="turns.json")
    args = ap.parse_args()
    token = Path.home().joinpath(".config/lan_speech_token").read_text().strip()
    client = AgentClient(args.url, token)
    history: list[dict[str, str]] = []
    results = []
    for question, expect, forbid, why in TURNS:
        t0 = time.time()
        first: float | None = None
        parts: list[str] = []
        for frame in client.frames([*history, {"role": "user", "content": question}]):
            if frame.get("text"):
                first = first if first is not None else time.time() - t0
                parts.append(frame["text"])
            elif frame.get("event") == "tool":
                parts.append(f"[tool:{frame.get('name')}]")
            elif frame.get("error"):
                parts.append(f"[error:{frame['error']}]")
        answer = " ".join(parts)
        low = answer.lower()
        ok = (not expect or any(e.lower() in low for e in expect)) and not any(
            f.lower() in low for f in forbid
        )
        row = {
            "q": question,
            "why": why,
            "first_s": round(first, 2) if first else None,
            "total_s": round(time.time() - t0, 2),
            "answer": answer,
            "ok_hint": ok,
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    print(f"{sum(r['ok_hint'] for r in results)}/{len(results)} ok_hint; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
