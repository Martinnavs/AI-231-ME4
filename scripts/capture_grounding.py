"""Save real Gemini grounded responses as test fixtures (tests/fixtures/grounding/<id>.json).

    .venv/bin/python scripts/capture_grounding.py nvidia_ceo="Who's the CEO of Nvidia?" ... [--yes]

Each case is one live request, billed as tokens (about $0.0003) plus, when Gemini searches, one of
the month's free search queries. Without --yes nothing is sent: the script prints the case count
and the estimate. `--legacy-prompt` sends the prompt from before the search rule (to capture the
answer from memory the rule exists for); `--force` adds the retry instruction. The key is read by
`gemini_grounding.load_key`, sent only as a header, and a file that contains it is not written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from voice_agent.mcp_servers import gemini_grounding as g

OUT = Path(__file__).resolve().parents[1] / "tests/fixtures/grounding"
LEGACY_SYSTEM = (  # the prompt before 2026-11-14, kept to reproduce an answer from memory
    "You are a voice assistant. Today is {today}. Your answer is spoken aloud: at most two short "
    "sentences, plain words, no lists, no markdown. Search the web before answering whenever the "
    "answer could have changed since your training: office holders, sports results, prices and "
    "rates, records, rankings, events described as latest, last, next, current or new. Do not "
    "search for settled facts."
)
TOKEN_COST = 0.0003  # dollars per call, from the ledger's median of 138 tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="+", help='id="question" (the id names the file)')
    parser.add_argument("--legacy-prompt", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--place", default=g.DEFAULT_PLACE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--yes", action="store_true", help="send the requests (otherwise a dry run)"
    )
    args = parser.parse_args(argv)
    cases = [c.split("=", 1) for c in args.cases]
    if any(len(c) != 2 for c in cases):
        parser.error('each case is id="question"')
    if not args.yes:
        print(f"{len(cases)} requests, about ${len(cases) * TOKEN_COST:.4f} plus free searches; "
              "nothing sent (add --yes)")  # fmt: skip
        return 0
    key = g.load_key()
    if not key:
        print("no Gemini key found")
        return 1
    today = f"{datetime.now(ZoneInfo(g.TIMEZONE)):%A, %B %-d, %Y}"
    args.out.mkdir(parents=True, exist_ok=True)
    ledger = g.Ledger(g.LEDGER)
    with httpx.Client(timeout=15, transport=httpx.HTTPTransport(local_address="0.0.0.0")) as http:
        for case_id, query in cases:
            body = g.request_body(query, today, args.place, force=args.force)
            if args.legacy_prompt:
                body["systemInstruction"]["parts"][0]["text"] = LEGACY_SYSTEM.format(today=today)
            reply = http.post(g.GEMINI_URL, json=body, headers={"x-goog-api-key": key})
            reply.raise_for_status()
            record = {
                "id": case_id,
                "query": query,
                "system": body["systemInstruction"]["parts"][0]["text"],
                "force": args.force,
                "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "response": reply.json(),
            }
            text = json.dumps(record, indent=2, ensure_ascii=False)
            if key in text:  # the response never carries it; refuse rather than assume
                print(f"{case_id}: the key is in the output, not written")
                return 1
            (args.out / f"{case_id}.json").write_text(text + "\n")
            parsed = g.parse(record["response"])
            ledger.add(
                g.Grounded(parsed[0], parsed[1], parsed[2], parsed[3], 0.0), query, args.force
            )
            print(f"{case_id}: {parsed[1]} searches, {len(parsed[0])} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
