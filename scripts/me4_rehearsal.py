"""Offline rehearsal of the document-answer path against a zip, over the real docs server.

cd pipeline && uv run --no-sync python ../scripts/me4_rehearsal.py --zip Z --corpus DIR \
    --questions Q.jsonl --mode offline --report OUT.json
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
from voice_pipeline import bench_fakes, doc_answer
from voice_pipeline.doc_route import NOT_FOUND_OFFER

KINDS = ("single", "multi", "table", "aggregate", "docweb", "unanswerable")
ABSTAIN_PHRASE = "documents don't say"
REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Question:
    id: str
    q: str
    kind: str
    gold_files: tuple[str, ...]
    gold_quote: str
    answer: str
    answer_regex: str


def load_questions(path: Path) -> list[Question]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["kind"] not in KINDS:
            raise ValueError(f"unknown kind {row['kind']!r} in question {row['id']!r}")
        questions.append(
            Question(
                row["id"],
                row["q"],
                row["kind"],
                tuple(row["gold_files"]),
                row["gold_quote"],
                row["answer"],
                row["answer_regex"],
            )
        )
    return questions


def gold_in_passages(gold_files: tuple[str, ...], passages: list[dict[str, Any]]) -> bool:
    sources = [str(p["source"]) for p in passages]
    return all(any(s == g or s.endswith("/" + g) for s in sources) for g in gold_files)


def answer_matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def abstained(text: str) -> bool:
    return ABSTAIN_PHRASE in text.lower().replace("\u2019", "'")


def spoken_text(result: doc_answer.DocAnswer) -> str:
    return (result.text or "") if result.status == "accepted" else NOT_FOUND_OFFER


def score_question(
    q: Question, passages: list[dict[str, Any]], result: doc_answer.DocAnswer
) -> dict[str, Any]:
    spoken = spoken_text(result)
    accepted = result.status == "accepted"
    found = gold_in_passages(q.gold_files, passages)
    answer_ok = accepted and answer_matches(q.answer_regex, spoken)
    if q.kind == "unanswerable":
        ok = abstained(spoken) and not accepted
    else:
        ok = found and accepted and answer_ok
    return {
        "id": q.id,
        "kind": q.kind,
        "gold_in_passages": found,
        "status": result.status,
        "answer_ok": answer_ok,
        "abstained": abstained(spoken),
        "failed": (result.rejected or {}).get("failed"),
        "pass": ok,
    }


def server_params(server_python: Path, corpus: Path, retriever: str) -> StdioServerParameters:
    env = {
        **get_default_environment(),
        **{k: v for k, v in os.environ.items() if k.startswith("VOICE_AGENT_")},
        "PYTHONPATH": str(REPO / "src"),
    }
    args = ["-m", "voice_agent.mcp_servers.docs", "--corpus", str(corpus), "--retriever", retriever]
    return StdioServerParameters(command=str(server_python), args=args, env=env)


async def poll_ready(session: ClientSession, timeout_s: float) -> dict[str, Any]:
    """Poll docs_status once a second; first_searchable_s is when a file first shows as ok."""
    t0 = time.monotonic()
    first: float | None = None
    status: dict[str, Any] = {}
    while time.monotonic() - t0 < timeout_s:
        status = (await session.call_tool("docs_status", {})).structured_content or {}
        now = round(time.monotonic() - t0, 1)
        if first is None and any(f["status"] == "ok" for f in status["files"]):
            first = now
        if status["ready"] and not status["indexing"]:
            return _ready_report(status, first, now)
        await asyncio.sleep(1.0)
    return _ready_report(status, first, None)


def _ready_report(
    status: dict[str, Any], first: float | None, ready: float | None
) -> dict[str, Any]:
    files = status.get("files", [])
    return {
        "first_searchable_s": first,
        "ready_s": ready,
        "files": len(files),
        "files_ok": sum(f["status"] == "ok" for f in files),
        "failures": status.get("failures", []),
    }


def oracle_request(q: Question) -> doc_answer.Request:
    """The offline model: it answers with the question's gold answer and gold quote."""
    reply = "NONE" if q.kind == "unanswerable" else f"ANSWER: {q.answer}\nQUOTE: {q.gold_quote}"
    return bench_fakes.fake_doc_request(reply)


ORACLE = "gold"


def ideal_request(
    q: Question, passages: list[dict[str, Any]], restate: bool = False
) -> doc_answer.Request:
    """An ideal model: quote a window of the RETRIEVED text around the gold answer."""
    if q.kind == "unanswerable":
        return oracle_request(q)
    pattern = re.compile(q.answer_regex)
    for p in passages:
        if not any(g in str(p.get("source", "")) for g in q.gold_files):
            continue
        text = str(p.get("text", ""))
        m = pattern.search(text)
        if m is None:
            continue
        words = text.split()
        before = len(text[: m.start()].split())
        start = max(0, before - 4)
        quote = " ".join(words[start : start + 11])
        value = m.group(0)
        return bench_fakes.fake_doc_request(f"ANSWER: {value}\nQUOTE: {quote}")
    return oracle_request(q)


async def run_question(
    session: ClientSession,
    q: Question,
    top_k: int | None = None,
    *,
    request=None,
    stamp=None,
    budget=None,
    run_id="",
) -> dict[str, Any]:
    args: dict[str, Any] = {"query": q.q}
    if top_k is not None:
        args["top_k"] = top_k
    t0 = time.monotonic()
    found = await session.call_tool("search_docs", args)
    t1 = time.monotonic()
    passages = doc_answer.passages_with_meta(
        (found.structured_content or {}).get("result", []),
    )
    if request is not None:
        result = await doc_answer.answer(request, passages, q.q, stamp)
    else:
        rq = (
            ideal_request(q, passages, restate=ORACLE == "restate")
            if ORACLE in ("ideal", "restate")
            else oracle_request(q)
        )
        result = await doc_answer.answer(rq, passages, q.q)
    t2 = time.monotonic()
    row = score_question(q, passages, result)
    row["search_ms"] = round((t1 - t0) * 1000, 1)
    row["answer_ms"] = round((t2 - t1) * 1000, 1)
    row["sources"] = [p["source"] for p in passages]
    row["passages"] = [{"source": p["source"], "text": p["text"]} for p in passages]
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind = {}
    for kind in KINDS:
        sub = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = {"n": len(sub), "pass": sum(r["pass"] for r in sub)}
    search = sorted(r["search_ms"] for r in rows)
    p95 = search[min(len(search) - 1, int(len(search) * 0.95))] if search else None
    return {
        "n": len(rows),
        "pass": sum(r["pass"] for r in rows),
        "by_kind": by_kind,
        "search_ms_p95": p95,
        "search_ms_max": search[-1] if search else None,
    }


def prepare_corpus(zip_path: Path, corpus: Path) -> None:
    """Drop the zip into the corpus folder; the docs server extracts it (WP1 path)."""
    corpus.mkdir(parents=True, exist_ok=True)
    shutil.copy(zip_path, corpus / zip_path.name)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    questions = load_questions(args.questions)
    prepare_corpus(args.zip, args.corpus)
    params = server_params(args.server_python, args.corpus, args.retriever)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        ready = await poll_ready(session, args.ready_timeout)
        if args.mode == "live":
            from voice_pipeline.dryrun import Budget, BudgetStop

            request = await doc_answer.gemini_request()
            budget = Budget(args.max_usd, args.ledger)
            stamp = f"[Now: {datetime.datetime.now():%A, %B %-d, %Y, %-I:%M %p}]"
            run_id = "me4-live-" + time.strftime("%Y%m%d-%H%M%S")
            rows: list[dict[str, Any]] = []
            stopped = None
            for q in questions:
                t0 = time.monotonic()
                a = {"query": q.q}
                if args.top_k is not None:
                    a["top_k"] = args.top_k
                found = await session.call_tool("search_docs", a)
                t1 = time.monotonic()
                passages = doc_answer.passages_with_meta(
                    (found.structured_content or {}).get("result", []),
                )
                est_in = (len(doc_answer.SYSTEM) + sum(len(p["text"]) for p in passages)) // 4
                try:
                    budget.check(
                        {
                            "prompt": est_in,
                            "completion": 300,
                            "requests": 1,
                            "searches": 0,
                        }
                    )
                except BudgetStop:
                    stopped = "spend cap"
                    break
                raw_reply = ""

                async def _wrap(system: str, content: str) -> str:
                    nonlocal raw_reply
                    raw_reply = await request(system, content)
                    return raw_reply

                result = await doc_answer.answer(_wrap, passages, q.q, stamp)
                t2 = time.monotonic()
                usage = {
                    "prompt": est_in,
                    "completion": len(raw_reply) // 4,
                    "requests": 1,
                    "searches": 0,
                }
                budget.commit(run_id, usage)
                row = score_question(q, passages, result)
                row["search_ms"] = round((t1 - t0) * 1000, 1)
                row["answer_ms"] = round((t2 - t1) * 1000, 1)
                row["sources"] = [p["source"] for p in passages]
                row["reply"] = raw_reply[:400]
                row["passages"] = [{"source": p["source"], "text": p["text"]} for p in passages]
                rows.append(row)
            report = {
                "mode": args.mode,
                "ready": ready,
                "summary": summarize(rows),
                "rows": rows,
            }
            if stopped:
                report["stopped"] = stopped
            report["spend"] = {
                "usd": round(budget.spent, 4),
                "cap": args.max_usd,
            }
            return report
        rows = [await run_question(session, q, args.top_k) for q in questions]
    return {
        "mode": args.mode,
        "ready": ready,
        "summary": summarize(rows),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--mode", choices=("offline", "live"), default="offline")
    ap.add_argument("--max-usd", type=float, default=0.05)
    ap.add_argument(
        "--ledger",
        type=Path,
        default=REPO / "out" / "me4_live" / "spend.json",
    )
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--server-python", type=Path, default=REPO / ".venv" / "bin" / "python")
    ap.add_argument("--retriever", default="fts5")
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--oracle", choices=("gold", "ideal", "restate"), default="gold")
    args = ap.parse_args(argv)
    global ORACLE
    ORACLE = args.oracle
    report = asyncio.run(run(args))
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
