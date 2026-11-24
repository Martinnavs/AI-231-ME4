"""Drive AgentServer.handle with held-out questions; score per turn."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from me4_scoring import (
    load_questions,
    score_question,
    summarize,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER = REPO / "out" / "me4_live" / "spend.json"


class _FakeAgent:
    """Records agent reach; returns canned text."""

    def __init__(self) -> None:
        self.reached = False
        self.tap = SimpleNamespace(usage={"requests": 0, "prompt": 0, "completion": 0, "cached": 0})

    async def ask(
        self, messages: Any, *, reset: bool, on_text: Any = None, **_: Any
    ) -> dict[str, Any]:
        self.reached = True
        text = "AGENT: I cannot answer that."
        if on_text:
            await on_text(text)
        return {"status": "ok", "text": text, "cancelled": False}

    def hide(self, names: frozenset[str]) -> None:
        pass


class _OracleRequest:
    """Offline oracle that checks passages before answering."""

    _TAG_RE = re.compile(r"<untrusted_document[^>]*>(.+?)</untrusted_document>", re.DOTALL)

    def __init__(self, q: dict[str, Any]) -> None:
        self._q = q

    async def __call__(self, system: str, content: str) -> str:
        facts = self._q.get("facts", [])
        if not facts:
            return "NONE"
        if not content:
            return "NONE"
        raw_texts = self._TAG_RE.findall(content)
        passages = [t.strip() for t in raw_texts if t.strip()]
        if not passages:
            return "NONE"
        for fact in facts:
            contains = fact.get("contains", "")
            if not contains:
                continue
            for pt in passages:
                if contains.lower() in pt.lower():
                    return self._build_answer(pt)
        return "NONE"

    def _build_answer(self, passage: str) -> str:
        accept = self._q.get("accept", [])
        answer = self._q.get("answer", "")
        target = accept[0] if accept else answer
        idx = passage.lower().find(target.lower())
        if idx < 0:
            target = ""
            idx = 0
        words = passage.split()
        start = 0
        for i, _w in enumerate(words):
            if idx <= sum(len(p) + 1 for p in words[:i]):
                start = i
                break
        wcount = 3
        end = min(start + wcount, len(words))
        quote = " ".join(words[start:end])
        return f"ANSWER: {answer}\nQUOTE: {quote}"


class _Live:
    """Real doc-answer calls (one Gemini request per question) under a hard spend cap."""

    def __init__(self, request: Any, budget: Any, run_id: str) -> None:
        self._request = request
        self._budget = budget
        self._run_id = run_id
        self.last_reply = ""
        self.stopped = False

    def wrap(self, question: str) -> Any:
        from voice_pipeline.dryrun import BudgetStop

        async def call(system: str, content: str) -> str:
            est_in = (len(system) + len(content)) // 4
            try:
                self._budget.check(
                    {"prompt": est_in, "completion": 300, "requests": 1, "searches": 0}
                )
            except BudgetStop:
                self.stopped = True
                self.last_reply = ""
                return "NONE"
            self.last_reply = await self._request(system, content)
            usage = {
                "prompt": est_in,
                "completion": len(self.last_reply) // 4,
                "requests": 1,
                "searches": 0,
            }
            self._budget.commit(self._run_id, usage)
            return self.last_reply

        return call


async def _run_replay(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    old_rows = report.get("rows", [])
    q_path = Path(report.get("questions_path", ""))
    qs = load_questions(q_path) if q_path and q_path.exists() else []
    qmap = {q["id"]: q for q in qs}
    scored = []
    for row in old_rows:
        q = qmap.get(row["id"], row)
        if q is None:
            q = row
        scored.append(score_question(q, row))
    return {"mode": "replay", "summary": summarize(scored), "rows": scored}


async def _run_offline(
    corpus: Path,
    questions_path: Path,
    *,
    router_script: str = "",
    db: Path | None = None,
    live: Any = None,
) -> dict[str, Any]:
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema
    from voice_pipeline import agent_tools, question
    from voice_pipeline.agent_server import AgentServer
    from voice_pipeline.bench_fakes import (
        RUN_STEM,
        _FakeRouter,
    )
    from voice_pipeline.benchmark_cases import RouterScript
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    if db:
        from voice_agent.rag.dbindex import DbIndex

        index = DbIndex(db)
    else:
        from voice_agent.rag.backends import make_retriever_factory, resolve_mode
        from voice_agent.rag.docindex import DocIndex

        # the retriever the docs server serves (fts5), not DocIndex's default python BM25
        factory = make_retriever_factory(
            mode=resolve_mode(None, os.environ), cache_dir=corpus / ".rag_cache"
        )
        index = DocIndex(corpus, retriever_factory=factory)
        index.refresh()

    questions = load_questions(questions_path)
    workdir = corpus / ".me4_eval_work"
    workdir.mkdir(exist_ok=True)
    log_path = workdir / f"{RUN_STEM}-entry.jsonl"
    ledger_path = workdir / "ledger.jsonl"
    ledger_path.write_text("", encoding="utf-8")

    record: list[dict[str, Any]] = []
    prev_chain = agent_tools.CHAIN
    prev_scratchpad = agent_tools.SCRATCHPAD
    prev_question = question.current
    agent_tools.CHAIN = None
    agent_tools.SCRATCHPAD = None
    agent_tools.reset_turn_lookups()

    try:
        captured: list[list[dict[str, str]]] = []
        captured_queries: list[str] = []

        async def _search_handler(params: Any) -> None:
            result = await asyncio.to_thread(
                index.search,
                params.arguments["query"][:2000],
                params.arguments.get("top_k", 8),
            )
            captured_queries.append(str(params.arguments["query"]))
            captured.append([{"source": r["source"], "text": r["text"]} for r in result])
            await params.result_callback(result)

        search_schema = FunctionSchema(
            name="search_docs",
            description="Search docs",
            properties={"query": {"type": "string"}, "top_k": {"type": "integer"}},
            required=["query"],
            handler=_search_handler,
        )
        agent_tools._recorded(search_schema, record)

        async def _status_handler(params: Any) -> None:
            payload = await asyncio.to_thread(_build_status)
            await params.result_callback(payload)

        def _build_status() -> dict[str, Any]:
            files = [
                {
                    "name": rec.name,
                    "status": rec.status,
                    "file_date": "",
                    "mtime": rec.mtime,
                    "size": rec.size,
                    **({"note": rec.error} if rec.error else {}),
                }
                for rec in index.records()
            ]
            failures = list(index.failures())
            active = index.retriever
            status_fn = getattr(active, "status", None)
            retriever_status = str(status_fn()) if callable(status_fn) else "bm25"
            return {
                "folder": str(corpus),
                "files": files,
                "failures": failures,
                "retriever": retriever_status,
                "indexing": False,
                "ready": True,
                "parsing": index.parsing(),
            }

        status_schema = FunctionSchema(
            name="docs_status",
            description="Docs status",
            properties={},
            required=[],
            handler=_status_handler,
        )
        agent_tools._recorded(status_schema, record)

        tools = ToolsSchema(standard_tools=[search_schema, status_schema])

        if router_script:
            router = _FakeRouter(RouterScript(lines=router_script))
        else:
            router = _FakeRouter(RouterScript(reason="normal"))

        rows: list[dict[str, Any]] = []
        for q in questions:
            captured.clear()
            captured_queries.clear()
            req: Any = _OracleRequest(q)
            if live is not None:
                req = live.wrap(q["q"])
            agent = _FakeAgent()
            server = AgentServer(
                agent,
                record,
                log_path,
                tools,
                router,
                doc_request=req,
            )
            server.ledger = ledger_path
            server.stamp = lambda: "[Now: test]"

            t_start = time.perf_counter()
            ws_server = await serve(server.handle, "127.0.0.1", 0)
            port = next(iter(ws_server.sockets)).getsockname()[1]
            try:
                async with connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(
                        json.dumps(
                            {
                                "messages": [{"role": "user", "content": q["q"]}],
                            }
                        )
                    )
                    while True:
                        frame = json.loads(await ws.recv())
                        if "done" in frame or "error" in frame:
                            break
            finally:
                ws_server.close()
                await ws_server.wait_closed()

            if server.checks:
                await asyncio.gather(*server.checks, return_exceptions=True)

            turn_row = json.loads(log_path.read_text().splitlines()[-1])
            turn_row["reached_agent"] = agent.reached
            turn_row["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            turn_row["passages"] = captured[0] if captured else []
            turn_row["search_query"] = captured_queries[0] if captured_queries else ""
            if live is not None:
                turn_row["reply"] = live.last_reply[:400]
                if live.stopped:
                    rows.append(turn_row)
                    break
            rows.append(turn_row)
            log_path.write_text("", encoding="utf-8")

        scored = [score_question(q, row) for q, row in zip(questions, rows, strict=False)]
        for srow, trow in zip(scored, rows, strict=False):
            srow["search_query"] = trow.get("search_query", "")
        summary = summarize(scored)
        report = {
            "mode": "offline",
            "corpus": str(corpus),
            "questions_path": str(questions_path),
            "summary": summary,
            "rows": scored,
        }
        return report
    finally:
        agent_tools.CHAIN = prev_chain
        agent_tools.SCRATCHPAD = prev_scratchpad
        question.current = prev_question
        if db:
            index.close()


def _run_live_main(args: argparse.Namespace) -> int:
    if not args.i_have_approval_to_spend:
        print("error: live mode requires --i-have-approval-to-spend")
        return 2
    if not args.ledger.exists():
        print("error: ledger file not found; create it or pass --ledger")
        return 2
    if not args.corpus and not args.db:
        print("error: --corpus or --db required")
        return 2
    import time as _time

    from voice_pipeline import doc_answer
    from voice_pipeline.dryrun import Budget

    async def go() -> dict[str, Any]:
        request = await doc_answer.gemini_request()
        budget = Budget(args.max_usd, args.ledger)
        live = _Live(request, budget, "me4-entry-" + _time.strftime("%Y%m%d-%H%M%S"))
        report = await _run_offline(
            args.corpus or Path("."),
            args.questions,
            router_script=args.router_script,
            db=args.db,
            live=live,
        )
        report["mode"] = "live"
        report["spend"] = {"usd": round(budget.spent, 4), "cap": args.max_usd}
        if live.stopped:
            report["stopped"] = "spend cap"
        return report

    report = asyncio.run(go())
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Drive AgentServer.handle with held-out questions.")
    ap.add_argument("--mode", choices=("offline", "live", "replay"), default="offline")
    ap.add_argument("--corpus", type=Path, default=None, help="Corpus folder (offline)")
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--db", type=Path, default=None, help="Pre-built ingest DB")
    ap.add_argument("--router-script", default="", help="Router script lines")
    ap.add_argument("--max-usd", type=float, default=0.02)
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--live-agent", action="store_true")
    ap.add_argument("--i-have-approval-to-spend", action="store_true")
    args = ap.parse_args(argv)

    if args.mode == "replay":
        report = asyncio.run(_run_replay(args.report))
    elif args.mode == "live":
        return _run_live_main(args)
    else:
        if not args.corpus and not args.db:
            print("error: --corpus or --db required for offline mode")
            return 2
        report = asyncio.run(
            _run_offline(
                args.corpus or Path("."),
                args.questions,
                router_script=args.router_script,
                db=args.db,
            )
        )

    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
