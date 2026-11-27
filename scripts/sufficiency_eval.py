"""Offline evaluation harness for the speak-first sufficiency check (ticket 255).

Runs a labelled question set (default eval/sufficiency/implicit_compound.jsonl) against the
frozen release's two parts, in order, over the shared Qwen relay:

  1. the document lane: voice_pipeline.ask_documents over DbIndex.file_context retrieval,
     with the same cheap JSON model (Qwen via the relay: temperature 0, thinking off,
     JSON-only response format),
  2. the sufficiency verdict: voice_pipeline.sufficiency.verdict with the same Qwen request,
     source_year from the row's file card date when available (else None) and a fixed
     --today.

The verdict is scored against the row's label: a needs-web row (web_needed true) passes
when the verdict is not "sufficient" (stale / missing_half / wrong_question as
appropriate); a docs-only control row passes when the verdict is "sufficient". A row the
lane does not accept is recorded with its doc_status and still scored on the document
half of the gold ideal answer (the web-hedge sentences dropped), so the verdict is
measurable independently of lane failures; --gold-answer does that for every row (the
second mode).

Corpus-agnostic: no corpus words in this file; the data file carries them. The relay is
shared with other runs, so at most one call is in flight (strictly sequential, plus a lock
inside the transport), each call is retried once, and any failure that survives the retry
stops the run loudly. The key travels only in the Authorization header (the judge_eval
convention) and is never printed or logged.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = REPO_ROOT / "eval" / "sufficiency" / "implicit_compound.jsonl"
V2_JSONL = REPO_ROOT / "eval" / "sufficiency" / "implicit_compound_v2.jsonl"
SET_PATHS: dict[str, Path] = {"v1": DEFAULT_JSONL, "v2": V2_JSONL}
DEFAULT_DB = Path("/tmp/ev218/db44_new")
DEFAULT_RAW = Path("/tmp/ev218/raw44")
DEFAULT_TODAY = "2026-11-27"
DEFAULT_BASE_URL = "http://127.0.0.1:18080/v1"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
MAX_TOKENS = 700
RETRY_PAUSE_S = 1.0
RELAY_TIMEOUT_S = 120.0
DOC_BUDGET_S = 60.0
# the working example (offdoc) appends this to the document lane's system prompt; the
# verdict's own prompt already asks for JSON-only output
JSON_ONLY_SUFFIX = "\nReply with JSON only."

VERDICTS = ("sufficient", "stale", "missing_half", "wrong_question", "none")
TRIGGERED = frozenset({"stale", "missing_half", "wrong_question"})

_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


class Transport(Protocol):
    """Anything that can send one relay request and return the parsed response."""

    def post(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class RelayError(RuntimeError):
    """A transport-level failure that survived the one retry."""


class RunAborted(RuntimeError):
    """The row layer would swallow a transport failure; the harness must not."""


class Relay:
    """The OpenAI-compatible relay: at most one call in flight (the relay is shared with
    other runs), one retry on HTTP or connection errors, a loud RelayError otherwise. The
    key lives only in the request headers."""

    def __init__(
        self, base_url: str, api_key: str, model: str, timeout: float = RELAY_TIMEOUT_S
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self._lock = threading.Lock()
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def request(self, payload: Mapping[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )

    def post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        # the lock also serialises a call whose waiter already gave up (a timeout): the
        # next row's call waits for it instead of overlapping it on the shared relay
        with self._lock:
            for attempt in (1, 2):
                try:
                    with urllib.request.urlopen(
                        self.request(payload), timeout=self.timeout
                    ) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                except urllib.error.HTTPError as exc:
                    if attempt == 2:
                        raise RelayError(
                            f"the relay answered HTTP {exc.code} (retried once)"
                        ) from None
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    if attempt == 2:
                        raise RelayError(
                            f"the relay is unreachable ({type(exc).__name__})"
                        ) from None
                else:
                    if not isinstance(data, dict):
                        raise RelayError("the relay answered with a non-JSON object")
                    return data
                time.sleep(RETRY_PAUSE_S)
        raise AssertionError("unreachable: the last attempt always returns or raises")


def relay_payload(model: str, system: str, content: str) -> dict[str, Any]:
    """The shared Qwen JSON request: the same cheap JSON model and the same request for the
    document lane and the sufficiency verdict (the frozen release wires one doc JSON
    request through both)."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }


@dataclass
class TokenCounter:
    doc_prompt: int = 0
    doc_completion: int = 0
    verdict_prompt: int = 0
    verdict_completion: int = 0

    def add(self, stage: str, data: Mapping[str, Any]) -> None:
        usage = data.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        if stage == "doc":
            self.doc_prompt += prompt
            self.doc_completion += completion
        else:
            self.verdict_prompt += prompt
            self.verdict_completion += completion


async def _relay_call(
    transport: Transport,
    model: str,
    system: str,
    content: str,
    stage: str,
    tokens: TokenCounter,
    failures: list[str],
) -> str:
    payload = relay_payload(model, system, content)
    try:
        data = await asyncio.to_thread(transport.post, payload)
    except Exception as exc:
        failures.append(f"{stage}: {exc}")
        raise
    tokens.add(stage, data)
    choices = data.get("choices") or []
    message = choices[0].get("message") if choices else None
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str):
        failures.append(f"{stage}: the relay returned no message content")
        raise RelayError(f"{stage}: the relay returned no message content")
    return text


def make_requests(
    transport: Transport, model: str, tokens: TokenCounter, failures: list[str]
) -> tuple[Callable[..., Awaitable[str]], Callable[..., Awaitable[str]]]:
    """The two request callables (document lane, verdict): the same Qwen request, tagged
    only for the token counts. A transport failure is recorded (so the row layer, which
    swallows errors by design, cannot hide it) and re-raised. Both accept the optional
    `schema` of the frozen JsonRequest protocol: the relay is OpenAI-format, so JSON-only
    comes from the suffix and the schema itself is not sent (ticket 259: the R1-1 schema
    argument of the verdict call must not fail-open here)."""

    async def doc_request(system: str, content: str, schema: Any = None) -> str:
        return await _relay_call(
            transport, model, system + JSON_ONLY_SUFFIX, content, "doc", tokens, failures
        )

    async def verdict_request(system: str, content: str, schema: Any = None) -> str:
        return await _relay_call(transport, model, system, content, "verdict", tokens, failures)

    return doc_request, verdict_request


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise RuntimeError(f"{path}:{i + 1}: not a JSON line ({exc})") from None
        if not isinstance(row, dict):
            raise RuntimeError(f"{path}:{i + 1}: the row is not an object")
        for key in ("id", "kind", "question"):
            if not str(row.get(key) or "").strip():
                raise RuntimeError(f"{path}:{i + 1}: missing {key!r}")
        rows.append(row)
    if not rows:
        raise RuntimeError(f"{path}: no rows")
    return rows


def resolve_set_paths(set_name: str, jsonl_override: Path | None) -> list[Path]:
    """Which row-set file(s) to load: an explicit --jsonl wins, else --set picks the
    v1 set, the v2 set, or both (v1 followed by v2)."""
    if jsonl_override is not None:
        return [jsonl_override]
    if set_name == "both":
        return [DEFAULT_JSONL, V2_JSONL]
    return [SET_PATHS[set_name]]


def filter_split(rows: Sequence[Mapping[str, Any]], split_name: str) -> list[dict[str, Any]]:
    """Select the ticket-260 frozen split of a loaded row set. 'all' keeps every row;
    when a split is selected, a row missing the field is a loud error (both shipped
    sets carry the split)."""
    if split_name == "all":
        return list(rows)  # type: ignore[return-value]
    out: list[dict[str, Any]] = []
    for row in rows:
        split = row.get("split")
        if not isinstance(split, str) or not split:
            raise RuntimeError(
                f"row {row.get('id')!r} has no 'split' field; cannot select --split {split_name}"
            )
        if split == split_name:
            out.append(dict(row))
    return out


def _open_db(db: Path) -> sqlite3.Connection:
    if not db.exists():
        raise RuntimeError(f"ingest DB not found: {db}")
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def load_file_names(db: Path) -> list[str]:
    con = _open_db(db)
    try:
        return [
            r[0]
            for r in con.execute("select rel_key from files where status = 'ok' order by rel_key")
        ]
    finally:
        con.close()


def load_cards(db: Path) -> dict[str, dict[str, Any]]:
    con = _open_db(db)
    try:
        cards: dict[str, dict[str, Any]] = {}
        for rel_key, card_text in con.execute("select rel_key, card_text from file_cards"):
            try:
                data = json.loads(card_text)
            except ValueError:
                continue
            if isinstance(data, dict):
                cards[str(rel_key)] = data
        return cards
    finally:
        con.close()


def source_year_from_card(card: Mapping[str, Any] | None) -> str | None:
    """The source year the verdict sees: the file card's date_max year, else None."""
    date_max = str((card or {}).get("date_max") or "")
    m = _YEAR_RE.search(date_max)
    return m.group(1) if m else None


# the web-hedge markers of the labelled set: the data file writes the part of an ideal
# answer that only the web could have supplied with these cues. Schema conventions of the
# data file, not corpus content.
WEB_HEDGE_MARKERS = ("verify live", "(web)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def gold_doc_answer(ideal: str) -> str:
    """The document half of the gold ideal answer: the sentences that state what the
    documents hold, with the web-hedge sentences dropped. For a docs-only ideal answer
    (no hedge marker) this is the answer itself; when every sentence is a hedge the whole
    answer is kept, so a row is never scored on nothing."""
    text = (ideal or "").strip()
    if not text:
        return ""
    sentences = _SENTENCE_SPLIT.split(text)
    kept = [s for s in sentences if not any(m in s.lower() for m in WEB_HEDGE_MARKERS)]
    return " ".join(kept).strip() or text


@dataclass
class Harness:
    transport: Transport
    model: str
    context_fn: Callable[[str], Awaitable[list[dict[str, Any]]]]
    ask_documents_fn: Callable[..., Awaitable[Any]]
    verdict_fn: Callable[..., Awaitable[Any]]
    file_names: list[str]
    cards: dict[str, dict[str, Any]]
    today: str
    gold_answer: bool = False
    doc_budget_s: float = DOC_BUDGET_S

    def source_year_for(self, file_rel_key: str) -> str | None:
        return source_year_from_card(self.cards.get(file_rel_key))


async def run_row(
    h: Harness,
    row: Mapping[str, Any],
    doc_request: Callable[[str, str], Awaitable[str]],
    verdict_request: Callable[[str, str], Awaitable[str]],
    failures: list[str],
) -> dict[str, Any]:
    question = str(row["question"])
    # the gold document half: the verdict must see only what the documents state, never the
    # web fact the gold answer carries (that would make a needs-web row read as complete)
    gold = gold_doc_answer(str(row.get("ideal_answer") or ""))
    n_failures = len(failures)
    t0 = time.perf_counter()
    doc = await h.ask_documents_fn(
        question,
        context=h.context_fn,
        request=doc_request,
        file_names=h.file_names,
        budget_s=h.doc_budget_s,
        allow_fallback=True,
    )
    doc_s = time.perf_counter() - t0
    doc_status = str(getattr(doc, "status", "error"))
    spoken = str(doc.spoken or "").strip() or None
    if h.gold_answer:
        answer, answer_source = (gold, "gold") if gold else ("", "none")
    elif doc_status == "accepted" and spoken:
        answer, answer_source = spoken, "lane"
    elif gold:
        # a row the lane did not accept is still scored: the verdict sees the gold answer,
        # so the verdict is measurable independently of lane failures
        answer, answer_source = gold, "gold"
    else:
        answer, answer_source = "", "none"
    year = h.source_year_for(str(row.get("doc_file") or ""))
    t1 = time.perf_counter()
    verdict_obj = None
    if answer:
        verdict_obj = await h.verdict_fn(verdict_request, question, answer, year, h.today)
    verdict_s = time.perf_counter() - t1
    verdict = str(getattr(verdict_obj, "verdict", "")) if verdict_obj is not None else "none"
    if verdict not in TRIGGERED and verdict != "sufficient":
        verdict = "none"  # an unparsable reply is no verdict (the frozen check's fail-open)
    web_needed = bool(row.get("web_needed"))
    if not answer:
        passed = False
    elif web_needed:
        passed = verdict in TRIGGERED
    else:
        passed = verdict == "sufficient"
    return {
        "id": str(row["id"]),
        "kind": str(row["kind"]),
        "question": question,
        "web_needed": web_needed,
        "doc_status": doc_status,
        "doc_spoken": spoken,
        "doc_file_cited": str(doc.file or "") or None,
        "doc_calls": int(doc.calls or 0),
        "doc_s": round(doc_s, 3),
        "verdict_answer": answer,
        "answer_source": answer_source,
        "source_year": year,
        "verdict": verdict,
        "web_question": str(getattr(verdict_obj, "web_question", "") or "") if verdict_obj else "",
        "reason": str(getattr(verdict_obj, "reason", "") or "") if verdict_obj else "",
        "verdict_s": round(verdict_s, 3),
        "passed": passed,
        "_failures_before": n_failures,
    }


async def run_round(
    h: Harness,
    rows: Sequence[Mapping[str, Any]],
    on_row: Callable[[int, dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], TokenCounter]:
    """One pass over the set, strictly one relay call at a time. A transport failure the
    row layer swallowed (the lane records it as an error, the verdict as no verdict) stops
    the run loudly."""
    tokens = TokenCounter()
    failures: list[str] = []
    doc_request, verdict_request = make_requests(h.transport, h.model, tokens, failures)
    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        rec = await run_row(h, row, doc_request, verdict_request, failures)
        before = rec.pop("_failures_before")
        if len(failures) > before:
            raise RunAborted(f"{rec['id']}: the relay failed: {'; '.join(failures[before:])}")
        records.append(rec)
        if on_row is not None:
            on_row(i, rec)
    return records, tokens


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return round(float(ordered[rank - 1]), 3)


def score_rows(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Confusion by kind, needs-web recall per kind, over-trigger on controls, verdict
    agreement (the verdict matches the row's label) and the failure table."""
    kinds = sorted({str(r["kind"]) for r in records})
    confusion: dict[str, dict[str, int]] = {kind: {v: 0 for v in VERDICTS} for kind in kinds}
    for r in records:
        confusion[str(r["kind"])][str(r["verdict"])] += 1
    recall_by_kind: dict[str, dict[str, Any]] = {}
    n_web = passed_web = 0
    for kind in kinds:
        web_rows = [r for r in records if r["kind"] == kind and r["web_needed"]]
        if not web_rows:
            continue
        passed = sum(1 for r in web_rows if r["passed"])
        recall_by_kind[kind] = {
            "n": len(web_rows),
            "passed": passed,
            "recall": round(passed / len(web_rows), 4),
        }
        n_web += len(web_rows)
        passed_web += passed
    controls = [r for r in records if not r["web_needed"]]
    over = sum(1 for r in controls if not r["passed"])
    passed_all = sum(1 for r in records if r["passed"])
    return {
        "confusion_by_kind": confusion,
        "recall_needs_web_by_kind": recall_by_kind,
        "recall_needs_web": {
            "n": n_web,
            "passed": passed_web,
            "recall": round(passed_web / n_web, 4) if n_web else None,
        },
        "over_trigger_controls": {
            "n": len(controls),
            "over_triggered": over,
            "rate": round(over / len(controls), 4) if controls else None,
        },
        "agreement": {
            "n": len(records),
            "passed": passed_all,
            "rate": round(passed_all / len(records), 4) if records else None,
        },
        "failures": [
            {
                "id": str(r["id"]),
                "kind": str(r["kind"]),
                "verdict": str(r["verdict"]),
                "reason": str(r["reason"]),
                "doc_status": str(r["doc_status"]),
                "answer_source": str(r["answer_source"]),
                "question": str(r["question"]),
            }
            for r in records
            if not r["passed"]
        ],
    }


def _latency_block(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "doc": {
            "p50": percentile([float(r["doc_s"]) for r in records], 50),
            "p95": percentile([float(r["doc_s"]) for r in records], 95),
        },
        "verdict": {
            "p50": percentile([float(r["verdict_s"]) for r in records], 50),
            "p95": percentile([float(r["verdict_s"]) for r in records], 95),
        },
    }


def _tokens_block(tokens: TokenCounter) -> dict[str, Any]:
    return {
        "doc": {
            "prompt": tokens.doc_prompt,
            "completion": tokens.doc_completion,
            "total": tokens.doc_prompt + tokens.doc_completion,
        },
        "verdict": {
            "prompt": tokens.verdict_prompt,
            "completion": tokens.verdict_completion,
            "total": tokens.verdict_prompt + tokens.verdict_completion,
        },
    }


def build_report(
    *,
    mode: str,
    today: str,
    model: str,
    db: Path,
    jsonl_path: Path,
    repeat: int,
    rounds: list[list[dict[str, Any]]],
    round_tokens: list[TokenCounter],
    judge_result: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records0 = rounds[0]
    scoring = score_rows(records0)
    stability: dict[str, Any] | None = None
    if repeat > 1:
        flips: list[dict[str, Any]] = []
        stable = 0
        for i, rec in enumerate(records0):
            verdicts = [rounds[k][i]["verdict"] for k in range(len(rounds))]
            if len(set(verdicts)) == 1:
                stable += 1
            else:
                flips.append({"id": str(rec["id"]), "kind": str(rec["kind"]), "verdicts": verdicts})
        stability = {
            "repeats": len(rounds),
            "n": len(records0),
            "stable": stable,
            "rate": round(stable / len(records0), 4) if records0 else None,
            "flips": flips,
        }
    rows_out: list[dict[str, Any]] = []
    for i, rec in enumerate(records0):
        out = dict(rec)
        if repeat > 1:
            out["verdicts"] = [rounds[k][i]["verdict"] for k in range(len(rounds))]
        rows_out.append(out)
    rounds_out = [
        {
            "round": k + 1,
            "verdict_counts": dict(Counter(str(r["verdict"]) for r in recs)),
            "agreement": score_rows(recs)["agreement"],
            "latency_s": _latency_block(recs),
            "tokens": _tokens_block(toks),
        }
        for k, (recs, toks) in enumerate(zip(rounds, round_tokens, strict=True))
    ]
    return {
        "ticket": 255,
        "mode": mode,
        "today": today,
        "model": model,
        "db": str(db),
        "jsonl": str(jsonl_path),
        "selection": dict(selection) if selection else None,
        "repeat": repeat,
        "n_rows": len(records0),
        "doc_status_counts": dict(Counter(str(r["doc_status"]) for r in records0)),
        "verdict_counts": dict(Counter(str(r["verdict"]) for r in records0)),
        "answer_source_counts": dict(Counter(str(r["answer_source"]) for r in records0)),
        **scoring,
        "stability": stability,
        "latency_s": _latency_block(records0),
        "tokens": _tokens_block(round_tokens[0]),
        "judge": judge_result,
        "rounds": rounds_out,
        "rows": rows_out,
    }


def grade_directness(
    records: list[dict[str, Any]], transport: Transport, model: str, judge: Any
) -> Mapping[str, Any] | None:
    """Optional: grade the spoken result of the speak-first flow (the accepted document
    answer) with judge_eval's directness metric - imported, not copied. Rows the lane did
    not accept are not spoken by the speak-first flow, so they are not graded."""
    targets = [r for r in records if r.get("doc_spoken")]
    if not targets:
        return None
    jobs: list[dict[str, Any]] = []
    for i, rec in enumerate(targets):
        job = judge.jobs_from_log_row(
            {"turn": i, "question": rec["question"], "answer": rec["doc_spoken"]}, i
        )
        if job is not None:
            job["turn"] = rec["id"]
            jobs.append(job)
    out = judge.run_jobs(jobs, ["directness"], transport, model)
    by_id = {str(r["turn"]): r for r in out}
    for rec in records:
        judged = by_id.get(str(rec["id"]))
        rec["judge_directness"] = (
            {
                "choice": str(judged["choice"]),
                "label": judge.label_for("directness", str(judged["choice"])),
                "conf": float(judged["conf"]),
            }
            if judged is not None
            else None
        )
    counts = Counter(str(r["choice"]) for r in out)
    return {
        "n": len(out),
        "counts": dict(sorted(counts.items())),
        "labels": {letter: judge.label_for("directness", letter) for letter in sorted(counts)},
        "avg_conf": round(sum(float(r["conf"]) for r in out) / len(out), 4) if out else None,
    }


def _fmt(value: float | None, fmt: str = "{:.2f}") -> str:
    return "n/a" if value is None else fmt.format(value)


def render_table(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append(
        f"sufficiency eval  mode={report['mode']}  today={report['today']}  "
        f"model={report['model']}  n={report['n_rows']}"
    )
    sel = report.get("selection")
    if sel:
        files = ", ".join(str(f) for f in sel.get("files", []))
        lines.append(f"set={sel.get('set')}  split={sel.get('split')}  files={files}")
    lines.append(
        "doc lane:   "
        + "  ".join(
            f"{k} {v}"
            for k, v in sorted(report["doc_status_counts"].items(), key=lambda kv: -kv[1])
        )
    )
    lines.append(
        "verdict:    "
        + "  ".join(
            f"{k} {v}" for k, v in sorted(report["verdict_counts"].items(), key=lambda kv: -kv[1])
        )
    )
    lines.append("")
    lines.append("confusion by kind:")
    lines.append(f"{'kind':<24} {'n':>3}  " + "  ".join(f"{v:>14}" for v in VERDICTS))
    for kind, counts in report["confusion_by_kind"].items():
        n = sum(counts.values())
        lines.append(
            f"{kind:<24} {n:>3}  " + "  ".join(f"{counts.get(v, 0):>14}" for v in VERDICTS)
        )
    lines.append("")
    lines.append("needs-web recall by kind:")
    for kind, m in report["recall_needs_web_by_kind"].items():
        lines.append(f"  {kind:<24} {m['passed']:>2}/{m['n']:<3} {_fmt(m['recall'])}")
    overall = report["recall_needs_web"]
    lines.append(
        f"  {'all needs_web':<24} {overall['passed']:>2}/{overall['n']:<3} "
        f"{_fmt(overall['recall'])}"
    )
    over = report["over_trigger_controls"]
    lines.append(
        f"over-trigger on controls: {over['over_triggered']}/{over['n']}  rate {_fmt(over['rate'])}"
    )
    agr = report["agreement"]
    lines.append(f"verdict agreement: {agr['passed']}/{agr['n']}  {_fmt(agr['rate'])}")
    st = report.get("stability")
    if st:
        flips = ", ".join(f"{f['id']} {'->'.join(f['verdicts'])}" for f in st["flips"][:10])
        lines.append(
            f"stability ({st['repeats']} runs): {st['stable']}/{st['n']} stable"
            + (f";  flips: {flips}" if flips else "")
        )
    lat = report["latency_s"]
    doc_lat, ver_lat = lat["doc"], lat["verdict"]
    lines.append(
        f"latency s: doc p50={_fmt(doc_lat['p50'], '{:.2f}')} "
        f"p95={_fmt(doc_lat['p95'], '{:.2f}')} | "
        f"verdict p50={_fmt(ver_lat['p50'], '{:.2f}')} "
        f"p95={_fmt(ver_lat['p95'], '{:.2f}')}"
    )
    tok = report["tokens"]
    lines.append(
        f"tokens: doc {tok['doc']['prompt']} in / {tok['doc']['completion']} out | "
        f"verdict {tok['verdict']['prompt']} in / {tok['verdict']['completion']} out"
    )
    j = report.get("judge")
    if j:
        lines.append(
            "judge directness: "
            + " ".join(f"{j['labels'][c]} {n}" for c, n in j["counts"].items())
            + f"  (n={j['n']}, avg_conf={_fmt(j['avg_conf'])})"
        )
    failures = report["failures"]
    lines.append("")
    lines.append(f"failures ({len(failures)}):")
    for f in failures:
        lines.append(
            f"  {f['id']:<8} {f['kind']:<22} {f['verdict']:<14} "
            f"[{f['doc_status']}/{f['answer_source']}] {f['reason'][:90]}"
        )
    return "\n".join(lines)


def _load_judge() -> Any:
    """scripts/judge_eval.py, imported (not copied): its read_key, jobs_from_log_row and
    run_jobs carry the judge's conventions."""
    module = sys.modules.get("judge_eval")
    if module is not None:
        return module
    path = Path(__file__).resolve().with_name("judge_eval.py")
    spec = importlib.util.spec_from_file_location("judge_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["judge_eval"] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="offline evaluation of the speak-first sufficiency check (ticket 255)"
    )
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="labelled question set (overrides --set); default: the v1 set",
    )
    ap.add_argument(
        "--set",
        dest="set_name",
        choices=("v1", "v2", "both"),
        default="v1",
        help="which labelled set to run: v1 (the released 60 rows), v2 (the ticket-260 "
        "40 rows), or both (the full 100, v1 followed by v2)",
    )
    ap.add_argument(
        "--split",
        dest="split_name",
        choices=("dev", "test", "all"),
        default="all",
        help="ticket-260 frozen dev/test split to run; 'all' (default) runs every row",
    )
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="ingest DB (DbIndex)")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="raw files root for the DB")
    ap.add_argument("--today", default=DEFAULT_TODAY, help="fixed date the verdict sees")
    ap.add_argument(
        "--gold-answer",
        action="store_true",
        help="score the verdict on the gold ideal answer for every row (second mode)",
    )
    ap.add_argument("--repeat", type=int, default=1, help="runs over the set (stability)")
    ap.add_argument(
        "--judge",
        action="store_true",
        help="grade the spoken lane answer with judge_eval's directness metric",
    )
    ap.add_argument(
        "--verdict-timeout",
        type=float,
        default=None,
        help="override the production verdict timeout in seconds (default: production value)",
    )
    ap.add_argument("--out", type=Path, default=None, help="JSON report path")
    ap.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", DEFAULT_MODEL))
    return ap.parse_args(argv)


async def amain(args: argparse.Namespace) -> int:
    judge = _load_judge()
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2
    paths = resolve_set_paths(args.set_name, args.jsonl)
    rows: list[dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            print(f"row set not found: {p}", file=sys.stderr)
            return 2
        rows.extend(load_rows(p))
    try:
        rows = filter_split(rows, args.split_name)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not rows:
        print(f"no rows for set={args.set_name} split={args.split_name}", file=sys.stderr)
        return 2
    selection = {
        "set": f"custom:{args.jsonl.name}" if args.jsonl is not None else args.set_name,
        "split": args.split_name,
        "files": [str(p) for p in paths],
    }
    file_names = load_file_names(args.db)
    cards = load_cards(args.db)
    mode = "gold" if args.gold_answer else "lane"
    # the heavy parts are imported late so the scoring code stays importable without the
    # pipeline venv (the root tests run on fakes)
    try:
        from voice_pipeline import ask_documents as ask_documents_mod
        from voice_pipeline import sufficiency as sufficiency_mod
        from voice_pipeline.doc_answer import passages_with_meta

        from voice_agent.rag.dbindex import DbIndex
    except ImportError as exc:
        print(
            f"this run needs the pipeline venv and the repo src on sys.path: {exc}", file=sys.stderr
        )
        return 2
    if args.verdict_timeout is not None:
        sufficiency_mod.VERDICT_TIMEOUT_S = args.verdict_timeout
    index = DbIndex(args.db, raw_root=args.raw)

    async def context_fn(question: str) -> list[dict[str, Any]]:
        return passages_with_meta(await asyncio.to_thread(index.file_context, question))

    transport = Relay(args.base_url, judge.read_key(), args.model)
    harness = Harness(
        transport=transport,
        model=args.model,
        context_fn=context_fn,
        ask_documents_fn=ask_documents_mod.ask_documents,
        verdict_fn=sufficiency_mod.verdict,
        file_names=file_names,
        cards=cards,
        today=args.today,
        gold_answer=mode == "gold",
    )
    n = len(rows)
    width = len(str(n))
    rounds: list[list[dict[str, Any]]] = []
    round_tokens: list[TokenCounter] = []
    for rep in range(1, args.repeat + 1):
        print(f"[mode={mode}] round {rep}/{args.repeat}: {n} rows", file=sys.stderr)
        records, tokens = await run_round(
            harness,
            rows,
            on_row=lambda i, rec: print(
                f"  [{i:>{width}}/{n}] {rec['id']} doc={rec['doc_status']:<12} "
                f"verdict={rec['verdict']:<14} ({rec['answer_source']})",
                file=sys.stderr,
            ),
        )
        rounds.append(records)
        round_tokens.append(tokens)
    judge_result: Mapping[str, Any] | None = None
    if args.judge:
        print("grading the spoken lane answers (judge directness)", file=sys.stderr)
        judge_result = grade_directness(rounds[0], transport, args.model, judge)
    report = build_report(
        mode=mode,
        today=args.today,
        model=args.model,
        db=args.db,
        jsonl_path=paths[0],
        repeat=args.repeat,
        rounds=rounds,
        round_tokens=round_tokens,
        judge_result=judge_result,
        selection=selection,
    )
    out_path = (
        args.out
        or Path("out/sufficiency_eval") / f"run-{time.strftime('%Y%m%d-%H%M%S')}-{mode}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(render_table(report))
    print(f"wrote {out_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        return asyncio.run(amain(args))
    except RunAborted as exc:
        print(f"aborted: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
