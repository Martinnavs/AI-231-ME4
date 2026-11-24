"""Score held-out questions against turn-trace output rows."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_ORDINAL_MAP = {"1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth"}


def _norm(s: str) -> str:
    s = re.sub(r"[^\w\s']", " ", s.lower().strip())
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _accept_norm(s: str) -> str:
    s = _norm(s)
    for short, long_ in _ORDINAL_MAP.items():
        s = re.sub(rf"\b{re.escape(short)}\b", long_, s)
    return s


def load_questions(path: Path) -> list[dict[str, Any]]:
    qs: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "facts" in row or "accept" in row:
            qs.append({**row, "legacy": False})
        else:
            qs.append(
                {
                    "id": row["id"],
                    "q": row["q"],
                    "kind": row["kind"],
                    "facts": [],
                    "answer": row.get("answer", ""),
                    "accept": row.get("accept", []),
                    "answer_regex": row.get("answer_regex", ""),
                    "gold_files": row.get("gold_files", []),
                    "web": row.get("web"),
                    "unanswerable": row.get("unanswerable", False),
                    "legacy": True,
                }
            )
    return qs


_CITE_PREFIX = re.compile(r"^\s*As per [^:]+:\s*", re.IGNORECASE)


def accept_match(accept: list[str], text: str) -> bool:
    """True when an accept string occurs as whole words in the spoken answer (citation stripped)."""
    if not accept:
        return False
    nt = _accept_norm(_CITE_PREFIX.sub("", text))
    for a in accept:
        na = _accept_norm(a)
        if na and re.search(rf"(?<![\w]){re.escape(na)}(?![\w])", nt):
            return True
    return False


def fact_in_passages(facts: list[dict[str, Any]], passages: list[dict[str, Any]]) -> bool:
    if not facts:
        return True
    texts = [_norm(p.get("text", "")) for p in passages]
    return all(any(_norm(f["contains"]) in t for t in texts) for f in facts)


def score_question(q: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    route = row.get("route", "")
    text = row.get("text", "")
    docs = row.get("docs")
    calls = row.get("calls", [])
    passages: list[dict] = list(row.get("passages") or [])
    for c in calls:
        if passages:
            break
        if c.get("tool") == "search_docs" and c.get("result_text"):
            pts = json.loads(c["result_text"])
            if isinstance(pts, list):
                passages = pts
            elif isinstance(pts, str):
                passages = json.loads(pts)
            break

    first_search = next((c for c in calls if c.get("tool") == "search_docs"), None)
    search_ms = round(float(first_search.get("s", 0)) * 1000, 1) if first_search else 0.0

    citation_text = ""
    if row.get("cited_by_server"):
        m = re.search(r"\[As per .+?:(?:\s|$)]", text)
        if m:
            citation_text = m.group(0).strip()

    accepted = docs == "accepted"
    correct = accept_match(q.get("accept", []), text) if q.get("accept") else False
    if q.get("legacy") and q.get("answer_regex") and accepted:
        correct = bool(re.search(q["answer_regex"], text, re.IGNORECASE))

    false_accept = False
    if q.get("unanswerable") and accepted or not q.get("unanswerable") and accepted and not correct:
        false_accept = True

    abstained = not accepted
    fp = fact_in_passages(q.get("facts", []), passages)
    sources = [p.get("source", "") for p in passages]
    file_in = (
        all(
            any(f["file"] in s or s.endswith("/" + f["file"]) for s in sources)
            for f in q.get("facts", [])
        )
        if q.get("facts")
        else True
    )

    out: dict[str, Any] = {
        "id": q["id"],
        "kind": q.get("kind", ""),
        "q": q["q"],
        "route": route,
        "docs": docs or "",
        "rejected": row.get("rejected", {}),
        "spoken": text,
        "citation": citation_text,
        "reached_agent": bool(row.get("reached_agent", route not in ("docs", "fast", ""))),
        "search_ms": search_ms,
        "doc_s": row.get("doc_s", 0),
        "total_ms": row.get("total_ms") or round(float(row.get("end_s", 0)) * 1000, 1),
        "passages": passages,
        "fact_in_passages": fp,
        "file_in_passages": file_in,
        "accepted": accepted,
        "correct": correct,
        "false_accept": false_accept,
        "abstained": abstained,
    }
    web = q.get("web")
    out["web_needed"] = bool(web)
    if web:
        accept = q.get("accept", [])
        out["correct_doc_half"] = accept_match(accept, text) if accept else False
        del out["correct"]
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = set(r["kind"] for r in rows)
    by_kind: dict[str, dict[str, Any]] = {}
    for k in sorted(kinds):
        sub = [r for r in rows if r["kind"] == k]
        by_kind[k] = {
            "n": len(sub),
            "routed_to_docs": sum(1 for r in sub if r["route"] == "docs"),
            "fact_in_passages": sum(1 for r in sub if r.get("fact_in_passages")),
            "accepted": sum(1 for r in sub if r["accepted"]),
            "correct": sum(1 for r in sub if r.get("correct", False)),
            "false_accept": sum(1 for r in sub if r.get("false_accept")),
            "abstained": sum(1 for r in sub if r.get("abstained")),
        }
    times = sorted(r["total_ms"] for r in rows)
    doc_times = sorted(r.get("doc_s", 0) for r in rows)
    n = len(rows)
    return {
        "n": n,
        "routed_to_docs": sum(1 for r in rows if r["route"] == "docs"),
        "fact_in_passages": sum(1 for r in rows if r.get("fact_in_passages")),
        "accepted": sum(1 for r in rows if r["accepted"]),
        "correct": sum(1 for r in rows if r.get("correct", False)),
        "false_accept": sum(1 for r in rows if r.get("false_accept")),
        "abstained": sum(1 for r in rows if r.get("abstained")),
        "by_kind": by_kind,
        "route_histogram": dict(Counter(r["route"] for r in rows)),
        "p50_total_ms": times[n // 2] if times else 0,
        "p95_total_ms": times[int(n * 0.95)] if times else 0,
        "p50_doc_s": doc_times[n // 2] if doc_times else 0,
        "p95_doc_s": doc_times[int(n * 0.95)] if doc_times else 0,
    }
