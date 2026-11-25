"""Offline eval of `ask_documents` with deterministic oracles (ticket 209).

Run in the pipeline venv (it imports voice_pipeline). Answerable rows get an oracle that
replies only when every fact is in the context and quotes a verbatim window of at most 28
words; unanswerable, decoy and off-topic rows get a "tempted" oracle that quotes the line
with the most question terms, so the count of accepted rows is what the checks let through.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from me4_context_ablate import fact_in, load_dev, load_golden, load_set_a, load_set_b, norm

from voice_agent.rag.dbindex import DbIndex

RAG = Path("~/rag-testsets").expanduser()
_NUM = re.compile(r"\d[\d,.]*")
_TAG = re.compile(r'<untrusted_document[^>]*file="([^"]*?)"[^>]*>(.+?)</untrusted_document>', re.S)
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
        "what",
        "which",
        "according",
        "documents",
        "document",
        "file",
        "many",
        "much",
    ]
)


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1}


def _passages(content: str) -> list[dict[str, str]]:
    return [{"source": m.group(1), "text": m.group(2).strip()} for m in _TAG.finditer(content)]


def _window(text: str, want: set[str], size: int = 28) -> str:
    """The verbatim run of at most `size` words with the most words from `want`."""
    words = text.split()
    best, at = -1, 0
    for i in range(max(1, len(words) - size + 1)):
        score = sum(1 for w in words[i : i + size] if norm(w) in want)
        if score > best:
            best, at = score, i
    return " ".join(words[at : at + size])


class AnswerOracle:
    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row
        self.called = False

    async def __call__(self, system: str, content: str) -> str:
        self.called = True
        ps = _passages(content)
        ctx = [(p["source"], p["text"]) for p in ps]
        facts = self.row["facts"]
        # Normalize: facts may be list[dict] or list[tuple]
        normalized = []
        for f in facts:
            if isinstance(f, dict):
                normalized.append(
                    (
                        f.get("file", ""),
                        f.get("contains", ""),
                        f.get("accept") or self.row.get("accept") or [],
                    )
                )
            else:
                normalized.append(f)
        facts = normalized
        if not facts or not all(fact_in(ctx, f) for f in facts if f[1] or f[2]):
            return json.dumps({"status": "none"})
        file_, contains, accept = facts[0]
        text = "\n".join(t for s, t in ctx if s == file_)
        if not text:
            return json.dumps({"status": "none"})
        quote = _window(text, set(norm(contains).split()) | {norm(a) for a in accept}, 28)
        answer = accept[0] if accept else " ".join(quote.split()[-4:])
        if self.row["web"]:
            return json.dumps(
                {
                    "status": "web",
                    "answer": " ".join(quote.split()[:6]),
                    "file": file_,
                    "quotes": [quote],
                    "web_question": self.row["web"]["needs"],
                }
            )
        return json.dumps({"status": "answer", "answer": answer, "file": file_, "quotes": [quote]})


class TemptedOracle:
    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row
        self.called = False

    async def __call__(self, system: str, content: str) -> str:
        self.called = True
        want = _terms(self.row["q"])
        best, pick = 0, None
        for p in _passages(content):
            for line in p["text"].split("\n"):
                score = sum(1 for t in want if t in line.lower())
                if score > best:
                    best, pick = score, (p["source"], line)
        if pick is None:
            return json.dumps({"status": "none"})
        nums = _NUM.findall(pick[1])
        proper = re.findall(r"(?<!^)(?<![.!?] )\b[A-Z][a-z]{2,}\b", pick[1])
        if not nums and not proper:
            return json.dumps({"status": "none"})
        answer = nums[0] if nums else proper[0]
        quote = _window(pick[1], {norm(answer)}, 12)
        return json.dumps(
            {"status": "answer", "answer": answer, "file": pick[0], "quotes": [quote]}
        )


def _rows(name: str, path: Path) -> list[dict[str, Any]]:
    """Every row of a set (answerable and not) in one shape."""
    loaders = {"dev": load_dev, "golden": load_golden, "A": load_set_a, "B": load_set_b}
    raw = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_id = {fid: facts for fid, _, facts in loaders[name](path)} if name != "nondoc" else {}
    out = []
    for r in raw:
        facts = by_id.get(r["id"], [])
        out.append(
            {
                "set": name,
                "id": r["id"],
                "q": r["q"],
                "facts": facts,
                "unanswerable": not facts,
                "web": r.get("web"),
                "expected": (r.get("web") or {}).get("expected", ""),
                "accept": (r.get("accept") or []) if name == "dev" else [],
            }
        )
    return out


async def _run(row: dict[str, Any], db: DbIndex) -> dict[str, Any]:
    from voice_pipeline.ask_documents import ask_documents  # needs the pipeline venv

    async def context(q: str) -> list[dict[str, Any]]:
        return db.file_context(q)

    async def web(q: str) -> str | None:
        return f"As per example.org: {row['expected']}" if row["expected"] else None

    async def catalog() -> list[str]:
        return [r.name for r in db.records()]

    oracle = TemptedOracle(row) if row["unanswerable"] else AnswerOracle(row)
    r = await ask_documents(
        row["q"],
        context=context,
        request=oracle,
        web=web,
        catalog=catalog,
        budget_s=10.0,
        card=db.file_card,
    )
    spoken = r.spoken or ""
    accepted = r.status == "accepted"
    want = row["accept"] + ([row["expected"]] if row["expected"] else [])
    return {
        "set": row["set"],
        "id": row["id"],
        "status": r.status,
        "accepted": accepted,
        "spoken": spoken,
        "rejected": r.rejected or {},
        "fallback": r.fallback,
        "calls": r.calls,
        "unanswerable": row["unanswerable"],
        "correct": accepted
        and not row["unanswerable"]
        and (not want or any(w.lower() in spoken.lower() for w in want)),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ans = [r for r in rows if not r["unanswerable"]]
    un = [r for r in rows if r["unanswerable"]]
    why = Counter(r["rejected"].get("failed", "") for r in ans if r["rejected"])
    return {
        "answerable": len(ans),
        "accepted_correct": sum(1 for r in ans if r["correct"]),
        "unanswerable": len(un),
        "false_accepts": sorted(r["id"] for r in un if r["accepted"]),
        "answerable_rejected_by": dict(why),
        "fallbacks": sum(1 for r in rows if r["fallback"]),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--db44", "--raw44", "--dbB", "--rawB", "--out"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()
    a_db = DbIndex(Path(args.db44), raw_root=Path(args.raw44))
    b_db = DbIndex(Path(args.dbB), raw_root=Path(args.rawB))
    sets = [
        ("dev", RAG / "heldout/dev.jsonl", a_db),
        ("golden", RAG / "ipo-2026/golden.jsonl", a_db),
        ("A", RAG / "ipo-2026/me4-frozen/q.jsonl", a_db),
        ("B", RAG / "ipo-2026-b/q2.jsonl", b_db),
    ]
    rows: list[dict[str, Any]] = []
    for name, path, db in sets:
        for row in _rows(name, path):
            rows.append(await _run(row, db))
    nondoc = (RAG / "heldout/nondoc25.txt").read_text(encoding="utf-8").splitlines()
    for i, q in enumerate(x for x in nondoc if x.strip()):
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
        rows.append(await _run(row, a_db))
    summary = {
        n: _summary([r for r in rows if r["set"] == n]) for n in [s[0] for s in sets] + ["nondoc25"]
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    for n, s in summary.items():
        print(n, s)
    a_db.close()
    b_db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
