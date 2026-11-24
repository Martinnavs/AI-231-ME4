"""Row-level ablation for file-first context (ticket 207).

Compares k8, 204, whole3, grep3 against per-set answerable rows.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from voice_agent.rag.dbindex import DbIndex


def norm(s: str) -> str:
    s = s.lower().replace("\u2019", "'").replace("\u2014", "-").replace("\u2013", "-")
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    return " ".join(re.findall(r"[a-z0-9$%.]+", s)).replace(" .", " ").strip(". ")


def fact_in(ctx: list[tuple[str, str]], fact: tuple[str, str, list[str]]) -> bool:
    """Port of Fable's checker."""
    f, contains, accept = fact
    texts = [norm(t) for r, t in ctx if r == f]
    if not texts:
        return False
    c = norm(contains)
    if c and any(c in t for t in texts):
        return True
    if "; " in contains and ": " in contains:
        vals = [kv.split(": ", 1)[1] for kv in contains.split("; ") if ": " in kv]
        ent, last = norm(vals[0]), norm(vals[-1])
        for r, t in ctx:
            if r != f:
                continue
            for line in t.split("\n"):
                nl = " " + norm(line) + " "
                if ent and last and f" {ent} " in nl and f" {last} " in nl:
                    return True
    ct = c.split()
    if len(ct) >= 4:
        w = len(ct) * 3
        need = set(ct)
        for t in texts:
            tt = t.split()
            step = max(1, len(ct) // 2)
            for i in range(0, max(1, len(tt) - w + 1), step):
                if len(need & set(tt[i : i + w])) >= 0.9 * len(need):
                    return True
    return any(norm(a) and any(norm(a) in t for t in texts) for a in accept)


def row_hit(
    ctx: list[tuple[str, str]],
    facts: list[tuple[str, str, list[str]]],
) -> bool:
    """Row is a hit when every fact with non-empty contains/accept returns True
    AND every fact's file is a source in context."""
    sources = {s for s, _ in ctx}
    for f, contains, accept in facts:
        if f not in sources:
            return False
        if (contains or accept) and not fact_in(ctx, (f, contains, accept)):
            return False
    return True


def load_dev(qfile: Path) -> list[tuple[str, str, list[tuple[str, str, list[str]]]]]:
    """(row_id, question, facts)."""
    rows: list[tuple[str, str, list[tuple[str, str, list[str]]]]] = []
    for line in qfile.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("unanswerable"):
            continue
        rid = row.get("id", str(len(rows)))
        q = row["q"]
        facts: list[tuple[str, str, list[str]]] = []
        for item in row.get("facts", []):
            facts.append((item["file"], item.get("contains", ""), row.get("accept", [])))
        if facts:
            rows.append((rid, q, facts))
    return rows


def load_golden(qfile: Path) -> list[tuple[str, str, list[tuple[str, str, list[str]]]]]:
    rows: list[tuple[str, str, list[tuple[str, str, list[str]]]]] = []
    for line in qfile.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        relevant = row.get("relevant", [])
        if not relevant:
            continue
        rid = row.get("id", str(len(rows)))
        q = row["q"]
        facts: list[tuple[str, str, list[str]]] = []
        for item in relevant:
            facts.append((item["file"], item.get("contains", ""), []))
        rows.append((rid, q, facts))
    return rows


def load_set_a(qfile: Path) -> list[tuple[str, str, list[tuple[str, str, list[str]]]]]:
    rows: list[tuple[str, str, list[tuple[str, str, list[str]]]]] = []
    for line in qfile.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        gold_files = row.get("gold_files", [])
        if not gold_files:
            continue
        rid = row.get("id", str(len(rows)))
        q = row["q"]
        facts: list[tuple[str, str, list[str]]] = []
        gold_quote = row.get("gold_quote", "")
        answer = row.get("answer", "")
        accept: list[str] = []
        if isinstance(answer, list):
            accept = answer
        elif answer:
            accept = [answer] if len(answer) < 25 else []
        for i, gf in enumerate(gold_files):
            if i == 0:
                facts.append((gf, gold_quote, accept))
            else:
                facts.append((gf, "", []))
        rows.append((rid, q, facts))
    return rows


def load_set_b(qfile: Path) -> list[tuple[str, str, list[tuple[str, str, list[str]]]]]:
    rows: list[tuple[str, str, list[tuple[str, str, list[str]]]]] = []
    for line in qfile.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        gold_files = row.get("gold_files", [])
        if not gold_files:
            continue
        rid = row.get("id", str(len(rows)))
        q = row["q"]
        facts: list[tuple[str, str, list[str]]] = []
        gold_quote = row.get("gold_quote", "")
        for i, gf in enumerate(gold_files):
            if i == 0:
                facts.append((gf, gold_quote, []))
            else:
                facts.append((gf, "", []))
        rows.append((rid, q, facts))
    return rows


def passages_to_ctx(passages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(p["source"], p["text"]) for p in passages]


def run_variant(
    variant: str,
    idx: DbIndex,
    questions: list[tuple[str, str, list[tuple[str, str, list[str]]]]],
) -> dict[str, Any]:
    hits = 0
    ctx_sizes: list[int] = []
    details: list[dict[str, Any]] = []
    for rid, q, facts in questions:
        if variant == "k8":
            passages = idx.search(q, 8)
        elif variant == "204":
            passages = idx.search(q, 8, per_file_cap=6, whole_files=2)
        elif variant == "whole3":
            passages = idx.file_context(q, per_big=8, grep_top=0)
        elif variant == "grep3":
            passages = idx.file_context(q)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        ctx = passages_to_ctx(passages)
        size = sum(len(t) for _, t in ctx)
        ctx_sizes.append(size)
        hit = row_hit(ctx, facts)
        if hit:
            hits += 1
        details.append({"id": rid, "hit": hit, "size": size, "facts": len(facts)})
    import statistics

    mean_sz = statistics.mean(ctx_sizes) if ctx_sizes else 0
    p95_sz = statistics.quantiles(ctx_sizes, n=20)[18] if ctx_sizes else 0
    return {
        "hits": hits,
        "rows": len(questions),
        "mean_chars": round(mean_sz, 1),
        "p95_chars": round(p95_sz, 1),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ME4 context ablation")
    parser.add_argument("--db44", required=True, help="DB for DEV/GOLDEN/A")
    parser.add_argument("--raw44", required=True, help="Raw dir for DEV/GOLDEN/A")
    parser.add_argument("--dbB", required=True, help="DB for B")
    parser.add_argument("--rawB", required=True, help="Raw dir for B")
    parser.add_argument("--out", required=True, help="Output JSON")
    args = parser.parse_args()

    home = Path.home()
    qfiles = {
        "DEV": home / "rag-testsets/heldout/dev.jsonl",
        "GOLDEN": home / "rag-testsets/ipo-2026/golden.jsonl",
        "A": home / "rag-testsets/ipo-2026/me4-frozen/q.jsonl",
        "B": home / "rag-testsets/ipo-2026-b/q2.jsonl",
    }

    loaders = {
        "DEV": load_dev,
        "GOLDEN": load_golden,
        "A": load_set_a,
        "B": load_set_b,
    }

    idx44 = DbIndex(Path(args.db44), raw_root=Path(args.raw44))
    idxB = DbIndex(Path(args.dbB), raw_root=Path(args.rawB))

    results: dict[str, dict[str, Any]] = {}
    variants = ["k8", "204", "whole3", "grep3"]

    for set_name in ("DEV", "GOLDEN", "A"):
        questions = loaders[set_name](qfiles[set_name])
        results[set_name] = {}
        for v in variants:
            results[set_name][v] = run_variant(v, idx44, questions)

    questions_b = loaders["B"](qfiles["B"])
    results["B"] = {}
    for v in variants:
        results["B"][v] = run_variant(v, idxB, questions_b)

    idx44.close()
    idxB.close()

    # Compute gained/lost vs k8
    for set_name in ("DEV", "GOLDEN", "A", "B"):
        k8_details = {d["id"]: d["hit"] for d in results[set_name]["k8"]["details"]}
        for v in variants:
            if v == "k8":
                continue
            v_details = {d["id"]: d["hit"] for d in results[set_name][v]["details"]}
            gained = [
                rid for rid, hit in v_details.items() if hit and not k8_details.get(rid, False)
            ]
            lost = [rid for rid, hit in k8_details.items() if hit and not v_details.get(rid, False)]
            results[set_name][v]["gained"] = sorted(gained)
            results[set_name][v]["lost"] = sorted(lost)

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                s: {v: f"{r['hits']}/{r['rows']}" for v, r in sets.items()}
                for s, sets in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
