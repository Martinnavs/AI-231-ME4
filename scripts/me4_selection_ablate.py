#!/usr/bin/env python3
"""Offline selection-variant ablation harness (ticket 204).

Usage:
    python scripts/me4_selection_ablate.py --db DB --questions Q.jsonl --out OUT.json [--set-name N]
    python scripts/me4_selection_ablate.py [--out OUT.json]  (auto-ingest default sets)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from voice_agent.rag.dbindex import DbIndex

HOME = Path.home()
TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
DEFAULT_SETS: list[tuple[str, Path, Path, str]] = [
    ("dev", HOME / "rag-testsets/heldout/dev.jsonl", HOME / "rag-testsets/heldout/corpus", "_dev"),
    (
        "A",
        HOME / "rag-testsets/ipo-2026/me4-frozen/q.jsonl",
        HOME / "rag-testsets/ablation/cA",
        "_cA",
    ),
    ("B", HOME / "rag-testsets/ipo-2026-b/q2.jsonl", HOME / "rag-testsets/ablation/cB", "_cB"),
    (
        "GOLDEN",
        HOME / "rag-testsets/ipo-2026/golden.jsonl",
        HOME / "rag-testsets/ipo-2026",
        "_golden",
    ),
]
VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("base", {}),
    ("cap4", {"per_file_cap": 4}),
    ("cap6", {"per_file_cap": 6}),
    ("adaptive", {"adaptive_cap": True}),
    ("whole2", {"per_file_cap": 6, "whole_files": 2}),
    ("adaptive+whole2", {"adaptive_cap": True, "per_file_cap": 6, "whole_files": 2}),
]


def _normalize(text: str) -> str:
    s = text.lower()
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _ingest_corpus(corpus: Path, db_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "voice_agent.rag.ingest", str(corpus), str(db_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"[WARN] ingest failed for {corpus}: {result.stderr}", file=sys.stderr)


def _load_questions(qfile: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in qfile.read_text(encoding="utf-8").strip().splitlines():
        row = json.loads(line)
        is_answerable = row.get("answerable")
        if is_answerable is None:
            is_answerable = not row.get("unanswerable", False)
        if is_answerable:
            rows.append(row)
    return rows


def _get_facts(row: dict[str, Any]) -> list[str]:
    if "contains" in row:
        return [row["contains"]]
    if "gold_quote" in row:
        return [row["gold_quote"]]
    facts = row.get("facts") or row.get("relevant") or []
    if isinstance(facts, dict):
        facts = [facts]
    return [f.get("contains", "") if isinstance(f, dict) else str(f) for f in facts]


def _ptext(p: Any) -> str:
    if isinstance(p, dict):
        return _normalize(f"{p.get('header', '')}\n{p.get('text', '')}")
    return _normalize(f"{getattr(p, 'header', '')}\n{getattr(p, 'text', '')}")


def _facts_in(facts: list[str], texts: list[str]) -> bool:
    nf = [_normalize(f) for f in facts if _normalize(f)]
    return bool(nf) and all(any(f in t for t in texts) for f in nf)


def _fact_in_passages(facts: list[str], passages: list[Any]) -> bool:
    return _facts_in(facts, [_ptext(p) for p in passages])


def _search(dbi: Any, query: str, kw: dict[str, Any]) -> list[dict[str, Any]]:
    return dbi.search(query, **kw)  # type: ignore[arg-type]


def _evaluate_set(
    name: str,
    dbi: Any,
    questions: list[dict[str, Any]],
    variants: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "name": name,
        "answerable": len(questions),
        "unscoreable": 0,
        "variants": {},
        "rows": {},
    }
    variant_rows: dict[str, list[dict[str, Any]]] = {v[0]: [] for v in variants}
    unscorable: set[str] = set()
    all_texts = [_ptext(c) for c in dbi.chunks]

    for qrow in questions:
        qid, query, fact = qrow["id"], qrow["q"], _get_facts(qrow)
        if not fact:
            unscorable.add(qid)
            for vn in variant_rows:
                variant_rows[vn].append({"id": qid, "status": "no_fact"})
            continue
        if not _facts_in(fact, all_texts):
            unscorable.add(qid)
            for vn in variant_rows:
                variant_rows[vn].append({"id": qid, "status": "unscoreable"})
            continue
        for vname, vkw in variants:
            kw = {"k": 8, **vkw}
            start = time.monotonic()
            passages = _search(dbi, query, kw)
            ms = (time.monotonic() - start) * 1000
            hit = _fact_in_passages(fact, passages)
            chars = sum(len(p.get("text", "")) for p in passages)
            variant_rows[vname].append(
                {
                    "id": qid,
                    "hit": hit,
                    "passages": len(passages),
                    "chars": chars,
                    "ms": round(ms, 2),
                }
            )

    results["unscoreable"] = len(unscorable)
    results["unscoreable_ids"] = sorted(unscorable)

    base_hits: set[str] | None = None
    for vname, _vkw in variants:
        rows = variant_rows[vname]
        scored = [r for r in rows if r.get("status") not in ("unscoreable", "no_fact")]
        hit_count = sum(1 for r in scored if r.get("hit"))
        n = len(scored) or 1
        results["variants"][vname] = {
            "fact@8": hit_count,
            "scored": len(scored),
            "mean_passages": round(sum(r["passages"] for r in scored) / n, 2),
            "mean_chars": round(sum(r["chars"] for r in scored) / n, 0),
            "mean_ms": round(sum(r["ms"] for r in scored) / n, 2),
        }
        if vname == "base":
            base_hits = {r["id"] for r in scored if r.get("hit")}
            results["variants"][vname].update({"gained": [], "lost": []})
        elif base_hits is not None:
            this = {r["id"] for r in scored if r.get("hit")}
            results["variants"][vname]["gained"] = sorted(this - base_hits)
            results["variants"][vname]["lost"] = sorted(base_hits - this)

    scored_ids = [
        r["id"]
        for r in variant_rows.get("base", [])
        if r.get("status") not in ("unscoreable", "no_fact")
    ][:50]
    for qid in scored_ids:
        entry: dict[str, Any] = {"id": qid}
        for vn in variant_rows:
            for r in variant_rows[vn]:
                if r["id"] == qid and r.get("status") not in ("unscoreable", "no_fact"):
                    entry[vn] = r.get("hit", False)
                    break
        results["rows"][qid] = entry
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--questions", type=Path, default=None)
    p.add_argument("--out", type=Path, default=TMPDIR / "sel.json")
    p.add_argument("--set-name", type=str, default="custom")
    p.add_argument("--variants", type=str, default=None)
    args = p.parse_args()

    selected = (
        [(n, kw) for n, kw in VARIANTS if n.strip() in args.variants.split(",")]
        if args.variants
        else VARIANTS
    )

    if args.db and args.questions:
        dbi = DbIndex(args.db)
        res = _evaluate_set(args.set_name, dbi, _load_questions(args.questions), selected)
        dbi.close()
        output = {"sets": {args.set_name: res}, "total": res}
    else:
        output = {"sets": {}, "total": defaultdict(int)}
        for sname, qfile, corpus, _suffix in DEFAULT_SETS:
            if not qfile.exists():
                print(f"[SKIP] {sname}: {qfile} not found", file=sys.stderr)
                continue
            db_path = TMPDIR / f"sel_{sname}.db"
            if not db_path.exists():
                _ingest_corpus(corpus, db_path)
            if not db_path.exists():
                continue
            dbi = DbIndex(db_path)
            qs = _load_questions(qfile)
            print(f"[EVAL] {sname}: {len(qs)} rows", file=sys.stderr)
            res = _evaluate_set(sname, dbi, qs, selected)
            dbi.close()
            output["sets"][sname] = res
            for vn, vd in res["variants"].items():
                t = output["total"]
                t.setdefault(vn, {"fact@8": 0, "scored": 0})
                t[vn]["fact@8"] += vd["fact@8"]
                t[vn]["scored"] += vd["scored"]
        output["total"] = dict(output["total"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nSelection Ablation Results")
    print(f"{'Set':>10} {'Variant':>16} {'fact@8':>8} {'scored':>8}")
    for sname in sorted(output.get("sets", {})):
        for vname, _ in selected:
            vd = output["sets"][sname]["variants"].get(vname)
            if vd:
                print(f"{sname:>10} {vname:>16} {vd['fact@8']:>8} {vd['scored']:>8}")
    for sname in sorted(output.get("sets", {})):
        for vn, vd in output["sets"][sname]["variants"].items():
            if vn == "base" or not (vd.get("gained") or vd.get("lost")):
                continue
            print(f"  {sname}/{vn}: gained={vd['gained']}, lost={vd['lost']}")
    print(f"\nOutput: {args.out}")


if __name__ == "__main__":
    main()
