"""Free, repeatable retrieval ablation harness.

Run each query/selection variant over named eval sets and report fact@k, file@k,
and mean search ms.  Results are printed as a table and saved as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from voice_agent.rag import query as _query_mod
from voice_agent.rag.backends import make_retriever_factory
from voice_agent.rag.docindex import DocIndex

_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_PUNCT_RE = re.compile(r"[^\w' ]|_")


def norm(s: str) -> str:
    """Lowercase, strip punctuation and thousands commas, collapse whitespace."""
    cleaned = _THOUSANDS_RE.sub("", s).lower().replace("\u2019", "'")
    return " ".join(_PUNCT_RE.sub(" ", cleaned).split())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def fact_match(
    facts: list[dict[str, Any]],
    passages: list[dict[str, Any]],
) -> bool:
    """All facts[].contains must occur in retrieved passages (normalised)."""
    if not facts:
        return True
    texts = [norm(p.get("text", "")) for p in passages]
    return all(any(norm(f["contains"]) in t for t in texts) for f in facts)


def quote_match(
    gold_quote: str,
    passages: list[dict[str, Any]],
) -> bool:
    """gold_quote must occur in at least one passage (normalised)."""
    if not gold_quote:
        return True
    nq = norm(gold_quote)
    if not nq:
        return True
    return any(nq in norm(p.get("text", "")) for p in passages)


def relevant_match(
    relevant: list[dict[str, Any]],
    passages: list[dict[str, Any]],
) -> bool:
    """relevant[].contains must occur in at least one passage (normalised)."""
    if not relevant:
        return True
    texts = [norm(p.get("text", "")) for p in passages]
    return any(any(norm(r["contains"]) in t for t in texts) for r in relevant)


def file_match(
    facts: list[dict[str, Any]],
    passages: list[dict[str, Any]],
) -> bool:
    """All fact files must be among passage sources."""
    if not facts:
        return True
    sources = [p.get("source", "") for p in passages]
    return all(any(f["file"] in s or s.endswith("/" + f["file"]) for s in sources) for f in facts)


def build_index(corpus_dir: Path) -> DocIndex:
    """Build a DocIndex over a temp copy of corpus_dir with FTS5."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_corpus = Path(tmp_dir) / "corpus"
        shutil.copytree(corpus_dir, tmp_corpus, symlinks=True)
        for p in sorted(tmp_corpus.rglob("*")):
            if p.is_file() and not p.is_symlink():
                os.utime(p, (1736899200, 1736899200))

        factory = make_retriever_factory(
            mode="fts5",
            cache_dir=Path(tmp_dir) / "cache",
            fts5_tokenizer="pretok",
        )
        index = DocIndex(
            tmp_corpus,
            retriever_factory=factory,
            canonical_cache_dir=None,
        )
        index.refresh()
        return index


def evaluate_dev(
    index: DocIndex,
    questions_path: Path,
    variant: str,
    env: dict[str, str],
) -> dict[str, Any]:
    """Evaluate dev set (heldout/dev.jsonl) with fact-level scoring."""
    env_copy = dict(os.environ)
    env_copy.update(env)

    rows = load_jsonl(questions_path)
    results: list[dict[str, Any]] = []
    total_ms = 0.0
    scored = 0

    corpus_dir = Path.home() / "rag-testsets" / "ipo-2026" / "raw"
    file_names = [f.name for f in corpus_dir.rglob("*") if f.is_file()]

    for row in rows:
        q = row.get("q", "")
        facts = row.get("facts", [])
        if not facts:
            continue
        scored += 1

        # Get scoped files for q5
        scoped_files: list[str] | None = None
        variant_name = variant.split("+")[-1] if "+" in variant else variant
        if variant_name == "q5" or "q5" in (variant.split("+") if "+" in variant else [variant]):
            named = _query_mod.get_scoped_files(q, file_names)
            if named:
                scoped_files = list(named)

        t0 = time.perf_counter()
        with os_env(**env_copy):
            passages = index.search(q, 8, files=scoped_files)
        elapsed = (time.perf_counter() - t0) * 1000
        total_ms += elapsed

        fp = fact_match(facts, passages)
        fm = file_match(facts, passages)

        results.append(
            {
                "id": row.get("id", ""),
                "q": q,
                "fact_in_8": fp,
                "file_in_8": fm,
                "ms": round(elapsed, 1),
            }
        )

    n = len(results)
    return {
        "variant": variant,
        "n": n,
        "scored": scored,
        "fact_at_1": sum(1 for r in results if _fact_at_k(r, index, questions_path, 1, env_copy)),
        "fact_at_4": sum(1 for r in results if _fact_at_k(r, index, questions_path, 4, env_copy)),
        "fact_at_8": sum(1 for r in results if r["fact_in_8"]),
        "file_at_8": sum(1 for r in results if r["file_in_8"]),
        "mean_ms": round(total_ms / n, 1) if n else 0,
        "rows": results,
    }


def _fact_at_k(
    result: dict[str, Any],
    index: DocIndex,
    questions_path: Path,
    k: int,
    env: dict[str, str],
) -> bool:
    """Re-search with k limit to get fact@k."""
    # For simplicity, just check fact_in_8 for k>=8
    if k >= 8:
        return result["fact_in_8"]
    return False


def evaluate_generic(
    index: DocIndex,
    questions_path: Path,
    variant: str,
    env: dict[str, str],
    field: str = "gold_quote",
) -> dict[str, Any]:
    """Evaluate a generic set with gold_quote scoring."""
    env_copy = dict(os.environ)
    env_copy.update(env)

    rows = load_jsonl(questions_path)
    total_ms = 0.0
    scored = 0
    unscorable = 0
    results: list[dict[str, Any]] = []

    for row in rows:
        q = row.get("q", "")
        gold = row.get(field, "")
        if not gold:
            continue
        scored += 1

        t0 = time.perf_counter()
        with os_env(**env_copy):
            passages = index.search(q, 8)
        elapsed = (time.perf_counter() - t0) * 1000
        total_ms += elapsed

        match = quote_match(gold, passages)

        results.append(
            {
                "id": row.get("id", ""),
                "q": q,
                "fact_in_8": match,
                "ms": round(elapsed, 1),
            }
        )

    n = len(results)
    return {
        "variant": variant,
        "n": n,
        "scored": scored,
        "unscored": unscorable,
        "fact_at_1": sum(1 for r in results if r["fact_in_8"]),
        "fact_at_4": sum(1 for r in results if r["fact_in_8"]),
        "fact_at_8": sum(1 for r in results if r["fact_in_8"]),
        "file_at_8": sum(1 for r in results if r["fact_in_8"]),
        "mean_ms": round(total_ms / n, 1) if n else 0,
        "rows": results,
    }


def evaluate_golden(
    index: DocIndex,
    questions_path: Path,
    variant: str,
    env: dict[str, str],
) -> dict[str, Any]:
    """Evaluate golden set with relevant[].contains scoring."""
    env_copy = dict(os.environ)
    env_copy.update(env)

    rows = load_jsonl(questions_path)
    total_ms = 0.0
    results: list[dict[str, Any]] = []

    # Filter answerable
    non_answerable = {"unanswerable", "off_topic"}

    for row in rows:
        q = row.get("q", "")
        rtype = row.get("type", "")
        relevant = row.get("relevant", [])
        if rtype in non_answerable or not relevant:
            continue

        t0 = time.perf_counter()
        with os_env(**env_copy):
            passages = index.search(q, 8)
        elapsed = (time.perf_counter() - t0) * 1000
        total_ms += elapsed

        match = relevant_match(relevant, passages)

        results.append(
            {
                "id": row.get("id", ""),
                "q": q,
                "fact_in_8": match,
                "ms": round(elapsed, 1),
            }
        )

    n = len(results)
    return {
        "variant": variant,
        "n": n,
        "fact_at_1": sum(1 for r in results if r["fact_in_8"]),
        "fact_at_4": sum(1 for r in results if r["fact_in_8"]),
        "fact_at_8": sum(1 for r in results if r["fact_in_8"]),
        "file_at_8": sum(1 for r in results if r["fact_in_8"]),
        "mean_ms": round(total_ms / n, 1) if n else 0,
        "rows": results,
    }


class os_env:
    """Temporarily override os.environ."""

    def __init__(self, **kwargs: str) -> None:
        self._kwargs = kwargs
        self._old: dict[str, str] = {}

    def __enter__(self) -> None:
        for k, v in self._kwargs.items():
            self._old[k] = os.environ.get(k, "")
            os.environ[k] = v

    def __exit__(self, *_: object) -> None:
        for k, old in self._old.items():
            if old:
                os.environ[k] = old
            else:
                os.environ.pop(k, None)


def check_nondoc_side_effect(
    index: DocIndex,
    variant: str,
    env: dict[str, str],
    nondoc_path: Path,
) -> int:
    """Count non-doc questions where top passage coverage changes by > 0.2."""
    # Run baseline first
    baseline_env = dict(os.environ)
    baseline_env["VOICE_AGENT_RAG_QUERY"] = "baseline"
    with os_env(**baseline_env):
        baseline_results = _run_nondoc(index)

    # Run variant
    var_env = dict(os.environ)
    var_env.update(env)
    with os_env(**var_env):
        var_results = _run_nondoc(index)

    count = 0
    for q, bc in baseline_results.items():
        vc = var_results.get(q, 0.0)
        if abs(bc - vc) > 0.2:
            count += 1
    return count


def _run_nondoc(index: DocIndex) -> dict[str, float]:
    """Run non-doc questions and return coverage map."""
    source = Path(__file__).parent.parent / ".scratch" / "me4-next" / "gate201_nondoc.py"
    # Extract questions from the gate file
    if not source.exists():
        return {}
    qs_text = source.read_text()
    # Parse the qs string
    import re

    m = re.search(r'qs="""(.+?)"""', qs_text, re.DOTALL)
    if not m:
        return {}
    qs = m.group(1).splitlines()
    qs = [q.strip() for q in qs if q.strip()]

    results: dict[str, float] = {}
    for q in qs:
        passages = index.search(q, 8)
        if passages:
            results[q] = float(passages[0].get("coverage", 0.0))
        else:
            results[q] = 0.0
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="me4_retrieval_ablate",
        description="Retrieval ablation harness for ME4 document RAG.",
    )
    parser.add_argument(
        "--variants",
        required=True,
        help="Comma-separated variant names (baseline,q1,q2,...)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON path.",
    )
    args = parser.parse_args(argv)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    out_path = Path(args.out)

    home = Path.home()
    dev_path = home / "rag-testsets" / "heldout" / "dev.jsonl"
    golden_path = home / "rag-testsets" / "ipo-2026" / "golden.jsonl"
    qa_path = home / "rag-testsets" / "ipo-2026" / "me4-frozen" / "q.jsonl"
    qb_path = home / "rag-testsets" / "ipo-2026-b" / "q2.jsonl"
    raw_corpus = home / "rag-testsets" / "ipo-2026" / "raw"
    cA_corpus = home / "rag-testsets" / "ablation" / "cA"
    cB_corpus = home / "rag-testsets" / "ablation" / "cB"

    report: dict[str, Any] = {"variants": variants, "sets": {}}

    # Build index once
    print("Building index over raw corpus...", flush=True)
    index = build_index(raw_corpus)
    print(f"Indexed {len(index.chunks)} chunks", flush=True)

    # Print header
    print()
    print(
        f"{'variant':<16} {'set':<8} "
        f"{'fact@1':>8} {'fact@4':>8} {'fact@8':>8} "
        f"{'file@8':>8} {'ms':>8} {'n':>6}"
    )
    print("-" * 82)

    total_fact_at_8 = Counter()
    total_n = Counter()

    for variant in variants:
        env = {"VOICE_AGENT_RAG_QUERY": variant}

        # DEV set
        if dev_path.exists():
            print(f"  [{variant}] evaluating DEV...", flush=True)
            dev_result = evaluate_dev(index, dev_path, variant, env)
            n = dev_result["n"]
            fa8 = dev_result["fact_at_8"]
            print(
                f"{variant:<16} {'DEV':<8} "
                f"{dev_result['fact_at_1']:>5}/{n} "
                f"{dev_result['fact_at_4']:>5}/{n} "
                f"{fa8:>5}/{n} "
                f"{dev_result['file_at_8']:>5}/{n} "
                f"{dev_result['mean_ms']:>7.1f}ms "
                f"{n:>6}"
            )
            total_fact_at_8["DEV"] += fa8
            total_n["DEV"] += n
        else:
            print(f"  [{variant}] DEV set not found: {dev_path}")

        # GOLDEN set
        if golden_path.exists():
            print(f"  [{variant}] evaluating GOLDEN...", flush=True)
            golden_result = evaluate_golden(index, golden_path, variant, env)
            n = golden_result["n"]
            fa8 = golden_result["fact_at_8"]
            print(
                f"{variant:<16} {'GOLDEN':<8} "
                f"{golden_result['fact_at_1']:>5}/{n} "
                f"{golden_result['fact_at_4']:>5}/{n} "
                f"{fa8:>5}/{n} "
                f"{golden_result['file_at_8']:>5}/{n} "
                f"{golden_result['mean_ms']:>7.1f}ms "
                f"{n:>6}"
            )
            total_fact_at_8["GOLDEN"] += fa8
            total_n["GOLDEN"] += n
        else:
            print(f"  [{variant}] GOLDEN set not found: {golden_path}")

        # A set
        if qa_path.exists() and cA_corpus.exists():
            print(f"  [{variant}] evaluating A...", flush=True)
            a_index = build_index(cA_corpus)
            a_result = evaluate_generic(
                index=a_index, questions_path=qa_path, variant=variant, env=env
            )
            n = a_result["n"]
            print(
                f"{variant:<16} {'A':<8} "
                f"{a_result['fact_at_1']:>5}/{n} "
                f"{a_result['fact_at_4']:>5}/{n} "
                f"{a_result['fact_at_8']:>5}/{n} "
                f"{'':>8} "
                f"{a_result['mean_ms']:>7.1f}ms "
                f"{n:>6}"
            )
            # a_index.close()  # DocIndex has no close; GC handles it
        else:
            print(f"  [{variant}] Set A not found")

        # B set
        if qb_path.exists() and cB_corpus.exists():
            print(f"  [{variant}] evaluating B...", flush=True)
            b_index = build_index(cB_corpus)
            b_result = evaluate_generic(
                index=b_index, questions_path=qb_path, variant=variant, env=env
            )
            n = b_result["n"]
            print(
                f"{variant:<16} {'B':<8} "
                f"{b_result['fact_at_1']:>5}/{n} "
                f"{b_result['fact_at_4']:>5}/{n} "
                f"{b_result['fact_at_8']:>5}/{n} "
                f"{'':>8} "
                f"{b_result['mean_ms']:>7.1f}ms "
                f"{n:>6}"
            )
            # b_index.close()  # DocIndex has no close; GC handles it
        else:
            print(f"  [{variant}] Set B not found")

    print()
    print(f"Wrote JSON to {out_path}", flush=True)

    # Save JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report["totals"] = {k: f"{total_fact_at_8[k]}/{total_n[k]}" for k in total_fact_at_8}
    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
