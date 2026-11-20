# ruff: noqa
"""Calibrate our retrievers on a BEIR dataset (default use: scifact, nfcorpus, fiqa, arguana, scidocs), one chunk per document.

usage: python scripts/experiments/scifact_calibrate.py <data-dir> <beir-name> [bm25|fts5|hybrid ...]  (data from beir_prepare.py)
Expects <data-dir>/<name>/{corpus.jsonl,queries.jsonl,qrels/test.tsv}. Prints nDCG@10,
Recall@10/100, MRR@10 for the 300 test queries, plus build time, query p50/p95 and peak RSS.
BEIR BM25 reference (paper Table 2): scifact 0.665, fiqa 0.236, trec-covid 0.656.
"""

from __future__ import annotations

import hashlib, json, math, resource, statistics, sys, time, urllib.request, zipfile
from pathlib import Path

from voice_agent.rag.docindex import Chunk


def fetch(d: Path, name: str) -> Path:
    root = d / name
    if not (root / "corpus.jsonl").exists():
        raise SystemExit(
            f"missing {root}/corpus.jsonl, queries.jsonl, qrels/test.tsv: run scripts/experiments/beir_prepare.py (see docs/RAG-EXPERIMENTS.md, E11)"
        )
    return root


def load(root: Path):
    corpus = [json.loads(l) for l in (root / "corpus.jsonl").open()]
    queries = {j["_id"]: j["text"] for j in map(json.loads, (root / "queries.jsonl").open())}
    qrels: dict[str, dict[str, int]] = {}
    for i, line in enumerate((root / "qrels" / "test.tsv").open()):
        if i == 0:
            continue
        q, d, s = line.rstrip("\n").split("\t")
        if int(s) > 0:
            qrels.setdefault(q, {})[d] = int(s)
    return corpus, queries, qrels


def mk_chunks(corpus):
    out = []
    for i, d in enumerate(corpus):
        text = (d.get("title", "") + ". " + d["text"]).strip()
        h = hashlib.sha256(text.encode()).hexdigest()
        out.append(
            Chunk(
                id=d["_id"],
                file_sha256=h,
                chunk_sha256=h,
                position=0,
                page=None,
                section="",
                file_date="",
                source=d["_id"],
                header="",
                text=text,
            )
        )
    return out


def ndcg10(ranked, rel):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


CACHE = Path(sys.argv[1]).expanduser() / "cache" / sys.argv[2]


def run(mode, chunks, queries, qrels):
    from voice_agent.rag.backends import make_retriever_factory as mf

    if mode == "hybrid":
        from voice_agent.rag.hybrid import make_retriever_factory as hf

        r = hf(mode="hybrid", cache_dir=CACHE, background=False)()
    else:
        r = mf(mode=mode, cache_dir=None)()
    t = time.perf_counter()
    r.build(chunks)
    build = time.perf_counter() - t
    if hasattr(r, "wait_ready"):
        r.wait_ready(3600)
    nd, r10, r100, mrr, lat = [], [], [], [], []
    for q, rel in qrels.items():
        t = time.perf_counter()
        res = r.rank(queries[q], 100)
        lat.append(time.perf_counter() - t)
        ids = [chunks[i].id for i, _ in res]
        nd.append(ndcg10(ids, rel))
        r10.append(len(set(ids[:10]) & set(rel)) / len(rel))
        r100.append(len(set(ids[:100]) & set(rel)) / len(rel))
        first = next((k for k, d in enumerate(ids[:10]) if d in rel), None)
        mrr.append(1 / (first + 1) if first is not None else 0)
    lat.sort()
    m = statistics.mean
    print(
        f"{mode}: n={len(nd)} ndcg@10={m(nd):.3f} R@10={m(r10):.3f} R@100={m(r100):.3f} MRR@10={m(mrr):.3f} "
        f"build={build:.1f}s p50={lat[len(lat) // 2] * 1000:.1f}ms p95={lat[int(len(lat) * 0.95)] * 1000:.1f}ms "
        f"maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}MB",
        flush=True,
    )


if __name__ == "__main__":
    name = sys.argv[2]
    root = fetch(Path(sys.argv[1]).expanduser(), name)
    corpus, queries, qrels = load(root)
    chunks = mk_chunks(corpus)
    print(f"{name}: docs={len(corpus)} test queries={len(qrels)}")
    for mode in sys.argv[3:] or ["bm25", "fts5"]:
        run(mode, chunks, queries, qrels)
