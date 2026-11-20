# ruff: noqa
"""Experiment E9a: raw BM25/FTS5 rank of the answer-bearing chunk for hand-picked golden rows.

Usage: uv run python scripts/experiments/deep_rank.py  (needs ~/rag-testsets/ipo-2026; edit ids below).
Prints, per row, the 1-based rank of the first chunk that matches the anchor (--match tokens rule) in the
full ranking, and how many chunks the answer file has. Read-only; writes nothing.
"""

import json, os, sys

sys.path.insert(0, "src")
from pathlib import Path
from voice_agent.rag.docindex import DocIndex
from voice_agent.rag import metrics
from voice_agent.rag.backends import make_retriever_factory

folder = Path(os.path.expanduser("~/rag-testsets/ipo-2026/raw"))
gold = {}
for l in open(os.path.expanduser("~/rag-testsets/ipo-2026/golden.jsonl")):
    d = json.loads(l)
    gold[d["id"]] = d
ids = ["para-1", "tab-4", "num-3", "num-4", "lex-2", "para-2", "tab-11", "num-11", "para-3"]
conv = {
    "pdf": "legacy",
    "docx": "legacy",
    "pptx": "direct",
    "xlsx": "direct",
    "html": "direct",
    "epub": "direct",
}
idx = DocIndex(
    folder,
    converters=conv,
    canonical_cache_dir=None,
    retriever_factory=make_retriever_factory("fts5"),
)
idx.refresh()
print("chunks", len(idx.chunks))
for i in ids:
    g = gold[i]
    q = g["q"]
    ranked = idx._retriever.rank(q, len(idx.chunks))
    found = None
    for pos, (ci, score) in enumerate(ranked, 1):
        ch = idx.chunks[ci]
        if metrics.is_relevant(
            {"source": ch.source, "text": ch.text}, g["relevant"], match="tokens"
        ):
            found = pos
            break
    f = g["relevant"][0]["file"]
    nfile = sum(1 for c in idx.chunks if c.source == f)
    print(f"{i:8} answer-chunk raw rank={found} of {len(ranked)}; file has {nfile} chunks")
