# ruff: noqa
"""Experiment E9c: sweep the per-file cap / score gate / dedup constants of DocIndex.search offline.

Result (2026-11-20): flat plateau; the committed 2 / 4 / 0.75 / 0.75 is within one row of the best.
Usage: uv run python scripts/experiments/sweep_cap.py (one index build, monkeypatches module constants).
"""

import json, os, sys, itertools

sys.path.insert(0, "src")
from pathlib import Path
import voice_agent.rag.docindex as d
from voice_agent.rag.docindex import DocIndex
from voice_agent.rag import metrics
from voice_agent.rag.backends import make_retriever_factory

folder = Path(os.path.expanduser("~/rag-testsets/ipo-2026/raw"))
gold = [json.loads(l) for l in open(os.path.expanduser("~/rag-testsets/ipo-2026/golden.jsonl"))]
gold = [g for g in gold if g["type"] not in ("unanswerable", "off_topic")]
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


# cache the ranked lists once (search ranks internally; call it per config but retriever is fast)
def run(soft, hard, ratio, jac):
    d.MAX_CHUNKS_PER_FILE, d.HARD_CHUNKS_PER_FILE, d.OVERFLOW_SCORE_RATIO, d.DEDUP_JACCARD = (
        soft,
        hard,
        ratio,
        jac,
    )
    r = {}
    for g in gold:
        hits = idx.search(g["q"], k=4)
        h = [metrics.is_relevant(x, g["relevant"], match="tokens") for x in hits]
        r[g["id"]] = any(h)
    return r


base = run(2, 4, 0.75, 0.75)
print("committed config hit4", sum(base.values()))
grid = [
    (2, 4, 0.75, 0.75),
    (2, 4, 0.5, 0.75),
    (2, 4, 0.6, 0.75),
    (2, 4, 0.9, 0.75),
    (2, 6, 0.6, 0.75),
    (3, 6, 0.6, 0.75),
    (2, 4, 0.0, 0.75),
    (2, 4, 0.75, 0.6),
    (2, 4, 0.75, 0.9),
    (2, 4, 0.75, 2.0),
    (1, 4, 0.75, 0.75),
    (2, 2, 0.75, 0.75),
    (3, 4, 0.75, 0.75),
    (2, 8, 0.5, 0.75),
]
for cfg in grid:
    r = run(*cfg)
    gain = [i for i in r if r[i] and not base[i]]
    loss = [i for i in r if base[i] and not r[i]]
    print(cfg, "hit4", sum(r.values()), "gained", gain, "lost", loss)
