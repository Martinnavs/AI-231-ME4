# ruff: noqa
"""Experiment E9b: prototype second-stage lexical reranker (query-term coverage x proximity window).

Negative result (2026-11-20): hits at k=4 on the 112 answerable external rows fell from 92 to 81-87.
Kept for reproducibility; not wired into the product. Usage: uv run python scripts/experiments/rerank_proto.py
"""

import json, os, sys, time

sys.path.insert(0, "src")
from pathlib import Path
from voice_agent.rag.docindex import DocIndex
from voice_agent.rag import metrics
from voice_agent.rag.backends import make_retriever_factory
from doc_chat.corpus import tokenize

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


class Rerank:
    def __init__(self, inner, getchunks, n=100, lam=1.0, win=40):
        self.inner, self.getchunks, self.n, self.lam, self.win = inner, getchunks, n, lam, win
        self.cache = {}

    def build(self, chunks):
        self.inner.build(chunks)
        self.chunks = list(chunks)
        self.cache = {}

    def rank(self, query, k):
        base = self.inner.rank(query, max(k, self.n))
        qt = list(dict.fromkeys(tokenize(query)))
        if not qt:
            return base[:k]
        top = base[: self.n]
        if not top:
            return base[:k]
        mx = max(s for _, s in top) or 1.0
        out = []
        for ci, s in top:
            toks = self.cache.get(ci)
            if toks is None:
                ch = self.chunks[ci]
                toks = tokenize(ch.header + " " + ch.text)
                self.cache[ci] = toks
            pos = {}
            for i, t in enumerate(toks):
                if t in qt:
                    pos.setdefault(t, []).append(i)
            cov = len(pos) / len(qt)
            # smallest window containing the most distinct query tokens (greedy by sorted positions)
            ev = sorted((i, t) for t, ps in pos.items() for i in ps)
            best = len(toks) or 1
            if ev:
                need = len(pos)
                cnt = {}
                have = 0
                l = 0
                for r, (ri, rt) in enumerate(ev):
                    cnt[rt] = cnt.get(rt, 0) + 1
                    if cnt[rt] == 1:
                        have += 1
                    while have == need:
                        best = min(best, ev[r][0] - ev[l][0] + 1)
                        lt = ev[l][1]
                        cnt[lt] -= 1
                        if cnt[lt] == 0:
                            have -= 1
                        l += 1
            prox = 1.0 / (1.0 + best / self.win)
            out.append((ci, s / mx + self.lam * cov * prox))
        out.sort(key=lambda x: -x[1])
        return out + base[self.n :]

    def close(self):
        self.inner.close() if hasattr(self.inner, "close") else None

    def status(self):
        return "rerank"

    def __getattr__(self, n):
        return getattr(self.inner, n)


def run(name, factory):
    idx = DocIndex(folder, converters=conv, canonical_cache_dir=None, retriever_factory=factory)
    idx.refresh()
    hit4 = hit3 = 0
    flips = []
    res = {}
    for g in gold:
        hits = idx.search(g["q"], k=4)
        h = [metrics.is_relevant(x, g["relevant"], match="tokens") for x in hits]
        res[g["id"]] = (any(h), any(h[:3]))
    return res


t0 = time.time()
base = run("base", make_retriever_factory("fts5"))
print(
    "base hit4",
    sum(v[0] for v in base.values()),
    "hit3",
    sum(v[1] for v in base.values()),
    "of",
    len(base),
    round(time.time() - t0),
)
for lam, win, n in [(1.0, 40, 100), (2.0, 40, 100), (1.0, 20, 100), (2.0, 20, 200)]:

    def fac(lam=lam, win=win, n=n):
        inner = make_retriever_factory("fts5")()
        return Rerank(inner, None, n=n, lam=lam, win=win)

    r = run("rr", fac)
    gain = [i for i in r if r[i][0] and not base[i][0]]
    loss = [i for i in r if base[i][0] and not r[i][0]]
    print(
        f"lam={lam} win={win} n={n}: hit4 {sum(v[0] for v in r.values())} hit3 {sum(v[1] for v in r.values())} gained {gain} lost {loss}",
        round(time.time() - t0),
    )
