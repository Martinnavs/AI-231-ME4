# ruff: noqa
"""Build a BEIR-format passage-retrieval set from SQuAD v2 validation (verbatim answer spans, cc-by-sa-4.0).

usage: uvx --from huggingface_hub --with pandas --with pyarrow python scripts/experiments/squad_prepare.py <data-dir>
Corpus = every unique validation context (one chunk each); query = answerable question; qrel = its own context.
Run with: python scripts/experiments/beir_calibrate.py <data-dir> squad2dev bm25 fts5
"""
import hashlib, json, sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download as h, list_repo_files

out = Path(sys.argv[1]).expanduser() / "squad2dev"
(out / "qrels").mkdir(parents=True, exist_ok=True)
f = next(x for x in list_repo_files("rajpurkar/squad_v2", repo_type="dataset") if "validation" in x and x.endswith(".parquet"))
df = pd.read_parquet(h("rajpurkar/squad_v2", f, repo_type="dataset"))
ids, qn = {}, 0
with (out / "queries.jsonl").open("w") as fq, (out / "qrels" / "test.tsv").open("w") as fr:
    fr.write("query-id\tcorpus-id\tscore\n")
    for r in df.to_dict("records"):
        if len(r["answers"]["text"]) == 0:
            continue
        cid = ids.setdefault(r["context"], "c" + hashlib.sha1(r["context"].encode()).hexdigest()[:12])
        qn += 1
        fq.write(json.dumps({"_id": f"q{qn}", "text": r["question"]}) + "\n")
        fr.write(f"q{qn}\t{cid}\t1\n")
with (out / "corpus.jsonl").open("w") as fc:
    for ctx, cid in ids.items():
        fc.write(json.dumps({"_id": cid, "title": "", "text": ctx}) + "\n")
print("squad2dev", len(ids), "passages", qn, "answerable queries")
