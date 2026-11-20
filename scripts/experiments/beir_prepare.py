# ruff: noqa
"""Build <data-dir>/<name>/{corpus.jsonl,queries.jsonl,qrels/test.tsv} from HF BeIR/<name> + BeIR/<name>-qrels.

usage: uvx --from huggingface_hub --with pandas --with pyarrow python scripts/experiments/beir_prepare.py <data-dir> <name> [<name> ...]
Names: scifact nfcorpus fiqa arguana scidocs trec-covid webis-touche2020 quora ... (data stays outside the repo).
"""
import json, shutil, sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download as h, list_repo_files

d = Path(sys.argv[1]).expanduser()
for name in sys.argv[2:]:
    out = d / name
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    files = list_repo_files(f"BeIR/{name}", repo_type="dataset")
    c = pd.read_parquet(h(f"BeIR/{name}", next(f for f in files if f.startswith("corpus/")), repo_type="dataset"))
    q = pd.read_parquet(h(f"BeIR/{name}", next(f for f in files if f.startswith("queries/")), repo_type="dataset"))
    with (out / "corpus.jsonl").open("w") as f:
        for r in c.to_dict("records"):
            f.write(json.dumps({"_id": str(r["_id"]), "title": r.get("title") or "", "text": r["text"]}) + "\n")
    with (out / "queries.jsonl").open("w") as f:
        for r in q.to_dict("records"):
            f.write(json.dumps({"_id": str(r["_id"]), "text": r["text"]}) + "\n")
    shutil.copy(h(f"BeIR/{name}-qrels", "test.tsv", repo_type="dataset"), out / "qrels" / "test.tsv")
    print(name, len(c), "docs", len(q), "queries")
