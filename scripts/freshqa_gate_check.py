"""How well does the changeable-question gate match FreshQA's labels? No network, no model calls.

    curl -sL "https://docs.google.com/spreadsheets/d/<id>/export?format=csv" -o freshqa.csv
    .venv/bin/python scripts/freshqa_gate_check.py freshqa.csv [--lr]

(<id>: the newest sheet linked from the README of github.com/freshllms/freshqa.) FreshQA (Vu et al.
2023) labels each question never-, slow- or fast-changing, and false-premise or not. The gate
(`gemini_grounding.changeable`) should flag the last two. --lr adds a bag-of-words logistic
regression under 5-fold cross-validation as a yardstick for a learned gate: 600 questions from one
team of writers, so an optimistic figure for our traffic, not an estimate of it.
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
from pathlib import Path

from voice_agent import gate
from voice_agent.mcp_servers.gemini_grounding import changeable_regex as changeable


def load(path: Path) -> list[dict[str, str]]:
    rows = list(csv.reader(path.open()))
    start = next(i for i, r in enumerate(rows) if r and r[0] == "id")
    return [
        dict(zip(rows[start], r, strict=False)) for r in rows[start + 1 :] if r and r[0].isdigit()
    ]


def rate(data: list[dict[str, str]], label: str) -> None:
    flagged = sum(changeable(d["question"]) for d in data)
    print(
        f"{label:34s} n={len(data):3d} flagged={flagged:3d} "
        f"({100 * flagged / max(len(data), 1):.0f}%)"
    )


def rate_flag(data: list[dict[str, str]], flag: dict[str, bool], label: str) -> None:
    flagged = sum(flag[d["question"]] for d in data)
    print(f"{label:34s} n={len(data):3d} flagged={flagged:3d} ({100 * flagged / len(data):.0f}%)")


def logistic_cv(data: list[dict[str, str]]) -> None:
    import numpy as np  # only for --lr

    questions = [d["question"] for d in data]
    y = np.array([d["fact_type"] != "never-changing" for d in data], float)

    def feats(s: str) -> list[str]:
        t = re.findall(r"[a-z0-9']+", s.lower())
        return t + [a + "_" + b for a, b in zip(t, t[1:], strict=False)]

    count = collections.Counter(f for q in questions for f in set(feats(q)))
    vocab = {f: i for i, f in enumerate(f for f, c in count.items() if c >= 2)}
    x = np.zeros((len(questions), len(vocab) + 1))
    for i, q in enumerate(questions):
        for f in feats(q):
            if f in vocab:
                x[i, vocab[f]] = 1
        x[i, -1] = 1
    folds = np.array_split(np.random.default_rng(0).permutation(len(questions)), 5)
    prob = np.zeros(len(questions))
    for held in folds:
        train = np.setdiff1d(np.arange(len(questions)), held)
        w = np.zeros(x.shape[1])
        for _ in range(400):
            p = 1 / (1 + np.exp(-x[train] @ w))
            w -= 0.5 * (x[train].T @ (p - y[train]) / len(train) + w / len(train))
        prob[held] = 1 / (1 + np.exp(-x[held] @ w))
    rule = np.array([changeable(q) for q in questions], float)
    learned = (prob >= 0.5) * 1.0
    gates = {
        "regex": rule,
        "logistic regression, cv, 0.5": learned,
        "regex OR logistic regression": np.maximum(rule, learned),
    }
    for name, flag in gates.items():
        tp, fp = ((flag == 1) & (y == 1)).sum(), ((flag == 1) & (y == 0)).sum()
        fn, tn = ((flag == 0) & (y == 1)).sum(), ((flag == 0) & (y == 0)).sum()
        print(
            f"{name:34s} recall(changing)={tp / (tp + fn):.2f} "
            f"false alarm(never)={fp / (fp + tn):.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--lr", action="store_true")
    args = parser.parse_args()
    data = load(args.csv)
    for kind in ("never-changing", "slow-changing", "fast-changing"):
        rate([d for d in data if d["fact_type"] == kind], kind)
    changing = [d for d in data if d["fact_type"] != "never-changing"]
    rate([d for d in changing if d["false_premise"] != "TRUE"], "changing, valid premise")
    model = gate.load()
    if model is None:
        print("EverGreen gate: not installed (scripts/export_gate_onnx.sh)")
    else:
        flag = {d["question"]: model.p_mutable(d["question"]) >= gate.THRESHOLD for d in data}
        for kind in ("never-changing", "slow-changing", "fast-changing"):
            rate_flag([d for d in data if d["fact_type"] == kind], flag, "EverGreen gate: " + kind)
    if args.lr:
        logistic_cv(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
