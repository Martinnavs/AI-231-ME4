"""Regression guard: ASCII questions must find accented passages (WP6 folding)."""

from __future__ import annotations

import json
from pathlib import Path

from voice_agent.rag.docindex import DocIndex

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_ascii_questions_find_accented_passages() -> None:
    index = DocIndex(FIXTURES / "rag_accent", canonical_cache_dir=None)
    index.refresh()
    lines = (FIXTURES / "rag_accent_golden.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    hits = sum(
        any(row["relevant"][0]["contains"] in hit["text"] for hit in index.search(row["q"], 4))
        for row in rows
    )
    assert len(rows) == 10
    assert hits >= 8
