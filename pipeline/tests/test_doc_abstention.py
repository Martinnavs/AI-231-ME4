"""Offline abstention and verbatim-quote contract tests over the real passage snapshot."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from voice_pipeline import doc_answer as da

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = Path(__file__).resolve().parent / "fixtures" / "rag_passages.json"
GOLDEN_PATH = REPO_ROOT / "config" / "eval" / "rag_golden.jsonl"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _load_snapshot() -> dict[str, Any]:
    data = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _load_golden() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _fake_request(reply: str) -> da.Request:
    async def request(_system: str, _content: str) -> str:
        return reply

    return request


def _norm_ws(text: str) -> str:
    return " ".join(text.split())


def _matching_anchor(
    passages: Sequence[dict[str, Any]],
    relevant: Sequence[dict[str, Any]],
) -> dict[str, str] | None:
    for rel in relevant:
        file_name = str(rel["file"])
        contains = str(rel["contains"])
        norm_anchor = _norm_ws(contains)
        for passage in passages:
            if passage["source"] == file_name and norm_anchor in _norm_ws(passage["text"]):
                return {"file": file_name, "contains": contains}
    return None


def _first_real_sentence(text: str) -> str:
    candidates = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    for candidate in candidates:
        if len(candidate.split()) >= 5:
            return candidate
    return candidates[0] if candidates else text.strip()


async def test_answerable_questions_with_anchor_in_snapshot_are_accepted() -> None:
    snapshot = _load_snapshot()
    golden = _load_golden()

    checked: list[str] = []
    old_generic = 0
    strict_generic = 0
    middle_generic_ok: list[str] = []
    middle_generic_overlap_fail: list[str] = []
    old_restated = 0
    strict_restated = 0

    for row in golden:
        if row["type"] in {"unanswerable", "off_topic"}:
            continue
        question = str(row["q"])
        passages = da.passages_with_meta(snapshot["questions"][question])
        anchor = _matching_anchor(passages, row["relevant"])
        if anchor is None:
            continue
        row_id = str(row["id"])
        checked.append(row_id)
        restated = f"{question.rstrip('?').lower()}."
        reply = f"ANSWER: {restated}\nQUOTE: {anchor['contains']}"
        got = await da.answer(_fake_request(reply), passages, question)
        assert got.status == "accepted", f"{row_id} failed: {got}"
        assert got.text == restated
        assert got.passage is not None
        assert got.passage["file"] == anchor["file"]

        matching_passages = [p for p in passages if da.quote_ok(anchor["contains"], p["text"])]
        generic_answer = "The document states this detail."
        strict_generic_text = f"{generic_answer} {anchor['contains']}"
        strict_restated_text = f"{restated} {anchor['contains']}"
        if any(da.overlap_ok(p["text"], question) for p in matching_passages):
            old_generic += 1
        if any(da.overlap_ok(strict_generic_text, question) for p in matching_passages):
            strict_generic += 1
        if any(da.overlap_ok(p["text"], question) for p in matching_passages):
            old_restated += 1
        if any(da.overlap_ok(strict_restated_text, question) for p in matching_passages):
            strict_restated += 1

        generic_reply = f"ANSWER: {generic_answer}\nQUOTE: {anchor['contains']}"
        generic_got = await da.answer(_fake_request(generic_reply), passages, question)
        if generic_got.status == "accepted":
            middle_generic_ok.append(row_id)
        else:
            assert generic_got.status == "check_failed", f"{row_id}: {generic_got}"
            assert generic_got.rejected is not None and generic_got.rejected["failed"] == "overlap"
            middle_generic_overlap_fail.append(row_id)

    assert len(checked) >= 15
    assert "num-7" in checked
    # para-1 left this list when the cap lift and tokenizer changes reordered its snapshot passages
    # (its passage now shares enough words with the question to pass the overlap check).
    assert middle_generic_overlap_fail == ["para-5"]
    n = len(checked)
    print(
        "\n[doc_abstention distribution]"
        f" checked={n}/29"
        f" generic(old={old_generic}/{n}, strict={strict_generic}/{n},"
        f" middle={len(middle_generic_ok)}/{n}, fail={middle_generic_overlap_fail})"
        f" restated(old={old_restated}/{n}, strict={strict_restated}/{n}, middle={n}/{n})"
    )


async def test_unanswerable_and_off_topic_invented_quotes_are_all_rejected() -> None:
    snapshot = _load_snapshot()
    golden = _load_golden()

    targets = [row for row in golden if row["type"] in {"unanswerable", "off_topic"}]
    assert len(targets) == 7

    invented_reply = (
        "ANSWER: The deadline is 31 December 2031.\nQUOTE: The deadline is 31 December 2031."
    )
    accepted = 0
    for row in targets:
        question = str(row["q"])
        passages = da.passages_with_meta(snapshot["questions"][question])
        got = await da.answer(_fake_request(invented_reply), passages, question)
        if got.status == "accepted":
            accepted += 1
        if passages:
            assert got.status == "check_failed", f"{row['id']} unexpected status: {got}"
            assert got.rejected is not None and got.rejected["failed"] == "quote"
        else:
            assert got.status == "none", f"{row['id']} unexpected status: {got}"
        assert got.text is None

    assert accepted == 0


async def test_unanswerable_real_top_passage_quote_limit_is_logged() -> None:
    snapshot = _load_snapshot()
    golden = _load_golden()

    unanswerable = [row for row in golden if row["type"] == "unanswerable"]
    assert len(unanswerable) == 4

    accepted = 0
    details: list[str] = []
    for row in unanswerable:
        question = str(row["q"])
        passages = da.passages_with_meta(snapshot["questions"][question])
        if not passages:
            details.append(f"{row['id']}:none(no_passages)")
            continue
        quote = _first_real_sentence(passages[0]["text"])
        reply = f"ANSWER: The document mentions this detail.\nQUOTE: {quote}"
        got = await da.answer(_fake_request(reply), passages, question)
        if got.status == "accepted":
            accepted += 1
        details.append(f"{row['id']}:{got.status}")

    print(
        f"\n[doc_abstention limit] unanswerable with real top-passage quote accepted="
        f"{accepted}/{len(unanswerable)} ({', '.join(details)})"
    )
