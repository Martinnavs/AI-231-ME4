"""scripts/bench_mmlongdoc.py: the MMLongBench-Doc offline bench (ticket 272).

All through fakes: synthetic rows (no pyarrow, no parquet, no network), a fake
relay transport, a scripted document lane. The logic under test: the seeded
sha256 sampling and split (deterministic, stable, matches the frozen rule), the
stratum assignment, the manifest counts adding up, the official-rules bridge on
a fixed (gold, extracted, format) table covering Str/Int/Float/List/Not
answerable, the extraction status mapping (a row the lane did not accept never
costs a relay call), the resumable runner, the status mapping onto run records,
and the aggregate buckets (the headline excludes vision and unanswerable).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "bench_mmlongdoc", Path(__file__).resolve().parents[1] / "scripts" / "bench_mmlongdoc.py"
)
assert _SPEC and _SPEC.loader
bm = importlib.util.module_from_spec(_SPEC)
sys.modules["bench_mmlongdoc"] = bm  # so late module references can resolve
_SPEC.loader.exec_module(bm)

OFFICIAL = bm.load_official()


def _official_score(gold: Any, pred: Any, fmt: str) -> float:
    with contextlib.redirect_stdout(io.StringIO()):
        return float(OFFICIAL.eval_score(gold, pred, fmt))


def make_raw_rows(n: int = 260) -> list[dict[str, Any]]:
    """Synthetic parquet-shaped rows: every 10th is unanswerable, the rest cycle
    through the evidence sources (so all four strata are populated)."""
    srcs = [
        "['Pure-text (Plain-text)']",
        "['Table']",
        "['Generalized-text (Layout)']",
        "['Chart']",
        "['Figure']",
        "['Chart', 'Pure-text (Plain-text)']",
        "['Table', 'Chart']",
    ]
    rows: list[dict[str, Any]] = []
    for i in range(n):
        unans = i % 10 == 0
        rows.append(
            {
                "doc_id": f"doc{i % 25}.pdf",
                "doc_type": ["Brochure", "Academic paper", "Financial report"][i % 3],
                "question": f"question {i}?",
                "answer": "Not answerable" if unans else f"answer {i}",
                "evidence_pages": "[1]" if not unans else "[0]",
                "evidence_sources": "[]" if unans else srcs[i % len(srcs)],
                "answer_format": "None" if unans else ["Str", "Int", "Float", "List"][i % 4],
            }
        )
    return rows


def selected() -> list[dict[str, Any]]:
    rows = bm.annotate(make_raw_rows())
    return bm.select_sample(rows, seed="7", total=200, unanswerable_quota=40)


# --- sampling & split -----------------------------------------------------------
def test_sampling_is_deterministic() -> None:
    s1 = selected()
    s2 = selected()
    assert [(r["qid"], r["split"]) for r in s1] == [(r["qid"], r["split"]) for r in s2]
    assert len(s1) == 200


def test_split_matches_the_frozen_rule() -> None:
    for r in selected():
        digest = hashlib.sha256(f"mmlongdoc-split:7:{r['qid']}".encode()).hexdigest()
        want = "dev" if int(digest, 16) % 2 == 0 else "test"
        assert r["split"] == want


def test_split_is_roughly_half_half() -> None:
    c = bm.count_sample(selected())
    assert 0.35 <= c["by_split"]["dev"] / 200 <= 0.65


def test_other_seed_selects_different_rows() -> None:
    rows = bm.annotate(make_raw_rows())
    s7 = {r["qid"] for r in bm.select_sample(rows, seed="7", total=200)}
    s8 = {r["qid"] for r in bm.select_sample(rows, seed="8", total=200)}
    assert s7 != s8


def test_stratum_assignment() -> None:
    assert bm.evidence_stratum(["Pure-text (Plain-text)"]) == "text"
    assert bm.evidence_stratum(["Table", "Chart"]) == "table"
    assert bm.evidence_stratum(["Generalized-text (Layout)"]) == "layout"
    assert bm.evidence_stratum(["Chart"]) == "chart_image"
    assert bm.evidence_stratum(["Figure", "Chart"]) == "chart_image"
    assert bm.is_vision_only(["Chart"])
    assert bm.is_vision_only(["Figure", "Chart"])
    assert not bm.is_vision_only(["Chart", "Pure-text (Plain-text)"])
    assert not bm.is_vision_only([])


def test_vision_only_rows_are_flagged_not_unflagged() -> None:
    for r in selected():
        if r["out_of_scope_vision"]:
            assert r["evidence_stratum"] == "chart_image"
        if r["evidence_stratum"] == "chart_image":
            assert r["out_of_scope_vision"]
        if r["unanswerable"]:
            assert not r["out_of_scope_vision"]
            assert r["evidence_stratum"] == "unanswerable"


# --- manifest counts --------------------------------------------------------------
def test_manifest_counts_add_up() -> None:
    sel = selected()
    c = bm.count_sample(sel)
    assert c["total"] == 200 == len(sel)
    assert c["by_split"]["dev"] + c["by_split"]["test"] == 200
    assert c["in_scope"] + c["out_of_scope_vision"] + c["unanswerable"] == 200
    assert sum(c["by_stratum"].values()) == 200
    assert sum(c["by_evidence"].values()) == 200
    assert sum(c["by_doc_type"].values()) == 200
    assert sum(c["by_answer_format"].values()) == 200
    # the unanswerable quota is min(quota, available)
    available_unans = sum(1 for r in bm.annotate(make_raw_rows()) if r["unanswerable"])
    assert c["unanswerable"] == min(40, available_unans)


def test_quotas_never_exceed_the_population() -> None:
    q = bm.allocate_quotas({"a": 100, "b": 30, "c": 5}, 50, minimum=1)
    assert sum(q.values()) == 50
    assert all(0 <= q[k] <= n for k, n in {"a": 100, "b": 30, "c": 5}.items())
    q2 = bm.allocate_quotas({"a": 1, "b": 1, "c": 1}, 2, minimum=1)
    assert sum(q2.values()) == 2


# --- the official-rules bridge ------------------------------------------------------
# The fixed table: (gold, extracted, answer_format, expected official score),
# covering Str / Int / Float / List / Not answerable.
TABLE = [
    ("Less well-off", "less well-off", "Str", 1.0),
    ("Not answerable", "Not answerable", "None", 1.0),
    ("Not answerable", "42", "None", 0.0),
    ("10", "10", "Int", 1.0),
    ("10", "10.4", "Int", 1.0),
    ("10", "9", "Int", 0.0),
    ("25.5", "25.5%", "Float", 1.0),
    ("25.5", "26.0", "Float", 0.0),
    ("['a', 'b']", "['b', 'a']", "List", 1.0),
    ("['a', 'b']", "['a']", "List", 0.0),
]


@pytest.mark.parametrize("gold,pred,fmt,want", TABLE, ids=lambda v: str(v).replace(" ", ""))
def test_scorer_wrapper_gives_the_official_score(
    gold: Any, pred: Any, fmt: str, want: float
) -> None:
    got = bm.score_row(gold, pred, fmt, OFFICIAL.eval_score)
    assert got == want
    # and it is the official rule, called as the official pipeline calls it
    assert got == _official_score(gold, pred, fmt)


# --- extraction: status mapping and the fake relay -----------------------------------
class FakeTransport:
    """In-memory relay: canned responses in order, every payload recorded."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def qwen_reply(text: str) -> dict[str, Any]:
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_non_accepted_rows_map_to_not_answerable_without_a_relay_call() -> None:
    t = FakeTransport([])
    for status in ("none", "check_failed", "timeout", "error"):
        row = {"id": "q0001", "status": status, "spoken": None, "question": "q?"}
        extracted, fmt, extractor = bm.extract_for_row(row, t, "model")
        assert extracted == "Not answerable"
        assert fmt == "String"
        assert extractor == f"lane:{status}"
    assert t.payloads == []  # no relay call, no spend


def test_accepted_spoken_rows_go_through_the_relay() -> None:
    reply = qwen_reply(json.dumps({"extracted_answer": "10", "answer_format": "Integer"}))
    t = FakeTransport([reply])
    row = {"id": "q0001", "status": "accepted", "spoken": "The answer is 10.", "question": "q?"}
    extracted, fmt, extractor = bm.extract_for_row(row, t, "model")
    assert (extracted, fmt, extractor) == ("10", "Integer", "qwen")
    assert len(t.payloads) == 1
    payload = t.payloads[0]
    assert payload["temperature"] == 0
    content = payload["messages"][-1]["content"]
    # the official prompt, the official question/analysis layout, and the JSON ask
    assert content.startswith(bm.OFFICIAL_EXTRACTOR_PROMPT)
    assert "Question:q?" in content
    assert "Analysis:The answer is 10." in content


def test_list_extraction_stays_a_list() -> None:
    t = FakeTransport(
        [qwen_reply(json.dumps({"extracted_answer": ["a", "b"], "answer_format": "List"}))]
    )
    row = {"id": "q0002", "status": "accepted", "spoken": "a and b", "question": "q?"}
    extracted, fmt, _ = bm.extract_for_row(row, t, "model")
    assert extracted == ["a", "b"]
    assert fmt == "List"


def test_score_rows_end_to_end_with_fakes() -> None:
    reply = qwen_reply(json.dumps({"extracted_answer": "10", "answer_format": "Integer"}))
    t = FakeTransport([reply])
    rows = [
        {
            "id": "q0000",
            "doc_id": "d.pdf",
            "question": "q0?",
            "answer": "10",
            "answer_format": "Int",
            "evidence_sources": ["Table"],
            "split": "dev",
            "out_of_scope_vision": False,
            "stratum": "table",
            "doc_type": "Brochure",
            "status": "accepted",
            "spoken": "The answer is 10.",
            "file": "d.pdf",
            "file_matches_doc_id": True,
        },
        {
            "id": "q0001",
            "doc_id": "d.pdf",
            "question": "q1?",
            "answer": "Not answerable",
            "answer_format": "None",
            "evidence_sources": ["Pure-text (Plain-text)"],
            "split": "dev",
            "out_of_scope_vision": False,
            "stratum": "unanswerable",
            "doc_type": "Brochure",
            "status": "none",
            "spoken": None,
            "file": None,
            "file_matches_doc_id": None,
        },
    ]
    scored = bm.score_rows(rows, t, "model", OFFICIAL.eval_score)
    assert scored[0]["score"] == 1.0 and scored[0]["extractor"] == "qwen"
    assert scored[1]["extracted"] == "Not answerable"
    assert scored[1]["extractor"] == "lane:none"
    assert scored[1]["score"] == 1.0
    assert scored[1]["unanswerable"] is True


# --- the resumable runner ------------------------------------------------------------
class FakeLane:
    """A DocResult-shaped lane that answers from the row id (no network)."""

    def __init__(self, fail_ids: frozenset[str] = frozenset()) -> None:
        self.fail_ids = fail_ids
        self.questions: list[str] = []

    async def __call__(self, question: str, row: Mapping[str, Any]) -> SimpleNamespace:
        self.questions.append(question)
        if row["id"] in self.fail_ids:
            return SimpleNamespace(
                status="error",
                spoken=None,
                file=None,
                quotes=[],
                cited_pages=[],
                quote_files=[],
                calls=1,
            )
        return SimpleNamespace(
            status="accepted",
            spoken=f"answer for {row['id']}",
            file=row["doc_id"],
            quotes=["quote"],
            cited_pages=[1],
            quote_files=[row["doc_id"]],
            calls=1,
        )


def run_rows(n: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"q{i:04d}",
            "doc_id": "d.pdf",
            "split": "dev",
            "question": f"q{i}?",
        }
        for i in range(n)
    ]


async def test_runner_writes_one_line_per_row(tmp_path: Path) -> None:
    out = tmp_path / "run-x.jsonl"
    lane = FakeLane()
    new, skipped = await bm.run_questions(run_rows(3), lane, out)
    assert skipped == 0 and len(new) == 3
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(ln)["id"] for ln in lines] == ["q0000", "q0001", "q0002"]
    assert lane.questions == ["q0?", "q1?", "q2?"]


async def test_runner_skips_done_ids(tmp_path: Path) -> None:
    out = tmp_path / "run-x.jsonl"
    await bm.run_questions(run_rows(3), FakeLane(), out)
    lane = FakeLane()
    new, skipped = await bm.run_questions(run_rows(4), lane, out)
    assert skipped == 3
    assert len(new) == 1 and new[0]["id"] == "q0003"
    assert lane.questions == ["q3?"]  # only the new id was answered
    ids = [json.loads(ln)["id"] for ln in out.read_text(encoding="utf-8").splitlines()]
    assert ids == ["q0000", "q0001", "q0002", "q0003"]


async def test_runner_records_errors_and_stops_point_is_the_callers(tmp_path: Path) -> None:
    out = tmp_path / "run-x.jsonl"
    lane = FakeLane(fail_ids=frozenset({"q0000"}))
    new, _ = await bm.run_questions(run_rows(2), lane, out)
    assert new[0]["status"] == "error" and new[0]["spoken"] is None
    assert new[1]["status"] == "accepted"


def test_record_from_result_maps_the_docresult() -> None:
    row = {"id": "q0000", "doc_id": "PH_x.pdf", "split": "dev", "question": "q?"}
    res = SimpleNamespace(
        status="accepted",
        spoken="As per PH_x.pdf: 10.",
        file="PH_x.pdf",
        quotes=["10"],
        cited_pages=[3],
        quote_files=["PH_x.pdf"],
        calls=2,
    )
    rec = bm.record_from_result(row, res, 1.2345)
    assert rec["status"] == "accepted"
    assert rec["spoken"] == "As per PH_x.pdf: 10."
    assert rec["file"] == "PH_x.pdf"
    assert rec["file_matches_doc_id"] is True
    assert rec["quotes"] == ["10"]
    assert rec["cited_pages"] == [3]
    assert rec["quote_files"] == ["PH_x.pdf"]
    assert rec["calls"] == 2
    assert rec["seconds"] == 1.234  # rounded to ms
    # a citation of another file does not match the question's doc_id
    other = SimpleNamespace(
        status="accepted",
        spoken="x",
        file="other.pdf",
        quotes=[],
        cited_pages=[],
        quote_files=[],
        calls=1,
    )
    assert bm.record_from_result(row, other, 0.5)["file_matches_doc_id"] is False
    # nothing cited at all
    nothing = SimpleNamespace(
        status="none",
        spoken=None,
        file=None,
        quotes=[],
        cited_pages=[],
        quote_files=[],
        calls=1,
    )
    rec3 = bm.record_from_result(row, nothing, 0.5)
    assert rec3["file_matches_doc_id"] is None and rec3["spoken"] is None


# --- aggregates --------------------------------------------------------------------
def scored_rows_fixture() -> list[dict[str, Any]]:
    return [
        {
            "id": "q0000",
            "gold": "10",
            "extracted": "10",
            "score": 1.0,
            "answer_format": "Int",
            "evidence_sources": ["Table"],
            "split": "dev",
            "out_of_scope_vision": False,
            "unanswerable": False,
            "stratum": "table",
            "file_matches_doc_id": True,
            "extractor": "qwen",
        },
        {
            "id": "q0001",
            "gold": "x",
            "extracted": "y",
            "score": 0.0,
            "answer_format": "Str",
            "evidence_sources": ["Pure-text (Plain-text)"],
            "split": "dev",
            "out_of_scope_vision": False,
            "unanswerable": False,
            "stratum": "text",
            "file_matches_doc_id": False,
            "extractor": "qwen",
        },
        {
            "id": "q0002",
            "gold": "Not answerable",
            "extracted": "Not answerable",
            "score": 1.0,
            "answer_format": "None",
            "evidence_sources": ["Pure-text (Plain-text)"],
            "split": "dev",
            "out_of_scope_vision": False,
            "unanswerable": True,
            "stratum": "unanswerable",
            "file_matches_doc_id": None,
            "extractor": "lane:none",
        },
        {
            "id": "q0003",
            "gold": "v",
            "extracted": "v",
            "score": 1.0,
            "answer_format": "Str",
            "evidence_sources": ["Chart"],
            "split": "test",
            "out_of_scope_vision": True,
            "unanswerable": False,
            "stratum": "chart_image",
            "file_matches_doc_id": True,
            "extractor": "qwen",
        },
    ]


def test_aggregate_buckets_exclude_vision_from_the_headline() -> None:
    agg = bm.aggregate_scores(scored_rows_fixture(), OFFICIAL.eval_acc_and_f1)
    assert agg["n_rows"] == 4
    assert agg["headline"]["n"] == 2 and agg["headline"]["acc"] == 0.5
    assert agg["core"]["n"] == 3 and agg["core"]["acc"] == round(2 / 3, 4)
    assert agg["unanswerable"] == {"n": 1, "acc": 1.0}
    assert agg["out_of_scope_vision"]["n"] == 1
    assert agg["out_of_scope_vision"]["acc"] == 1.0
    assert agg["official_overall"]["acc"] == 0.75
    # by_split / by_source / by_format are computed on the in-scope rows only
    assert agg["by_split"] == {"dev": {"n": 3, "acc": round(2 / 3, 4)}}
    assert "Chart" not in agg["by_evidence_source"]
    assert agg["by_evidence_source"]["Table"] == {"n": 1, "acc": 1.0}
    assert agg["by_answer_format"]["Int"] == {"n": 1, "acc": 1.0}
    assert agg["cited_file_match"] == {"n": 2, "matches": 1}
    assert agg["extractors"] == {"lane:none": 1, "qwen": 3}


def test_official_f1_is_unanswerable_aware() -> None:
    # the official F1 over the in-scope rows: recall/precision per eval_acc_and_f1
    agg = bm.aggregate_scores(scored_rows_fixture(), OFFICIAL.eval_acc_and_f1)
    # scores of answerable-gold rows: 1.0 + 0.0 = 1.0; n answerable = 2 -> recall 0.5
    # preds != 'Not answerable': 2 -> precision 0.5 -> F1 0.5
    assert agg["core"]["f1"] == 0.5


def test_calibration_selection_is_stratified_and_seeded() -> None:
    rows = scored_rows_fixture()
    sel = bm.select_calibration(rows, n=3, seed="7")
    assert len(sel) == 3
    # seeded: stable
    assert [r["id"] for r in sel] == [r["id"] for r in bm.select_calibration(rows, n=3, seed="7")]
    # stratified: the two strata present each get >= 1
    strata = {r["id"]: r["stratum"] for r in rows}
    assert len({strata[r["id"]] for r in sel}) >= 2


def test_join_run_with_questions_raises_on_unknown_id() -> None:
    questions = [
        {
            "id": "q0000",
            "doc_id": "d.pdf",
            "question": "q0?",
            "answer": "10",
            "answer_format": "Int",
            "evidence_sources": ["Table"],
            "split": "dev",
            "flags": {"out_of_scope_vision": False, "stratum": "table", "doc_type": "Brochure"},
        }
    ]
    joined = bm.join_run_with_questions(
        [{"id": "q0000", "status": "accepted", "spoken": "s", "question": "q0?"}], questions
    )
    assert joined[0]["answer"] == "10"
    with pytest.raises(RuntimeError, match="not in"):
        bm.join_run_with_questions([{"id": "q9999", "status": "none"}], questions)


def test_read_done_ids_ignores_torn_lines(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    p.write_text('{"id": "q0000"}\n{"id": "q0001"}\n{"id": "q0002"', encoding="utf-8")
    assert bm.read_done_ids(p) == {"q0000", "q0001"}
    assert bm.read_done_ids(tmp_path / "missing.jsonl") == set()
