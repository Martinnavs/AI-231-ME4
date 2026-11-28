"""scripts/bench_mmlongdoc_diagnose.py: the MMLongBench-Doc no-answer diagnosis
(ticket 274).

All through fakes (no relay, no corpus DB, no PDFs): a fake ``context_fn``
passage list and a fake lane result (a `DocResult`-shaped SimpleNamespace)
drive the pure classifier, one row per class R0-R4 plus the accepted row.
Also covered: the gold-answer-in-text matcher (exact string, normalised
number, List majority), the R0 judge prompt (question text alone - no gold
answer, no file name) and the judge's error path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "bench_mmlongdoc", _REPO / "scripts" / "bench_mmlongdoc.py"
)
assert _SPEC and _SPEC.loader
bm = importlib.util.module_from_spec(_SPEC)
sys.modules["bench_mmlongdoc"] = bm  # so the diagnose module's `import` resolves
_SPEC.loader.exec_module(bm)

_SPEC2 = importlib.util.spec_from_file_location(
    "bench_mmlongdoc_diagnose", _REPO / "scripts" / "bench_mmlongdoc_diagnose.py"
)
assert _SPEC2 and _SPEC2.loader
bd = importlib.util.module_from_spec(_SPEC2)
sys.modules["bench_mmlongdoc_diagnose"] = bd
_SPEC2.loader.exec_module(bd)

DOC = "docA.pdf"


def passage(source: str, text: str, page: Any = None, part: str = "chunk") -> dict[str, Any]:
    """One passage of the fake ``context_fn`` output (the passages_with_meta shape)."""
    return {
        "source": source,
        "id": f"{source}#whole" if part == "whole" else f"{source}#{part}-{page}",
        "section": "",
        "page": page,
        "file_date": "",
        "text": text,
        "score": 0.0,
        "coverage": 0.0,
    }


def classify(
    passages: list[dict[str, Any]],
    *,
    self_contained: bool | None = True,
    status: str = "none",
    rejected: Any = None,
    gold: str = "Radley Horton",
    answer_format: str = "Str",
    evidence_pages: list[int] | None = None,
    doc_id: str = DOC,
    status_reask: str | None = None,
) -> dict[str, Any]:
    """The classifier with a fake context (``passages``) and a fake lane result
    (``status``/``rejected``)."""
    return bd.classify_row(
        doc_id=doc_id,
        evidence_pages=evidence_pages if evidence_pages is not None else [3],
        gold=gold,
        answer_format=answer_format,
        passages=passages,
        self_contained=self_contained,
        status=status,
        rejected=rejected,
        status_reask=status_reask,
    )


def test_r0_question_not_self_contained_wins_over_good_retrieval() -> None:
    # the answer is in the retrieved gold page, but the judge says the question
    # only makes sense with the document in hand: R0, the shared-corpus ceiling
    out = classify(
        [passage(DOC, "Program led by Radley Horton", page=3)],
        self_contained=False,
    )
    assert out["class"] == "R0"
    assert out["page_in_meta"] is True
    assert out["answer_in_text"] is True


def test_r1_gold_document_not_retrieved_at_all() -> None:
    out = classify([passage("other.pdf", "unrelated text", page=1)])
    assert out["class"] == "R1"
    assert out["doc_retrieved"] is False
    assert out["retrieved_pages_of_gold_doc"] == []


def test_r1_doc_retrieved_without_its_evidence() -> None:
    # the gold doc's page 1 is in the context, but the evidence is page 3 and
    # the gold answer is nowhere in the gold doc's passages: a retrieval miss
    out = classify(
        [passage(DOC, "a page of the right document without the answer", page=1)],
        evidence_pages=[3],
    )
    assert out["class"] == "R1"
    assert out["doc_retrieved"] is True
    assert out["page_in_meta"] is False
    assert out["answer_in_text"] is False


def test_r2_evidence_page_retrieved_answer_not_in_text() -> None:
    # the evidence page is in the passages (e.g. a scanned page: text is
    # fragments), the gold answer string is absent: the text, not the
    # retrieval, is the problem
    out = classify(
        [passage(DOC, "fig. 3 ...", page=3)],
        evidence_pages=[3],
        gold="Radley Horton",
    )
    assert out["class"] == "R2"
    assert out["page_in_meta"] is True
    assert out["answer_in_text"] is False


def test_r3_answer_in_passages_lane_declined() -> None:
    # the gold answer IS in the retrieved evidence page, yet the lane failed
    # the quote check: the most actionable class
    out = classify(
        [passage(DOC, "Program led by Radley Horton", page=3)],
        evidence_pages=[3],
        status="check_failed",
        rejected={"failed": "quote", "answer": "Radley Horton", "quote": "x"},
    )
    assert out["class"] == "R3"
    assert out["rejected_failed"] == "quote"


def test_r3_answer_in_gold_passage_text_but_page_metadata_absent() -> None:
    # a grep/whole line of the gold doc carries the answer but the page
    # metadata is absent (page=None, not the whole file): still R3 - the lane
    # had the answer text - with a note saying so
    out = classify(
        [passage(DOC, "revenue was 1,234.5 million", page=None)],
        evidence_pages=[7],
        gold="1234.5",
        answer_format="Float",
    )
    assert out["class"] == "R3"
    assert out["page_in_meta"] is False
    assert out["whole_file"] is False
    assert any("grep/whole" in n for n in out["notes"])


def test_r4_lane_call_errored() -> None:
    out = classify(
        [passage(DOC, "Program led by Radley Horton", page=3)],
        evidence_pages=[3],
        status="error",
    )
    assert out["class"] == "R4"


def test_accepted_rows_are_class_accepted() -> None:
    out = classify(
        [passage(DOC, "Program led by Radley Horton", page=3)],
        evidence_pages=[3],
        status="accepted",
    )
    assert out["class"] == "accepted"
    assert out["rejected_failed"] == ""


def test_status_reask_flip_is_noted() -> None:
    out = classify(
        [passage(DOC, "Program led by Radley Horton", page=3)],
        evidence_pages=[3],
        status="none",
        status_reask="accepted",
    )
    assert out["class"] == "R3"
    assert any("re-ask status accepted" in n for n in out["notes"])


def test_gold_answer_in_text_exact_string() -> None:
    found, how = bd.gold_answer_in_text("Radley Horton", "Str", "led by RADLEY  horton ok")
    assert found and how == "string"


def test_gold_answer_in_text_normalised_number() -> None:
    found, how = bd.gold_answer_in_text("1234.5", "Float", "the figure was 1,234.50 m")
    assert found and how == "number"


def test_gold_answer_in_text_list_majority() -> None:
    gold = "['alpha', 'beta', 'gamma']"
    found, how = bd.gold_answer_in_text(gold, "List", "the items are alpha and beta here")
    assert found and how == "list 2/3"
    found, _ = bd.gold_answer_in_text(gold, "List", "only gamma is here")
    assert not found


def test_judge_prompt_sees_the_question_text_alone() -> None:
    prompt = bd.judge_prompt("Which individual led this program?")
    assert "Which individual led this program?" in prompt
    assert DOC not in prompt  # no file name
    assert "Radley" not in prompt  # no gold answer


def test_judge_error_returns_none_without_raising() -> None:
    class Dead:
        def post(self, payload: Any) -> dict[str, Any]:
            raise RuntimeError("relay down")

    flag, why = bd.judge_self_contained(Dead(), "model", "Which individual led this program?")
    assert flag is None
    assert "judge error" in why


def test_gold_doc_analysis_whole_file_and_pages() -> None:
    passages = [
        passage(DOC, "whole file text with the answer 1,234.5", part="whole", page=None),
        passage("other.pdf", "noise", page=2),
    ]
    a = bd.gold_doc_analysis(DOC, [7], "1234.5", "Float", passages)
    assert a["doc_retrieved"] is True
    assert a["whole_file"] is True
    assert a["page_in_meta"] is True
    assert a["retrieved_pages_of_gold_doc"] == []
    assert a["answer_in_text"] is True
    a2 = bd.gold_doc_analysis("missing.pdf", [7], "1234.5", "Float", passages)
    assert a2["doc_retrieved"] is False
    assert a2["answer_in_text"] is False


def test_db_helpers(tmp_path: Path) -> None:
    """zero_chunk_docs and answer_in_gold_doc_db on a synthetic corpus DB."""
    import sqlite3

    db = tmp_path / "corpus.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE files (rel_key TEXT, status TEXT)")
    con.execute("CREATE TABLE chunk_data (rel_key TEXT, page INTEGER, text TEXT)")
    con.execute("INSERT INTO files VALUES ('a.pdf', 'ok')")
    con.execute("INSERT INTO files VALUES ('slides.pdf', 'ok')")  # zero chunks
    con.execute("INSERT INTO files VALUES ('broken.pdf', 'failed')")
    con.execute("INSERT INTO chunk_data VALUES ('a.pdf', 3, 'the total is 1,234.5 million')")
    con.commit()
    con.close()
    assert bd.zero_chunk_docs(db) == ["slides.pdf"]
    assert bd.answer_in_gold_doc_db(db, "a.pdf", "1234.5", "Float") is True
    assert bd.answer_in_gold_doc_db(db, "a.pdf", "999.9", "Float") is False
    assert bd.answer_in_gold_doc_db(db, "slides.pdf", "1234.5", "Float") is None
