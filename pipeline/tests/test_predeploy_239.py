"""239 next-year guard: a question about the NEXT year ("What will the Social Security wage
base be next year?") cannot be answered by the current year's documents; the answer is
accepted only when the target year (current year + 1, from an injectable `now`) appears in
the answer or a cited quote, else the check fails with failed='next_year'."""

from __future__ import annotations

from datetime import datetime

from voice_pipeline.ask_documents import _run_checks

P15 = "p15_employers_tax_guide.pdf"


def _p(source: str, text: str) -> dict:
    return {"source": source, "id": f"{source}#0", "text": text, "part": "whole"}


def _reply(answer: str, quote: str) -> dict:
    return {"status": "answer", "answer": answer, "file": P15, "quotes": [quote]}


def _failed(question: str, passage: dict, answer: str, quote: str, now: datetime) -> str | None:
    return _run_checks(_reply(answer, quote), [passage], question, [], now=now)[4]


def _status(question: str, passage: dict, answer: str, quote: str, now: datetime) -> str:
    return _run_checks(_reply(answer, quote), [passage], question, [], now=now)[0]


# 2026 documents: they can only answer about 2026.
PASSAGE_2026 = _p(P15, "The social security wage base limit for 2026 is $184,500.")
A_2026 = "The social security wage base limit for 2026 is $184,500"
Q_2026 = "The social security wage base limit for 2026 is $184,500"

# 2027 documents: they can answer the next-year question.
PASSAGE_2027 = _p(P15, "The social security wage base limit for 2027 is $190,000.")
A_2027 = "The social security wage base limit for 2027 is $190,000"
Q_2027 = "The social security wage base limit for 2027 is $190,000"

NEXT_YEAR_Q = "What will the Social Security wage base be next year?"


# (1) next-year question, answer/quotes only mention 2026 -> fails next_year


def test_next_year_question_2026_only_fails() -> None:
    now = datetime(2026, 11, 26)
    assert _failed(NEXT_YEAR_Q, PASSAGE_2026, A_2026, Q_2026, now) == "next_year"
    assert _status(NEXT_YEAR_Q, PASSAGE_2026, A_2026, Q_2026, now) == "next_year"


def test_next_year_question_variants_fail() -> None:
    now = datetime(2026, 11, 26)
    for q in (
        "What will the Social Security wage base be next fiscal year?",
        "What is the Social Security wage base in the coming year?",
        "What was the wage base year after this?",
        "What will the wage base be next year's?",
    ):
        assert _failed(q, PASSAGE_2026, A_2026, Q_2026, now) == "next_year", q


# (2) next-year question where a quote contains "2027" -> accepted


def test_next_year_question_quote_has_2027_accepted() -> None:
    now = datetime(2026, 11, 26)
    assert _failed(NEXT_YEAR_Q, PASSAGE_2027, A_2027, Q_2027, now) is None
    assert _status(NEXT_YEAR_Q, PASSAGE_2027, A_2027, Q_2027, now) == "ok"


def test_next_year_question_fy_forms_accepted() -> None:
    now = datetime(2026, 11, 26)
    passage = _p(P15, "The FY2027 social security wage base limit is $190,000.")
    a = "The FY2027 social security wage base limit is $190,000"
    quote = "The FY2027 social security wage base limit is"
    assert _failed(NEXT_YEAR_Q, passage, a, quote, now) is None
    passage2 = _p(P15, "The social security wage base limit for FY27 is $190,000.")
    a2 = "The social security wage base limit for FY27 is $190,000"
    quote2 = "The social security wage base limit for FY27"
    assert _failed(NEXT_YEAR_Q, passage2, a2, quote2, now) is None


# (3) "this year" / explicit-year questions unchanged


def test_this_year_question_unchanged() -> None:
    now = datetime(2026, 11, 26)
    q = "What is the Social Security wage base this year?"
    assert _failed(q, PASSAGE_2026, A_2026, Q_2026, now) is None
    assert _status(q, PASSAGE_2026, A_2026, Q_2026, now) == "ok"


def test_last_year_question_unchanged() -> None:
    now = datetime(2026, 11, 26)
    passage = _p(P15, "The social security wage base limit for 2025 is $168,600.")
    a = "The social security wage base limit for 2025 is $168,600"
    assert _failed("What was the Social Security wage base last year?", passage, a, a, now) is None


def test_explicit_year_question_unchanged() -> None:
    now = datetime(2026, 11, 26)
    q = "What is the 2026 Social Security wage base?"
    assert _failed(q, PASSAGE_2026, A_2026, Q_2026, now) is None
    assert _status(q, PASSAGE_2026, A_2026, Q_2026, now) == "ok"


# (4) a question with no year words unchanged


def test_no_year_words_unchanged() -> None:
    now = datetime(2026, 11, 26)
    q = "What is the Social Security wage base?"
    assert _failed(q, PASSAGE_2026, A_2026, Q_2026, now) is None
    assert _status(q, PASSAGE_2026, A_2026, Q_2026, now) == "ok"


# (5) now injected as 2027 -> target 2028


def test_now_2027_targets_2028() -> None:
    now = datetime(2027, 1, 1)
    # 2027-only documents cannot answer the 2028 next-year question
    assert _failed(NEXT_YEAR_Q, PASSAGE_2027, A_2027, Q_2027, now) == "next_year"
    # ...but 2028 documents can
    passage = _p(P15, "The social security wage base limit for 2028 is $195,000.")
    a = "The social security wage base limit for 2028 is $195,000"
    assert _failed(NEXT_YEAR_Q, passage, a, a, now) is None
    assert _status(NEXT_YEAR_Q, passage, a, a, now) == "ok"
