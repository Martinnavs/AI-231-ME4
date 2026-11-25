"""Live golden-v2 failures: whole-row quotes and up/down comparisons."""

from __future__ import annotations

from voice_pipeline.ask_documents import _eval_calc, _run_checks, _trim_quote


def _p(source: str, text: str) -> dict:
    return {"source": source, "id": f"{source}#0", "text": text, "part": "whole"}


def test_whole_row_quote_is_trimmed_to_a_verbatim_window() -> None:
    row = " ".join(f"c{i}: {i * 7}" for i in range(40)) + " Total: 26535"
    cut = _trim_quote(row, "26535")
    assert len(cut.split()) == 28 and "26535" in cut and cut in row


def test_short_quote_and_number_free_answer_are_left_alone() -> None:
    assert _trim_quote("a b c", "x") == "a b c"
    long = " ".join(["w"] * 40)
    assert _trim_quote(long, "none") == long


def test_comparison_calc_is_safe_and_boolean() -> None:
    val, nums, safe = _eval_calc("25885 < 25984")
    assert safe and val == 1.0 and nums == ["25885", "25984"]
    assert _eval_calc("25885 > 25984")[0] == 0.0
    assert _eval_calc("__import__('os').system('x') < 3")[2] is False


def test_comparison_answer_needs_no_number_in_the_answer() -> None:
    qa = _p(
        "qcas.xlsx",
        "2026 Q1; Total: 25984\n2026 Q2; Total: 25885",
    )
    reply = {
        "status": "answer",
        "answer": "It went down.",
        "file": "qcas.xlsx",
        "quotes": ["2026 Q1; Total: 25984", "2026 Q2; Total: 25885"],
        "calc": "25885 < 25984",
    }
    failed = _run_checks(reply, [qa], "Did the total go up or down from 2026 Q1 to 2026 Q2?", [])[4]
    assert failed is None


def test_trim_quote_one_digit_answer() -> None:
    row = " ".join(f"c{i}: {i * 3}" for i in range(2, 40)) + " total 1"
    cut = _trim_quote(row, "1")
    assert len(cut.split()) == 28 and "1" in cut.split() and cut in row
