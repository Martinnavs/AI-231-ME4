"""236 pre-deploy fixes: (A) a passed calc's result is accepted by names/numbers and calc
expressions parse with thousands commas; (B) the quotes:[] bypass is gated behind
REQUIRE_QUOTES (off by default)."""

from __future__ import annotations

import pytest

from voice_pipeline import ask_documents
from voice_pipeline.ask_documents import _calc_result_forms, _eval_calc, _run_checks


def _p(source: str, text: str) -> dict:
    return {"source": source, "id": f"{source}#0", "text": text, "part": "whole"}


def _gs_passage() -> dict:
    # 37917 (and 38000/55555) appear nowhere in this text: any acceptance of them proves the
    # calc result was injected as known, not found by coincidence.
    return _p("gs.xlsx", "GS-15 step 1: 126384; GS-15 step 10: 164301")


def _failed(reply: dict) -> str | None:
    return _run_checks(reply, [_gs_passage()], "how much more does step 10 earn than step 1", [])[4]


# --- FIX A: calc results accepted by names_numbers_ok -------------------------


def test_eval_calc_strips_thousands_commas() -> None:
    val, nums, safe = _eval_calc("164,301 - 126,384")
    assert safe and val == 37917.0 and nums == ["164301", "126384"]


def test_eval_calc_strips_commas_in_multi_group_numbers() -> None:
    val, nums, safe = _eval_calc("1,000,000 - 1")
    assert safe and val == 999999.0 and nums == ["1000000", "1"]


def test_calc_result_accepted_by_names_numbers() -> None:
    reply = {
        "status": "answer",
        "answer": "A GS-15 step 10 earns 37,917 dollars more than step 1",
        "file": "gs.xlsx",
        "quotes": ["GS-15 step 10: 164301"],
        "calc": "164,301 - 126,384",
    }
    assert _failed(reply) is None


def test_calc_wrong_result_in_the_answer_still_rejected() -> None:
    reply = {
        "status": "answer",
        "answer": "A GS-15 step 10 earns 38,000 dollars more than step 1",
        "file": "gs.xlsx",
        "quotes": ["GS-15 step 10: 164301"],
        "calc": "164,301 - 126,384",
    }
    assert _failed(reply) == "calc"


def test_calc_invented_number_still_rejected_by_names_numbers() -> None:
    # 37917 is in the answer (calc check passes) but 55555 is neither in the passages nor the
    # calc result: the names/numbers guard must still fire.
    reply = {
        "status": "answer",
        "answer": "A GS-15 step 10 earns 37,917 dollars more, plus 55,555",
        "file": "gs.xlsx",
        "quotes": ["GS-15 step 10: 164301"],
        "calc": "164,301 - 126,384",
    }
    assert _failed(reply) == "names_numbers"


def test_calc_result_forms_plain_when_operands_are_integers() -> None:
    assert _calc_result_forms(37917.0, ["164301", "126384"]) == ["37917", "37,917", "37917.0"]


def test_calc_result_forms_add_dollar_cents_only_with_decimal_operands() -> None:
    forms = _calc_result_forms(37917.0, ["100.50", "200.25"])
    assert forms == ["37917", "37,917", "$37,917.00", "37917.0"]
    assert "$" not in " ".join(_calc_result_forms(37917.0, ["164301", "126384"]))


# --- FIX B: quotes:[] bypass behind REQUIRE_QUOTES ----------------------------


def test_require_quotes_defaults_to_false() -> None:
    assert ask_documents.REQUIRE_QUOTES is False


def test_require_quotes_off_keeps_empty_quotes_passing() -> None:
    # default behaviour, unchanged: no quote check runs when there are no quotes
    reply = {
        "status": "answer",
        "answer": "A GS-15 step 1 is 126,384",
        "file": "gs.xlsx",
        "quotes": [],
    }
    failed = _run_checks(reply, [_gs_passage()], "how much is a GS-15 step 1", [])[4]
    assert failed is None


def test_require_quotes_on_rejects_answer_without_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ask_documents, "REQUIRE_QUOTES", True)
    reply = {
        "status": "answer",
        "answer": "A GS-15 step 1 is 126,384",
        "file": "gs.xlsx",
        "quotes": [],
    }
    failed = _run_checks(reply, [_gs_passage()], "how much is a GS-15 step 1", [])[4]
    assert failed == "quote_missing"


def test_require_quotes_on_still_allows_answer_with_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ask_documents, "REQUIRE_QUOTES", True)
    reply = {
        "status": "answer",
        "answer": "A GS-15 step 1 is 126,384",
        "file": "gs.xlsx",
        "quotes": ["GS-15 step 1: 126384"],
    }
    failed = _run_checks(reply, [_gs_passage()], "how much is a GS-15 step 1", [])[4]
    assert failed is None


def test_require_quotes_on_does_not_affect_none_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ask_documents, "REQUIRE_QUOTES", True)
    reply = {"status": "none", "answer": None, "file": "gs.xlsx", "quotes": []}
    status, _, _, _, failed = _run_checks(reply, [_gs_passage()], "how much is a GS-15 step 1", [])
    assert failed != "quote_missing" and status == "answer_empty"
