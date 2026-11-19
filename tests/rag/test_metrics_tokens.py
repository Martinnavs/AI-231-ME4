"""Converter-independent ``--match tokens`` anchor matching."""

from __future__ import annotations

import time

import pytest

from voice_agent.rag.metrics import (
    MATCH_MODES,
    evaluate,
    is_relevant,
    tokenize_for_match,
    tokens_match,
)


def _hit(text: str, anchor: str, *, match: str = "tokens", source: str = "a.docx") -> bool:
    return is_relevant(
        {"source": source, "text": text}, [{"file": "a.docx", "contains": anchor}], match=match
    )


def test_mode_registered() -> None:
    assert "tokens" in MATCH_MODES
    assert MATCH_MODES[:2] == ("exact", "normalized")


@pytest.mark.parametrize(
    "text",
    ["Grade: A-; GPA: 3.667", "| Grade | GPA |\n| A- | 3.667 |", "A- 3.667"],
)
def test_table_renderings_match(text: str) -> None:
    assert _hit(text, "A- 3.667")


def test_row_rendering_with_headers() -> None:
    assert _hit(
        "Date: October 5; Event: Final changes due noon", "October 5 Final changes due noon"
    )


def test_order_missing_and_span() -> None:
    anchor = "October 5 Final changes due noon"
    assert not _hit("noon due changes Final 5 October", anchor)
    assert not _hit("October 5 Final changes due", anchor)
    far = "October 5 " + "filler " * 200 + "Final changes due noon"
    assert not _hit(far, anchor)


def test_source_must_match() -> None:
    assert not _hit("A- 3.667", "A- 3.667", source="b.docx")


def test_thousands_and_decimals() -> None:
    assert tokenize_for_match("1,599 and 12.5") == ["1599", "and", "12.5"]
    assert _hit("Price: 1599 USD", "1,599 USD")
    assert _hit("Price: 1,599 USD", "1599 USD")
    assert _hit("ratio 12.5 percent", "12.5 percent")
    assert not _hit("ratio 12 5 percent", "12.5 percent")
    assert tokenize_for_match("a, b 1,23") == ["a", "b", "1", "23"]


def test_weak_single_token_ignored() -> None:
    assert not _hit("is a to be", "a")
    assert not _hit("it is", "is")
    assert _hit("the budget", "budget")
    assert _hit("year 42", "42")
    assert not tokens_match([], ["x"])


def test_exact_and_normalized_unchanged() -> None:
    assert _hit("Total  cost: 1,599", "Total cost: 1,599", match="exact")
    assert not _hit("Total cost: 1,599", "Total cost: 1599", match="exact")
    assert _hit("Total cost: 1,599", "total cost 1599", match="normalized")
    assert not _hit("Grade: A-; GPA: 3.667", "A- 3.667", match="exact")


def test_unknown_mode_message() -> None:
    with pytest.raises(ValueError, match="tokens"):
        is_relevant({"source": "a", "text": "b"}, [], match="bogus")
    with pytest.raises(ValueError, match="tokens"):
        evaluate(lambda q, k: [], [], match="bogus")


def test_huge_passage_fast() -> None:
    text = "lorem ipsum dolor " * 280_000
    assert len(text) > 5_000_000
    t0 = time.perf_counter()
    assert not _hit(text, "lorem ipsum zebra")
    assert _hit(text, "lorem ipsum dolor")
    assert time.perf_counter() - t0 < 5


def test_pathological_repeated_anchor_fast() -> None:
    text = "foo " * 200_000 + "bar"
    anchor = "foo " * 300 + "bar"
    t0 = time.perf_counter()
    assert _hit(text, anchor)  # anchor capped at 200 tokens: only the leading foo run is used
    assert time.perf_counter() - t0 < 5
    t0 = time.perf_counter()
    assert _hit("foo " * 400 + "bar", anchor)
    assert time.perf_counter() - t0 < 5


def test_tokens_match_scan_budget_counts_scanned_elements() -> None:
    anchor = ["a", "zz"] + [f"c{i}" for i in range(998)]
    passage = ["a"] * 199_000 + ["zz"] + [f"c{i}" for i in range(998)]
    start = time.monotonic()
    assert tokens_match(anchor, passage) in (True, False)
    assert time.monotonic() - start < 5


def test_tokens_match_budget_exceeded_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    from voice_agent.rag import metrics

    monkeypatch.setattr(metrics, "TOKENS_MAX_SCAN_STEPS", 1_000)
    anchor = ["a", "b"]
    passage = ["a"] * 5_000 + ["x"] * 5_000 + ["a", "b"]
    assert tokens_match(anchor, passage) is False
    monkeypatch.setattr(metrics, "TOKENS_MAX_SCAN_STEPS", 2_000_000)
    assert tokens_match(anchor, passage) is True


def test_tokens_match_anchor_capped_at_200_tokens() -> None:
    anchor = [f"t{i}" for i in range(300)]
    passage = [f"t{i}" for i in range(200)] + ["other"] * 10
    assert tokens_match(anchor, passage) is True  # tokens past the 200th are ignored
    assert tokens_match(anchor[:199] + ["missing"], passage) is False
