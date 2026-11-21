"""Tests for doc_chat.corpus.fold (accent and ligature folding)."""

from __future__ import annotations

import pytest

from doc_chat.corpus import fold


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Café NAÏVE", "cafe naive"),
        ("Straße ﬁnance œuvre", "strasse finance oeuvre"),
        ("Łódź Æsir", "lodz aesir"),
    ],
)
def test_fold_strips_accents_and_ligatures(raw: str, expected: str) -> None:
    assert fold(raw) == expected


def test_fold_ascii_is_plain_lowercase() -> None:
    assert fold("Hello, World 1,599.00") == "hello, world 1,599.00"
