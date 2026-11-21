"""Tests for tokenize with accent and ligature folding."""

from __future__ import annotations

from doc_chat.corpus import tokenize


def test_accented_and_ascii_forms_tokenize_alike() -> None:
    assert tokenize("Café résumé façade") == ["cafe", "resume", "facade"]
    assert tokenize("cafe resume facade") == ["cafe", "resume", "facade"]


def test_ligatures_and_special_letters_fold() -> None:
    assert tokenize("ﬁnance Straße œuvre") == ["finance", "strasse", "oeuvre"]


def test_ascii_tokens_are_unchanged() -> None:
    text = "Boxes's churches wishes passes cats' running walked 2026 1,599.00 3.14 the and about"
    expected = ["box", "church", "wish", "pass", "cat", "runn", "walk", "2026", "1599.00", "3.14"]
    assert tokenize(text) == expected
