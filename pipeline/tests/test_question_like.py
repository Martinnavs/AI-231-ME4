"""question_like: what goes to the document path first when there is no local router."""

from __future__ import annotations

import pytest

from voice_pipeline.doc_route import question_like


@pytest.mark.parametrize(
    "said",
    [
        "How quickly do I need to notify GSA IT after a suspected incident?",
        "what is the base schedule and where does it live",
        "According to the documents, which file mentions the deadline?",
        "Is the pet policy in the handbook?",
        "okay, who is the contact person",
    ],
)
def test_questions_go_to_the_documents(said: str) -> None:
    assert question_like(said)


@pytest.mark.parametrize(
    "said",
    [
        "Turn on the living room lights",
        "Play some jazz",
        "set a timer for ten minutes",
        "thanks",
        "please turn it down",
    ],
)
def test_commands_and_chitchat_do_not(said: str) -> None:
    assert not question_like(said)


def test_non_document_intent_patterns() -> None:
    from voice_pipeline.doc_route import non_document_intent

    for q in (
        "What's the weather today?",
        "What's the weather?",
        "How's the weather like?",
        "Is it going to rain?",
        "what time is it",
        "what's today's date",
        "set a timer for ten minutes",
        "remind me to stretch",
        "play some music",
        "turn up the volume",
        "turn on the lights",
        "any news today?",
    ):
        assert non_document_intent(q), q
    for q in (
        "What is the GSA budget for 2025?",
        "When is the midterm in section 3?",
        "Who wrote the incident report?",
        "What time does the exam start?",
        "What date does fall semester begin?",
        "What temperature range does the datasheet give?",
        "What do the weather minimums in the handbook say?",
    ):
        assert not non_document_intent(q), q
