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
