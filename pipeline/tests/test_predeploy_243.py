"""Pre-deploy 243: a different web query searches again; the docs-miss prefix is spoken only
for document questions; a duplicate filler line is not repeated.

Bug 1 (2026-11-26 session, Miss Universe turn 42): one real lookup per turn; the reworded
retries got the first lookup's (stale pre-result) text back, so the agent spoke "reports on
the winner are unavailable". Now: an identical query (case/space/punctuation-normalised)
returns the cached text with the prefix; a different query runs a real lookup, at most two
extras per turn (3 total); beyond that the newest result comes back with the prefix.

Bug 2: web answers to non-document questions got "Your documents don't cover that." first
(session turns 32, 42, 43, 44) and a duplicate "Checking ..." sentence. Now: the miss prefix
is spoken only when the question names documents/files (a generic phrase, a document word, or
a stem of a file in the loaded catalog - nothing corpus-specific), and a filler line already
spoken this turn is not repeated.

The tests use synthetic names so they prove the behaviour on an unseen corpus; the
session-utterance replay at the end is marked regression data.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from voice_pipeline import doc_route
from voice_pipeline.agent import TurnTap
from voice_pipeline.agent_server import AgentServer
from voice_pipeline.ask_documents import DocResult
from voice_pipeline.spoken_filter import filler_repeats, is_filler

# ---------------------------------------------------------------------------
# Bug 2a: the miss prefix is spoken only for document questions
# ---------------------------------------------------------------------------


class _MissAgent:
    def __init__(self, text: str = "The capital is Lima.") -> None:
        self.text = text
        self.tap = SimpleNamespace(
            usage={"requests": 0, "prompt": 0, "completion": 0, "cached": 0}
        )

    async def ask(
        self, messages: Any, *, reset: bool, on_text: Any = None, **_: Any
    ) -> dict[str, Any]:
        if on_text:
            await on_text(self.text)
        return {"status": "ok", "text": self.text, "cancelled": False}

    def hide(self, names: frozenset[str]) -> None:
        pass


async def _miss(
    tmp_path: Path, said: str, status: str, catalog: list[str] | None = None
) -> dict[str, Any]:
    """One docs-miss turn through the real `_docs_miss_turn`, with a scripted agent."""
    server = AgentServer(_MissAgent(), [], tmp_path / "243.jsonl")
    if catalog is not None:
        doc_route.set_cached_files(server, catalog)

    async def on_text(text: str) -> None:
        pass

    return await server._docs_miss_turn(
        DocResult(status), 0.1, said, [{"role": "user", "content": said}], on_text, None, None
    )


@pytest.mark.parametrize(
    "said",
    [
        "Who won the 2026 World Cup?",
        "Who is the current CEO of Nvidia",
        "What is the weather today?",
        "Sorry, can you broadly chat?",
    ],
)
async def test_a_non_document_miss_has_no_prefix(tmp_path: Path, said: str) -> None:
    res = await _miss(tmp_path, said, "none")
    assert res["text"] == "The capital is Lima."
    assert res["docs"] == "none"  # the docs verdict is still logged


@pytest.mark.parametrize(
    "said",
    [
        "According to the Alpha Travel Policy PDF, what's the rate for Denver?",
        "Is the Denver rate in my files?",
        "What's in the folder with the beta program?",
        "The beta rate is in my notes, right?",
        "According to the quarterly memo, what's the rate for Denver?",
    ],
)
async def test_a_document_phrase_keeps_the_prefix(tmp_path: Path, said: str) -> None:
    res = await _miss(tmp_path, said, "none")
    assert res["text"].startswith("Your documents don't cover that. ")
    assert res["docs"] == "none"


@pytest.mark.parametrize(
    "said",
    [
        "Is the Denver rate in my documents?",
        "What does the document say about the beta program?",
        "Where is the beta rate, in the file?",
    ],
)
async def test_a_document_word_keeps_the_unconfirmed_prefix(tmp_path: Path, said: str) -> None:
    res = await _miss(tmp_path, said, "check_failed")
    assert res["text"].startswith("I couldn't confirm that in your documents. ")
    assert res["docs"] == "check_failed"


async def test_a_file_from_the_loaded_catalog_keeps_the_prefix(tmp_path: Path) -> None:
    res = await _miss(
        tmp_path,
        "What does alpha_report_2031.pdf say about the beta program?",
        "check_failed",
        catalog=["alpha_report_2031.pdf"],
    )
    assert res["text"].startswith("I couldn't confirm that in your documents. ")


async def test_a_file_not_in_the_loaded_catalog_is_not_a_cue(tmp_path: Path) -> None:
    res = await _miss(
        tmp_path,
        "What does gamma_notes_2041.pdf say about the beta program?",
        "none",
        catalog=["alpha_report_2031.pdf"],
    )
    assert res["text"] == "The capital is Lima."  # no prefix: that file is not in the corpus
    assert res["docs"] == "none"


# ---------------------------------------------------------------------------
# Bug 2b: a filler line already spoken this turn is not repeated
# ---------------------------------------------------------------------------


def test_filler_detection_is_generic() -> None:
    for text in (
        "Checking the weather.",
        "I'm checking the beta program.",
        "Looking up the answer.",
        "I'm looking for the winner of the 2026 Zorblat pageant.",
        "Let me look that up.",
        "Searching for the rate.",
    ):
        assert is_filler(text), text
    for text in (
        "The weather is clear.",
        "As per CNN, it is Jensen Huang.",
        "The capital is Lima.",
    ):
        assert not is_filler(text), text


def test_the_same_thing_means_two_shared_content_words() -> None:
    assert filler_repeats(
        "I'm looking for the winner of the 2026 Zorblat pageant.",
        "Checking who won the 2026 Zorblat pageant.",
    )
    assert not filler_repeats("Checking the weather.", "Looking up the weather forecast.")
    assert not filler_repeats("The capital is Lima.", "Checking who won the 2026 Zorblat pageant.")


async def test_the_duplicate_filler_line_is_not_spoken_twice() -> None:
    spoken: list[str] = []

    async def on_say(text: str) -> None:
        spoken.append(text)

    tap = TurnTap()
    tap.begin(on_text=None, on_say=on_say)
    tap.say_text = "Checking who won the 2026 Zorblat pageant."
    await tap._speak_say()
    tap.say_text = "I'm looking for the winner of the 2026 Zorblat pageant."  # a retry's say
    await tap._speak_say()
    assert len(spoken) == 1
    assert "".join(tap.text) == "Checking who won the 2026 Zorblat pageant. "
    # a filler about something else is still spoken
    tap.say_text = "Looking up the weather forecast."
    await tap._speak_say()
    assert len(spoken) == 2
    # a new turn starts fresh
    tap.begin(on_text=None, on_say=on_say)
    tap.say_text = "Checking who won the 2026 Zorblat pageant."
    await tap._speak_say()
    assert len(spoken) == 3


# ---------------------------------------------------------------------------
# Regression data: utterances from the 2026-11-26 Pi session (20261126-173512), replayed
# as test data only. Their names and topics exist only in that session's 48-file corpus;
# the tests above already prove the rule on an unseen corpus.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "said",
    [
        "S Miss Universe Twenty Twenty Six.",
        "Sorry, can you broadly chat?",
        "Who is the current CEO of Nvidia",
        "Got it, what's the Office Wi Pi password?",
        "Who won the latest mayor race in LA?",
    ],
)
async def test_session_non_document_questions_get_no_prefix(tmp_path: Path, said: str) -> None:
    res = await _miss(tmp_path, said, "none")
    assert res["text"] == "The capital is Lima."
    assert res["docs"] == "none"


async def test_session_document_question_keeps_the_prefix(tmp_path: Path) -> None:
    # session turn 32: the user named a policy PDF, so the miss was about their documents
    res = await _miss(
        tmp_path,
        "According to the TTS Travel Policy PDF, what's the per diamond for Denver?",
        "none",
    )
    assert res["text"].startswith("Your documents don't cover that. ")
