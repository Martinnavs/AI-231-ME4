"""Ticket 212: main prompt edits, baby-onit removal, CHANGEABLE_NOTE_DOCS."""

from pathlib import Path

import pytest

from voice_pipeline import call_policy
from voice_pipeline.agent_server import AgentServer

PROMPTS = Path(__file__).resolve().parents[1] / "config" / "eval"
BASE = (PROMPTS / "system_agent.txt").read_text()
SEARCH = (PROMPTS / "system_agent_search.txt").read_text()
BLOCK = (PROMPTS / "system_agent_case_f.txt").read_text()


def test_forbidden_strings_absent() -> None:
    for forbidden in (
        "baby-onit",
        "onit",
        "never web_answer",
        "say you don't know",
        "a fact that can change is never Case G",
    ):
        assert forbidden not in BASE, f"{forbidden!r} in BASE"
        assert forbidden not in SEARCH, f"{forbidden!r} in SEARCH"


def test_case_c_has_abstain_and_ask_documents() -> None:
    for prompt in (BASE, SEARCH):
        assert "The documents don't say." in prompt
        assert "ask_documents" in prompt


def test_case_f_out_of_base_but_in_block_and_steer() -> None:
    # Distinctive phrase from the old long Case F line is absent from BASE
    assert "Call delegate with `task` written as a complete request" not in BASE
    # steer(case_f=True) still adds the block
    server = AgentServer(None, [], Path("/dev/null"), case_f_block="PROTOCOL")
    out = server.steer([{"role": "user", "content": "Build a site."}], case_f=True)
    assert out[-1]["content"].endswith("PROTOCOL")


def test_base_prompt_word_count() -> None:
    assert len(BASE.split()) <= 1000


@pytest.mark.parametrize(
    "question",
    [
        "Which U.S. president signed the federal plain-writing law highlighted in ATSDR Module 3?",
        "What is the last day to withdraw from classes in Spring 2027 at WMU?",
        "What version does the sample project declare in its pyproject.toml?",
    ],
)
def test_changeable_note_docs_used_when_docs_tool_present(
    question: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert call_policy.changeable_question(question)
    server = AgentServer(None, [], Path("/dev/null"))
    # Patch self.tools directly (AgentServer expects ToolsSchema on init)
    server.tools = {"ask_documents": object()}
    out = server.stamped([{"role": "user", "content": question}])
    content = out[-1]["content"]
    assert "check the documents first" in content
    assert "look it up with web_answer" not in content


def test_changeable_note_stays_without_docs_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    server = AgentServer(None, [], Path("/dev/null"))
    out = server.stamped([{"role": "user", "content": "Who's the CEO of Nvidia?"}])
    content = out[-1]["content"]
    assert call_policy.CHANGEABLE_NOTE in content
    assert "check the documents first" not in content


def test_document_topic_removed() -> None:
    # These names no longer exist in call_policy (removed with baby-onit)
    assert not hasattr(call_policy, "DOCUMENT_TOPIC")
    assert not hasattr(call_policy, "about_document_topic")


def test_router_path_does_not_crash_on_corpus_query() -> None:
    from voice_pipeline.router import spec

    calls = spec.parse_output("web_answer|how many tools does the harness ship")
    decision = call_policy.decide(
        calls, "How many tools does the harness ship?", {"web_answer", "search_docs"}
    )
    # No "about the loaded documents" reason (DOCUMENT_TOPIC removed)
    assert decision.reason != "about the loaded documents"
