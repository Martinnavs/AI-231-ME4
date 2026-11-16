"""The changeable-question gate: the regex floor, the EverGreen model, the fallback between them.

`tests/fixtures/freshqa_sample.csv`: 120 questions (40 per class) drawn from FreshQA (Vu et al.
2023, Apache-2.0, github.com/freshllms/freshqa), kept with its fact_type labels so the gate's
numbers are asserted here, not narrated in a ticket.
"""

import csv
from collections.abc import Callable
from pathlib import Path

import pytest

from voice_agent import gate
from voice_agent.mcp_servers import gemini_grounding

SAMPLE = Path(__file__).parent / "fixtures/freshqa_sample.csv"


def sample() -> list[tuple[str, bool]]:
    """(question, changes over time) from FreshQA's fact_type."""
    with SAMPLE.open() as f:
        return [(r["question"], r["fact_type"] != "never-changing") for r in csv.DictReader(f)]


def rates(flag: Callable[[str], bool]) -> tuple[float, float]:
    rows = sample()
    changing = [q for q, c in rows if c]
    never = [q for q, c in rows if not c]
    return (
        sum(bool(flag(q)) for q in changing) / len(changing),
        sum(bool(flag(q)) for q in never) / len(never),
    )


def test_the_regex_floor_on_the_freshqa_sample() -> None:
    recall, false_alarm = rates(gemini_grounding.changeable_regex)
    assert recall >= 0.45 and false_alarm <= 0.30  # measured on all 600: 0.55 and 0.21


def test_the_gate_is_the_regex_when_there_is_no_model() -> None:
    # conftest.regex_gate: no model loaded
    assert gate.mutable("Who is the CEO of Nvidia?") is None
    assert gemini_grounding.changeable("Who is the CEO of Nvidia?")
    assert not gemini_grounding.changeable("What is the capital of France?")


def test_a_missing_model_directory_or_library_falls_back(tmp_path: Path) -> None:
    gate.reset()
    assert gate.load(tmp_path) is None  # no files
    assert gate.mutable("Who is the CEO of Nvidia?") is None


def test_a_loaded_gate_decides_and_a_failing_one_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Fake:
        def __init__(self, p: float | Exception) -> None:
            self.p = p

        def p_mutable(self, question: str) -> float:
            if isinstance(self.p, Exception):
                raise self.p
            return self.p

    gate.reset()
    gate._loaded.append(Fake(0.9))
    assert gemini_grounding.changeable(
        "What is the capital of France?"
    )  # the model wins over the regex
    gate.reset()
    gate._loaded.append(Fake(0.1))
    assert not gemini_grounding.changeable("Who is the CEO of Nvidia?")
    gate.reset()
    gate._loaded.append(Fake(RuntimeError("onnx failed")))
    assert gemini_grounding.changeable("Who is the CEO of Nvidia?")  # failed: the regex answers


def test_the_installed_model_beats_the_regex_on_the_freshqa_sample(real_gate: object) -> None:
    recall, false_alarm = rates(gemini_grounding.changeable)
    regex_recall, _ = rates(gemini_grounding.changeable_regex)
    assert recall >= 0.70 and false_alarm <= 0.25  # measured on all 600: 0.80 and 0.18 (int8)
    assert recall > regex_recall


def test_the_installed_model_knows_roles_and_settled_facts(real_gate: object) -> None:
    changing = [
        "Who's the CEO of Nvidia?",
        "Who is the mayor here?",
        "What is the latest version of Python?",
        "Who won the last NBA championship?",
    ]
    settled = [
        "What is the capital of France?",
        "Who wrote Noli Me Tangere?",
        "Who painted the Mona Lisa?",
    ]
    for question in changing:
        assert gate.mutable(question) is True, question
    for question in settled:
        assert gate.mutable(question) is False, question


def test_the_installed_model_is_fast_enough(real_gate: object) -> None:
    import time  # noqa: PLC0415

    gate.mutable("warm up")
    start = time.perf_counter()
    for question, _ in sample()[:40]:
        gate.mutable(question)
    assert (time.perf_counter() - start) / 40 < 0.05  # the budget; measured about 4 ms
