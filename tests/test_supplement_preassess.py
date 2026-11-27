"""scripts/supplement_preassess.py: the free Qwen pre-assessment (ticket 271).

All through fakes: a fake transport (no network, no relay, no spend) and an injected
gate pair (the root env has no pipeline, and the production gates are covered by the
pipeline suite). The script's own logic is under test - the judge's content (no
labels), the flag parsing, the three measurement blocks, the relay request shape
(same Relay class as sufficiency_eval), and the report rendering."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_HERE = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "supplement_preassess", _HERE / "scripts" / "supplement_preassess.py"
)
assert _SPEC and _SPEC.loader
pa = importlib.util.module_from_spec(_SPEC)
sys.modules["supplement_preassess"] = pa
_SPEC.loader.exec_module(pa)

se = sys.modules["sufficiency_eval"]  # loaded by the script module

TODAY = "2026-11-27"
HEDGE_TAIL = "I couldn't confirm that with a search, so I won't guess."


class FakeTransport:
    """In-memory relay: returns canned responses in order, records every payload."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if not isinstance(item, dict):
            raise TypeError("fake transport: expected a canned response dict or an exception")
        return item


def qwen_reply(text: str, *, prompt_tokens: int = 10, completion_tokens: int = 5) -> dict[str, Any]:
    """A minimal relay response with a message content and a usage block."""
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def flags_text(
    answers: bool = True,
    period: bool = True,
    scope: bool = True,
    not_yet: bool = False,
) -> dict[str, Any]:
    return qwen_reply(
        json.dumps(
            {
                "answers_asked_question": answers,
                "right_period": period,
                "right_scope": scope,
                "states_not_yet_or_unavailable": not_yet,
            }
        )
    )


@dataclass
class FakeVerdict:
    verdict: str
    web_question: str
    reason: str
    fast: bool = False


class FakeSufficiency:
    """The gate pair the script consumes: the hedge tail, the time cue, the verdict
    type and a scripted supplement_ok (reply text -> (kept, why, lead))."""

    def __init__(self, ok_by_reply: dict[str, tuple[bool, str, str]]) -> None:
        self.HEDGE_TAIL = HEDGE_TAIL
        self.Verdict = FakeVerdict
        self.ok_by_reply = dict(ok_by_reply)
        self.calls: list[dict[str, Any]] = []

    def time_sensitive(self, question: str) -> bool:
        q = question or ""
        return "right now" in q or "current" in q

    def supplement_ok(
        self,
        verdict: FakeVerdict,
        reply: str,
        now: Any,
        question: str = "",
        doc_answer: str = "",
    ) -> tuple[bool, str, str]:
        self.calls.append(
            {"verdict": verdict.verdict, "question": question, "doc_answer": doc_answer}
        )
        if reply not in self.ok_by_reply:
            raise AssertionError(f"gate not scripted for reply {reply!r}")
        return self.ok_by_reply[reply]


def write_fixture(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def make_rows() -> list[dict[str, Any]]:
    """One bad (caught and dropped), one clean ok (kept), one ok the judge wrongly
    flags (dropped by the gates anyway) and one hedge (silent to the gates)."""
    return [
        {
            "id": "r-bad1",
            "question": "What's the rate right now?",
            "web_question": "What is the current rate?",
            "reply": "The rate is 209 a night.",
            "label": "bad",
            "doc_answer": "The sheet lists 209 a night.",
        },
        {
            "id": "r-ok1",
            "question": "What is the capital of France?",
            "web_question": "",
            "reply": "Paris is the capital of France.",
            "label": "ok",
        },
        {
            "id": "r-ok2",
            "question": "Who is the coordinator?",
            "web_question": "Who is the current coordinator?",
            "reply": "It varies by institution.",
            "label": "ok",
        },
        {
            "id": "r-hedge1",
            "question": "How many seats are there right now?",
            "web_question": "How many seats are available?",
            "reply": "The building has 40 seats. " + HEDGE_TAIL,
            "label": "hedge",
        },
    ]


JUDGE_RESPONSES = [
    flags_text(answers=False),  # r-bad1: the question is not answered
    flags_text(),  # r-ok1: clean
    flags_text(scope=False),  # r-ok2: wrongly flagged (a false flag)
    flags_text(not_yet=True),  # r-hedge1: flagged, but hedges are silent to the gates
]


def run_fake_fixture(tmp_path: Path) -> tuple[dict[str, Any], FakeTransport, FakeSufficiency]:
    path = write_fixture(
        tmp_path / "fixture.jsonl",
        make_rows(),
    )
    fake = FakeSufficiency(
        {
            "The rate is 209 a night.": (False, "figure:stale", ""),
            "Paris is the capital of France.": (True, "", "From the web: "),
            "It varies by institution.": (False, "scope:varies", ""),
        }
    )
    transport = FakeTransport(JUDGE_RESPONSES)
    block = pa.run_fixture(path, transport, "fake/model", TODAY, se.manila_now(TODAY), fake)
    return block, transport, fake


# --- the judge's content and flag parsing ---


def test_judge_content_sees_only_the_question_web_question_reply_and_today() -> None:
    row = make_rows()[0]
    content = pa.judge_content(row, TODAY)
    assert row["question"] in content
    assert row["web_question"] in content
    assert row["reply"] in content
    assert TODAY in content
    # nothing the judge must not see: no label, no document answer, no gold
    assert "bad" not in content
    assert "The sheet lists 209 a night." not in content


def test_judge_flags_parses_the_four_booleans_and_variants() -> None:
    flags = pa.judge_flags(
        '{"answers_asked_question": true, "right_period": false, "right_scope": true, '
        '"states_not_yet_or_unavailable": "true"}'
    )
    assert flags == {
        "answers_asked_question": True,
        "right_period": False,
        "right_scope": True,
        "states_not_yet_or_unavailable": True,
    }
    # a missing key is an unparsable judge answer: None, never guessed
    assert pa.judge_flags('{"answers_asked_question": true}') is None
    assert pa.judge_flags("[1, 2, 3]") is None
    assert pa.judge_flags("not json at all") is None
    assert pa.judge_flags(None) is None


def test_flagged_issues_marks_the_problems() -> None:
    clean = pa.judge_flags(
        '{"answers_asked_question": true, "right_period": true, '
        '"right_scope": true, "states_not_yet_or_unavailable": false}'
    )
    assert pa.flagged_issues(clean) == []
    flagged = pa.judge_flags(
        '{"answers_asked_question": false, "right_period": false, '
        '"right_scope": true, "states_not_yet_or_unavailable": true}'
    )
    assert pa.flagged_issues(flagged) == [
        "answers_asked_question",
        "right_period",
        "states_not_yet_or_unavailable",
    ]


# --- the three blocks, all through fakes ---


def test_run_fixture_blocks_judge_gates_misleading(tmp_path: Path) -> None:
    block, transport, fake = run_fake_fixture(tmp_path)
    assert block["fixture"] == "fixture.jsonl"
    assert block["n"] == 4
    assert block["labels"] == {"bad": 1, "ok": 2, "hedge": 1, "unsure": 0}
    # (a) the judge vs the hand label: the bad row caught, one false flag on ok
    j = block["judge"]
    assert j["bad"] == {"n": 1, "caught": 1, "ids_flagged": ["r-bad1"], "ids_clean": []}
    assert j["ok"] == {"n": 2, "false_flags": 1, "ids_flagged": ["r-ok2"]}
    assert j["flag_counts"] == {
        "answers_asked_question": 1,
        "right_period": 0,
        "right_scope": 1,
        "states_not_yet_or_unavailable": 1,
    }
    assert j["unparsed"] == []
    assert [r["id"] for r in j["rows"]] == ["r-bad1", "r-ok1", "r-ok2", "r-hedge1"]
    # (b) the gates vs the hand label: bad dropped, ok kept, hedge silent
    g = block["gates"]
    assert g["bad"] == {"n": 1, "dropped": 1, "kept": []}
    assert g["ok"]["kept"] == 1 and g["ok"]["n"] == 2
    assert g["ok"]["dropped"] == [{"id": "r-ok2", "why": "scope:varies"}]
    assert g["hedges_silent"] == 1
    assert g["unsure"] == []
    # (c) the predicted misleading rate: labelled bad among the gate survivors
    m = block["misleading"]
    assert m == {"survivors": 1, "labelled_bad": 0, "rate": 0.0, "bad_ids_surviving": []}
    # the gate saw the verdict derived from the row (stale here: the question is
    # time-sensitive and a document answer stands behind the supplement) and the
    # hedge never reached it
    assert fake.calls[0]["verdict"] == "stale"
    assert [c["question"] for c in fake.calls] == [r["question"] for r in make_rows()[:3]]
    assert all(HEDGE_TAIL not in json.dumps(c) for c in fake.calls)


def test_run_fixture_counts_an_unparsed_judge_and_kept_bad(tmp_path: Path) -> None:
    rows = make_rows()
    rows[0] = dict(rows[0], reply="The 2026 rate is 210 a night.")
    path = write_fixture(tmp_path / "fixture2.jsonl", rows)
    fake = FakeSufficiency(
        {
            "The 2026 rate is 210 a night.": (True, "", "The web has a newer figure: "),
            "Paris is the capital of France.": (True, "", "From the web: "),
            "It varies by institution.": (False, "scope:varies", ""),
        }
    )
    transport = FakeTransport(
        [
            qwen_reply("this is not json"),  # r-bad1: unparsable (reported, not guessed)
            flags_text(),
            flags_text(),
            flags_text(not_yet=True),
        ]
    )
    block = pa.run_fixture(path, transport, "fake/model", TODAY, se.manila_now(TODAY), fake)
    bad = block["judge"]["bad"]
    assert bad == {"n": 1, "caught": 0, "ids_flagged": [], "ids_clean": ["r-bad1"]}
    assert block["judge"]["unparsed"] == ["r-bad1"]
    # the gates kept the bad row (that is the point of measuring): it shows up in the
    # predicted misleading rate
    assert block["gates"]["bad"] == {"n": 1, "dropped": 0, "kept": ["r-bad1"]}
    assert block["misleading"] == {
        "survivors": 2,
        "labelled_bad": 1,
        "rate": 0.5,
        "bad_ids_surviving": ["r-bad1"],
    }


def test_one_relay_call_per_reply_with_the_shared_qwen_request(tmp_path: Path) -> None:
    _, transport, _ = run_fake_fixture(tmp_path)
    assert len(transport.payloads) == 4  # one per reply, hedges included
    for payload in transport.payloads:
        assert payload["model"] == "fake/model"
        assert payload["temperature"] == 0
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["response_format"] == {"type": "json_object"}
        system, user = [m["content"] for m in payload["messages"]]
        assert system == pa.JUDGE_SYSTEM
        # the key is never in the payload: it travels only in the Bearer header
    assert "209" in transport.payloads[0]["messages"][1]["content"]


def test_row_verdict_derives_stale_from_question_web_question_or_doc_answer() -> None:
    rows = make_rows()
    assert pa._row_verdict(FakeSufficiency({}), rows[0]).verdict == "stale"  # question
    assert pa._row_verdict(FakeSufficiency({}), rows[2]).verdict == "stale"  # web question
    bare = {
        "id": "x",
        "question": "plain question",
        "web_question": "",
        "reply": "r",
        "label": "ok",
    }
    assert pa._row_verdict(FakeSufficiency({}), bare).verdict == "missing_half"
    with_doc = dict(bare, doc_answer="The sheet says 209.")
    assert pa._row_verdict(FakeSufficiency({}), with_doc).verdict == "stale"  # doc answer


def test_load_rows_fails_loudly_on_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "x", "question": "q?"}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing 'reply'"):
        pa._load_rows(path)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no rows"):
        pa._load_rows(empty)


# --- combined and rendering ---


def test_combined_sums_the_blocks(tmp_path: Path) -> None:
    block1, _, _ = run_fake_fixture(tmp_path)
    path2 = write_fixture(
        tmp_path / "second.jsonl",
        [
            {
                "id": "s-bad1",
                "question": "What's the rate right now?",
                "web_question": "",
                "reply": "No current figure.",
                "label": "bad",
            }
        ],
    )
    fake2 = FakeSufficiency({"No current figure.": (False, "figure:missing", "")})
    transport2 = FakeTransport([flags_text(answers=False, not_yet=True)])
    block2 = pa.run_fixture(path2, transport2, "fake/model", TODAY, se.manila_now(TODAY), fake2)
    combo = pa._combined([block1, block2])
    assert combo["judge"]["bad"] == {"n": 2, "caught": 2}
    assert combo["judge"]["ok"] == {"n": 2, "false_flags": 1}
    assert combo["gates"] == {
        "bad": {"n": 2, "dropped": 2},
        "ok": {"n": 2, "kept": 1},
        "hedges_silent": 1,
    }
    assert combo["misleading"] == {"survivors": 1, "labelled_bad": 0, "rate": 0.0}


def test_render_prints_the_three_blocks_per_fixture_and_the_combined(tmp_path: Path) -> None:
    block, _, _ = run_fake_fixture(tmp_path)
    lines = pa.render_fixture(block)
    text = "\n".join(lines)
    assert "[fixture: fixture.jsonl]  n=4" in text
    assert "(a) judge vs hand label" in text
    assert "bad caught:     1/1" in text
    assert "ok false flags: 1/2" in text
    assert "r-ok2" in text
    assert "(b) gates vs hand label" in text
    assert "bad dropped:    1/1" in text
    assert "ok kept:        1/2" in text
    assert "hedges silent:  1" in text
    assert "(c) predicted misleading rate" in text
    assert "0/1 = 0.0%" in text
    combo_lines = pa.render_combined(pa._combined([block]))
    combo_text = "\n".join(combo_lines)
    assert "[combined]" in combo_text
    assert "(a) judge vs hand label" in combo_text
    assert "(b) gates vs hand label" in combo_text
    assert "(c) predicted misleading rate" in combo_text


# --- the command line: usage errors are loud, no relay needed ---


def test_main_rejects_a_missing_fixture(tmp_path: Path) -> None:
    assert pa.main([str(tmp_path / "nope.jsonl")]) == 2


def test_main_rejects_a_bad_today(tmp_path: Path) -> None:
    path = write_fixture(tmp_path / "fixture.jsonl", make_rows())
    assert pa.main([str(path), "--today", "not-a-date"]) == 2
