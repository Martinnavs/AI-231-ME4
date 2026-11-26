"""scripts/judge_eval.py: logprob parsing, labels, explain mode, summary, secrets, calibrate.

All through a fake in-memory transport: no network, no relay, no spend.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import urllib.error
from pathlib import Path
from typing import Any

import pytest

_spec = importlib.util.spec_from_file_location(
    "judge_eval", Path(__file__).resolve().parents[1] / "scripts" / "judge_eval.py"
)
assert _spec and _spec.loader
je = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(je)

FAKE_KEY = "sk-super-secret-never-print-123"


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


def relay_response(
    token: str, logprobs: list[tuple[str, float]], content: str | None = None
) -> dict[str, Any]:
    """A minimal OpenAI-compatible chat completion whose first token carries the given logprobs."""
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content if content is not None else token,
                },
                "logprobs": {
                    "content": [
                        {
                            "token": token,
                            "logprob": logprobs[0][1],
                            "top_logprobs": [{"token": t, "logprob": lp} for t, lp in logprobs],
                        }
                    ]
                },
            }
        ]
    }


def job(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "turn": 1,
        "question": "What day is the team meeting?",
        "answer": "The team meeting is on Tuesday.",
        "evidence": ["The team meeting is on Tuesday at 10:00."],
        "gold": ["the team meeting is on Tuesday"],
        "key_points": [],
        "reference": "The team meeting is on Tuesday at 10:00 in room 5.",
        "excerpts": ["The team meeting is on Tuesday.", "at 10:00 in room 5"],
    }
    base.update(overrides)
    return base


class _FakeResp:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_logprobs_softmax_over_option_letters_only() -> None:
    raw = {"A": math.exp(-0.11), "B": math.exp(-2.31), "C": math.exp(-7.2), "D": math.exp(-9.2)}
    total = sum(raw.values())
    expected = {letter: value / total for letter, value in raw.items()}
    data = relay_response(
        "A",
        [("A", -0.11), ("B", -2.31), ("C", -7.2), ("D", -9.2), ("E", -11.5)],
    )
    choice, probs = je.parse_letter(data, je.METRICS["faithfulness"])
    assert choice == "A"
    # "E" is not an option letter: excluded from the softmax entirely
    assert "E" not in probs and set(probs) == set("ABCD")
    for letter in "ABCD":
        assert probs[letter] == pytest.approx(expected[letter], abs=1e-9)
    assert sum(probs.values()) == pytest.approx(1.0)
    # confidence = the probability of the chosen letter
    assert probs[choice] == pytest.approx(max(probs.values()))


def test_logprobs_letters_missing_from_top5_get_zero() -> None:
    data = relay_response("B", [("B", -0.2), ("the", -1.0), ("A", -4.0), ("I", -5.0), ("it", -6.0)])
    choice, probs = je.parse_letter(data, je.METRICS["directness"])
    assert choice == "B"
    assert probs["C"] == 0.0  # below the top-5 cut
    assert probs["B"] > probs["A"] > 0.0
    assert sum(probs.values()) == pytest.approx(1.0)


def test_letter_parsing_lower_and_punctuated() -> None:
    data = relay_response("a.", [("a.", -0.1), ("b", -3.0), ("c", -6.0)])
    choice, _ = je.parse_letter(data, je.METRICS["directness"])
    assert choice == "A"
    data = relay_response("C.", [("C.", -0.5), ("A", -0.9)])
    choice, _ = je.parse_letter(data, je.METRICS["completeness"])
    assert choice == "C"


def test_choice_falls_back_to_argmax_when_token_is_not_a_letter() -> None:
    data = relay_response(
        "The", [("The", -0.3), ("A", -0.5), ("B", -3.0), ("C", -6.0), ("D", -8.0)]
    )
    choice, probs = je.parse_letter(data, je.METRICS["faithfulness"])
    assert choice == "A" and probs[choice] == max(probs.values())


def test_no_letter_at_all_is_a_loud_failure() -> None:
    data = relay_response(
        "I", [("I", -0.1), ("am", -0.5), ("not", -1.0), ("sure", -2.0), ("x", -3.0)]
    )
    with pytest.raises(je.JudgeError, match="without an option letter"):
        je.parse_letter(data, je.METRICS["directness"])


def test_letter_to_label_mapping() -> None:
    assert je.label_for("directness", "A") == "answers"
    assert je.label_for("directness", "B") == "describes_only"
    assert je.label_for("directness", "C") == "not_found"
    assert je.label_for("faithfulness", "D") == "contradicted"
    assert je.label_for("correctness", "C") == "missing"
    assert je.label_for("completeness", "A") == "covered"
    assert je.label_for("recall", "B") == "found_across_several"
    assert je.label_for("precision", "A") == "key_information"
    # the closed sets are exactly the ticket's
    assert {k: v[0] for k, v in je.METRICS["directness"].items()} == {
        "A": "answers",
        "B": "describes_only",
        "C": "not_found",
    }
    assert {k: v[0] for k, v in je.METRICS["faithfulness"].items()} == {
        "A": "supported",
        "B": "partial",
        "C": "unsupported",
        "D": "contradicted",
    }
    assert {k: v[0] for k, v in je.METRICS["correctness"].items()} == {
        "A": "supported",
        "B": "partial",
        "C": "missing",
        "D": "contradicted",
    }
    assert {k: v[0] for k, v in je.METRICS["completeness"].items()} == {
        "A": "covered",
        "B": "partial",
        "C": "not_covered",
    }
    assert {k: v[0] for k, v in je.METRICS["recall"].items()} == {
        "A": "found_in_one_excerpt",
        "B": "found_across_several",
        "C": "partly_found",
        "D": "not_found",
    }
    assert {k: v[0] for k, v in je.METRICS["precision"].items()} == {
        "A": "key_information",
        "B": "side_detail",
        "C": "irrelevant",
    }


def test_judge_payload_shape() -> None:
    payload = je.judge_payload("Qwen/Qwen3.8-27B", "some prompt")
    assert payload["model"] == "Qwen/Qwen3.8-27B"
    assert payload["logprobs"] is True and payload["top_logprobs"] == 5
    assert payload["temperature"] == 0 and payload["max_tokens"] == 4
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    # the system text: judge only from the given material, answer with one letter
    system = payload["messages"][0]["content"]
    assert "ONLY" in system and "letter" in system


def test_prompts_carry_the_given_material() -> None:
    prompt = je.build_user_prompt("faithfulness", job())
    assert "The team meeting is on Tuesday." in prompt  # answer
    assert "at 10:00" in prompt  # evidence
    assert "What day is the team meeting?" in prompt  # question
    prompt = je.build_user_prompt("recall", job())
    assert "room 5" in prompt  # reference
    assert "[1]" in prompt and "[2]" in prompt  # numbered excerpts
    prompt = je.build_user_prompt("precision", job(), point="at 10:00 in room 5")
    assert "at 10:00 in room 5" in prompt


def test_explain_mode_triggers_and_stores_why() -> None:
    low = relay_response("A", [("A", -2.0), ("B", -0.01), ("C", -6.0), ("D", -8.0)])
    reason = relay_response(
        "ok", [("ok", -0.0)], content="Because the evidence is missing the day."
    )
    fake = FakeTransport([low, reason])
    rows = je.run_jobs([job()], ["faithfulness"], fake, "fake/model", explain=True)
    assert len(rows) == 1 and rows[0]["conf"] < je.EXPLAIN_BELOW
    # the re-ask ran with thinking on and a long budget
    assert fake.payloads[1]["chat_template_kwargs"] == {"enable_thinking": True}
    assert fake.payloads[1]["max_tokens"] == 400
    assert "Because the evidence is missing the day." in rows[0]["why"]


def test_explain_mode_not_needed_for_confident_call() -> None:
    high = relay_response("A", [("A", -0.01), ("B", -3.0), ("C", -6.0), ("D", -8.0)])
    fake = FakeTransport([high])
    rows = je.run_jobs([job()], ["faithfulness"], fake, "fake/model", explain=True)
    assert rows[0]["conf"] >= je.EXPLAIN_BELOW
    assert rows[0]["why"] is None and len(fake.payloads) == 1


def test_explain_mode_triggers_when_choice_is_not_the_best_letter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the sampled token is B but A is the top-1 letter: the mismatch must trigger explain even
    # when the confidence threshold alone would not
    mismatch = relay_response("B", [("A", -0.05), ("B", -0.06), ("C", -6.0), ("D", -8.0)])
    reason = relay_response("ok", [("ok", -0.0)], content="The token and the top letter disagree.")
    fake = FakeTransport([mismatch, reason])
    monkeypatch.setattr(je, "EXPLAIN_BELOW", 0.0)  # kill the conf branch: only the mismatch counts
    rows = je.run_jobs([job()], ["faithfulness"], fake, "fake/model", explain=True)
    assert rows[0]["choice"] == "B"
    assert len(fake.payloads) == 2 and rows[0]["why"]


def test_runs_one_call_at_a_time_and_rows_shape() -> None:
    canned = relay_response("A", [("A", -0.01), ("B", -4.0), ("C", -8.0), ("D", -10.0)])
    fake = FakeTransport([canned] * 7)
    rows = je.run_jobs([job()], list(je.METRICS), fake, "fake/model")
    # directness 1 + faithfulness 1 + correctness 1 + completeness 1 + recall 1 + precision 2
    assert len(rows) == 7
    assert [r["metric"] for r in rows] == [
        "directness",
        "faithfulness",
        "correctness",
        "completeness",
        "recall",
        "precision",
        "precision",
    ]
    # strictly sequential: payload count equals call count, in the same order
    assert len(fake.payloads) == 7
    for row in rows:
        assert list(row) == ["turn", "question", "metric", "choice", "conf", "probs", "why"]
        assert row["why"] is None and row["turn"] == 1


def test_missing_inputs_make_no_row_not_a_skipped_call() -> None:
    bare = job(evidence=[], gold=[], key_points=[], reference="", excerpts=[])
    fake = FakeTransport([relay_response("A", [("A", -0.1), ("B", -2.0)]) for _ in range(2)])
    rows = je.run_jobs([bare], list(je.METRICS), fake, "fake/model")
    # only directness can be judged without evidence/gold/reference
    assert [r["metric"] for r in rows] == ["directness"]
    assert len(fake.payloads) == 1


def test_turns_mode_flattens_search_docs_evidence(tmp_path: Path) -> None:
    log = tmp_path / "agent_server.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "turn": 1,
                        "question": "What day is the team meeting?",
                        "text": "The team meeting is on Tuesday.",
                        "calls": [
                            {
                                "tool": "search_docs",
                                "result_text": json.dumps(
                                    [
                                        {
                                            "text": "The team meeting is on Tuesday at 10:00.",
                                            "source": "notes.md",
                                        }
                                    ]
                                ),
                            }
                        ],
                    }
                ),
                json.dumps({"turn": 2, "question": "Hi.", "text": "Hi.", "calls": []}),
                json.dumps({"turn": 3, "question": "", "text": "", "calls": []}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    jobs, skipped = je.load_log_rows(log)
    assert skipped == 1  # the empty turn
    assert len(jobs) == 2
    assert jobs[0]["evidence"] == ["The team meeting is on Tuesday at 10:00."]
    assert jobs[1]["evidence"] == []
    fake = FakeTransport([relay_response("A", [("A", -0.01), ("B", -3.0), ("C", -6.0)])] * 4)
    rows = je.run_jobs(jobs, ["directness", "faithfulness"], fake, "fake/model")
    # turn 2 has no evidence -> directness only
    assert {r["turn"] for r in rows} == {1, 2}
    assert [(r["turn"], r["metric"]) for r in rows] == [
        (1, "directness"),
        (1, "faithfulness"),
        (2, "directness"),
    ]


def test_rows_mode_input(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text(
        json.dumps(
            [
                {"id": "r1", "question": "Q?", "answer": "A.", "evidence": "E."},
                {"question": "Q2?", "answer": "A2."},
            ]
        ),
        encoding="utf-8",
    )
    jobs = je.load_rows_input(path)
    assert [j["turn"] for j in jobs] == ["r1", 1]  # id, then the index
    assert jobs[0]["evidence"] == ["E."]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"question": "", "answer": ""}]), encoding="utf-8")
    with pytest.raises(je.JudgeError, match="row 0 has no question or answer"):
        je.load_rows_input(bad)


def test_relay_retry_once_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    relay = je.Relay("http://127.0.0.1:18080/v1", FAKE_KEY, "fake/model")
    calls = {"n": 0}

    def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeResp:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(str(req), 500, "boom", {}, None)
        body = json.dumps(relay_response("A", [("A", -0.1), ("B", -2.0)])).encode()
        return _FakeResp(body)

    monkeypatch.setattr(je.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(je.time, "sleep", lambda _s: None)
    data = relay.post(je.judge_payload("fake/model", "hi"))
    assert calls["n"] == 2 and data["choices"][0]["logprobs"]


def test_relay_fails_loudly_after_one_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    relay = je.Relay("http://127.0.0.1:18080/v1", FAKE_KEY, "fake/model")
    calls = {"n": 0}

    def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeResp:
        calls["n"] += 1
        raise urllib.error.HTTPError(str(req), 503, "busy", {}, None)

    monkeypatch.setattr(je.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(je.time, "sleep", lambda _s: None)
    with pytest.raises(je.JudgeError, match="HTTP 503"):
        relay.post(je.judge_payload("fake/model", "hi"))
    assert calls["n"] == 2  # one attempt + one retry, then loud


def test_relay_retries_connection_errors_once(monkeypatch: pytest.MonkeyPatch) -> None:
    relay = je.Relay("http://127.0.0.1:18080/v1", FAKE_KEY, "fake/model")
    calls = {"n": 0}

    def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeResp:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("connection refused")
        return _FakeResp(json.dumps(relay_response("B", [("B", -0.1)])).encode())

    monkeypatch.setattr(je.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(je.time, "sleep", lambda _s: None)
    data = relay.post(je.judge_payload("fake/model", "hi"))
    assert calls["n"] == 2 and data["choices"][0]["message"]["content"] == "B"


def test_key_header_and_never_in_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # the key travels only in the Authorization header, built from the request
    relay = je.Relay("http://127.0.0.1:18080/v1", FAKE_KEY, "fake/model")
    request = relay.request({"model": "fake/model"})
    assert request.get_header("Authorization") == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in request.full_url

    # end to end: key file is read, and the key never appears in any produced artifact
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "api_key.txt").write_text(FAKE_KEY + "\n", encoding="utf-8")
    assert je.read_key(secrets) == FAKE_KEY
    rows_in = tmp_path / "rows.json"
    rows_in.write_text(json.dumps([{"question": "Q?", "answer": "A."}]), encoding="utf-8")
    out = tmp_path / "out.jsonl"
    fake = FakeTransport([relay_response("A", [("A", -0.01), ("B", -3.0), ("C", -6.0)])])
    rc = je.main(
        ["rows", "--in", str(rows_in), "--metrics", "directness", "--out", str(out)],
        transport=fake,
        model="fake/model",
    )
    assert rc == 0
    produced = out.read_text(encoding="utf-8") + capsys.readouterr().out
    assert FAKE_KEY not in produced
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["metric"] == "directness" and row["choice"] == "A"


def test_read_key_from_env_and_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUDGE_API_KEY", "  env-key \n")
    assert je.read_key() == "env-key"
    monkeypatch.delenv("JUDGE_API_KEY")
    with pytest.raises(je.JudgeError, match="cannot read the relay key"):
        je.read_key(tmp_path / "nowhere")


def test_summary_table() -> None:
    rows = [
        {
            "turn": 1,
            "question": "q1",
            "metric": "directness",
            "choice": "A",
            "conf": 0.9,
            "probs": {},
            "why": None,
        },
        {
            "turn": 1,
            "question": "q1",
            "metric": "faithfulness",
            "choice": "B",
            "conf": 0.4,
            "probs": {},
            "why": "low",
        },
        {
            "turn": 2,
            "question": "q2",
            "metric": "directness",
            "choice": "C",
            "conf": 0.95,
            "probs": {},
            "why": None,
        },
    ]
    text = je.summarize(rows)
    assert re.search(r"directness\s+1\s+0\s+1\s+0", text)
    assert re.search(r"faithfulness\s+0\s+1\s+0\s+0", text)
    assert "lowest confidence" in text
    # the lowest-confidence row is listed first
    first = text.split("lowest confidence:")[1].strip().splitlines()[0]
    assert "turn 1" in first and "faithfulness" in first and "B" in first


def test_calibrate_agreement() -> None:
    rows = [
        {"turn": 1, "metric": "directness", "choice": "A", "conf": 0.9, "probs": {}, "why": None},
        {"turn": 2, "metric": "directness", "choice": "B", "conf": 0.7, "probs": {}, "why": None},
        {
            "turn": 2,
            "metric": "faithfulness",
            "choice": "A",
            "conf": 0.99,
            "probs": {},
            "why": None,
        },
    ]
    result = je.calibrate(rows, {1: "A", 2: "B", 3: "A"}, metric="directness")
    assert result["n"] == 3 and result["matched"] == 2
    assert result["agreement"] == 1.0
    assert result["missing"] == [3]
    assert result["disagreements"] == []
    off = je.calibrate(rows, {2: "A"}, metric="directness")
    assert off["agreement"] == 0.0
    assert off["disagreements"] == [{"turn": 2, "expected": "A", "got": "B"}]
    # the same turn with several metrics: without a metric filter the first row wins
    mixed = je.calibrate(rows, {2: "A"}, metric="faithfulness")
    assert mixed["agreement"] == 1.0
