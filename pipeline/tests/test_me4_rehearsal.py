"""scripts/me4_rehearsal.py: loader, scoring, and an offline run over the real docs server."""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from voice_pipeline import doc_answer

_spec = importlib.util.spec_from_file_location(
    "me4_rehearsal", Path(__file__).resolve().parents[2] / "scripts" / "me4_rehearsal.py"
)
assert _spec is not None and _spec.loader is not None
rh = importlib.util.module_from_spec(_spec)
sys.modules["me4_rehearsal"] = rh
_spec.loader.exec_module(rh)

DOCS = {
    "kb/a.md": "# Zoo\n\nThe platypus is a venomous mammal found in Australia.\n",
    "kb/b.md": "# Birds\n\nThe kiwi is a flightless bird from New Zealand.\n",
}
ROWS = [
    {
        "id": "s1",
        "q": "What kind of animal is a platypus?",
        "kind": "single",
        "gold_files": ["kb/a.md"],
        "gold_quote": "The platypus is a venomous mammal",
        "answer": "A venomous mammal.",
        "answer_regex": "venomous",
    },
    {
        "id": "u1",
        "q": "What is the capital of Atlantis?",
        "kind": "unanswerable",
        "gold_files": [],
        "gold_quote": "",
        "answer": "",
        "answer_regex": "x",
    },
]


def _write_questions(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_load_questions_reads_rows(tmp_path: Path) -> None:
    questions = rh.load_questions(_write_questions(tmp_path / "q.jsonl", ROWS))
    assert [q.id for q in questions] == ["s1", "u1"]
    assert questions[0].gold_files == ("kb/a.md",)


def test_an_unknown_kind_is_rejected(tmp_path: Path) -> None:
    path = _write_questions(tmp_path / "q.jsonl", [{**ROWS[0], "kind": "bogus"}])
    with pytest.raises(ValueError, match="unknown kind"):
        rh.load_questions(path)


def _accepted(text: str) -> doc_answer.DocAnswer:
    return doc_answer.DocAnswer("accepted", text)


def test_gold_file_matches_by_path_suffix() -> None:
    passages = [{"source": "ipo-2026/kb/a.md"}]
    assert rh.gold_in_passages(("kb/a.md",), passages)
    assert not rh.gold_in_passages(("kb/a.md", "kb/b.md"), passages)
    assert not rh.gold_in_passages(("a.md",), [{"source": "xa.md"}])


def test_answer_regex_and_abstain_phrase() -> None:
    assert rh.answer_matches(r"\b184,?500\b", "The wage base is $184,500.")
    assert rh.abstained("The documents don\u2019t say. Want me to look it up on the web?")
    assert not rh.abstained("As per a.md: a venomous mammal.")


def test_score_question_pass_and_fail_cases() -> None:
    questions = [rh.Question(**{**r, "gold_files": tuple(r["gold_files"])}) for r in ROWS]
    single, unanswerable = questions
    hit = [{"source": "kb/a.md"}]
    assert rh.score_question(single, hit, _accepted("A venomous mammal."))["pass"]
    assert not rh.score_question(single, [], _accepted("A venomous mammal."))["pass"]
    assert not rh.score_question(single, hit, _accepted("A bird."))["pass"]
    assert rh.score_question(unanswerable, [], doc_answer.DocAnswer("none"))["pass"]
    assert not rh.score_question(unanswerable, hit, _accepted("Paris."))["pass"]


def _make_corpus(corpus: Path) -> Path:
    for name, text in DOCS.items():
        (corpus / name).parent.mkdir(parents=True, exist_ok=True)
        (corpus / name).write_text(text, encoding="utf-8")
    return corpus


async def test_poll_ready_reports_first_searchable_and_ready(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path / "corpus")
    params = rh.server_params(rh.REPO / ".venv" / "bin" / "python", corpus, "fts5")
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        ready = await rh.poll_ready(session, 60.0)
    assert ready["files_ok"] == 2 and ready["failures"] == []
    assert ready["first_searchable_s"] is not None and ready["ready_s"] is not None


async def test_run_question_scores_a_hit_and_an_abstention(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path / "corpus")
    questions = rh.load_questions(_write_questions(tmp_path / "q.jsonl", ROWS))
    params = rh.server_params(rh.REPO / ".venv" / "bin" / "python", corpus, "fts5")
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        await rh.poll_ready(session, 60.0)
        rows = [await rh.run_question(session, q) for q in questions]
    assert [r["pass"] for r in rows] == [True, True]
    assert rows[0]["sources"] == ["kb/a.md"] and rows[0]["search_ms"] >= 0


def _make_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in DOCS.items():
            zf.writestr(name, text)
    return path


def test_summarize_counts_by_kind() -> None:
    rows = [
        {"kind": "single", "pass": True, "search_ms": 5.0},
        {"kind": "single", "pass": False, "search_ms": 9.0},
    ]
    summary = rh.summarize(rows)
    assert summary["by_kind"]["single"] == {"n": 2, "pass": 1}
    assert summary["search_ms_max"] == 9.0 and summary["pass"] == 1


def test_main_writes_a_json_report(tmp_path: Path) -> None:
    report = tmp_path / "rep.json"
    argv = [
        "--zip", str(_make_zip(tmp_path / "t.zip")),
        "--corpus", str(tmp_path / "corpus"),
        "--questions", str(_write_questions(tmp_path / "q.jsonl", ROWS)),
        "--report", str(report),
    ]  # fmt: skip
    assert rh.main(argv) == 0
    data = json.loads(report.read_text())
    assert data["summary"]["n"] == 2 and data["summary"]["pass"] == 2
    assert data["ready"]["files_ok"] == 2


def test_prepare_corpus_drops_the_zip_for_the_server_to_extract(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    rh.prepare_corpus(_make_zip(tmp_path / "t.zip"), corpus)
    assert [p.name for p in corpus.iterdir()] == ["t.zip"]


def test_live_mode_fake_request_and_reply(tmp_path: Path) -> None:
    _write_questions(tmp_path / "q.jsonl", ROWS)
    ledger = tmp_path / "ledger.json"

    async def _fake_gemini(*a, **kw):
        async def _req(system: str, content: str) -> str:
            return "ANSWER: test answer\nQUOTE: test quote"

        return _req

    rh.doc_answer.gemini_request = _fake_gemini
    argv = [
        "--zip", str(_make_zip(tmp_path / "t.zip")),
        "--corpus", str(tmp_path / "corpus"),
        "--questions", str(tmp_path / "q.jsonl"),
        "--mode", "live", "--report", str(tmp_path / "rep.json"),
        "--ledger", str(ledger), "--max-usd", "100",
    ]  # fmt: skip
    assert rh.main(argv) == 0
    data = json.loads((tmp_path / "rep.json").read_text())
    assert data["mode"] == "live" and len(data["rows"]) == 2
    assert data["rows"][0]["reply"] == "ANSWER: test answer\nQUOTE: test quote"
    assert "spend" in data and "stopped" not in data


def test_live_budget_cap_stops_before_calls(tmp_path: Path) -> None:
    _write_questions(tmp_path / "q.jsonl", ROWS)
    ledger = tmp_path / "ledger2.json"
    called = [False]

    async def _fake_gemini(*a, **kw):
        async def _req(system: str, content: str) -> str:
            called[0] = True
            return "ANSWER: should not reach\nQUOTE: x"

        return _req

    rh.doc_answer.gemini_request = _fake_gemini
    argv = [
        "--zip", str(_make_zip(tmp_path / "t.zip")),
        "--corpus", str(tmp_path / "corpus2"),
        "--questions", str(tmp_path / "q.jsonl"),
        "--mode", "live", "--report", str(tmp_path / "rep2.json"),
        "--ledger", str(ledger), "--max-usd", "0.0000001",
    ]  # fmt: skip
    assert rh.main(argv) == 0
    data = json.loads((tmp_path / "rep2.json").read_text())
    assert data["mode"] == "live" and data["stopped"] == "spend cap"
    assert data["rows"] == [] and not called[0]


def test_offline_default_unchanged(tmp_path: Path) -> None:
    report = tmp_path / "rep3.json"
    argv = [
        "--zip", str(_make_zip(tmp_path / "t.zip")),
        "--corpus", str(tmp_path / "corpus3"),
        "--questions", str(_write_questions(tmp_path / "q.jsonl", ROWS)),
        "--report", str(report),
    ]  # fmt: skip
    assert rh.main(argv) == 0
    data = json.loads(report.read_text())
    assert data["summary"]["n"] == 2 and data["summary"]["pass"] == 2
    assert data["mode"] == "offline" and "spend" not in data
