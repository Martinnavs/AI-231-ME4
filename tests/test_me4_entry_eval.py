"""Tests for me4_scoring and me4_entry_eval helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pytest

scripts = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(scripts))

from me4_scoring import (  # noqa: E402
    accept_match,
    fact_in_passages,
    load_questions,
    score_question,
    summarize,
)


def _make_tmp_corpus(tmp_path: Path) -> Path:
    """Create 3 markdown files for the entry test."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "security_incidents.md").write_text(
        "## Security Incident Response\n\n"
        "It is critical that you notify GSA IT within 1 hour of suspected incident.\n"
        "Report all incidents to the GSA IT Service Desk immediately.\n"
    )
    (corpus / "clarity_guidance.md").write_text(
        "## ATSDR Module 3 Clarity\n\n"
        "Federal clarity guidance requires plain writing at an 8th-grade level.\n"
        "For general communications, this is typically an 8th-grade level.\n"
    )
    (corpus / "electricity_rates.md").write_text(
        "## Electricity Rates Table\n\n"
        "Hawaii residential: 47.18 cents/kWh. Oregon: 10.50 cents/kWh.\n"
        "Census Division and State: Hawaii; August 2026: 47.18.\n"
    )
    return corpus


def _make_index(corpus: Path) -> Any:
    from voice_agent.rag.docindex import DocIndex

    return DocIndex(corpus)


class TestLoader:
    def test_new_format(self, tmp_path: Path) -> None:
        qf = tmp_path / "qs.jsonl"
        qf.write_text(
            json.dumps(
                {
                    "id": "t1",
                    "q": "What?",
                    "kind": "single",
                    "facts": [{"file": "a.md", "contains": "fact text"}],
                    "answer": "Yes",
                    "accept": ["yes"],
                    "web": None,
                    "unanswerable": False,
                }
            )
            + "\n"
        )
        qs = load_questions(qf)
        assert len(qs) == 1
        assert qs[0]["legacy"] is False

    def test_old_format_adapter(self, tmp_path: Path) -> None:
        qf = tmp_path / "qs.jsonl"
        qf.write_text(
            json.dumps(
                {
                    "id": "t1",
                    "q": "What?",
                    "kind": "single",
                    "answer": "Yes",
                    "answer_regex": r"(?i)yes",
                    "gold_files": ["a.md"],
                    "gold_quote": "fact text",
                }
            )
            + "\n"
        )
        qs = load_questions(qf)
        assert len(qs) == 1
        assert qs[0]["legacy"] is True
        assert qs[0]["facts"] == []

    def test_bad_row_raises(self, tmp_path: Path) -> None:
        qf = tmp_path / "qs.jsonl"
        qf.write_text("not json\n")
        with pytest.raises(json.JSONDecodeError):
            load_questions(qf)


class TestScoring:
    def test_accept_list_normalisation(self) -> None:
        assert accept_match(["175300", "175,300"], "175,300")
        assert accept_match(["175300", "175,300"], "175300")
        assert accept_match(["Within 1 hour", "within one hour"], "Within 1 hour.")
        assert not accept_match(["175300"], "175400")
        assert accept_match(["60-22"], "As per faa/ch2.pdf: AC 60-22.")
        assert not accept_match(["8"], "As per file8.pdf: eighteen")
        assert not accept_match(["60-22"], "As per 60-22.pdf: no idea")
        # the ticket 242 spoken leads are stripped before the accept check too
        assert accept_match(["60-22"], "From your faa chapter 2 document: AC 60-22.")
        assert accept_match(["60-22"], "From your document on FAA Chapter 2: AC 60-22.")
        assert accept_match(["60-22"], "From your documents: AC 60-22.")

    def test_fact_in_passages_with_whitespace_diff(self) -> None:
        fact = {"file": "a.md", "contains": "within 1 hour of suspected incident"}
        passage = {"source": "a.md", "text": "within one hour of suspected incident"}
        assert not fact_in_passages([fact], [passage])
        passage2 = {"source": "a.md", "text": "notify within 1 hour of suspected incident here"}
        assert fact_in_passages([fact], [passage2])

    def test_false_accept_for_decoy(self) -> None:
        q = {
            "id": "d1",
            "kind": "decoy",
            "q": "decoy?",
            "facts": [],
            "unanswerable": True,
            "accept": [],
        }
        row = {"route": "docs", "text": "It is 42.", "docs": "accepted", "calls": [], "end_s": 1.0}
        s = score_question(q, row)
        assert s["false_accept"] is True
        assert s["accepted"] is True

    def test_abstain_counts(self) -> None:
        q = {
            "id": "a1",
            "kind": "single",
            "q": "Q?",
            "facts": [],
            "unanswerable": False,
            "accept": ["yes"],
        }
        row = {
            "route": "routed",
            "text": "I don't know.",
            "docs": "none",
            "calls": [],
            "end_s": 0.5,
        }
        s = score_question(q, row)
        assert s["abstained"] is True
        assert s["accepted"] is False

    def test_summary_counts(self) -> None:
        rows = [
            {
                "kind": "single",
                "route": "docs",
                "accepted": True,
                "correct": True,
                "fact_in_passages": True,
                "false_accept": False,
                "abstained": False,
                "total_ms": 500,
            },
            {
                "kind": "single",
                "route": "docs",
                "accepted": True,
                "correct": False,
                "fact_in_passages": True,
                "false_accept": True,
                "abstained": False,
                "total_ms": 800,
            },
            {
                "kind": "table",
                "route": "routed",
                "accepted": False,
                "correct": False,
                "fact_in_passages": False,
                "false_accept": False,
                "abstained": True,
                "total_ms": 1000,
            },
        ]
        summary = summarize(rows)
        assert summary["n"] == 3
        assert summary["accepted"] == 2
        assert summary["correct"] == 1
        assert summary["false_accept"] == 1
        assert summary["abstained"] == 1
        assert summary["by_kind"]["single"]["n"] == 2
        assert summary["by_kind"]["table"]["n"] == 1

    def test_docweb_row(self) -> None:
        q = {
            "id": "w1",
            "kind": "docweb",
            "q": "Q?",
            "facts": [],
            "unanswerable": False,
            "accept": ["obama"],
            "web": {"needs": "x", "expected": "obama"},
        }
        row = {"route": "docs", "text": "obama", "docs": "accepted", "calls": [], "end_s": 1.0}
        s = score_question(q, row)
        assert s["web_needed"] is True
        assert "correct" not in s
        assert s.get("correct_doc_half") is True


class TestRetrievalWiring:
    def test_index_returns_passages(self, tmp_path: Path) -> None:
        corpus = _make_tmp_corpus(tmp_path)
        idx = _make_index(corpus)
        idx.refresh()
        results = idx.search("security incident", k=8)
        assert isinstance(results, list) and len(results) > 0
        assert any("security_incidents.md" in p.get("source", "") for p in results)

    def test_index_status_has_files(self, tmp_path: Path) -> None:
        corpus = _make_tmp_corpus(tmp_path)
        idx = _make_index(corpus)
        idx.refresh()
        recs = idx.records()
        assert len(recs) == 3
        names = [r.name for r in recs]
        assert "security_incidents.md" in names
        assert "clarity_guidance.md" in names
        assert "electricity_rates.md" in names

    def test_oracle_checks_passages(self) -> None:
        import asyncio

        from me4_entry_eval import _OracleRequest

        q = {
            "id": "x1",
            "facts": [{"file": "a.md", "contains": "within 1 hour"}],
            "answer": "Within 1 hour",
            "accept": ["within 1 hour"],
        }
        oracle = _OracleRequest(q)
        match_content = (
            'Passages:\n[1] <untrusted_document file="a.md">'
            "notify GSA IT within 1 hour of suspected incident"
            "</untrusted_document>\n\nQuestion: test?"
        )
        content_empty = "Passages:\n\nQuestion: test?"
        reply = asyncio.run(oracle("", match_content))
        assert reply.startswith("ANSWER:")
        reply2 = asyncio.run(oracle("", content_empty))
        assert reply2 == "NONE"
        reply3 = asyncio.run(oracle("", "no match here"))
        assert reply3 == "NONE"


@pytest.mark.asyncio
async def test_offline_entry_mode(tmp_path: Path) -> None:
    pytest.importorskip("pipecat")
    corpus = _make_tmp_corpus(tmp_path)
    _facts = [{"file": "security_incidents.md", "contains": "notify GSA IT within 1 hour"}]
    _sec_q = "According to the documents, what is the notification window for security incidents?"
    questions = [
        {
            "id": "t1",
            "q": _sec_q,
            "kind": "single",
            "facts": _facts,
            "answer": "Within 1 hour",
            "accept": ["within 1 hour"],
            "unanswerable": False,
            "web": None,
        },
        {
            "id": "t2",
            "q": "What is the notification window for security incidents?",
            "kind": "single",
            "facts": _facts,
            "answer": "Within 1 hour",
            "accept": ["within 1 hour"],
            "unanswerable": False,
            "web": None,
        },
        {
            "id": "t3",
            "q": "What level is recommended?",
            "kind": "reworded",
            "facts": [{"file": "clarity_guidance.md", "contains": "8th-grade level"}],
            "answer": "8th-grade",
            "accept": ["8th-grade"],
            "unanswerable": False,
            "web": None,
        },
        {
            "id": "t4",
            "q": "What is the electricity price in Washington?",
            "kind": "decoy",
            "facts": [],
            "answer": "N/A",
            "accept": [],
            "unanswerable": True,
            "web": None,
        },
        {
            "id": "t5",
            "q": "According to the documents, what is the GSA budget change for 2025?",
            "kind": "unanswerable",
            "facts": [],
            "answer": "N/A",
            "accept": [],
            "unanswerable": True,
            "web": None,
        },
        {
            "id": "t6",
            "q": "What color is the sky?",
            "kind": "offtopic",
            "facts": [],
            "answer": "Blue",
            "accept": ["blue"],
            "unanswerable": True,
            "web": None,
        },
    ]

    qf = tmp_path / "qs.jsonl"
    qf.write_text("\n".join(json.dumps(q) for q in questions) + "\n")

    from me4_entry_eval import _run_offline

    report = await _run_offline(corpus, qf)
    assert len(report["rows"]) == 6
    rows = {r["id"]: r for r in report["rows"]}

    assert rows["t1"]["route"] == "docs"
    assert rows["t3"]["reached_agent"] is True
    assert rows["t4"]["accepted"] is False or rows["t4"]["route"] != "docs"
    assert rows["t6"]["route"] != "docs"

    for rid in ("t1", "t2"):
        assert rows[rid]["passages"], f"{rid}: passages should not be empty"

    for rid in ("t1", "t2"):
        assert rows[rid]["file_in_passages"], f"{rid}: file_in_passages should be True"

    docs_rows = [r for r in rows.values() if r["route"] == "docs"]
    assert any(r["accepted"] for r in docs_rows), "At least one docs row should be accepted"

    summary = report["summary"]
    assert summary["n"] == 6
    assert summary["route_histogram"] is not None


@pytest.mark.asyncio
async def test_replay_equal(tmp_path: Path) -> None:
    report = {
        "mode": "offline",
        "questions_path": str(tmp_path / "qs.jsonl"),
        "rows": [
            {
                "id": "r1",
                "kind": "single",
                "q": "Q?",
                "route": "docs",
                "text": "ANSWER",
                "docs": "accepted",
                "calls": [],
                "end_s": 1.0,
                "doc_s": 0.5,
            },
            {
                "id": "r2",
                "kind": "single",
                "q": "Q2?",
                "route": "routed",
                "text": "no answer",
                "docs": "none",
                "calls": [],
                "end_s": 0.5,
                "doc_s": 0.0,
            },
        ],
    }
    rp = tmp_path / "report.json"
    rp.write_text(json.dumps(report))

    from me4_entry_eval import _run_replay

    replay = await _run_replay(rp)
    assert replay["mode"] == "replay"
    assert replay["summary"]["n"] == 2
    assert replay["summary"]["accepted"] == replay["rows"].count(True) + 1


def test_live_mode_refuses_without_approval(tmp_path: Path, monkeypatch: Any) -> None:
    exited: list[int] = []

    def fake_exit(code: int) -> None:
        exited.append(code)

    monkeypatch.setattr("sys.exit", fake_exit)

    from me4_entry_eval import _run_live_main

    args = argparse.Namespace(
        i_have_approval_to_spend=False,
        ledger=tmp_path / "ledger.json",
        corpus=tmp_path,
        db=None,
    )
    assert _run_live_main(args) == 2


def test_live_mode_refuses_without_ledger(tmp_path: Path) -> None:
    from me4_entry_eval import _run_live_main

    args = argparse.Namespace(
        i_have_approval_to_spend=True,
        ledger=tmp_path / "missing.json",
        corpus=tmp_path,
        db=None,
    )
    assert _run_live_main(args) == 2


def test_plain_loading(tmp_path: Path) -> None:
    """--plain loads one question per line as nondoc kind."""
    from me4_entry_eval import _build_plain_questions

    pf = tmp_path / "plain.txt"
    pf.write_text("What's the weather?\nWhat time is it?\n")
    qs = _build_plain_questions(pf)
    assert len(qs) == 2
    assert qs[0]["id"] == "nondoc-01"
    assert qs[0]["kind"] == "nondoc"
    assert qs[0]["facts"] == []
    assert qs[0]["unanswerable"] is True
    assert qs[0]["q"] == "What's the weather?"
    assert qs[1]["id"] == "nondoc-02"
    assert qs[1]["q"] == "What time is it?"


def test_plain_skips_empty_lines(tmp_path: Path) -> None:
    """Empty lines in --plain are skipped."""
    from me4_entry_eval import _build_plain_questions

    pf = tmp_path / "plain.txt"
    pf.write_text("Q1\n\n  \nQ2\n")
    qs = _build_plain_questions(pf)
    assert len(qs) == 2
    assert qs[0]["q"] == "Q1"
    assert qs[1]["q"] == "Q2"


def test_router_fake_default(tmp_path: Path) -> None:
    """--router defaults to fake; existing behaviour unchanged."""
    pytest.importorskip("pipecat")
    from me4_entry_eval import main

    corpus = _make_tmp_corpus(tmp_path)
    _facts = [{"file": "security_incidents.md", "contains": "notify GSA IT within 1 hour"}]
    qf = tmp_path / "qs.jsonl"
    qf.write_text(
        json.dumps(
            {
                "id": "t1",
                "q": "What is the notification window?",
                "kind": "single",
                "facts": _facts,
                "answer": "Within 1 hour",
                "accept": ["within 1 hour"],
                "unanswerable": False,
                "web": None,
            }
        )
        + "\n"
    )
    report_path = tmp_path / "report.json"

    # Default --router is fake (no --router flag at all)
    rv = main(["--corpus", str(corpus), "--questions", str(qf), "--report", str(report_path)])
    assert rv == 0
    report = json.loads(report_path.read_text())
    assert report["router_mode"] == "fake"
    assert len(report["rows"]) == 1


def test_summary_table_with_fixture(tmp_path: Path) -> None:
    """Summary table groups questions correctly by route."""
    from me4_entry_eval import _build_summary_table

    questions = [
        {"id": "d1", "kind": "single", "unanswerable": False, "facts": []},
        {"id": "d2", "kind": "decoy", "unanswerable": True, "facts": []},
        {"id": "n1", "kind": "nondoc", "unanswerable": True, "facts": []},
    ]
    scored = [
        {
            "id": "d1",
            "route": "docs",
            "docs_called": True,
            "accepted": True,
            "correct": True,
            "router_lines": "",
        },
        {
            "id": "d2",
            "route": "routed",
            "docs_called": False,
            "accepted": False,
            "correct": False,
            "router_lines": "none|decoy",
        },
        {
            "id": "n1",
            "route": "routed",
            "docs_called": False,
            "accepted": False,
            "correct": False,
            "router_lines": "weather|Manila|today",
            "agent_called": False,
        },
    ]
    table = _build_summary_table(scored, questions)
    assert "dev_answerable" in table
    assert table["dev_answerable"]["n"] == 1
    assert table["dev_answerable"]["docs_called"] == 1
    assert table["dev_answerable"]["accepted"] == 1
    assert table["dev_answerable"]["correct"] == 1

    assert "dev_other" in table
    assert table["dev_other"]["n"] == 1
    assert table["dev_other"]["docs_called"] == 0

    assert "nondoc25" in table
    assert table["nondoc25"]["n"] == 1
    assert table["nondoc25"]["docs_called"] == 0
    assert table["nondoc25"]["home_device"] == 1  # weather is a home/device action


def test_real_router_unreachable_exits(tmp_path: Path) -> None:
    """--router real with unreachable URL exits non-zero with a message."""
    pytest.importorskip("pipecat")
    from me4_entry_eval import main

    corpus = _make_tmp_corpus(tmp_path)
    qf = tmp_path / "qs.jsonl"
    qf.write_text(
        json.dumps(
            {
                "id": "t1",
                "q": "test?",
                "kind": "nondoc",
                "facts": [],
                "answer": "",
                "accept": [],
                "unanswerable": True,
                "web": None,
            }
        )
        + "\n"
    )
    report_path = tmp_path / "report.json"

    # Use an unreachable port (19999) — no real router there
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--corpus",
                str(corpus),
                "--questions",
                str(qf),
                "--report",
                str(report_path),
                "--router",
                "real",
                "--router-url",
                "http://127.0.0.1:19999",
            ]
        )
    assert exc_info.value.code == "error: real router at http://127.0.0.1:19999 is unreachable"


def test_search_ms_measured(tmp_path: Path) -> None:
    """search_ms is measured from actual search_docs time, not 0.0."""
    import asyncio

    pytest.importorskip("pipecat")
    corpus = _make_tmp_corpus(tmp_path)
    questions = [
        {
            "id": "t1",
            "q": "What is the notification window for security incidents?",
            "kind": "single",
            "facts": [{"file": "security_incidents.md", "contains": "notify GSA IT within 1 hour"}],
            "answer": "Within 1 hour",
            "accept": ["within 1 hour"],
            "unanswerable": False,
            "web": None,
        },
    ]
    qf = tmp_path / "qs.jsonl"
    qf.write_text("\n".join(json.dumps(q) for q in questions) + "\n")

    from me4_entry_eval import _run_offline

    report = asyncio.run(_run_offline(corpus, qf))
    row = report["rows"][0]
    # The docs route triggers search_docs, so search_ms should be > 0
    assert row["search_ms"] > 0, f"search_ms should be > 0, got {row['search_ms']}"
    assert row["docs_called"] is True


def test_docs_called_and_agent_called_fields(tmp_path: Path) -> None:
    """Per-row docs_called and agent_called fields are present."""
    import asyncio

    pytest.importorskip("pipecat")
    corpus = _make_tmp_corpus(tmp_path)
    questions = [
        {
            "id": "t1",
            "q": "What is the notification window for security incidents?",
            "kind": "single",
            "facts": [{"file": "security_incidents.md", "contains": "notify GSA IT within 1 hour"}],
            "answer": "Within 1 hour",
            "accept": ["within 1 hour"],
            "unanswerable": False,
            "web": None,
        },
        {
            "id": "t2",
            "q": "What color is the sky?",
            "kind": "offtopic",
            "facts": [],
            "answer": "Blue",
            "accept": ["blue"],
            "unanswerable": True,
            "web": None,
        },
    ]
    qf = tmp_path / "qs.jsonl"
    qf.write_text("\n".join(json.dumps(q) for q in questions) + "\n")

    from me4_entry_eval import _run_offline

    report = asyncio.run(_run_offline(corpus, qf))
    rows = {r["id"]: r for r in report["rows"]}

    # docs question: oracle should be called
    assert "docs_called" in rows["t1"]
    assert rows["t1"]["docs_called"] is True

    # offtopic: oracle should not be called (no facts to check)
    assert "docs_called" in rows["t2"]

    # agent_called should be present
    assert "agent_called" in rows["t1"]
    assert "agent_called" in rows["t2"]

    # router fields should be present
    assert "router_s" in rows["t1"]
    assert "router_reason" in rows["t1"]
    assert "router_lines" in rows["t1"]
