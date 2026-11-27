"""Ticket 270: post-lookup gates on the sufficiency supplement (figure, period, scope)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from voice_pipeline import sufficiency
from voice_pipeline.agent_server import AgentServer
from voice_pipeline.ask_documents import DocResult
from voice_pipeline.sufficiency import Verdict, supplement_ok

MANILA = ZoneInfo("Asia/Manila")
NOW = datetime(2026, 11, 27, 12, 0, tzinfo=MANILA)
FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "eval" / "sufficiency" / "supplement_replies_dev.jsonl"
)


def _row_verdict(row: dict[str, Any]) -> Verdict:
    """Derive stale vs missing_half for dev fixture rows lacking an explicit verdict field."""
    kind = (
        "stale"
        if sufficiency.time_sensitive(row["question"]) or bool(row.get("doc_answer"))
        else "missing_half"
    )
    return Verdict(verdict=kind, web_question=row["web_question"], reason="")


# --- Fixture test on the 34 dev replies ---


def test_dev_fixture_drops_all_6_bad_keeps_at_least_22_ok_and_4_hedges_silent() -> None:
    rows = [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 34

    bad_rows = [r for r in rows if r["label"] == "bad"]
    ok_rows = [r for r in rows if r["label"] == "ok"]
    hedge_rows = [r for r in rows if r["label"] == "hedge"]
    assert len(bad_rows) == 6
    assert len(ok_rows) == 24
    assert len(hedge_rows) == 4

    bad_reasons: dict[str, str] = {}
    for row in bad_rows:
        ok, why, _ = supplement_ok(
            _row_verdict(row),
            row["reply"],
            NOW,
            question=row["question"],
            doc_answer=row["doc_answer"],
        )
        assert not ok, f"expected bad row {row['id']} to be dropped"
        bad_reasons[row["id"]] = why

    for row_id in ("imp-011", "imp-015", "imp-013"):
        assert bad_reasons[row_id].startswith("figure:"), row_id
    for row_id in ("imp-001", "imp-061"):
        assert bad_reasons[row_id].startswith("period:"), row_id
    assert bad_reasons["imp-004"].startswith("scope:")

    kept_ok_ids: list[str] = []
    for row in ok_rows:
        ok, _, _ = supplement_ok(
            _row_verdict(row),
            row["reply"],
            NOW,
            question=row["question"],
            doc_answer=row["doc_answer"],
        )
        if ok:
            kept_ok_ids.append(row["id"])

    assert len(kept_ok_ids) >= 22
    for must_keep in ("imp-017", "imp-024", "imp-067", "imp-068", "imp-092"):
        assert must_keep in kept_ok_ids

    for row in hedge_rows:
        assert sufficiency.HEDGE_TAIL in row["reply"]


# --- Gate 1: Figure gate unit tests ---


def test_figure_gate_edge_cases() -> None:
    v_stale = Verdict("stale", "What is the cap for 2027?", "")
    v_half = Verdict("missing_half", "Who publishes it?", "")

    ok, why, _ = supplement_ok(
        v_stale,
        "The cap is $192,300. Wages above this amount are not subject to the tax.",
        NOW,
    )
    assert ok and why == ""

    ok, why, _ = supplement_ok(
        v_stale,
        "The October 2026 growth figure has not yet been released.",
        NOW,
    )
    assert not ok and why.startswith("figure:")

    ok, why, _ = supplement_ok(
        v_half,
        "The requested data is not yet available.",
        NOW,
    )
    assert not ok and why.startswith("figure:")

    ok, why, _ = supplement_ok(
        v_stale,
        "The holiday break took place in October 2026 and has already concluded.",
        NOW,
    )
    assert not ok and why.startswith("figure:")

    ok, why, _ = supplement_ok(
        v_stale,
        "As per 24Timezones, the campus is in the Central Time zone.",
        NOW,
    )
    assert not ok and why.startswith("figure:")

    ok, why, _ = supplement_ok(
        v_half,
        "The Office of Personnel Management publishes the schedule.",
        NOW,
    )
    assert ok and why == ""


# --- Gate 2: Period gate unit tests ---


def test_period_gate_edge_cases() -> None:
    v = Verdict("stale", "What is the current per diem rate?", "")

    # Current fiscal year (FY2027 on 2026-11-27) is kept.
    ok, _, _ = supplement_ok(v, "For the 2027 fiscal year, the rate is $225.", NOW)
    assert ok
    ok, _, _ = supplement_ok(v, "The FY2027 rate is $225.", NOW)
    assert ok

    # Ended fiscal year (FY2026 ended 2026-09-30) is dropped.
    ok, why, _ = supplement_ok(
        v, "The rate is $209 for November 2026 under the 2026 fiscal year.", NOW
    )
    assert not ok and why == "period:stale"
    ok, why, _ = supplement_ok(
        v, "The rate is $342 for the fiscal year ending September 30, 2026.", NOW
    )
    assert not ok and why == "period:stale"

    # Observation/origin dates are exempt.
    ok, _, _ = supplement_ok(v, "The index reached 709.97 billion pounds as of June 2026.", NOW)
    assert ok
    ok, _, _ = supplement_ok(v, "As of estimates from July 2025, the population is 910,638.", NOW)
    assert ok
    ok, _, _ = supplement_ok(v, "He has served since his confirmation in May 2025.", NOW)
    assert ok

    # Validity calendar year / month ended before now is dropped for present questions,
    # while current year/month and fixed-past questions are kept.
    ok, why, _ = supplement_ok(v, "The 2025 rate is $180.", NOW)
    assert not ok and why == "period:stale"
    ok, why, _ = supplement_ok(v, "For the month of October 2026, the rate is $180.", NOW)
    assert not ok and why == "period:stale"
    ok, _, _ = supplement_ok(v, "The 2026 rate is $209.", NOW)
    assert ok
    ok, _, _ = supplement_ok(
        Verdict("stale", "", ""),
        "The 2025 rate is $180.",
        NOW,
        question="What was the rate in 2025?",
    )
    assert ok


# --- Gate 3: Scope prompt, source label, and country gate ---


def test_scope_prompt_and_source_label() -> None:
    s = sufficiency.SYSTEM
    assert "keeps every public scope qualifier" in s
    assert "General Schedule / OPM" in s


async def test_verdict_includes_source_label_only_when_given() -> None:
    seen: list[str] = []

    async def fake(_system: str, content: str, _schema: dict[str, Any] | None = None) -> str:
        seen.append(content)
        return json.dumps({"verdict": "sufficient", "web_question": "", "reason": "ok"})

    await sufficiency.verdict(
        fake,
        "How much does a GS-15 step 1 make next January?",
        "As per gs_salary_table_2026.pdf: $126,384",
        "2026",
        NOW,
        source_label="us/gs_salary_table_2026.pdf",
    )
    assert "Source label: us/gs_salary_table_2026.pdf" in seen[0]


def test_country_scope_gate_edge_cases() -> None:
    v = Verdict("stale", "What is the GS-15 step 1 salary for 2027?", "")

    # Unasked country in reply when doc_answer is present -> dropped.
    ok, why, _ = supplement_ok(
        v,
        "Under the Philippine Salary Grade system, Salary Grade 15 is 44,148 pesos for 2027.",
        NOW,
        question="How much does a GS-15 step 1 make next January?",
        doc_answer="As per gs_salary_table_2026.pdf (2026): $126,384",
    )
    assert not ok and why == "scope:philippines"

    # Cross-variant match (United States in question, U.S. in reply) -> kept.
    ok, _, _ = supplement_ok(
        Verdict("stale", "What is the United States GS-15 salary for 2027?", ""),
        "The U.S. General Schedule GS-15 step 1 salary for 2027 is $130,000.",
        NOW,
        question="What is the United States GS-15 salary next year?",
        doc_answer="As per gs_salary_table_2026.pdf: $126,384",
    )
    assert ok

    # Lowercase pronoun "us" must not trigger United States.
    ok, _, _ = supplement_ok(
        v,
        "The latest figures give us a salary of $130,000 for 2027.",
        NOW,
        question="How much does a GS-15 step 1 make next January?",
        doc_answer="As per gs_salary_table_2026.pdf: $126,384",
    )
    assert ok

    # Empty doc_answer skips the country check.
    ok, _, _ = supplement_ok(
        v,
        "In Canada, the 2027 figure is $130,000.",
        NOW,
        question="What is the figure next year?",
        doc_answer="",
    )
    assert ok


# --- AgentServer integration: dropped supplement is silent and logged ---


async def test_agent_server_logs_sufficiency_dropped_and_stays_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sufficiency, "SUFFICIENCY", True)
    seen_verdict_inputs: list[str] = []

    async def wa(params: Any) -> None:
        await params.result_callback(
            "As per Parliament, the October 2026 figure has not yet been released."
        )

    async def fake_json(_system: str, content: str, _schema: dict[str, Any] | None = None) -> str:
        seen_verdict_inputs.append(content)
        return json.dumps(
            {
                "verdict": "stale",
                "web_question": "What was the UK economic growth last month?",
                "reason": "source is old",
            }
        )

    tools = SimpleNamespace(standard_tools=[SimpleNamespace(name="web_answer", handler=wa)])
    server = AgentServer(
        object(), [], tmp_path / "270.jsonl", tools=tools, doc_json_request=fake_json
    )
    doc = DocResult(
        "accepted",
        spoken="As per uk_gdp_2025.xlsx (2025): 1.1%.",
        answer="1.1%.",
        file="uk/uk_gdp_2025.xlsx",
        sources=["uk/uk_gdp_2025.xlsx"],
        calls=1,
    )
    emitted: list[str] = []

    async def on_text(text: str) -> None:
        emitted.append(text)

    with caplog.at_level(logging.INFO, logger="voice_pipeline.agent_server"):
        res = await server._maybe_supplement(
            "What was the UK's economic growth right now?",
            doc,
            server.spoken(doc.spoken or "", on_text, docs="accepted", route="docs"),
            on_text,
            NOW,
        )

    assert emitted == [doc.spoken]
    assert res["sufficiency"] == "stale"
    assert res["sufficiency_web"] is False
    assert "Source label: uk/uk_gdp_2025.xlsx" in seen_verdict_inputs[0]
    assert any("sufficiency_dropped=figure:unavailable" in m for m in caplog.messages)
