"""Ticket 271: supplement gates, second pass (yes/no and named answers, period mismatch,
the opposite-yes/no contradiction) on both labelled fixtures, plus gate edge cases.

The test-half replies are SEEN from this ticket on (they drove it): both fixtures are
development data; the counts asserted here are computed from the files.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from voice_pipeline import sufficiency
from voice_pipeline.sufficiency import (
    LEAD_FROM_WEB,
    LEAD_NEWER_FIGURE,
    Verdict,
    supplement_ok,
)

MANILA = ZoneInfo("Asia/Manila")
NOW = datetime(2026, 11, 27, 12, 0, tzinfo=MANILA)
REPO = Path(__file__).resolve().parents[2]
DEV_PATH = REPO / "eval" / "sufficiency" / "supplement_replies_dev.jsonl"
TEST_SEEN_PATH = REPO / "eval" / "sufficiency" / "supplement_replies_test_seen.jsonl"


def _load(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _row_verdict(row: dict[str, Any]) -> Verdict:
    """Derive stale vs missing_half for fixture rows lacking an explicit verdict field:
    the question is time-sensitive when the question or the web question (the gate's own
    fallback) carries a time cue, or a document answer stands behind the supplement."""
    kind = (
        "stale"
        if sufficiency.time_sensitive(row["question"])
        or sufficiency.time_sensitive(row.get("web_question", ""))
        or bool(row.get("doc_answer"))
        else "missing_half"
    )
    return Verdict(verdict=kind, web_question=row["web_question"], reason="")


def _gate(row: dict[str, Any]) -> tuple[bool, str, str]:
    return supplement_ok(
        _row_verdict(row),
        row["reply"],
        NOW,
        question=row["question"],
        doc_answer=row["doc_answer"],
    )


# --- Fixture tests ---


def test_dev_fixture_drops_all_bad_and_keeps_every_ok() -> None:
    rows = _load(DEV_PATH)
    labels = {label: [r for r in rows if r["label"] == label] for label in ("bad", "ok", "hedge")}
    assert len(rows) == 34
    assert len(labels["bad"]) == 6
    assert len(labels["ok"]) == 24
    assert len(labels["hedge"]) == 4

    # all 6 bad dropped, with the 270 reasons intact
    reasons = {row["id"]: _gate(row)[1] for row in labels["bad"]}
    for row in labels["bad"]:
        assert not _gate(row)[0], f"expected bad row {row['id']} to be dropped"
    for row_id in ("imp-011", "imp-015", "imp-013"):
        assert reasons[row_id].startswith("figure:"), row_id
    for row_id in ("imp-001", "imp-061"):
        assert reasons[row_id].startswith("period:"), row_id
    assert reasons["imp-004"].startswith("scope:")

    # at least 22 of 24 ok kept (the two stale no-figure named answers are kept now)
    kept = [row["id"] for row in labels["ok"] if _gate(row)[0]]
    assert len(kept) >= 22
    for must_keep in ("imp-017", "imp-024", "imp-067", "imp-068", "imp-092"):
        assert must_keep in kept
    # the 270 figure gate's two drops are now kept with the neutral lead
    for named_id in ("imp-034", "imp-097"):
        assert named_id in kept
        assert _gate(next(r for r in labels["ok"] if r["id"] == named_id))[2] == LEAD_FROM_WEB

    # hedges stay silent (the flow never lets them reach the gate)
    for row in labels["hedge"]:
        assert sufficiency.HEDGE_TAIL in row["reply"]


def test_test_seen_fixture_drops_all_bad_keeps_imp096_and_ok_rows() -> None:
    rows = _load(TEST_SEEN_PATH)
    labels = {
        label: [r for r in rows if r["label"] == label]
        for label in ("bad", "ok", "hedge", "unsure")
    }
    # the file's own counts drive the assertions (the replies are now development data)
    assert len(rows) == 35
    assert len(labels["unsure"]) >= 1

    # all bad rows dropped; the period-mismatch rows by the new gate
    bad_reasons = {row["id"]: _gate(row)[1] for row in labels["bad"]}
    for row in labels["bad"]:
        ok, why = _gate(row)[:2]
        assert not ok, f"expected bad row {row['id']} to be dropped ({why})"
    for row_id in ("imp-012", "imp-014", "imp-016"):
        assert bad_reasons[row_id] == "period:wrong", row_id

    # imp-096: a good yes/no reply on a stale verdict, kept with the neutral lead
    row_096 = next(r for r in rows if r["id"] == "imp-096")
    ok, why, lead = _gate(row_096)
    assert ok and why == "" and lead == LEAD_FROM_WEB

    # at least 19 of the ok rows kept (computed from the file)
    ok_rows = labels["ok"]
    kept = [row["id"] for row in ok_rows if _gate(row)[0]]
    assert len(kept) >= 19
    assert len(kept) <= len(ok_rows)
    # no-figure keeps use the neutral lead, figure keeps the newer-figure lead
    for row in ok_rows:
        if _gate(row)[0]:
            assert _gate(row)[2] in (LEAD_FROM_WEB, LEAD_NEWER_FIGURE)

    # unsure rows are ignored: no assertion on their fate
    # hedges stay silent (the flow never lets them reach the gate)
    for row in labels["hedge"]:
        assert sufficiency.HEDGE_TAIL in row["reply"]


# --- Gate edge cases: yes/no and named answers (Part 1) ---


def test_yes_no_reply_kept_with_neutral_lead() -> None:
    v = Verdict("stale", "Is the ATSDR primer currently maintained?", "")
    ok, why, lead = supplement_ok(
        v,
        "As per Iiab and CDC, No, the primer is no longer maintained and is provided only "
        "as an historical reference.",
        NOW,
        question="Is the primer still maintained?",
        doc_answer="As per the module: No, it is no longer maintained.",
    )
    assert ok and why == "" and lead == LEAD_FROM_WEB

    # a yes/no reply to a non-who question needs no name: the first sentence starts with Yes
    ok, why, lead = supplement_ok(
        Verdict("stale", "Is the schedule final?", ""),
        "Yes, the schedule is final.",
        NOW,
        question="Is the schedule final yet?",
    )
    assert ok and why == "" and lead == LEAD_FROM_WEB


def test_no_figure_non_answer_still_dropped() -> None:
    # "No data available" starts with No but is not an answer
    ok, why, lead = supplement_ok(
        Verdict("missing_half", "Who publishes it?", ""), "No data available.", NOW
    )
    assert not ok and why == "figure:unavailable" and lead == ""

    # "varies / check your ..." with no figure
    ok, why, _ = supplement_ok(
        Verdict("stale", "What's the residential electricity rate in Maine right now?", ""),
        "Residential electricity rates vary by provider. You should check your utility's "
        "website for the current rate applicable to your account.",
        NOW,
        question="What's the residential electricity rate in Maine right now?",
    )
    assert not ok and why == "figure:missing"

    # a who question whose reply never names the person is not a named answer
    ok, why, _ = supplement_ok(
        Verdict("stale", "Who currently teaches the community engagement module?", ""),
        "The program is a self-paced certificate series. It does not have a single "
        "instructor who teaches the community engagement component.",
        NOW,
        question="Who teaches the community engagement module?",
    )
    assert not ok and why == "figure:missing"


def test_named_answer_kept_with_neutral_lead() -> None:
    v = Verdict("stale", "Who is the current chair?", "")
    ok, why, lead = supplement_ok(
        v,
        "As per IAB (the only source I found), the current chair is Roman Danyliw.",
        NOW,
        question="Who's the current chair of the IETF?",
    )
    assert ok and why == "" and lead == LEAD_FROM_WEB

    # "<Name> is the <role>" order, with a title
    ok, why, lead = supplement_ok(
        v,
        "As per the university, Dr. Kevin Carmody is the current Title IX coordinator.",
        NOW,
        question="Who's the title IX coordinator at GVSU right now?",
    )
    assert ok and why == "" and lead == LEAD_FROM_WEB

    # a named answer to a non-who question does not apply, but a figure does
    ok, why, lead = supplement_ok(
        Verdict("stale", "Which state has the largest population today?", ""),
        "California is the most populous state.",
        NOW,
        question="Which state has the largest population right now?",
    )
    assert ok and why == "" and lead == LEAD_FROM_WEB


# --- Gate edge cases: period mismatch (Part 2) ---


def test_future_period_question_current_period_reply_dropped() -> None:
    v = Verdict("stale", "What will the per diem be next month?", "")
    # the reply's period (November 2026) ended before the period asked (next month)
    ok, why, lead = supplement_ok(
        v,
        "For November 2026, the per diem rate is $165.",
        NOW,
        question="What will the per diem be next month?",
    )
    assert not ok and why == "period:wrong" and lead == ""

    # its only date is a past event ("began on"), the current year's start
    ok, why, _ = supplement_ok(
        Verdict("stale", "When do classes start now?", ""),
        "Classes for the current school year began on June 8, 2026. Check with your school "
        "for its unique academic calendar.",
        NOW,
        question="When do classes start next fall?",
    )
    assert not ok and why == "period:wrong"

    # the academic year that already began, asked about next year
    ok, why, _ = supplement_ok(
        Verdict("stale", "When does the fall term begin now?", ""),
        "Most universities began their first semester for the 2026-2027 academic year "
        "between June and August 2026.",
        NOW,
        question="When does the fall term begin next academic year?",
    )
    assert not ok and why == "period:wrong"


def test_future_period_reply_kept() -> None:
    v = Verdict("stale", "What will the per diem be next month?", "")
    # the reply's period IS the period asked
    ok, why, lead = supplement_ok(
        v,
        "For December 2026, the per diem rate is $170.",
        NOW,
        question="What will the per diem be next month?",
    )
    assert ok and why == "" and lead == LEAD_NEWER_FIGURE

    # a future year's start date answers next fall
    ok, why, lead = supplement_ok(
        Verdict("stale", "When do fall classes start now?", ""),
        "Fall classes for the 2027-2028 academic year will begin in August 2027.",
        NOW,
        question="When do fall classes start next year?",
    )
    assert ok and why == "" and lead == LEAD_NEWER_FIGURE

    # a reply that also states a future date is not "only past events"
    ok, why, _ = supplement_ok(
        Verdict("stale", "When do classes start now?", ""),
        "Classes began on June 8, 2026, and will begin again on August 24, 2027.",
        NOW,
        question="When do classes start next fall?",
    )
    assert ok


# --- Gate edge cases: the opposite-yes/no contradiction ---


def test_opposite_yesno_to_doc_answer_dropped() -> None:
    v = Verdict("stale", "Is there a confirmed fall break in October 2026?", "")
    ok, why, lead = supplement_ok(
        v,
        "No, there is no confirmed fall break in October 2026. The calendar for the "
        "2026-2027 school year does not include a standalone break for that month.",
        NOW,
        question="Is there a fall break in October this semester?",
        doc_answer="As per the academic calendar: Yes, there is a fall break on October 12, 2026.",
    )
    assert not ok and why == "contradiction:yesno" and lead == ""


def test_same_polarity_yesno_to_doc_answer_kept() -> None:
    v = Verdict("stale", "Is the primer currently maintained?", "")
    ok, why, lead = supplement_ok(
        v,
        "No, the primer is no longer maintained.",
        NOW,
        question="Is the primer still maintained?",
        doc_answer="As per the module: No, it is no longer maintained by ATSDR.",
    )
    assert ok and why == "" and lead == LEAD_FROM_WEB
