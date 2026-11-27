"""Ticket 263: the date-based staleness decision (source period vs target period).

The misses the check left: answers about an earlier period (a fiscal-year-labelled
schedule or rate, a "newest figure", a dated survey) that the LLM reads as covering the
present, because the answer text alone does not say so. The decision is made in code
from dates (``temporal.py``): the period the source covers vs the period the question
targets; a confident decision always pays for the check, rides to the verdict prompt as
the structured 'Period facts:' line, and — when the source ended and the target is the
present or the next period — is a `stale` verdict without an LLM call (``FAST_STALE``).
All transports are fakes (no network).
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import pytest

from voice_pipeline import sufficiency, temporal
from voice_pipeline.temporal import (
    period_facts,
    present_web_question,
    question_target_period,
    source_period,
)

NOW = datetime(2026, 11, 27)  # after the fiscal year's September-30 end (FY26)
BEFORE_FY_END = datetime(2026, 6, 15)  # before September 30: FY26 may still be running


def _facts(
    answer: str,
    question: str,
    now: datetime = NOW,
    *,
    file: str = "",
    date_min: str | None = None,
    date_max: str | None = None,
    title: str = "",
) -> Any:
    source = source_period(answer, date_min, date_max, file, title)
    target = question_target_period(question, now)
    return period_facts(source, target, now)


# --- the source period: the label shapes, in priority order ---


def test_a_calendar_year_in_the_answer_is_a_medium_year_period() -> None:
    f = _facts("The 2025 sheet lists the rate as 9.75.", "What's the rate right now?")
    assert f.source is not None
    assert (f.source.kind, f.source.confidence, f.source.basis) == ("year", "medium", "answer")
    assert (f.source.start, f.source.end) == (date(2025, 1, 1), date(2025, 12, 31))
    # an ended year for a present question: stale, but a bare year is only medium, so no
    # fast path — the facts ride to the LLM, which confirms
    assert f.decision.stale is True
    assert f.decision.confident is False and f.decision.fast_path is False


def test_a_fiscal_year_label_with_two_digits_is_high_confidence() -> None:
    f = _facts("The FY26 schedule lists the rate.", "What's the rate right now?")
    assert f.source is not None
    assert (f.source.kind, f.source.label, f.source.confidence) == ("fiscal", "FY2026", "high")
    # a fiscal year labelled N runs October 1 of the previous year through September 30
    # of N (it has ended by September 30 of its label under every common convention)
    assert (f.source.start, f.source.end) == (date(2025, 10, 1), date(2026, 9, 30))


def test_a_fiscal_year_label_with_four_digits_is_the_same_period() -> None:
    f = _facts("The FY2026 schedule lists the rate.", "What's the rate right now?")
    assert f.source is not None and f.source.label == "FY2026"
    assert (f.source.start, f.source.end) == (date(2025, 10, 1), date(2026, 9, 30))


def test_the_word_fiscal_labels_the_source_period_too() -> None:
    f = _facts(
        "The 2025 budget bill appropriates court funding through fiscal 2026.",
        "What's the budget this year?",
    )
    assert f.source is not None
    assert (f.source.kind, f.source.label, f.source.confidence) == ("fiscal", "FY2026", "high")
    assert (f.source.start, f.source.end) == (date(2025, 10, 1), date(2026, 9, 30))


def test_an_academic_year_runs_august_through_july() -> None:
    f = _facts("The 2025-26 calendar lists the term dates.", "What's the rate right now?")
    assert f.source is not None
    assert (f.source.kind, f.source.confidence) == ("academic", "high")
    assert (f.source.start, f.source.end) == (date(2025, 8, 1), date(2026, 7, 31))
    assert f.decision.stale is True and f.decision.confident is True
    assert f.decision.fast_path is True  # ended, present target, high confidence


def test_a_quarter_is_a_high_confidence_period() -> None:
    f = _facts(
        "The Q3 2026 survey has the top response.", "What's the most common answer right now?"
    )
    assert f.source is not None
    assert (f.source.kind, f.source.confidence) == ("quarter", "high")
    assert (f.source.start, f.source.end) == (date(2026, 7, 1), date(2026, 9, 30))
    assert f.decision.stale is True and f.decision.fast_path is True


def test_a_month_and_a_month_range_are_high_confidence_periods() -> None:
    f = _facts("The newest month in the table is September 2026.", "What's the rate right now?")
    assert f.source is not None and (f.source.kind, f.source.confidence) == ("month", "high")
    assert (f.source.start, f.source.end) == (date(2026, 9, 1), date(2026, 9, 30))
    f = _facts("The reading covers July-September 2026.", "What's the rate right now?")
    assert f.source is not None
    assert (f.source.start, f.source.end) == (date(2026, 7, 1), date(2026, 9, 30))


def test_a_full_date_in_the_answer_text_is_a_high_confidence_point() -> None:
    f = _facts("The rate changed on 2026-03-15.", "What's the rate right now?")
    assert f.source is not None
    assert (f.source.kind, f.source.confidence) == ("date", "high")
    assert (f.source.start, f.source.end) == (date(2026, 3, 15), date(2026, 3, 15))
    assert f.decision.stale is True and f.decision.fast_path is True


def test_no_date_at_all_is_no_source_period() -> None:
    f = _facts("The rate is 9.75.", "What's the rate right now?", file="notes.txt")
    assert f.source is None
    assert f.decision.stale is False and f.decision.fast_path is False


def test_a_year_reading_through_a_month_stops_at_that_month() -> None:
    # a year-to-date total: "2026 ... through August" is not the whole year 2026
    f = _facts(
        "In the table, 2026 generation through August totals 3,073,105.",
        "How much has been generated this year so far?",
    )
    assert f.source is not None
    assert (f.source.kind, f.source.confidence) == ("year", "medium")
    assert (f.source.start, f.source.end) == (date(2026, 1, 1), date(2026, 8, 31))
    # the year's reading ended before today while the question asks for the present
    assert f.decision.stale is True
    assert f.decision.confident is False and f.decision.fast_path is False


def test_the_newest_year_stated_in_the_answer_is_the_source_period() -> None:
    f = _facts(
        "The guide gives $176,100 for 2025 and $184,500 for 2026.",
        "What is the cap next year?",
    )
    assert f.source is not None and f.source.label == "2026"
    assert (f.source.start, f.source.end) == (date(2026, 1, 1), date(2026, 12, 31))
    assert f.decision.stale is True  # 2026 does not reach next year (2027)


def test_a_web_hedge_sentence_does_not_state_the_source_period() -> None:
    # the hedge points at the period to check; the figure's own period is 2026
    f = _facts(
        "The 2026 table shows the rate as 12.5. The rate that starts in January 2027 "
        "isn't in that table, so verify live.",
        "What will the rate be next January?",
    )
    assert f.source is not None and f.source.label == "2026"
    assert (f.source.start, f.source.end) == (date(2026, 1, 1), date(2026, 12, 31))
    assert f.decision.stale is True and f.decision.fast_path is False


def test_a_web_hedge_clause_in_the_figure_sentence_keeps_the_figure_period() -> None:
    # the hedge shares the sentence with the figure: cut at the clause that leads in
    f = _facts(
        "The 2026-2027 calendar shows the fall 2026 semester starting September 2, "
        "but next fall's dates aren't in it, so verify live.",
        "When's fall break next fall?",
    )
    assert f.source is not None
    assert (f.source.kind, f.source.label) == ("academic", "2026-2027")
    assert f.decision.stale is True and f.decision.fast_path is True


def test_the_card_date_range_is_the_fallback_when_the_answer_states_no_period() -> None:
    f = _facts(
        "The rate is 9.75.",
        "What's the rate right now?",
        date_min="2024-01-01",
        date_max="2024-06-30",
    )
    assert f.source is not None
    assert (f.source.kind, f.source.confidence, f.source.basis) == ("range", "medium", "card")
    assert (f.source.start, f.source.end) == (date(2024, 1, 1), date(2024, 6, 30))
    assert f.decision.stale is True
    assert f.decision.confident is False and f.decision.fast_path is False  # medium only


def test_a_period_label_in_the_file_name_is_the_last_resort() -> None:
    f = _facts("The rate is 9.75.", "What's the rate right now?", file="fy26_rates.xlsx")
    assert f.source is not None
    assert (f.source.kind, f.source.label, f.source.basis) == ("fiscal", "FY2026", "name")
    assert f.decision.stale is True and f.decision.fast_path is True
    # the answer text beats the file name
    f = _facts("The 2025 sheet lists the rate.", "What's the rate right now?", file="fy26.xlsx")
    assert f.source is not None and f.source.basis == "answer" and f.source.label == "2025"


def test_room_numbers_and_long_digit_runs_are_not_years() -> None:
    assert source_period("The room is 3005.", None, None, "", "") is None
    assert source_period("The code 19934 applies.", None, None, "", "") is None


# --- the target period: present / next / past / unspecified ---


def test_present_targets() -> None:
    assert question_target_period("What's the rate right now?", NOW).kind == "present"
    assert question_target_period("What is the latest figure?", NOW).kind == "present"
    f = question_target_period("What's the rate this month?", NOW)
    assert f.kind == "present" and f.unit == "month"
    assert f.span == (date(2026, 11, 1), date(2026, 11, 30))


def test_recently_and_at_the_moment_are_present_cues() -> None:
    for q in (
        "Have a lot of households run out of food recently?",
        "Where does the index stand at the moment?",
        "What is the rate at present?",
    ):
        assert temporal.has_temporal_cue(q), q
        assert question_target_period(q, NOW).kind == "present"


def test_a_period_word_inside_another_word_is_not_a_cue() -> None:
    # the unit word needs its this/next/last prefix and a word boundary
    assert not temporal.has_temporal_cue("Where is the agency headquartered?")
    assert not question_target_period("Where is the agency headquartered?", NOW).match
    assert temporal.has_temporal_cue("When do classes start next fall?")
    assert not temporal.has_temporal_cue("When is the next meeting?")


def test_next_targets_resolve_to_the_next_occurrence() -> None:
    f = question_target_period("What will the rate be next year?", NOW)
    assert f.kind == "next" and f.unit == "year"
    assert f.span == (date(2027, 1, 1), date(2027, 12, 31))
    f = question_target_period("What will the rate be next January?", NOW)
    assert f.kind == "next" and f.unit == "january"
    assert f.span == (date(2027, 1, 1), date(2027, 1, 31))
    f = question_target_period("What's the rate next quarter?", NOW)
    assert f.kind == "next" and f.unit == "quarter"
    assert f.span == (date(2027, 1, 1), date(2027, 3, 31))
    f = question_target_period("What's the rate next month?", NOW)
    assert f.span == (date(2026, 12, 1), date(2026, 12, 31))


def test_past_targets_name_a_fixed_period() -> None:
    f = question_target_period("What was the rate in 2025?", NOW)
    assert f.kind == "past" and f.unit == "year"
    assert f.span == (date(2025, 1, 1), date(2025, 12, 31))
    f = question_target_period("What was the rate last year?", NOW)
    assert f.kind == "past" and f.span == (date(2025, 1, 1), date(2025, 12, 31))
    f = question_target_period("What was the budget last fiscal year?", NOW)
    assert f.kind == "past" and f.unit == "fiscal"
    # in 2026-11 the last completed fiscal year is FY2026 (October 2025 to September
    # 2026)
    assert f.span == (date(2025, 10, 1), date(2026, 9, 30))


def test_this_fiscal_year_is_the_fiscal_period_in_progress() -> None:
    # in 2026-11 the fiscal year in progress is labelled 2027 (October 2026 through
    # September 2027); an FY2026 source ended before it
    f = question_target_period("How much does the budget give this fiscal year?", NOW)
    assert f.kind == "present" and f.unit == "fiscal"
    assert f.span == (date(2026, 10, 1), date(2027, 9, 30))
    f = _facts(
        "The 2025 budget bill appropriates court funding through fiscal 2026.",
        "How much does the budget give the courts this fiscal year?",
    )
    assert f.decision.stale is True
    assert f.decision.confident is True and f.decision.fast_path is True


def test_a_season_word_refines_a_next_year_target() -> None:
    # "fall semester next year" asks for the fall term of 2027, not the whole year
    f = question_target_period("When's the first day of fall semester next year?", NOW)
    assert f.kind == "next" and f.unit == "season"
    assert f.span == (date(2027, 9, 1), date(2027, 12, 31))
    # the 2026-2027 academic list (August 2026 through July 2027) does not reach it
    f = _facts(
        "The 2026-2027 list has the fall semester beginning August 31, 2026.",
        "When's the first day of fall semester next year?",
    )
    assert f.decision.stale is True and f.decision.fast_path is True
    assert (
        present_web_question("When's the first day of fall semester next year?", f.target)
        == "When's the first day of fall semester now?"
    )


def test_the_current_year_is_the_present() -> None:
    # the in-progress current year is not a fixed past period: a question about it
    # targets the present, and an earlier-year source does not cover it
    f = question_target_period("What is the rate in 2026?", NOW)
    assert f.kind == "present" and f.unit == "year"
    assert f.span == (date(2026, 1, 1), date(2026, 12, 31))
    f = _facts("The 2025 sheet lists the rate.", "What is the rate in 2026?")
    assert f.decision.stale is True


def test_a_question_with_no_period_cue_is_unspecified() -> None:
    assert question_target_period("What is the rate?", NOW).kind == "unspecified"
    assert (
        question_target_period("How many guiding questions are there?", NOW).kind == "unspecified"
    )


# --- the staleness decision: the code-side rules ---


def test_a_fiscal_year_ended_after_september_30_is_stale_for_the_present() -> None:
    f = _facts("The FY2026 schedule lists the rate.", "What's the rate right now?", now=NOW)
    assert f.decision.stale is True
    assert f.decision.confident is True and f.decision.fast_path is True


def test_a_fiscal_year_before_september_30_is_possibly_still_running() -> None:
    f = _facts(
        "The FY2026 schedule lists the rate.", "What's the rate right now?", now=BEFORE_FY_END
    )
    assert f.decision.stale is False
    assert f.decision.confident is True  # a confident not-stale still pays for the check
    assert "may still be running" in f.decision.hint


def test_a_source_ending_inside_the_asked_span_is_stale_but_not_fast() -> None:
    # a mid-month reading for a month question: stale (it ended before today) but the
    # fast path needs a source that ends before the asked span
    f = _facts(
        "The rate was set on 2026-11-15.",
        "What's the rate this month?",
    )
    assert f.decision.stale is True
    assert f.decision.confident is True and f.decision.fast_path is False


def test_a_source_covering_today_is_not_stale() -> None:
    f = _facts("The November 2026 table lists the rate.", "What's the rate right now?")
    assert f.decision.stale is False
    assert f.decision.confident is True and f.decision.covers is True
    f = _facts("The 2026 table lists the rate.", "What's the rate right now?")
    assert f.decision.stale is False and f.decision.covers is True


def test_next_year_against_the_newest_current_year_figure_is_stale() -> None:
    f = _facts("The 2026 table lists the cap.", "What will the cap be next year?")
    assert f.target.kind == "next"
    assert f.decision.stale is True
    assert f.decision.confident is False  # a bare year is medium: the LLM confirms
    assert f.decision.fast_path is False


def test_a_fixed_past_period_the_source_covers_is_never_stale() -> None:
    f = _facts("The FY2025 schedule covers the year 2025.", "What was the rate in 2025?")
    assert f.target.kind == "past"
    assert f.decision.stale is False
    assert f.decision.confident is True  # a confident not-stale still pays for the check


def test_a_fixed_past_period_the_source_does_not_reach_is_stale() -> None:
    f = _facts("The FY2024 schedule lists the rate.", "What was the rate in 2025?")
    assert f.decision.stale is True
    assert f.decision.confident is True  # high-confidence source: the gate always runs


def test_a_future_source_is_never_confidently_stale() -> None:
    f = _facts("The 2028 plan lists the rate.", "What's the rate right now?")
    assert f.source is not None and f.source.start > NOW.date()
    assert f.decision.stale is False and f.decision.confident is False


def test_an_unspecified_target_is_never_stale_even_with_an_old_source() -> None:
    f = _facts(
        "The rate is 9.75.",
        "How many guiding questions are there?",
        date_min="2024-01-01",
        date_max="2024-06-30",
    )
    assert f.target.kind == "unspecified"
    assert f.decision.stale is False and f.decision.confident is False


# --- the fast path: a stale verdict without an LLM call ---


def _fiscal_facts_next_january() -> Any:
    return _facts("The FY2026 schedule lists the rate.", "What will the rate be next January?")


async def test_the_fast_path_returns_stale_without_an_llm_call() -> None:
    calls: list[tuple[str, str]] = []

    async def fake(system: str, content: str, schema: dict[str, Any] | None = None) -> str:
        calls.append((system, content))
        return json.dumps({"verdict": "sufficient", "web_question": "", "reason": "x"})

    got = await sufficiency.verdict(
        fake,
        "What will the rate be next January?",
        "The FY2026 schedule lists the rate.",
        "FY2026",
        "2026-11-27",
        facts=_fiscal_facts_next_january(),
    )
    assert calls == []  # no LLM call: the date facts alone decide
    assert got is not None and got.verdict == "stale" and got.fast is True
    # the web question is the question restated for the present, same words
    assert got.web_question == "What will the rate be now?"
    assert got.reason


async def test_the_fast_path_flag_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sufficiency, "FAST_STALE", False)
    calls: list[str] = []

    async def fake(system: str, content: str, schema: dict[str, Any] | None = None) -> str:
        calls.append(content)
        return json.dumps({"verdict": "sufficient", "web_question": "", "reason": "x"})

    got = await sufficiency.verdict(
        fake,
        "What will the rate be next January?",
        "The FY2026 schedule lists the rate.",
        "FY2026",
        "2026-11-27",
        facts=_fiscal_facts_next_january(),
    )
    assert len(calls) == 1  # the flag off: the LLM confirms instead
    assert got is not None and got.verdict == "sufficient" and got.fast is False


async def test_the_medium_confidence_stale_rides_to_the_llm_as_period_facts() -> None:
    seen: list[str] = []

    async def fake(system: str, content: str, schema: dict[str, Any] | None = None) -> str:
        seen.append(content)
        return json.dumps({"verdict": "stale", "web_question": "q?", "reason": "old"})

    got = await sufficiency.verdict(
        fake,
        "What's the rate right now?",
        "The 2025 sheet lists the rate as 9.75.",
        "2025",
        "2026-11-27",
        facts=_facts("The 2025 sheet lists the rate as 9.75.", "What's the rate right now?"),
    )
    assert len(seen) == 1  # medium confidence: no fast path, the LLM confirms
    assert got is not None and got.verdict == "stale" and got.fast is False
    user = seen[0]
    assert "Period facts:" in user
    assert "the source covers 2025-01-01 to 2025-12-31" in user
    assert "today is 2026-11-27" in user
    assert "the question targets the present" in user
    assert "it ended before today" in user


def test_the_gate_always_runs_on_a_confident_decision() -> None:
    # the legacy gate exempts a pinned-past question from the source-year leg; a
    # confident date decision (the source cannot reach the named period) overrides that
    f = _facts("The FY2024 schedule lists the rate.", "What was the rate in 2025?")
    assert f.decision.confident is True
    assert (
        sufficiency.should_check(
            "What was the rate in 2025?", "2024", "The rate listed for the year is 9.75.", now=NOW
        )
        is False
    )
    assert (
        sufficiency.should_check(
            "What was the rate in 2025?",
            "2024",
            "The rate listed for the year is 9.75.",
            now=NOW,
            facts=f,
        )
        is True
    )
    # a confident not-stale (the source covers the named period) also pays for the
    # check, so the LLM confirms on the structured facts
    f = _facts("The table covers January-December 2026.", "What was the rate in 2026?")
    assert f.target.kind == "present"  # the in-progress current year is the present
    assert f.decision.confident is True and f.decision.stale is False
    assert (
        sufficiency.should_check(
            "What was the rate in 2026?", "2026", "The rate is 9.75 for the year.", now=NOW
        )
        is False
    )
    assert (
        sufficiency.should_check(
            "What was the rate in 2026?", "2026", "The rate is 9.75 for the year.", now=NOW, facts=f
        )
        is True
    )


def test_a_control_with_a_timeless_source_and_unspecified_target_is_not_checked() -> None:
    f = _facts("The room is on the third floor.", "Where is the meeting room?")
    assert f.source is None and f.target.kind == "unspecified"
    assert f.decision.stale is False and f.decision.fast_path is False
    assert (
        sufficiency.should_check(
            "Where is the meeting room?",
            None,
            "The room is on the third floor of the building.",
            now=NOW,
            facts=f,
        )
        is False
    )


def test_present_web_question_restates_only_next_period_questions() -> None:
    target = question_target_period("What will the rate be next January?", NOW)
    assert present_web_question("What will the rate be next January?", target) == (
        "What will the rate be now?"
    )
    target = question_target_period("What's the rate right now?", NOW)
    q = "What's the rate right now?"
    assert present_web_question(q, target) == q


def test_the_fiscal_running_hint_reaches_the_evidence_line() -> None:
    f = _facts(
        "The FY2026 schedule lists the rate.", "What's the rate right now?", now=BEFORE_FY_END
    )
    line = f.evidence_line(BEFORE_FY_END.date())
    assert line.startswith("Period facts:")
    assert "the source covers 2025-10-01 to 2026-09-30" in line
    assert "may still be running" in line
    assert "the question targets the present" in line


def test_the_evidence_line_says_the_source_period_state() -> None:
    f = _facts("The FY2026 schedule lists the rate.", "What's the rate right now?")
    assert "the source period has ended" in f.evidence_line(NOW.date())
    f = _facts("The 2025 table lists the rate.", "What was the rate in 2025?")
    assert "the source period ended before today" in f.evidence_line(NOW.date())
    f = _facts("The November 2026 table lists the rate.", "What's the rate right now?")
    assert "the source period has not ended" in f.evidence_line(NOW.date())
    f = _facts("The rate is 9.75.", "What is the rate?")
    assert "no source period could be read" in f.evidence_line(NOW.date())


def test_an_unspecified_target_with_an_ended_source_is_neutral_not_stale() -> None:
    # dev round 1: a timeless question (no period cue) with an old source must not be
    # read as stale just because the source ended - the over-trigger guard. The decision
    # stays not-stale and the evidence line says the source period is not a staleness
    # signal, so the LLM does not flip a timeless control to stale.
    f = _facts(
        "No, the 2025 policy allows it only for the listed courses.",
        "Can a graduate course use a D grade?",
        date_min="2025-01-01",
        date_max="2025-12-31",
    )
    assert f.target.kind == "unspecified"
    assert (
        f.decision.stale is False
        and f.decision.confident is False
        and f.decision.fast_path is False
    )
    line = f.evidence_line(NOW.date())
    assert "does not by itself mean the answer is stale" in line
    assert "the source period ended before today" not in line


def test_an_unspecified_target_with_no_source_keeps_the_no_period_state() -> None:
    f = _facts("The rate is 9.75.", "What is the rate?")
    assert f.target.kind == "unspecified" and f.source is None
    assert "no source period could be read" in f.evidence_line(NOW.date())


def test_a_card_range_reaching_the_current_year_is_not_stale_for_the_present() -> None:
    # dev round 2: a file whose dated content reaches the current year (a card date range)
    # is not an earlier period - a present question about this year is not answered by an
    # earlier period on that ground, so the decision stays not-stale and the evidence line
    # does not read the ended range as one (this is the docs_only mileage-rate control)
    f = _facts(
        "72.5 cents a mile",
        "What's the standard mileage rate this year?",
        date_min="1992-12-31",
        date_max="2026-06-30",
    )
    assert f.target.kind == "present"
    assert f.source is not None and f.source.basis == "card"
    assert (
        f.decision.stale is False
        and f.decision.confident is False
        and f.decision.fast_path is False
    )
    line = f.evidence_line(NOW.date())
    assert "reaches the current year" in line
    assert "ended before today" not in line


def test_a_card_range_from_an_earlier_year_is_stale_for_the_present() -> None:
    # the same rule keeps the genuine case: a file whose content ends in a past year is an
    # earlier period for a present question (a medium-confidence source, so no fast path)
    f = _facts(
        "72.5 cents a mile",
        "What's the standard mileage rate this year?",
        date_min="1992-12-31",
        date_max="2024-12-31",
    )
    assert f.target.kind == "present"
    assert f.decision.stale is True
    assert f.decision.confident is False and f.decision.fast_path is False


def test_a_year_to_date_answer_text_is_still_stale_even_in_the_current_year() -> None:
    # an answer-text period (not a card range) that ends inside the current year is the
    # figure's own period: "this year so far" read through August is not the figure
    # through the present month, so it stays stale (unlike the card range above)
    f = _facts(
        "In the table, 2026 generation through August totals 3,073,105.",
        "How much has been generated this year so far?",
    )
    assert f.source is not None and f.source.basis == "answer"
    assert f.decision.stale is True


def test_the_shared_cue_list_is_unchanged_and_the_targets_share_it() -> None:
    # the gate's cue list (ticket 259, R2-3) is the same word groups the target-period
    # cues are built from: every cue the gate knows about resolves to a target kind
    for q in (
        "What is the rate right now?",
        "What's the latest figure?",
        "What's the rate this month?",
        "What will the rate be next year?",
        "What was the rate last month?",
    ):
        assert temporal.has_temporal_cue(q)
        assert question_target_period(q, NOW).kind != "unspecified"
