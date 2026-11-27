"""Ticket 268: one timezone-aware clock per turn (Asia/Manila), and the Pi clock sanity.

The hazard: the demo zone is Asia/Manila (UTC+8, no DST) but the process may run in any
system zone (a Pi can also boot with a wrong date). Before the fix, "today" was read
with a naive `datetime.now()`/`date.today()` in several modules, so a process in UTC or
America/Los_Angeles made every stale/current decision follow the process zone instead
of Manila. Manila runs AHEAD of both zones, so its day turns first: at 00:01 Manila on
a period end, the UTC and LA wall clocks are still on the previous day - the period that
just ended - and a naive system-zone read would keep deciding with the old date. After
the fix every decision takes the turn's clock as an explicit tz-aware `now` (the one
helper `fast_path.turn_now()` is the only clock read), so the decision follows Manila
under every process time zone.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from voice_pipeline import fast_path, sufficiency, temporal
from voice_pipeline.temporal import (
    SourcePeriod,
    period_facts,
    question_target_period,
    source_period,
    stale_decision,
)

MANILA = ZoneInfo(fast_path.TIMEZONE)
NOW = datetime(2026, 11, 27)  # a mid-year reference clock for the signature checks
# the process time zones the boundary must NOT follow (part 1.3): UTC and Los Angeles,
# plus Manila itself (a correctly zoned process, for the control)
PROCESS_TZS = ("UTC", "America/Los_Angeles", "Asia/Manila")

# the two boundaries, as the turn's clock reads them
FY_END_EVE = datetime(2026, 9, 30, 23, 59, tzinfo=MANILA)  # FY2026 still running
FY_END_PAST = datetime(2026, 10, 1, 0, 1, tzinfo=MANILA)  # FY2026 ended
YEAR_END_EVE = datetime(2026, 12, 31, 23, 59, tzinfo=MANILA)  # 2026 still running
YEAR_END_PAST = datetime(2027, 1, 1, 0, 1, tzinfo=MANILA)  # 2026 ended


class _process_tz:
    """Sets the process time zone for a `with` block (and restores it on exit). The
    decisions under test must come out identical for every process zone: they take the
    turn's clock as an argument and never read the system zone."""

    def __init__(self, tz: str) -> None:
        self.tz = tz
        self.old = os.environ.get("TZ")

    def __enter__(self) -> str:
        os.environ["TZ"] = self.tz
        time.tzset()
        return self.tz

    def __exit__(self, *exc: object) -> bool:
        if self.old is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = self.old
        time.tzset()
        return False


def _present(question: str, now: datetime) -> temporal.TargetPeriod:
    t = question_target_period(question, now)
    assert t.kind == "present"
    return t


def _fy2026() -> SourcePeriod:
    s = source_period("The FY2026 schedule lists the rate.", None, None, "", "")
    assert s is not None and (s.kind, s.confidence) == ("fiscal", "high")
    assert (s.start, s.end) == (date(2025, 10, 1), date(2026, 9, 30))
    return s


def _dec2026() -> SourcePeriod:
    s = source_period("The report covers December 2026.", None, None, "", "")
    assert s is not None and (s.kind, s.confidence) == ("month", "high")
    assert (s.start, s.end) == (date(2026, 12, 1), date(2026, 12, 31))
    return s


# --- part 1.3: the boundaries must follow Manila, not the process zone ---


def test_the_zone_divergence_straddles_both_boundaries() -> None:
    """The hazard is real at these exact moments: Manila runs ahead of UTC and
    America/Los_Angeles, so at 00:01 Manila (the demo's day has already turned) the
    process-zone wall clocks are still on the previous day - the period that just ended.
    A naive system-zone read would decide there with the old date. Pinned so the
    boundary tests below sit in the danger zone."""
    for moment, old_day in (
        (FY_END_PAST, date(2026, 9, 30)),
        (YEAR_END_PAST, date(2026, 12, 31)),
    ):
        for tz in ("UTC", "America/Los_Angeles"):
            assert moment.astimezone(ZoneInfo(tz)).date() == old_day
            assert moment.date() != old_day  # Manila is already on the next day


@pytest.mark.parametrize("process_tz", PROCESS_TZS)
def test_the_fiscal_year_boundary_follows_manila_under_every_process_tz(process_tz: str) -> None:
    source = _fy2026()
    with _process_tz(process_tz):
        for now, stale, fast in (
            (FY_END_EVE, False, False),  # 23:59 Manila: FY2026 may still be running
            (FY_END_PAST, True, True),  # 00:01 Manila: FY2026 ended before today
        ):
            got = stale_decision(source, _present("What's the rate right now?", now), now)
            assert got.stale is stale, f"process TZ {process_tz} @ {now}"
            assert got.fast_path is fast, f"process TZ {process_tz} @ {now}"


@pytest.mark.parametrize("process_tz", PROCESS_TZS)
def test_the_calendar_year_boundary_follows_manila_under_every_process_tz(process_tz: str) -> None:
    source = _dec2026()
    with _process_tz(process_tz):
        for now, stale, fast in (
            (YEAR_END_EVE, False, False),  # 23:59 Manila: today is inside December 2026
            (YEAR_END_PAST, True, True),  # 00:01 Manila: December 2026 ended before today
        ):
            got = stale_decision(source, _present("What's the rate right now?", now), now)
            assert got.stale is stale, f"process TZ {process_tz} @ {now}"
            assert got.fast_path is fast, f"process TZ {process_tz} @ {now}"


@pytest.mark.parametrize("process_tz", PROCESS_TZS)
def test_the_evidence_line_and_the_target_period_follow_manila_too(process_tz: str) -> None:
    """The same clock rides to the prompt: the 'today is' fact and the target period of
    a next-period question both derive from `now`, under every process zone."""
    source = _fy2026()
    with _process_tz(process_tz):
        for now in (FY_END_EVE, FY_END_PAST, YEAR_END_EVE, YEAR_END_PAST):
            facts = period_facts(source, _present("What's the rate right now?", now), now)
            assert f"today is {now.date().isoformat()}" in facts.evidence_line(now.date())
            nxt = question_target_period("What will the rate be next year?", now)
            assert nxt.kind == "next"
            assert nxt.span is not None and nxt.span[0].year >= now.year


@pytest.mark.parametrize("process_tz", PROCESS_TZS)
def test_the_verdict_today_line_follows_manila_under_every_process_tz(process_tz: str) -> None:
    """The prompt's 'Today:' line comes from the turn's clock, not the system zone."""
    with _process_tz(process_tz):
        for now in (FY_END_EVE, FY_END_PAST, YEAR_END_EVE, YEAR_END_PAST):
            assert sufficiency.today_line(now) == now.date().isoformat()


# --- the one clock helper, and the TypeError contract ---


def test_turn_now_is_the_real_clock_in_the_demo_zone() -> None:
    now = fast_path.turn_now()
    assert now.tzinfo is not None and now.tzname() is not None
    assert now.utcoffset() == timedelta(hours=8)  # UTC+8, no DST
    assert now.astimezone(MANILA).utcoffset() == timedelta(hours=8)
    # it is the real clock: within the last minute of it
    assert abs((fast_path.turn_now() - now).total_seconds()) < 60


def test_the_verdict_needs_the_turn_clock() -> None:
    """A missing `now` is a TypeError in tests, never a silent system-zone read."""
    with pytest.raises(TypeError):
        sufficiency.verdict(None, "q?", "a", None)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        temporal.stale_decision(_fy2026(), _present("What's the rate right now?", NOW))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        temporal.question_target_period("What's the rate right now?")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        temporal.period_facts(_fy2026(), _present("What's the rate right now?", NOW))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        sufficiency.pinned_past("What was the 2025 rate?")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        sufficiency.source_earlier("2025")  # type: ignore[call-arg]


# --- part 2: the start-up clock sanity (agent_server.check_clock) ---


def _check_clock(tmp_path: Path, now: datetime) -> tuple[bool, str]:
    from voice_pipeline import agent_server

    db = tmp_path / "docs.db"
    db.write_text("", encoding="utf-8")
    return agent_server.check_clock(db, now=now)


def test_check_clock_flags_a_year_below_2026(tmp_path: Path) -> None:
    sane, detail = _check_clock(tmp_path, datetime(1970, 1, 1, 0, 1, tzinfo=MANILA))
    assert sane is False
    assert "1970" in detail and "2026" in detail


def test_check_clock_is_sane_on_the_first_start_and_writes_last_seen(tmp_path: Path) -> None:
    now = datetime(2026, 11, 28, 12, 0, tzinfo=MANILA)
    sane, detail = _check_clock(tmp_path, now)
    assert sane is True
    assert f"clock {now:%Y-%m-%d %H:%M:%S}" in detail and f"zone={fast_path.TIMEZONE}" in detail
    last_seen = tmp_path / "docs.db.last_seen"
    assert last_seen.read_text(encoding="utf-8").strip() == now.isoformat()


def test_check_clock_flags_more_than_24h_off_last_seen(tmp_path: Path) -> None:
    seen = datetime(2026, 11, 27, 12, 0, tzinfo=MANILA)
    tmp_path.joinpath("docs.db").write_text("", encoding="utf-8")
    tmp_path.joinpath("docs.db.last_seen").write_text(seen.isoformat() + "\n", encoding="utf-8")
    # 25 h after the last-seen time: the clock (or the RTC) jumped
    sane, detail = _check_clock(tmp_path, seen + timedelta(hours=25))
    assert sane is False
    assert "24 h" in detail


def test_check_clock_accepts_a_small_forward_jitter(tmp_path: Path) -> None:
    seen = datetime(2026, 11, 27, 12, 0, tzinfo=MANILA)
    tmp_path.joinpath("docs.db").write_text("", encoding="utf-8")
    tmp_path.joinpath("docs.db.last_seen").write_text(seen.isoformat() + "\n", encoding="utf-8")
    sane, _detail = _check_clock(tmp_path, seen + timedelta(hours=3))
    assert sane is True


def test_check_clock_with_no_db_only_checks_the_year(tmp_path: Path) -> None:
    from voice_pipeline import agent_server

    sane, detail = agent_server.check_clock(None, now=datetime(2026, 11, 28, 12, 0, tzinfo=MANILA))
    assert sane is True
    sane, _ = agent_server.check_clock(None, now=datetime(2000, 1, 1, 12, 0, tzinfo=MANILA))
    assert sane is False
