# Ticket 268: one timezone-aware clock per turn (Philippine Standard Time) + Pi clock sync

Branch: exp/sufficiency (worktree ~/AI-231-ME4-w250). Flag `SUFFICIENCY` stays False.

## Problem
"Today" is read in several places with a naive `datetime.now()` (the Pi's system zone and clock):
`agent_server.py:878`, `sufficiency.py:125,152,380,416`, `temporal.py:272,521,629,683`,
`ask_documents.py:777`. The demo zone is Philippine Standard Time (`fast_path.TIMEZONE = "Asia/Manila"`,
UTC+8, no DST); other code already uses `datetime.now(ZoneInfo(fast_path.TIMEZONE))`. A Pi with no real-time
clock can also boot with a wrong date, and every stale/current decision then goes wrong silently.
Reference: onit builds `current_date` once per request from the clock and puts it in a marked context block
(`src/mcp/prompts/prompts.py`); it has no timezone handling and no hook, so there is nothing more to copy.

## Files
- `pipeline/src/voice_pipeline/agent_server.py` (take `now` once per turn, pass it on)
- `pipeline/src/voice_pipeline/sufficiency.py`, `temporal.py`, `ask_documents.py` (`now` required/injected, no naive default)
- `pipeline/src/voice_pipeline/fast_path.py` (a small `turn_now()` helper next to `TIMEZONE`, if it fits)
- `scripts/sufficiency_eval.py` (`--today` becomes the injected `now`, at 12:00 Asia/Manila)
- `scripts/pi_clock_check.sh` (new, see part 2) and tests

## Part 1: one clock per turn
1. One tz-aware `now = datetime.now(ZoneInfo(fast_path.TIMEZONE))` per turn (optional client override: if the
   websocket message carries `timestamp` + `tz`, use it; absent is the normal case). Pass the same `now` to
   `should_check`, `pinned_past`, `temporal.*`, `period_facts`, the verdict's today line and `ask_documents:777`.
2. Remove the naive `datetime.now()` / `date.today()` defaults in those modules (a missing `now` is a TypeError
   in tests, never a silent system-zone read). `today_line(now)` formats `YYYY-MM-DD` from that `now`.
3. Boundary tests: 23:59 vs 00:01 Asia/Manila on a fiscal-year end (Sep 30 / Oct 1) and on Dec 31 / Jan 1, with the
   process TZ set to UTC and to America/Los_Angeles: the decision must follow Manila, not the process zone.
4. The 100-row eval numbers must not change (`--today 2026-11-27` still maps to that Manila date); rerun the
   dev gold eval on Qwen and the test suite.

## Part 2: Pi clock sanity at start-up (no secrets, no new dependency)
- Make-up step: before a run, check `timedatectl show -p NTPSynchronized` (or `chronyc tracking`); if not
  synchronised, set the clock once from an external HTTPS `Date` header (`curl -sI https://www.google.com`) via
  `sudo date -s`, then re-check. Write it as `scripts/pi_clock_check.sh` that only reports by default and acts
  with `--fix`. The owner runs it (needs sudo and wifi); do not run it from the agent.
- Agent start-up: log one line with the clock, zone and NTP state; if the year is below 2026 or the clock is
  more than 24 h off the last-seen time (file next to the DB), log a loud warning and disable the sufficiency
  date fast path for that run (the verdict then abstains rather than deciding on a wrong date).

## Acceptance Criteria
- No naive `datetime.now()`/`date.today()` remain in the files above except inside the one helper.
- Boundary tests (Part 1.3) pass under three process time zones.
- Dev gold eval on Qwen unchanged from `out/sufficiency_eval` (stale 15/15, conflict 7/7, controls 0/14).
- `uv run pytest` in pipeline: only the known `test_docs_wiring` failure; ruff clean.
- `pi_clock_check.sh` reports and does nothing else unless `--fix`.

## Non-Goals
- Do not enable `SUFFICIENCY`; do not touch the frozen release branch; no DST logic (Manila has none).
- Do not change `bench/telemetry` files in this ticket (merge conflicts in `agent_server.py` come later).
- No new paid calls.
