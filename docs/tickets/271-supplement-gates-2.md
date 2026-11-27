# Ticket 271: supplement gates, second pass + a free Qwen pre-assessment

Branch: exp/sufficiency (worktree ~/AI-231-ME4-w250). `SUFFICIENCY` stays False. Builds on 270 (`bd7bfe0`).

## Problem (from the held-out look at the test half, docs/POTENTIAL-IMPROVEMENTS-PARKED.md)
1. The figure gate (270) drops a good yes/no reply on a `stale` verdict (imp-096: "No, no longer maintained").
2. Nothing catches a reply about the wrong period: imp-012 (asked "when do classes start next fall", reply gives
   the current year's start date), imp-014/016 (2026-27 year that already began, asked about next year).
3. A free way to estimate the misleading rate before spending on more live lookups.
NOTE: the test-half replies are now SEEN (they drove this ticket). `supplement_replies_test_seen.jsonl` is
therefore development data from here on, not a held-out set. A new held-out check needs new rows.

## Files
- `pipeline/src/voice_pipeline/sufficiency.py` (`supplement_ok`, lead selection)
- `pipeline/src/voice_pipeline/temporal.py` (a helper for the reply's period vs the question's target period)
- `pipeline/src/voice_pipeline/agent_server.py` (use the lead that `supplement_ok` returns)
- `scripts/supplement_preassess.py` (new): the Qwen pre-assessment, see Part 3
- tests: `pipeline/tests/test_supplement_gates_271.py`, `tests/test_supplement_preassess.py`

## Part 1: yes/no and named answers
`supplement_ok` returns `(ok, why, lead)`. A reply with a figure keeps "The web has a newer figure: ". A reply
with no figure that is a direct yes/no (its first sentence, after an "As per X," attribution, starts with
Yes/No) or names a person/organisation the question asks for ("who/which" question) is KEPT with the neutral
lead "From the web: ". "Not yet released / unavailable / has already passed / varies / check your ..." with no
figure is still dropped. imp-096 must be kept; imp-009 and imp-043 must stay dropped.

## Part 2: period mismatch
Use `temporal.question_target_period(question, now)` and the reply's validity period (270's extraction). If the
question targets a later period (next year / next fall / next month / next semester) and the reply's period
ends before that target begins, or its only dates are past events ("began on", "took place", "was held"),
drop with why `period:wrong`. Must drop imp-012, imp-014, imp-016 and keep every current-period `ok` row.

## Part 3: Qwen pre-assessment (free, offline)
`scripts/supplement_preassess.py <fixture.jsonl>...`: for each reply, one Qwen JSON call (relay
127.0.0.1:18080, `chat_template_kwargs.enable_thinking=false`, same Relay class as `sufficiency_eval.py`):
`{"answers_asked_question": bool, "right_period": bool, "right_scope": bool, "states_not_yet_or_unavailable": bool}`
given the question, web question and reply only (no labels). It prints: (a) the judge's flags vs the hand
label per fixture (bad caught, ok wrongly flagged), (b) the gate result (`supplement_ok`) vs the hand label,
(c) the predicted misleading rate = labelled bad among replies that survive the gates, per fixture and
combined. It writes `out/sufficiency_eval/supplement-preassess.json`. The judge is a measurement, not a gate.

## Acceptance Criteria
- Fixture tests: on `supplement_replies_dev.jsonl` all 6 bad still dropped and at least 22 of 24 ok kept; on
  `supplement_replies_test_seen.jsonl` all `bad` rows dropped, imp-096 kept, at least 19 of the 22 `ok` rows
  kept, `unsure` ignored, hedges silent (compute the real counts from the file; do not hard-code a pass).
- Gate edge-case tests (yes/no kept with the neutral lead, "No data available" dropped, a future-period question
  with a current-period reply dropped, a future-period reply kept).
- Dev gold verdict eval on Qwen unchanged (needs-web 35/36, stale 15/15, conflict 7/7, controls 0/14).
- The pre-assessment runs on both fixtures against the live Qwen relay and prints the three blocks; report the
  numbers in the final report, plus whether the Qwen judge catches at least 6 of the 14 hand-`bad` rows with at
  most 3 false flags among the `ok` rows (a report, not a pass/fail gate).
- `uv run pytest` in pipeline: only the known `test_docs_wiring` failure; ruff clean.

## Non-Goals
- No paid calls, no Gemini. No change to the verdict prompt or the date fast path. The judge does not gate the
  supplement. Do not edit the fixture files. No tuning of anything on facts outside the two fixtures.
