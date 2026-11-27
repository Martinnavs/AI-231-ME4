# Ticket 270: gate the supplement sentence (figure, period, scope)

Branch: exp/sufficiency (worktree ~/AI-231-ME4-w250). `SUFFICIENCY` stays False. Not dispatched until the owner says so.

## Problem (measured, see docs/POTENTIAL-IMPROVEMENTS-PARKED.md, "P5 measured")
The real grounded lookup on the 34 dev web questions: 4 hedges (silent, fine), 24 plausible, 6 the supplement
would speak wrongly. `_maybe_supplement` (agent_server.py, the block that speaks `"The web has a newer
figure: " + text`) checks only the hedge and the repeat marker. The 6, hand-labelled in
`eval/sufficiency/supplement_replies_dev.jsonl` (label bad/ok/hedge, real replies, facts not externally verified):
- not a figure, spoken as "newer figure": imp-011, imp-015
- wrong scope: imp-004 (US General Schedule asked, Philippine grade answered), imp-013 (no place asked, Cebu answered)
- the reply is for a period that already ended: imp-001, imp-061 (FY2026, ended 2026-09-30; today 2026-11-27)

## Files
- `pipeline/src/voice_pipeline/sufficiency.py` (new pure function `supplement_ok(verdict, reply, now) -> (bool, why)`;
  the SYSTEM prompt rule for `web_question` scope)
- `pipeline/src/voice_pipeline/temporal.py` (reuse `source_period` / `stale_decision` on the reply text)
- `pipeline/src/voice_pipeline/agent_server.py` (call it in the supplement block; log `sufficiency_dropped=<why>`)
- tests: `pipeline/tests/test_supplement_gates_270.py`

## The three gates
1. **Figure gate** (verdict `stale`, lead "newer figure"): the reply must contain a number, date or named
   quantity and must not match "not yet (been )?(released|published|available)", "has (already )?(passed|ended|
   concluded)", "took place", "no (data|information)", "unavailable". Fail: say nothing (never speak the lead).
   For `missing_half` (lead "From the web:") apply only the "not yet / unavailable" part.
2. **Period gate**: run the reply text through the existing period extraction, but only for periods that say
   what the figure is VALID for: a fiscal-year or academic-year label, "fiscal year ending <date>", a calendar
   year used as the figure's period ("for 2026", "the 2026 rate"), "for the month of <Month Year>". An
   observation or origin date is NOT a validity period and is exempt: "as of <date>", "since <date>",
   "estimates from <date>", "reached ... as of June 2026". If the validity period ended before `now` and the
   question asks for the present or a later period, drop the supplement (the web answer is itself stale).
   (Orchestrator note after the first agy run: the first wording dropped imp-068, imp-092 and imp-024, which are
   correct "as of" answers.)
3. **Scope**: (a) prompt: add to the verdict SYSTEM prompt that `web_question` keeps every qualifier the user's
   question or the document answer names (country, agency, schedule name such as General Schedule / OPM) and adds
   none that neither names; give the verdict the document's source label so such a qualifier survives.
   (b) code, country level only: a fixed set of about 30 country names and demonyms (Philippines/Philippine,
   United States/U.S./US/American, United Kingdom/UK/British, ...) that appears in the reply but in none of the
   question, the web question and the DOCUMENT ANSWER drops the supplement. The check runs only when a
   document answer is present (always so in production: the supplement follows an accepted document answer).
   Fixture rows have `question` and `doc_answer`; an empty `doc_answer` (a gold-mode artifact: no lane answer)
   means the check is skipped for that row. Do not try place/institution detection beyond the country list.

## Acceptance Criteria
- The fixture test loads `supplement_replies_dev.jsonl` (real replies) and asserts, with `now` = 2026-11-27 Manila:
  all 6 `bad` rows dropped (imp-011, imp-015, imp-013 by gate 1, imp-001, imp-061 by gate 2, imp-004 by gate 3); at least 22 of the 24 `ok` rows kept; all 4 `hedge` rows still silent. Numeric
  thresholds in the test, not prose.
- Gate unit tests with hand-written edge cases (a figure with "not" elsewhere in the sentence is kept; a reply
  for the current FY is kept; "has not yet been released" dropped).
- Dev gold verdict eval on Qwen unchanged: needs-web 35/36, stale 15/15, conflict 7/7, controls 0/14.
- `uv run pytest` in pipeline: only the known `test_docs_wiring` failure; ruff clean.

## Non-Goals
- No recovery design, no URL reading, no provider change (that is the parked critique). No change to the
  hedge handling. No tuning on the test half: the 34 dev replies are the only ones the implementer may read.
- No paid calls inside the ticket. Afterwards the orchestrator may rerun the 34 dev lookups (about $0.36 worst
  case) to see whether the scope prompt changed the web questions; a held-out run on the test half's web
  questions needs a new spend approval (the $0.50 cap is nearly used up by the first run).
